"""
Event-driven trading engine — the orchestrator.

Per-cycle flow (every CYCLE_INTERVAL_SECONDS):
  1. Fetch live market data (Kite)
  2. Fetch & normalize news
  3. Compute technical indicators
  4. For each symbol in universe:
     a. Build LLM context packet
     b. Get DeepSeek signal
     c. Validate signal
     d. Risk check
     e. Execute if approved
     f. Update portfolio state
  5. Snapshot equity curve
  6. Log cycle summary

The engine is completely decoupled from execution mechanics.
It orchestrates; it does not execute.
"""

from __future__ import annotations

import signal as signal_module
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import settings
from data.kite_client import KiteClient
from data.historical import HistoricalDataFetcher
from data.instruments import InstrumentManager
from execution.zerodha_broker import ZerodhaBroker
from features.indicators import IndicatorEngine
from llm.context_builder import ContextBuilder
from llm.deepseek_client import DeepSeekClient
from llm.signal_validator import SignalValidator
from news.fetcher import NewsFetcher
from news.normalizer import NewsNormalizer
from news.summarizer import NewsSummarizer
from portfolio.state import PortfolioState
from risk.risk_manager import RiskManager
from logger import get_logger

log = get_logger(__name__)

_HISTORY_BARS = 250          # candles per symbol for indicator computation
_HISTORY_INTERVAL = "day"


class TradeEngine:
    def __init__(
        self,
        kite: KiteClient,
        portfolio: PortfolioState,
        risk_manager: RiskManager,
        broker: ZerodhaBroker,
        instruments: InstrumentManager,
        historical: HistoricalDataFetcher,
    ) -> None:
        self._kite = kite
        self._portfolio = portfolio
        self._risk = risk_manager
        self._broker = broker
        self._instruments = instruments
        self._historical = historical

        self._news_fetcher = NewsFetcher()
        self._news_normalizer = NewsNormalizer()
        self._news_summarizer = NewsSummarizer()
        self._indicators = IndicatorEngine()
        self._llm = DeepSeekClient()
        self._context = ContextBuilder()
        self._validator = SignalValidator()

        self._running = False
        self._cycle_count = 0

        # Handle SIGINT/SIGTERM gracefully
        signal_module.signal(signal_module.SIGINT, self._shutdown_handler)
        signal_module.signal(signal_module.SIGTERM, self._shutdown_handler)

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the live trading loop. Blocks until stopped."""
        log.info(
            "engine_starting",
            universe=settings.symbol_list,
            cycle_interval=settings.cycle_interval_seconds,
        )
        self._running = True
        while self._running:
            try:
                self._run_cycle()
            except Exception as exc:
                log.error("cycle_error", error=str(exc), exc_info=True)
            if self._running:
                log.info(
                    "cycle_sleeping",
                    seconds=settings.cycle_interval_seconds,
                )
                time.sleep(settings.cycle_interval_seconds)

    def run_once(self) -> List[Dict[str, Any]]:
        """Run a single decision cycle. Returns list of cycle decisions."""
        return self._run_cycle()

    def stop(self) -> None:
        log.info("engine_stopping")
        self._running = False

    # ── Cycle ──────────────────────────────────────────────────────────────

    def _run_cycle(self) -> List[Dict[str, Any]]:
        self._cycle_count += 1
        cycle_start = datetime.now(timezone.utc)
        log.info(
            "cycle_start",
            cycle=self._cycle_count,
            time=cycle_start.isoformat(),
        )

        # ── Step 1: Market data ───────────────────────────────────────────
        ltp = self._safe_fetch_ltp(settings.symbol_list)
        self._portfolio.update_prices(ltp)

        # ── Step 2: News ──────────────────────────────────────────────────
        raw_news = self._news_fetcher.fetch_all()
        norm_news = self._news_normalizer.normalize(raw_news)

        # ── Step 3: Per-symbol decision loop ─────────────────────────────
        cycle_decisions: List[Dict[str, Any]] = []

        for symbol in settings.symbol_list:
            if self._risk.is_kill_switch_active():
                log.critical("kill_switch_active_skipping_remaining_symbols")
                break

            decision = self._process_symbol(symbol, ltp, norm_news)
            if decision:
                cycle_decisions.append(decision)

        # ── Step 4: Equity snapshot ───────────────────────────────────────
        self._portfolio.record_equity_point()
        equity = self._portfolio.snapshot_equity()

        cycle_end = datetime.now(timezone.utc)
        elapsed = (cycle_end - cycle_start).total_seconds()
        log.info(
            "cycle_complete",
            cycle=self._cycle_count,
            elapsed_s=round(elapsed, 1),
            equity=round(equity, 2),
            decisions=len(cycle_decisions),
        )
        return cycle_decisions

    def _process_symbol(
        self,
        symbol: str,
        ltp: Dict[str, float],
        norm_news,
    ) -> Optional[Dict[str, Any]]:
        last_price = ltp.get(symbol)
        if last_price is None or last_price <= 0:
            log.warning("no_price_for_symbol", symbol=symbol)
            return None

        # ── Indicators ────────────────────────────────────────────────────
        token = self._instruments.token(symbol)
        if token is None:
            return None

        from datetime import date, timedelta
        today = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")

        df = self._historical.fetch(
            symbol=symbol,
            from_date=start,
            to_date=today,
            interval=_HISTORY_INTERVAL,
        )
        indicator_data = self._indicators.compute(df, symbol)

        # ── Market snapshot ───────────────────────────────────────────────
        quote = self._safe_get_quote(symbol)
        market_snapshot: Dict[str, Any] = {
            "symbol": symbol,
            "last_price": last_price,
            "ohlc": quote.get("ohlc"),
            "volume": quote.get("volume"),
            "oi": quote.get("oi"),
        }

        # ── News context ──────────────────────────────────────────────────
        news_block = self._news_summarizer.build_context_block(symbol, norm_news)

        # ── LLM context ───────────────────────────────────────────────────
        portfolio_state = self._portfolio.get_state_dict()
        risk_limits = self._risk.get_risk_limits_dict()

        context_prompt = self._context.build(
            symbol=symbol,
            market_snapshot=market_snapshot,
            indicators=indicator_data,
            news_block=news_block,
            portfolio_state=portfolio_state,
            risk_limits=risk_limits,
        )

        # ── LLM call ──────────────────────────────────────────────────────
        raw_signal = self._llm.get_signal(context_prompt)

        # ── Validation ────────────────────────────────────────────────────
        signal = self._validator.validate(raw_signal, symbol)

        # ── Risk check ────────────────────────────────────────────────────
        open_positions = self._portfolio.get_open_positions()
        portfolio_value = self._portfolio.snapshot_equity()

        risk_decision = self._risk.evaluate(
            signal=signal,
            portfolio_value=portfolio_value,
            open_positions=open_positions,
            last_price=last_price,
        )

        decision_record = {
            "symbol": symbol,
            "action": signal.action,
            "confidence": signal.confidence,
            "risk_approved": risk_decision.approved,
            "risk_reason": risk_decision.reason,
            "quantity": int(risk_decision.adjusted_size),
            "reasoning": signal.reasoning,
        }

        # ── Execute ───────────────────────────────────────────────────────
        if risk_decision.approved:
            order_result = self._broker.execute(risk_decision)
            decision_record["order_id"] = order_result.order_id
            decision_record["order_status"] = order_result.status
            decision_record["fill_price"] = order_result.fill_price

            if order_result.is_filled():
                if signal.action == "BUY":
                    self._portfolio.open_position(
                        symbol=symbol,
                        quantity=order_result.quantity,
                        price=order_result.fill_price,
                        order_id=order_result.order_id,
                        reasoning=signal.reasoning,
                        confidence=signal.confidence,
                    )
                elif signal.action == "SELL":
                    realized_pnl = self._portfolio.close_position(
                        symbol=symbol,
                        price=order_result.fill_price,
                        order_id=order_result.order_id,
                    )
                    self._risk.record_pnl(realized_pnl)
                    decision_record["realized_pnl"] = round(realized_pnl, 2)

        log.info("symbol_cycle_done", **decision_record)
        return decision_record

    # ── Helpers ────────────────────────────────────────────────────────────

    def _safe_fetch_ltp(self, symbols: List[str]) -> Dict[str, float]:
        try:
            return self._kite.get_ltp(symbols)
        except Exception as exc:
            log.error("ltp_fetch_failed", error=str(exc))
            return {}

    def _safe_get_quote(self, symbol: str) -> Dict[str, Any]:
        try:
            data = self._kite.get_quote([symbol])
            key = f"{settings.exchange}:{symbol}"
            return data.get(key, {})
        except Exception as exc:
            log.warning("quote_fetch_failed", symbol=symbol, error=str(exc))
            return {}

    def _shutdown_handler(self, signum, frame) -> None:
        log.warning("shutdown_signal_received", signum=signum)
        self.stop()
        # Cancel all open orders on shutdown
        try:
            self._broker.cancel_all_open_orders()
        except Exception as exc:
            log.error("cancel_orders_on_shutdown_failed", error=str(exc))
