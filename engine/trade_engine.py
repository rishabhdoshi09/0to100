"""
Trade Engine — the live trading orchestrator.

Per-cycle flow (every CYCLE_INTERVAL_SECONDS):
  1. Fetch live market data (Kite)
  2. Fetch & normalize news (RSS + Marketaux)
  3. VADER sentiment scoring per symbol
  4. For each symbol:
     a. Compute technical indicators
     b. Build LLM context
     c. DeepSeek V3 → fast first-pass signal
     d. DeepSeek R1 validates / overrides (via DualLLMEngine)
     e. Devil's Advocate challenge (R1 challenges confirmed signals)
     f. Memory-based confidence adjustment (feedback from past trades)
     g. Agent Supervisor cross-validation (if enabled)
     h. Risk gate
     i. Execute if approved
     j. Record signal to memory DB
  5. Snapshot equity
  6. Log cycle summary

Invariants:
  - LLM never executes trades — JSON signals only.
  - Risk manager is the final gate.
  - Kill switch is immediate.
  - Every signal is stored in signal_memory.db for the feedback loop.
"""
from __future__ import annotations

import signal as signal_module
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config import settings
from data.kite_client import KiteClient
from data.historical import HistoricalDataFetcher
from data.instruments import InstrumentManager
from execution.zerodha_broker import ZerodhaBroker
from features.indicators import IndicatorEngine
from llm.context_builder import ContextBuilder
from llm.dual_engine import DualLLMEngine
from llm.signal_validator import SignalValidator
from news.fetcher import NewsFetcher
from news.normalizer import NewsNormalizer
from news.summarizer import NewsSummarizer
from portfolio.state import PortfolioState
from risk.risk_manager import RiskManager
from logger import get_logger

log = get_logger(__name__)

# ── Optional features — graceful degradation ──────────────────────────────────

try:
    from news.semantic_index import SemanticNewsIndex
    _semantic_available = True
except Exception:
    _semantic_available = False

try:
    from news.vader_scorer import batch_score as _vader_batch_score
    _vader_available = True
except Exception:
    _vader_available = False

try:
    from ai.mem0_store import get_memory as _get_memory
    _memory_available = True
except Exception:
    _memory_available = False

try:
    from agents.supervisor import AgentSupervisor
    _supervisor_available = True
except Exception:
    _supervisor_available = False

try:
    from llm.devil_advocate import challenge_signal as _challenge_signal
    _devil_available = True
except Exception:
    _devil_available = False

try:
    from ai.signal_memory import record_signal, query_similar, update_outcome, SignalRecord
    _signal_memory_available = True
except Exception:
    _signal_memory_available = False

_HISTORY_BARS     = 250
_HISTORY_INTERVAL = "day"

# Devil's advocate: only challenge signals above this confidence (saves R1 API cost)
_DEVIL_CONFIDENCE_THRESHOLD = 0.55


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
        self._kite        = kite
        self._portfolio   = portfolio
        self._risk        = risk_manager
        self._broker      = broker
        self._instruments = instruments
        self._historical  = historical

        self._news_fetcher    = NewsFetcher()
        self._news_normalizer = NewsNormalizer()
        self._news_summarizer = NewsSummarizer()
        self._indicators      = IndicatorEngine()
        self._llm             = DualLLMEngine()
        self._context         = ContextBuilder()
        self._validator       = SignalValidator()

        self._running     = False
        self._cycle_count = 0

        # Tracks open signal_memory IDs for P&L feedback on close
        self._pending_memory_ids: Dict[str, int] = {}

        # ── Optional AI features ──────────────────────────────────────────
        self._semantic_idx = None
        if _semantic_available:
            try:
                self._semantic_idx = SemanticNewsIndex()
                log.info("semantic_news_index_ready")
            except Exception as exc:
                log.warning("semantic_news_index_unavailable", error=str(exc))

        self._memory = None
        if _memory_available:
            try:
                self._memory = _get_memory()
                log.info("persistent_memory_ready")
            except Exception as exc:
                log.warning("persistent_memory_unavailable", error=str(exc))

        self._agent_supervisor = None
        if _supervisor_available and settings.enable_agent_supervisor:
            try:
                self._agent_supervisor = AgentSupervisor()
                log.info("agent_supervisor_ready")
            except Exception as exc:
                log.warning("agent_supervisor_unavailable", error=str(exc))

        signal_module.signal(signal_module.SIGINT,  self._shutdown_handler)
        signal_module.signal(signal_module.SIGTERM, self._shutdown_handler)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the live trading loop. Blocks until stopped."""
        log.info("engine_starting", universe=settings.symbol_list,
                 cycle_interval=settings.cycle_interval_seconds)
        self._running = True
        while self._running:
            try:
                self._run_cycle()
            except Exception as exc:
                log.error("cycle_error", error=str(exc), exc_info=True)
            if self._running:
                log.info("cycle_sleeping", seconds=settings.cycle_interval_seconds)
                time.sleep(settings.cycle_interval_seconds)

    def run_once(self) -> List[Dict[str, Any]]:
        return self._run_cycle()

    def stop(self) -> None:
        log.info("engine_stopping")
        self._running = False

    # ── Cycle ──────────────────────────────────────────────────────────────────

    def _run_cycle(self) -> List[Dict[str, Any]]:
        self._cycle_count += 1
        cycle_start = datetime.now(timezone.utc)
        log.info("cycle_start", cycle=self._cycle_count, time=cycle_start.isoformat())

        # 1. Live prices
        ltp = self._safe_fetch_ltp(settings.symbol_list)
        self._portfolio.update_prices(ltp)

        # 2. News pipeline
        raw_news = self._news_fetcher.fetch_all()
        norm_news = self._news_normalizer.normalize(raw_news)

        if self._semantic_idx is not None:
            try:
                self._semantic_idx.index(raw_news)
            except Exception as exc:
                log.warning("semantic_index_failed", error=str(exc))

        # 3. VADER per-symbol sentiment
        vader_score_map = self._build_vader_map(raw_news)

        # 4. Per-symbol decisions
        cycle_decisions: List[Dict[str, Any]] = []
        for symbol in settings.symbol_list:
            if self._risk.is_kill_switch_active():
                log.critical("kill_switch_active_halting_cycle")
                break
            decision = self._process_symbol(symbol, ltp, norm_news, vader_score_map)
            if decision:
                cycle_decisions.append(decision)

        # 5. Equity snapshot
        self._portfolio.record_equity_point()
        equity = self._portfolio.snapshot_equity()

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        log.info("cycle_complete", cycle=self._cycle_count,
                 elapsed_s=round(elapsed, 1), equity=round(equity, 2),
                 decisions=len(cycle_decisions))
        return cycle_decisions

    # ── Per-symbol decision pipeline ───────────────────────────────────────────

    def _process_symbol(
        self,
        symbol: str,
        ltp: Dict[str, float],
        norm_news,
        vader_score_map: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        last_price = ltp.get(symbol)
        if not last_price or last_price <= 0:
            log.warning("no_price_for_symbol", symbol=symbol)
            return None

        token = self._instruments.token(symbol)
        if token is None:
            return None

        # ── Indicators ────────────────────────────────────────────────────────
        today = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        df = self._historical.fetch(symbol=symbol, from_date=start,
                                    to_date=today, interval=_HISTORY_INTERVAL)
        indicator_data = self._indicators.compute(df, symbol)

        # ── Market snapshot ───────────────────────────────────────────────────
        quote = self._safe_get_quote(symbol)
        market_snapshot: Dict[str, Any] = {
            "symbol": symbol, "last_price": last_price,
            "ohlc": quote.get("ohlc"), "volume": quote.get("volume"),
        }

        # ── News context ──────────────────────────────────────────────────────
        vader_avg = vader_score_map.get(symbol)
        news_block = self._build_news_block(symbol, norm_news, vader_avg)

        # ── Persistent memory context ─────────────────────────────────────────
        memory_context = self._get_memory_context(symbol)

        # ── Build LLM context ─────────────────────────────────────────────────
        context_prompt = self._context.build(
            symbol=symbol,
            market_snapshot=market_snapshot,
            indicators=indicator_data,
            news_block=news_block,
            portfolio_state=self._portfolio.get_state_dict(),
            risk_limits=self._risk.get_risk_limits_dict(),
            vader_sentiment=vader_avg,
            memory_context=memory_context,
        )

        # ── Step A: DeepSeek V3 → R1 signal ─────────────────────────────────
        signal = self._llm.get_signal(context_prompt, symbol)

        # ── Step B: Memory feedback — adjust confidence from past outcomes ────
        memory_adj = 0.0
        if _signal_memory_available and signal.action in ("BUY", "SELL"):
            try:
                insight = query_similar(symbol, signal.action, indicator_data)
                memory_adj = insight.confidence_adjustment
                signal.confidence = max(0.0, min(1.0, signal.confidence + memory_adj))
                if abs(memory_adj) > 0.01:
                    signal.reasoning = (
                        f"[Memory: {insight.summary}] {signal.reasoning}"
                    )
                    log.info("memory_adjustment_applied", symbol=symbol,
                             adj=memory_adj, win_rate=insight.win_rate)
            except Exception as exc:
                log.warning("memory_query_failed", symbol=symbol, error=str(exc))

        # ── Step C: Devil's Advocate (R1 challenges strong signals) ──────────
        devil_result = None
        if (
            _devil_available
            and signal.action in ("BUY", "SELL")
            and signal.confidence >= _DEVIL_CONFIDENCE_THRESHOLD
        ):
            try:
                devil_result = _challenge_signal(
                    symbol=symbol,
                    action=signal.action,
                    confidence=signal.confidence,
                    context={
                        "indicators": indicator_data,
                        "news_summary": news_block[:500],
                        "fundamentals": {},
                    },
                    reasoning=signal.reasoning,
                )
                if not devil_result.survived:
                    log.warning("devil_advocate_rejected_signal",
                                symbol=symbol, concern=devil_result.strongest_concern)
                    signal.action = "HOLD"
                    signal.confidence *= (1 - devil_result.penalty)
                    signal.reasoning = (
                        f"[Devil's Advocate REJECTED: {devil_result.strongest_concern}] "
                        f"{signal.reasoning}"
                    )
                elif devil_result.penalty > 0:
                    original_conf = signal.confidence
                    signal.confidence *= (1 - devil_result.penalty)
                    signal.reasoning = (
                        f"[Devil's Advocate penalised -{devil_result.penalty:.0%}: "
                        f"{devil_result.strongest_concern}] {signal.reasoning}"
                    )
                    log.info("devil_advocate_penalised", symbol=symbol,
                             original=original_conf, new=signal.confidence,
                             concern=devil_result.strongest_concern[:60])
            except Exception as exc:
                log.warning("devil_advocate_failed", symbol=symbol, error=str(exc))

        # ── Step D: Agent Supervisor cross-validation ─────────────────────────
        if self._agent_supervisor is not None and signal.action in ("BUY", "SELL"):
            try:
                sv = self._agent_supervisor.evaluate_stock(symbol)
                sv_action = str(sv.get("action", "HOLD")).upper()
                if sv.get("risk_override"):
                    signal.confidence *= 0.4
                    signal.reasoning = f"[Supervisor RISK_OVERRIDE] {signal.reasoning}"
                    log.warning("supervisor_risk_override", symbol=symbol)
                elif sv_action == signal.action:
                    signal.confidence = min(signal.confidence * 1.05, 1.0)
                else:
                    signal.confidence *= 0.65
                    signal.reasoning = (
                        f"[Supervisor CONTRADICTION:{sv_action}] {signal.reasoning}"
                    )
            except Exception as exc:
                log.warning("supervisor_failed", symbol=symbol, error=str(exc))

        # ── Step E: Record signal to memory DB ───────────────────────────────
        mem_id: Optional[int] = None
        if _signal_memory_available and signal.action in ("BUY", "SELL"):
            try:
                mem_id = record_signal(SignalRecord(
                    symbol=symbol,
                    action=signal.action,
                    confidence=signal.confidence,
                    indicators=indicator_data,
                ))
            except Exception as exc:
                log.warning("signal_record_failed", symbol=symbol, error=str(exc))

        # ── Step F: Risk gate ─────────────────────────────────────────────────
        risk_decision = self._risk.evaluate(
            signal=signal,
            portfolio_value=self._portfolio.snapshot_equity(),
            open_positions=self._portfolio.get_open_positions(),
            last_price=last_price,
        )

        decision_record: Dict[str, Any] = {
            "symbol":           symbol,
            "action":           signal.action,
            "confidence":       round(signal.confidence, 4),
            "risk_approved":    risk_decision.approved,
            "risk_reason":      risk_decision.reason,
            "quantity":         int(risk_decision.adjusted_size),
            "reasoning":        signal.reasoning,
            "llm_decision_maker": signal.llm_decision_maker,
            "vader_sentiment":  vader_avg,
            "memory_adj":       round(memory_adj, 4),
            "devil_survived":   devil_result.survived if devil_result else None,
        }

        # ── Step G: Execute ───────────────────────────────────────────────────
        if risk_decision.approved:
            order_result = self._broker.execute(risk_decision)
            decision_record.update({
                "order_id":     order_result.order_id,
                "order_status": order_result.status,
                "fill_price":   order_result.fill_price,
            })

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
                    if mem_id:
                        self._pending_memory_ids[symbol] = mem_id

                elif signal.action == "SELL":
                    realized_pnl = self._portfolio.close_position(
                        symbol=symbol,
                        price=order_result.fill_price,
                        order_id=order_result.order_id,
                    )
                    self._risk.record_pnl(realized_pnl)
                    decision_record["realized_pnl"] = round(realized_pnl, 2)

                    # ── Feedback loop: record outcome to memory DB ────────────
                    pending_id = self._pending_memory_ids.pop(symbol, None)
                    if _signal_memory_available and pending_id:
                        try:
                            entry_price = self._portfolio.get_entry_price(symbol) or last_price
                            pnl_pct = (order_result.fill_price / entry_price - 1) * 100
                            outcome = "WIN" if pnl_pct > 0 else "LOSS"
                            update_outcome(pending_id, outcome, round(pnl_pct, 4))
                            log.info("feedback_loop_updated", symbol=symbol,
                                     outcome=outcome, pnl_pct=round(pnl_pct, 2))
                        except Exception as exc:
                            log.warning("feedback_update_failed",
                                        symbol=symbol, error=str(exc))

        log.info("symbol_cycle_done", **decision_record)
        return decision_record

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_vader_map(self, raw_news) -> Dict[str, float]:
        if not _vader_available or not raw_news:
            return {}
        try:
            scored = _vader_batch_score(
                [{"headline": a.headline, "summary": getattr(a, "summary", "")}
                 for a in raw_news]
            )
            result: Dict[str, float] = {}
            for symbol in settings.symbol_list:
                sym_l = symbol.lower()
                relevant = [
                    s["vader_score"] for s in scored
                    if sym_l in s.get("headline", "").lower()
                    or sym_l in s.get("summary", "").lower()
                ]
                if relevant:
                    result[symbol] = round(sum(relevant) / len(relevant), 4)
            return result
        except Exception as exc:
            log.warning("vader_scoring_failed", error=str(exc))
            return {}

    def _build_news_block(
        self, symbol: str, norm_news, vader_avg: Optional[float]
    ) -> str:
        if self._semantic_idx is not None:
            try:
                sem_articles = self._semantic_idx.search(symbol, top_k=8)
                return self._semantic_news_block(symbol, sem_articles, vader_avg)
            except Exception:
                pass
        return self._news_summarizer.build_context_block(symbol, norm_news)

    def _get_memory_context(self, symbol: str) -> str:
        if self._memory is None:
            return ""
        try:
            mems = self._memory.search(symbol, limit=3)
            return "\n".join(f"- {m['content']}" for m in mems) if mems else ""
        except Exception:
            return ""

    @staticmethod
    def _semantic_news_block(
        symbol: str, articles: List[Dict[str, Any]], vader_avg: Optional[float]
    ) -> str:
        if not articles:
            return f"No recent news for {symbol}."
        lines = [f"Top {len(articles)} relevant news for {symbol}:"]
        for a in articles:
            score = a.get("vader_score")
            s_str = f" [VADER: {score:+.2f}]" if score is not None else ""
            lines.append(f"• {a.get('title', a.get('headline', ''))}{s_str}")
        if vader_avg is not None:
            label = "BULLISH" if vader_avg > 0.05 else ("BEARISH" if vader_avg < -0.05 else "NEUTRAL")
            lines.append(f"\nAggregate sentiment: {vader_avg:+.3f} ({label})")
        return "\n".join(lines)

    def _safe_fetch_ltp(self, symbols: List[str]) -> Dict[str, float]:
        try:
            return self._kite.get_ltp(symbols)
        except Exception as exc:
            log.error("ltp_fetch_failed", error=str(exc))
            return {}

    def _safe_get_quote(self, symbol: str) -> Dict[str, Any]:
        try:
            data = self._kite.get_quote([symbol])
            return data.get(f"{settings.exchange}:{symbol}", {})
        except Exception as exc:
            log.warning("quote_fetch_failed", symbol=symbol, error=str(exc))
            return {}

    def _shutdown_handler(self, signum, frame) -> None:
        log.warning("shutdown_signal_received", signum=signum)
        self.stop()
        try:
            self._broker.cancel_all_open_orders()
        except Exception as exc:
            log.error("cancel_orders_on_shutdown_failed", error=str(exc))
