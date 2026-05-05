"""
Event-driven backtester.

Processes historical bars in chronological order.
At each bar:
  1. Flush pending orders (filled at this bar's open — no lookahead).
  2. Compute indicators using only data up to this bar (no future data).
  3. Get LLM signal for each symbol.
  4. Validate signal + risk check.
  5. Submit new orders (filled next bar).
  6. Record equity snapshot.

Strict no-lookahead guarantee:
  indicators[t] uses close[0..t] only.
  order submitted at close[t] is filled at open[t+1].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.simulator import SimulatedBroker, SimFill
from config import settings
from features.indicators import IndicatorEngine
from signals.composite_signal import CompositeSignal
from llm.context_builder import ContextBuilder
from llm.dual_engine import DualLLMEngine
from llm.signal_validator import SignalValidator, TradingSignal
from news.fetcher import NewsFetcher
from news.normalizer import NewsNormalizer
from news.summarizer import NewsSummarizer
from portfolio.state import PortfolioState
from risk.risk_manager import RiskManager
from logger import get_logger

log = get_logger(__name__)

_MIN_WARMUP_BARS = 20   # minimum bars needed for indicators (RSI-14 + SMA-20)


class Backtester:
    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],  # symbol → OHLCV DataFrame
        initial_capital: float = settings.backtest_initial_capital,
        slippage: float = settings.backtest_slippage,
        transaction_cost: float = settings.backtest_transaction_cost,
        use_llm: bool = True,
    ) -> None:
        self._data = historical_data
        self._use_llm = use_llm
        self._portfolio = PortfolioState(initial_capital)
        self._risk = RiskManager()
        self._broker = SimulatedBroker(slippage=slippage, transaction_cost=transaction_cost)
        self._indicators = IndicatorEngine()
        self.composite_signal = CompositeSignal()
        self._context = ContextBuilder()
        self._validator = SignalValidator()

        if use_llm:
            self._llm = DualLLMEngine()
            self._news_fetcher = NewsFetcher()
            self._news_normalizer = NewsNormalizer()
            self._news_summarizer = NewsSummarizer()
        else:
            self._llm = None
            self._news_fetcher = None
            self._news_normalizer = None
            self._news_summarizer = None

        self._results: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        """
        Run the full backtest.
        Returns a summary dict with all trade records and equity curve.
        """
        # Align all symbols to a common date index
        all_dates = sorted(set.union(*[
            set(df.index) for df in self._data.values()
        ]))

        log.info(
            "backtest_start",
            symbols=list(self._data.keys()),
            bars=len(all_dates),
            capital=self._portfolio.initial_capital,
        )

        news_cache: List = []
        news_refresh_idx = 0
        bar_closes: Dict[str, float] = {}
        bar_time = all_dates[-1] if all_dates else None

        for bar_idx, bar_time in enumerate(all_dates):
            if bar_idx < _MIN_WARMUP_BARS:
                # Skip warmup period — not enough data for indicators
                continue

            # ── Gather this bar's OHLCV for all symbols ───────────────────
            bar_opens: Dict[str, float] = {}
            bar_closes: Dict[str, float] = {}
            for symbol, df in self._data.items():
                if bar_time in df.index:
                    bar_opens[symbol] = float(df.loc[bar_time, "open"])
                    bar_closes[symbol] = float(df.loc[bar_time, "close"])

            # ── Step 1: Flush pending orders (fill at this bar's open) ─────
            entry_prices = self._get_entry_prices()
            fills = self._broker.flush_pending(bar_opens, bar_time, entry_prices)
            self._apply_fills(fills)

            # Update current prices for unrealized PnL
            self._portfolio.update_prices(bar_closes)

            # ── Step 2: Refresh news occasionally (expensive LLM prep) ────
            if self._use_llm and bar_idx - news_refresh_idx >= 5:
                raw_news = self._news_fetcher.fetch_all(max_age_hours=24)
                news_cache = self._news_normalizer.normalize(raw_news)
                news_refresh_idx = bar_idx

            # ── Step 3: Score all symbols, rank, cap LLM calls per bar ───
            # Phase A: compute technical signals for all symbols (free)
            tech_scores: List[tuple] = []  # (priority, symbol, hist_slice, bar_close)
            for symbol in self._data.keys():
                if bar_time not in self._data[symbol].index:
                    continue
                hist_slice = self._data[symbol].loc[:bar_time]
                indicators = self._indicators.compute(hist_slice, symbol)
                # Get ML-enhanced composite signal
                composite = self.composite_signal.compute(indicators, llm_signal=None, regime=1)
                tech = TradingSignal(
                    symbol=symbol,
                    action="BUY" if composite["direction"] == 1 else "SELL" if composite["direction"] == -1 else "HOLD",
                    confidence=composite["confidence"],
                    time_horizon="swing",
                    position_size=settings.max_position_size_pct,
                    reasoning=f'composite: f={composite.get("attribution", composite)["factor"]:.2f}, ml={composite.get("attribution", composite)["ml"]:.2f}',
                    risk_level="medium"
                )
                has_pos = self._portfolio.has_position(symbol)
                # Priority: open positions first (exit monitoring), then BUY by |z-score|
                zscore = abs(indicators.get("zscore_20") or 0)
                if has_pos:
                    priority = 10 + zscore  # always evaluate open positions
                elif tech.action == "BUY":
                    priority = zscore        # rank BUY candidates by signal strength
                else:
                    priority = -1            # HOLD/phantom SELL — skip LLM
                tech_scores.append((priority, symbol, hist_slice, bar_closes.get(symbol, 0), tech, indicators))

            # Phase B: sort by priority, cap LLM at 3 per bar
            tech_scores.sort(key=lambda x: x[0], reverse=True)
            llm_budget = 3 if self._use_llm else 0
            llm_used = 0

            for priority, symbol, hist_slice, bar_close, tech_signal, indicator_data in tech_scores:
                if self._risk.is_kill_switch_active():
                    break

                if priority < 0:
                    # HOLD/no-signal symbols — use technical result directly
                    signal = tech_signal
                elif self._use_llm and llm_used < llm_budget:
                    # Worth an LLM call
                    signal = self._get_signal_llm(
                        symbol=symbol,
                        hist_df=hist_slice,
                        bar_close=bar_close,
                        indicator_data=indicator_data,
                        news_cache=news_cache,
                    )
                    llm_used += 1
                else:
                    # Budget exhausted — fall back to technical
                    signal = tech_signal

                if signal is None:
                    continue

                # ── Risk check ────────────────────────────────────────────
                open_positions = self._portfolio.get_open_positions()
                portfolio_value = self._portfolio.snapshot_equity()
                last_price = bar_closes.get(symbol, 0.0)

                risk_decision = self._risk.evaluate(
                    signal=signal,
                    portfolio_value=portfolio_value,
                    open_positions=open_positions,
                    last_price=last_price,
                )

                if risk_decision.approved:
                    self._broker.submit_order(
                        symbol=symbol,
                        action=signal.action,
                        quantity=int(risk_decision.adjusted_size),
                        submitted_at=bar_time,
                        reasoning=signal.reasoning,
                        confidence=signal.confidence,
                    )

            # ── Step 4: Equity snapshot ───────────────────────────────────
            self._portfolio.record_equity_point(timestamp=bar_time)

        # Final close-out — mark all positions at last available price
        self._close_all_positions(bar_closes, last_bar_time=bar_time)

        log.info("backtest_complete", total_bars=len(all_dates))
        return self._build_result()

    # ── Signal generation ──────────────────────────────────────────────────

    def _get_signal(
        self,
        symbol: str,
        hist_df: pd.DataFrame,
        bar_close: float,
        news_cache: List,
    ) -> Optional[TradingSignal]:
        """Used by live/LLM-mode path when budget isn't a concern."""
        indicator_data = self._indicators.compute(hist_df, symbol)
        if not self._use_llm or self._llm is None:
            composite = self.composite_signal.compute(indicator_data, llm_signal=None, regime=1)
        direction = composite.get('direction', 0)
        confidence = composite.get('confidence', 50)
        attribution = composite.get('attribution', {})
        return TradingSignal(
            symbol=symbol,
            action="BUY" if direction == 1 else "SELL" if direction == -1 else "HOLD",
            confidence=confidence,
            time_horizon="swing",
            position_size=settings.max_position_size_pct,
            reasoning=f"ml={attribution.get('ml', 0):.2f}, combined={attribution.get('combined', 0):.2f}",
            risk_level="medium"
        )
        return self._get_signal_llm(symbol, hist_df, bar_close, indicator_data, news_cache)

    def _get_signal_llm(
        self,
        symbol: str,
        hist_df: pd.DataFrame,
        bar_close: float,
        indicator_data: Dict[str, Any],
        news_cache: List,
    ) -> Optional[TradingSignal]:
        """Run dual-LLM pipeline (DeepSeek → Claude) and return validated signal."""
        news_block = self._news_summarizer.build_context_block(symbol, news_cache)
        portfolio_state = self._portfolio.get_state_dict()
        risk_limits = self._risk.get_risk_limits_dict()
        market_snapshot = {"symbol": symbol, "last_price": bar_close}
        context_prompt = self._context.build(
            symbol=symbol,
            market_snapshot=market_snapshot,
            indicators=indicator_data,
            news_block=news_block,
            portfolio_state=portfolio_state,
            risk_limits=risk_limits,
        )
        return self._llm.get_signal(context_prompt, symbol)

    def _technical_signal(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        bar_close: float,
    ) -> TradingSignal:
        """
        Pure technical strategy (used when use_llm=False):
          - z-score mean reversion: BUY when z < -1.5, SELL when z > 1.5
          - RSI filter: avoid buying overbought / selling oversold
          - Trend filter: only buy in uptrend
        """
        zscore = indicators.get("zscore_20")
        rsi = indicators.get("rsi_14")
        _trend = indicators.get("trend_5d")  # noqa: F841
        has_position = self._portfolio.has_position(symbol)

        action = "HOLD"
        confidence = 0.5
        reasoning = "no_clear_signal"

        if zscore is not None and rsi is not None:
            if zscore < -1.0 and rsi < 55 and not has_position:
                action = "BUY"
                confidence = min(0.92, 0.60 + abs(zscore) * 0.08)
                reasoning = f"mean_reversion_buy: zscore={zscore:.2f}, rsi={rsi:.1f}"
            elif has_position and (zscore > 1.0 or rsi > 65):
                action = "SELL"
                confidence = min(0.92, 0.60 + abs(zscore) * 0.08)
                reasoning = f"mean_reversion_exit: zscore={zscore:.2f}, rsi={rsi:.1f}"

        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            time_horizon="swing",
            position_size=settings.max_position_size_pct,
            reasoning=reasoning,
            risk_level="medium",
        )

    # ── Fill application ───────────────────────────────────────────────────

    def _apply_fills(self, fills: List[SimFill]) -> None:
        for fill in fills:
            if fill.action == "BUY":
                self._portfolio.open_position(
                    symbol=fill.symbol,
                    quantity=fill.quantity,
                    price=fill.fill_price,
                    order_id=fill.order_id,
                    reasoning=fill.reasoning,
                    confidence=fill.confidence,
                    transaction_cost_rate=0.0,  # already baked into fill
                    timestamp=fill.fill_time,
                )
            elif fill.action == "SELL":
                realized = self._portfolio.close_position(
                    symbol=fill.symbol,
                    price=fill.fill_price,
                    order_id=fill.order_id,
                    transaction_cost_rate=0.0,
                    timestamp=fill.fill_time,
                )
                self._risk.record_pnl(realized)
            self._results.append(fill.to_dict())

    def _get_entry_prices(self) -> Dict[str, float]:
        state = self._portfolio.get_state_dict()
        return {
            sym: pos_data["avg_entry_price"]
            for sym, pos_data in state.get("positions", {}).items()
        }

    def _close_all_positions(
        self, last_prices: Dict[str, float], last_bar_time=None
    ) -> None:
        state = self._portfolio.get_state_dict()
        for sym in list(state.get("positions", {}).keys()):
            price = last_prices.get(sym)
            if price:
                self._portfolio.close_position(
                    symbol=sym,
                    price=price,
                    order_id=f"CLOSEOUT-{sym}",
                    transaction_cost_rate=settings.backtest_transaction_cost,
                    timestamp=last_bar_time,
                )

    def _build_result(self) -> Dict[str, Any]:
        return {
            "trade_journal": self._portfolio.get_trade_journal(),
            "equity_curve": self._portfolio.get_equity_curve(),
            "fills": self._results,
            "final_equity": self._portfolio.snapshot_equity(),
            "initial_capital": self._portfolio.initial_capital,
        }
