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

import uuid
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

import pickle
from pathlib import Path

import numpy as np

from backtest.simulator import SimulatedBroker, SimFill
from config import settings
from features.indicators import IndicatorEngine
from llm.context_builder import ContextBuilder
from llm.deepseek_client import DeepSeekClient
from llm.signal_validator import SignalValidator, TradingSignal
from news.fetcher import NewsFetcher
from news.normalizer import NewsNormalizer
from news.summarizer import NewsSummarizer
from portfolio.state import PortfolioState
from risk.risk_manager import RiskManager
from logger import get_logger

log = get_logger(__name__)

_MIN_WARMUP_BARS = 55   # need SMA-50 + regime (was 20)

_ATR_STOP_MULT = 2.0    # hard stop = entry - 2×ATR
_PROFIT_MULT   = 3.0    # profit target = entry + 3×ATR  (R:R = 1.5)
_MAX_HOLD_BARS = 20     # time stop: exit flat/losing position after N bars

# ── ML model path ─────────────────────────────────────────────────────────────
_ML_MODEL_PATHS = [
    Path("models/lgb_trading_model.pkl"),
    Path("../models/lgb_trading_model.pkl"),
    Path(__file__).parent.parent / "models" / "lgb_trading_model.pkl",
]


def _load_ml_model():
    """Load LightGBM model if it exists; return None otherwise."""
    for p in _ML_MODEL_PATHS:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    model = pickle.load(f)
                log.info("ml_model_loaded", path=str(p))
                return model
            except Exception as exc:
                log.warning("ml_model_load_failed", path=str(p), error=str(exc))
    log.info("ml_model_not_found_using_technical_only")
    return None


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
        self._context = ContextBuilder()
        self._validator = SignalValidator()

        if use_llm:
            self._llm = DeepSeekClient()
            self._news_fetcher = NewsFetcher()
            self._news_normalizer = NewsNormalizer()
            self._news_summarizer = NewsSummarizer()
        else:
            self._llm = None
            self._news_fetcher = None
            self._news_normalizer = None
            self._news_summarizer = None

        self._results: List[Dict[str, Any]] = []

        # ── Stop-loss / time-stop state ───────────────────────────────────
        # symbol → (stop_price, profit_target_price, entry_bar_idx, entry_atr)
        self._stops: Dict[str, tuple] = {}
        # Stops staged when a BUY order is submitted, applied on fill.
        self._pending_stops: Dict[str, tuple] = {}
        self._bar_idx: int = 0

        # ── ML model ─────────────────────────────────────────────────────
        self._ml_model = _load_ml_model()

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

        # Fetch news once at startup (historical backtest sees today's news anyway;
        # re-fetching every N bars adds HTTP latency with no accuracy gain)
        news_cache: List = []
        if self._use_llm and self._news_fetcher is not None:
            try:
                raw_news = self._news_fetcher.fetch_all(max_age_hours=24)
                news_cache = self._news_normalizer.normalize(raw_news)
                log.info("backtest_news_fetched", articles=len(news_cache))
            except Exception as exc:
                log.warning("backtest_news_fetch_failed", error=str(exc))

        bar_closes: Dict[str, float] = {}
        bar_time = all_dates[-1] if all_dates else None

        for bar_idx, bar_time in enumerate(all_dates):
            self._bar_idx = bar_idx
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

            # ── Step 1b: Check ATR stops and time stops ───────────────────
            self._check_stops(bar_closes, bar_time, bar_idx)

            # ── Step 3: Score all symbols, rank, cap LLM calls per bar ───
            # Phase A: compute technical signals for all symbols (free)
            tech_scores: List[tuple] = []  # (priority, symbol, hist_slice, bar_close)
            for symbol in self._data.keys():
                if bar_time not in self._data[symbol].index:
                    continue
                hist_slice = self._data[symbol].loc[:bar_time]
                indicators = self._indicators.compute(hist_slice, symbol)
                tech = self._technical_signal(symbol, indicators, bar_close=bar_closes.get(symbol, 0))
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

                from risk.risk_manager import RiskDecision
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
                    # Stage ATR stop for BUY orders
                    if signal.action == "BUY":
                        self._stage_stop(
                            symbol=symbol,
                            approx_entry=bar_closes.get(symbol, 0.0),
                            bar_idx=bar_idx,
                            indicators=indicator_data,
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
            return self._technical_signal(symbol, indicator_data, bar_close)
        return self._get_signal_llm(symbol, hist_df, bar_close, indicator_data, news_cache)

    def _get_signal_llm(
        self,
        symbol: str,
        hist_df: pd.DataFrame,
        bar_close: float,
        indicator_data: Dict[str, Any],
        news_cache: List,
    ) -> Optional[TradingSignal]:
        """Send to DeepSeek and validate response."""
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
        raw_signal = self._llm.get_signal(context_prompt)
        return self._validator.validate(raw_signal, symbol)

    def _technical_signal(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        bar_close: float,
    ) -> TradingSignal:
        """
        Composite signal (regime-filtered mean-reversion + optional ML model):

        Entry rules (ALL must pass):
          1. Regime ≥ 1  (bull or neutral — never buy in bear)
          2. z-score < -1.0  (oversold relative to 20-day mean)
          3. RSI 14 < 55  (not yet overbought)
          4. Not already long this symbol

        Exit rules (ANY triggers sell):
          • Stops handled separately in _check_stops (ATR + time + profit target)
          • z-score > 1.0  (mean reverted — take profit)
          • RSI > 68       (overbought exit)
          • Regime turned bear (risk-off exit)

        ML model (if loaded) adjusts confidence:
          • Predicts P(up) from the same indicators; blended 50/50 with technical.
          • If P(up) < 0.45, suppress the BUY even if technical says yes.
        """
        zscore    = indicators.get("zscore_20")
        rsi       = indicators.get("rsi_14")
        regime    = indicators.get("regime", 1)   # 0=bear 1=neutral 2=bull
        atr_pct   = indicators.get("atr_pct", 2.0)
        has_pos   = self._portfolio.has_position(symbol)

        action    = "HOLD"
        confidence = 0.5
        reasoning  = "no_clear_signal"

        # ── ML probability ─────────────────────────────────────────────
        ml_prob = self._ml_predict(indicators)

        # ── Exit checks (open positions only) ─────────────────────────
        if has_pos:
            if zscore is not None and zscore > 1.0:
                action = "SELL"
                confidence = min(0.90, 0.65 + zscore * 0.06)
                reasoning  = f"mean_reverted: zscore={zscore:.2f}"
            elif rsi is not None and rsi > 68:
                action = "SELL"
                confidence = 0.75
                reasoning  = f"rsi_overbought: rsi={rsi:.1f}"
            elif regime == 0:
                action = "SELL"
                confidence = 0.80
                reasoning  = "regime_turned_bear: risk_off_exit"
            return TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                time_horizon="swing",
                position_size=settings.max_position_size_pct,
                reasoning=reasoning,
                risk_level="medium",
            )

        # ── Entry checks (no open position) ───────────────────────────
        if zscore is None or rsi is None:
            return TradingSignal(
                symbol=symbol, action="HOLD", confidence=0.5,
                time_horizon="swing", position_size=0,
                reasoning="insufficient_indicators", risk_level="low",
            )

        # Gate 1: regime filter
        if regime == 0:
            return TradingSignal(
                symbol=symbol, action="HOLD", confidence=0.5,
                time_horizon="swing", position_size=0,
                reasoning=f"bear_regime_filter: regime={regime}", risk_level="low",
            )

        # Gate 2: mean-reversion entry
        if zscore < -1.0 and rsi < 55:
            tech_confidence = min(0.88, 0.60 + abs(zscore) * 0.08)

            # Gate 3: ML model veto
            if ml_prob is not None and ml_prob < 0.45:
                return TradingSignal(
                    symbol=symbol, action="HOLD", confidence=0.5,
                    time_horizon="swing", position_size=0,
                    reasoning=f"ml_veto: p_up={ml_prob:.2f}", risk_level="low",
                )

            # Blend ML confidence
            if ml_prob is not None:
                blended_conf = 0.5 * tech_confidence + 0.5 * ml_prob
            else:
                blended_conf = tech_confidence

            # Scale position by ATR volatility (smaller size in volatile stocks)
            vol_scalar = min(1.0, 2.0 / max(atr_pct, 0.5))
            pos_size = settings.max_position_size_pct * vol_scalar

            action     = "BUY"
            confidence = round(blended_conf, 3)
            reasoning  = (
                f"regime={regime} zscore={zscore:.2f} rsi={rsi:.1f} "
                f"ml_p={ml_prob:.2f if ml_prob else 'N/A'} atr_pct={atr_pct:.2f}"
            )
            return TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                time_horizon="swing",
                position_size=pos_size,
                reasoning=reasoning,
                risk_level="medium",
            )

        return TradingSignal(
            symbol=symbol, action="HOLD", confidence=0.5,
            time_horizon="swing", position_size=0,
            reasoning="no_entry_condition", risk_level="low",
        )

    # ── ML prediction ──────────────────────────────────────────────────────

    def _ml_predict(self, indicators: Dict[str, Any]) -> Optional[float]:
        """
        Returns P(up next bar) ∈ [0,1] or None if model unavailable.
        Builds feature vector matching the trained model's feature_names_in_.
        Unknown features default to 0 with a warning (once).
        """
        if self._ml_model is None:
            return None

        try:
            feat_names = list(getattr(self._ml_model, "feature_names_in_", []))
            if not feat_names:
                # Try booster feature names (raw lgb.Booster)
                feat_names = getattr(self._ml_model, "feature_name_", [])

            if not feat_names:
                return None

            row = []
            for name in feat_names:
                val = indicators.get(name, 0.0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                row.append(float(val))

            X = np.array(row).reshape(1, -1)
            proba = self._ml_model.predict_proba(X)[0]
            # sklearn classifiers: proba[1] = P(class=1=up)
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as exc:
            log.debug("ml_predict_failed", error=str(exc))
            return None

    # ── ATR stop / time stop / profit target ──────────────────────────────

    def _stage_stop(
        self,
        symbol: str,
        approx_entry: float,
        bar_idx: int,
        indicators: Dict[str, Any],
    ) -> None:
        """
        Stage stop levels when a BUY order is submitted.
        Actual registration happens in _apply_fills when the order is filled.
        approx_entry is the current bar close (actual fill will be next open).
        """
        atr    = indicators.get("atr_14") or (approx_entry * 0.02)
        stop   = approx_entry - _ATR_STOP_MULT * atr
        target = approx_entry + _PROFIT_MULT   * atr
        self._pending_stops[symbol] = (stop, target, bar_idx, atr)
        log.debug(
            "stop_staged",
            symbol=symbol,
            approx_entry=round(approx_entry, 2),
            stop=round(stop, 2),
            target=round(target, 2),
        )

    def _check_stops(
        self,
        bar_closes: Dict[str, float],
        bar_time,
        bar_idx: int,
    ) -> None:
        """
        Submit SELL orders for any position that has:
          • Breached the ATR hard stop
          • Hit the profit target
          • Been held > _MAX_HOLD_BARS with no profit
        """
        for sym, (stop, target, entry_bar, _atr) in list(self._stops.items()):
            if not self._portfolio.has_position(sym):
                del self._stops[sym]
                continue

            price = bar_closes.get(sym)
            if price is None:
                continue

            reason = None
            if price <= stop:
                reason = f"atr_stop: price={price:.2f} <= stop={stop:.2f}"
            elif price >= target:
                reason = f"profit_target: price={price:.2f} >= target={target:.2f}"
            else:
                bars_held = bar_idx - entry_bar
                if bars_held >= _MAX_HOLD_BARS:
                    state = self._portfolio.get_state_dict()
                    pos = state.get("positions", {}).get(sym, {})
                    entry_avg = pos.get("avg_entry_price", price)
                    if price <= entry_avg:
                        reason = f"time_stop: held={bars_held} bars, flat/losing"

            if reason:
                qty = self._portfolio.get_state_dict().get("positions", {}).get(sym, {}).get("quantity", 0)
                if qty > 0:
                    self._broker.submit_order(
                        symbol=sym,
                        action="SELL",
                        quantity=qty,
                        submitted_at=bar_time,
                        reasoning=reason,
                        confidence=0.99,
                    )
                    log.info("stop_triggered", symbol=sym, reason=reason)
                del self._stops[sym]

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
                # Register ATR stop for this new position.
                # We don't have fresh indicator data at fill time, so we use
                # the cached stop data set during signal generation (stored
                # via _pending_stops keyed by symbol).
                pending = self._pending_stops.pop(fill.symbol, None)
                if pending:
                    stop, target, entry_bar, atr = pending
                    self._stops[fill.symbol] = (stop, target, entry_bar, atr)
                else:
                    # Fallback: 2% hard stop
                    rough_atr = fill.fill_price * 0.02
                    self._stops[fill.symbol] = (
                        fill.fill_price - _ATR_STOP_MULT * rough_atr,
                        fill.fill_price + _PROFIT_MULT  * rough_atr,
                        self._bar_idx,
                        rough_atr,
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
                # Clean up any pending stop for this symbol
                self._stops.pop(fill.symbol, None)
                self._pending_stops.pop(fill.symbol, None)
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
