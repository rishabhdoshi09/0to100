"""APScheduler – 5-min decision loop + 23:00 IST overnight screener."""
from __future__ import annotations

import os
from datetime import datetime, time
from typing import Any, Callable

import pandas as pd
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from sq_ai.backend.claude_client import ClaudeClient
from sq_ai.backend.data_fetcher import (
    DEFAULT_WATCHLIST,
    KiteFetcher,
    fetch_news,
    fetch_yf_history,
)
from sq_ai.backend.executor import Executor, Order
from sq_ai.backend.risk_manager import RiskManager
from sq_ai.portfolio.tracker import PortfolioTracker
from sq_ai.signals.composite_signal import CompositeSignal


IST = pytz.timezone(os.environ.get("SQ_TIMEZONE", "Asia/Kolkata"))
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def is_market_hours(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:                       # Sat/Sun
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


# ---------------------------------------------------------------------------
class TradingScheduler:
    """Wraps APScheduler.  Holds shared singletons (composite, claude,
    executor, tracker)."""

    def __init__(
        self,
        watchlist: list[str] | None = None,
        on_cycle: Callable[[dict], None] | None = None,
    ) -> None:
        self.watchlist = watchlist or DEFAULT_WATCHLIST
        self.tracker = PortfolioTracker()
        self.composite = CompositeSignal()
        self.claude = ClaudeClient()
        self.kite = KiteFetcher()
        self.risk = RiskManager()
        self.executor = Executor(self.tracker)
        self._scheduler = BackgroundScheduler(timezone=IST)
        self._last_cycle: dict[str, Any] = {}
        self._on_cycle = on_cycle

        initial = float(os.environ.get("SQ_INITIAL_EQUITY", 1_000_000))
        if not self.tracker.equity_curve():
            self.tracker.record_equity(initial, initial)

    # ----------------------------------------------------------------- API
    def start(self) -> None:
        self._scheduler.add_job(
            self.run_cycle, trigger="interval", minutes=5,
            id="cycle", replace_existing=True,
        )
        self._scheduler.add_job(
            self.run_screener,
            trigger=CronTrigger(hour=23, minute=0, timezone=IST),
            id="screener", replace_existing=True,
        )
        self._scheduler.start()

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    @property
    def last_cycle(self) -> dict:
        return self._last_cycle

    # ------------------------------------------------------------- one cycle
    def run_cycle(self) -> dict:
        ts = datetime.now(IST)
        cycle: dict[str, Any] = {"timestamp": ts.isoformat(), "decisions": [],
                                 "events": [], "market_hours": is_market_hours(ts)}
        if not cycle["market_hours"]:
            cycle["note"] = "outside market hours – skipping"
            self._last_cycle = cycle
            return cycle

        # 1. fetch features per symbol (yfinance fallback)
        features_per_sym, last_prices = [], {}
        composite_results: dict[str, dict] = {}
        for sym in self.watchlist:
            try:
                df = fetch_yf_history(sym, period="6mo", interval="1d")
                if len(df) < 60:
                    continue
                feats = self.composite.compute_indicators(df)
                feats["symbol"] = sym
                # add ml proba for the brief (informational)
                if self.composite.ml.loaded:
                    order = self.composite.ml.feature_names or self.composite.DEFAULT_FEATURES
                    x = [float(feats.get(n, 1 if n == "regime" else 0.0)) for n in order]
                    feats["ml_proba_up"] = self.composite.ml.predict_proba_up(x)
                else:
                    feats["ml_proba_up"] = 0.5
                features_per_sym.append(feats)
                last_prices[sym] = feats["close"]
                composite_results[sym] = self.composite.compute(feats)
            except Exception as exc:                       # pragma: no cover
                cycle["events"].append({"symbol": sym, "error": str(exc)})

        # 2. exits before new entries
        exits = self.executor.check_exits(last_prices)
        cycle["events"].extend(exits)

        # 3. portfolio snapshot
        snap = self._snapshot(last_prices)

        # 4. ask Claude (or fallback)
        news = fetch_news("Indian stock market", top_n=3)
        decisions: dict[str, Any]
        brief = self.claude.build_brief(features_per_sym, news, snap)
        cycle["brief_size"] = len(brief)

        if self.claude.available:
            resp = self.claude.decide(brief)
            decisions = resp or self.claude.fallback_decisions(features_per_sym, composite_results)
            cycle["used_claude"] = bool(resp)
        else:
            decisions = self.claude.fallback_decisions(features_per_sym, composite_results)
            cycle["used_claude"] = False

        # 5. execute
        for d in decisions.get("decisions", []):
            sym = d["symbol"]
            self.tracker.log_signal(
                symbol=sym, action=d["action"],
                confidence=float(d.get("confidence", 0.0)),
                regime=int(next((f["regime"] for f in features_per_sym if f["symbol"] == sym), 1)),
                reasoning=d.get("reasoning", ""),
                extra={"size_pct": d.get("size_pct", 0.0)},
            )
            if d["action"] != "BUY":
                continue
            price = last_prices.get(sym)
            if price is None:
                continue
            atr_v = next((f["atr"] for f in features_per_sym if f["symbol"] == sym), 0.0)
            qty = self.risk.position_size(
                equity=snap["equity"], price=price, atr_value=atr_v,
                confidence=float(d.get("confidence", 0.0)),
            )
            if qty <= 0:
                continue
            stop = float(d.get("stop") or (price - 2 * atr_v))
            target = float(d.get("target") or (price + 3 * atr_v))
            res = self.executor.buy(Order(sym, "BUY", qty, price, stop, target))
            cycle["decisions"].append({**d, "execution": res})

        cycle["snapshot"] = snap
        self._last_cycle = cycle
        if self._on_cycle:
            try:
                self._on_cycle(cycle)
            except Exception:                              # pragma: no cover
                pass
        return cycle

    # ------------------------------------------------------------- screener
    def run_screener(self) -> dict:
        ts = datetime.now(IST).date().isoformat()
        ranked: list[dict] = []
        from sq_ai.backend.data_fetcher import nse500_symbols
        for sym in nse500_symbols():
            try:
                df = fetch_yf_history(sym, period="3mo", interval="1d")
                if len(df) < 60:
                    continue
                feats = self.composite.compute_indicators(df)
                # Simple rank score: regime weight + RSI proximity-to-50 + volume
                score = (feats["regime"] - 1) * 0.5 + (50 - abs(feats["rsi"] - 50)) / 100
                score += min(feats["volume_trend"] - 1, 1.0) * 0.2
                ranked.append({"symbol": sym, "score": float(score),
                               "reasoning": f"regime={feats['regime']} rsi={feats['rsi']:.1f}"})
            except Exception:
                continue
        ranked.sort(key=lambda r: r["score"], reverse=True)
        top10 = ranked[:10]
        self.tracker.save_screener(ts, top10)
        return {"date": ts, "top10": top10}

    # ------------------------------------------------------------- snapshot
    def _snapshot(self, last_prices: dict[str, float]) -> dict[str, Any]:
        positions = self.tracker.open_positions()
        equity_curve = self.tracker.equity_curve()
        cash = equity_curve[-1]["cash"] if equity_curve else float(
            os.environ.get("SQ_INITIAL_EQUITY", 1_000_000)
        )
        gross = 0.0
        unrealized = 0.0
        for p in positions:
            ltp = last_prices.get(p["symbol"], p["entry_price"])
            gross += ltp * p["qty"]
            unrealized += (ltp - p["entry_price"]) * p["qty"]
        equity = cash + gross
        equity_open = equity_curve[-1]["equity"] if equity_curve else equity
        return {
            "cash": cash,
            "equity": equity,
            "gross_exposure": gross,
            "exposure_pct": (gross / equity * 100) if equity else 0.0,
            "unrealized_pnl": unrealized,
            "daily_pnl_pct": ((equity - equity_open) / equity_open * 100) if equity_open else 0.0,
            "positions": positions,
        }
