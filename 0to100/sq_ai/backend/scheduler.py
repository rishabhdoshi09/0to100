"""APScheduler – orchestrates the four jobs of the dual-LLM cockpit.

* 08:00 IST daily   → ``universe.refresh_universe`` (Kite instrument master)
* 0/30 min          → ``Screener.run``               (DeepSeek pre-filter)
* every 5 min       → ``DecisionEngine.run``         (Claude + ensemble veto)
* 23:00 IST daily   → overnight summary screener     (kept for nightly cron)
"""
from __future__ import annotations

import os
from datetime import datetime, time
from pathlib import Path
from typing import Any, Callable

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from sq_ai.backend.decision import DecisionEngine
from sq_ai.backend.screener import Screener, load_config
from sq_ai.backend.universe import refresh_universe
from sq_ai.portfolio.tracker import PortfolioTracker
from sq_ai.signals.composite_signal import CompositeSignal


IST = pytz.timezone(os.environ.get("SQ_TIMEZONE", "Asia/Kolkata"))
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def is_market_hours(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


# ─────────────────────────────────────────────────────────────────────────────
class TradingScheduler:
    """Wraps APScheduler.  Holds shared singletons."""

    def __init__(self, on_cycle: Callable[[dict], None] | None = None) -> None:
        # ── config ───────────────────────────────────────────────────────
        cfg_path = os.environ.get("SQ_CONFIG_PATH",
                                  str(Path(__file__).resolve().parents[2] / "config.yaml"))
        self.cfg = load_config(cfg_path)

        # ── core services ────────────────────────────────────────────────
        self.tracker = PortfolioTracker()
        self.composite = CompositeSignal()
        self.screener = Screener(tracker=self.tracker, config=self.cfg)
        self.decision_engine = DecisionEngine(
            tracker=self.tracker,
            composite=self.composite,
            ensemble_threshold_pct=float(self.cfg.get("ensemble_threshold_percent", 10.0)),
        )

        self._scheduler = BackgroundScheduler(timezone=IST)
        self._last_cycle: dict[str, Any] = {}
        self._last_screener: dict[str, Any] = {}
        self._on_cycle = on_cycle

        initial = float(os.environ.get("SQ_INITIAL_EQUITY", 1_000_000))
        if not self.tracker.equity_curve():
            self.tracker.record_equity(initial, initial)

    # ----------------------------------------------------------------- API
    def start(self) -> None:
        # 1. daily universe refresh – 08:00 IST
        refresh_hour = int(self.cfg.get("universe_refresh_hour", 8))
        self._scheduler.add_job(
            self._refresh_universe,
            trigger=CronTrigger(hour=refresh_hour, minute=0, timezone=IST),
            id="universe_refresh", replace_existing=True,
        )
        # 2. 30-min DeepSeek screener (gated by market hours inside)
        screener_min = int(self.cfg.get("screener_interval_minutes", 30))
        self._scheduler.add_job(
            self.run_screener,
            trigger=IntervalTrigger(minutes=screener_min),
            id="screener", replace_existing=True,
        )
        # 3. 5-min Claude decision cycle
        decision_min = int(self.cfg.get("decision_interval_minutes", 5))
        self._scheduler.add_job(
            self.run_cycle,
            trigger=IntervalTrigger(minutes=decision_min),
            id="cycle", replace_existing=True,
        )
        # 4. 23:00 IST nightly screener (legacy summary)
        self._scheduler.add_job(
            self.run_screener,
            trigger=CronTrigger(hour=23, minute=0, timezone=IST),
            id="screener_nightly", replace_existing=True,
        )

        # one-shot warm-up: refresh universe immediately so the screener has data
        self._scheduler.add_job(self._refresh_universe, id="universe_warmup")

        self._scheduler.start()

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    # ---------------------------------------------------------- universe
    def _refresh_universe(self) -> dict[str, Any]:
        n = refresh_universe(self.tracker)
        return {"refreshed": n, "ts": datetime.now(IST).isoformat()}

    # ---------------------------------------------------------- screener
    def run_screener(self) -> dict[str, Any]:
        out = self.screener.run()
        self._last_screener = out
        return out

    # ---------------------------------------------------------- decision
    def run_cycle(self) -> dict[str, Any]:
        if not is_market_hours():
            cycle = {"timestamp": datetime.now(IST).isoformat(),
                     "market_hours": False,
                     "note": "outside market hours – skipping"}
            self._last_cycle = cycle
            return cycle
        cycle = self.decision_engine.run()
        cycle["market_hours"] = True
        self._last_cycle = cycle
        if self._on_cycle:
            try:
                self._on_cycle(cycle)
            except Exception:                              # pragma: no cover
                pass
        return cycle

    # ----------------------------------------------------------- views
    @property
    def last_cycle(self) -> dict:
        return self._last_cycle

    @property
    def last_screener(self) -> dict:
        return self._last_screener

    # ----------------------------------------------------------- legacy alias
    @property
    def executor(self):
        return self.decision_engine.executor

    def _snapshot(self, last_prices: dict[str, float]) -> dict[str, Any]:
        return self.decision_engine._snapshot(last_prices)
