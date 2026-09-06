from __future__ import annotations

import inspect
import threading

from research.auto_research.paper_book import PaperBook
from research.autonomy import job_store as JS
from research.autonomy.outcome_liveness import (
    _run_outcome_resolution_light,
    settle_paper_session,
)


class _RuntimeState:
    def __init__(self):
        self.reconciled = 0
        self.saved = 0

    def reconcile(self, _book):
        self.reconciled += 1
        return True

    def save(self):
        self.saved += 1


class _Brain:
    def __init__(self, bars):
        self.intel_book = PaperBook(capital=100_000.0)
        self._intel_lock = threading.Lock()
        self.bars_fn = lambda _day: dict(bars)
        self.runtime_state = _RuntimeState()
        self.event_store = []
        self.saved = 0
        self.full_cycles = 0

    def _save_intel_book(self):
        self.saved += 1

    def run_intelligence_cycle_day(self, *args, **kwargs):
        self.full_cycles += 1
        raise AssertionError("EOD settlement must not run the full intelligence cycle")


def test_eod_settlement_marks_once_without_full_intelligence_cycle():
    brain = _Brain({"AAA": (101.0, 94.0, 99.0)})
    opened = brain.intel_book.open_position(
        "S1", "AAA", 100.0, 95.0, 110.0, "2026-09-03", 5,
    )
    assert opened is not None

    first = settle_paper_session(brain, "2026-09-04")
    assert first["status"] == "EOD_SETTLED"
    assert first["full_intelligence_cycle"] is False
    assert first["historical_backtest"] is False
    assert len(first["positions_closed"]) == 1
    assert first["positions_closed"][0]["exit_reason"] == "STOP"
    assert brain.full_cycles == 0
    assert brain.saved == 1
    assert brain.runtime_state.reconciled == 1

    second = settle_paper_session(brain, "2026-09-04")
    assert second["status"] == "EOD_SETTLED"
    assert second["positions_closed"] == []
    assert brain.full_cycles == 0


def test_eod_settlement_fails_closed_when_open_positions_have_no_bars():
    brain = _Brain({})
    opened = brain.intel_book.open_position(
        "S1", "AAA", 100.0, 95.0, 110.0, "2026-09-03", 5,
    )
    assert opened is not None

    result = settle_paper_session(brain, "2026-09-04")
    assert result["status"] == "BARS_UNAVAILABLE"
    assert len(brain.intel_book.open) == 1
    assert brain.intel_book.open[("S1", "AAA")].bars_held == 0
    assert brain.full_cycles == 0


def test_eod_settlement_returns_busy_instead_of_overlapping_mutation():
    brain = _Brain({"AAA": (101.0, 99.0, 100.0)})
    assert brain._intel_lock.acquire(blocking=False)
    try:
        result = settle_paper_session(brain, "2026-09-04")
    finally:
        brain._intel_lock.release()
    assert result["status"] == "SKIPPED_LOCKED"
    assert brain.full_cycles == 0


def test_outcome_handler_contract_does_not_call_full_autonomous_loop():
    source = inspect.getsource(_run_outcome_resolution_light)
    assert "advance_loop(" not in source
    assert "run_intelligence_cycle_day" not in source
    assert "settle_official_outcomes" in source


def test_busy_paper_settlement_is_retryable_not_success(monkeypatch):
    class Deps:
        def now_ist(self):
            from datetime import datetime
            from zoneinfo import ZoneInfo
            return datetime(2026, 9, 6, 9, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

        def holidays(self):
            return set()

        def official_history(self):
            return {"current": True, "available_session": "2026-09-04"}

        def resolve_outcomes(self, *_args, **_kwargs):
            return {
                "status": "SKIPPED_LOCKED",
                "warnings": ["Paper mutation lock is busy; supervisor will retry."],
            }

    class Ctx:
        deps = Deps()
        active_failures = set()

    result = _run_outcome_resolution_light(Ctx())
    assert result.status == JS.RETRYABLE_FAILED
    assert result.error_code == "PAPER_SETTLEMENT_BUSY"
    assert not result.unblocks
