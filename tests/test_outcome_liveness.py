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


def test_entry_day_full_daily_bar_cannot_retroactively_close_new_position():
    # This bar crosses both stop and target. Because the paper position was opened
    # on this same session, none of this session's full-day OHLC may be used to exit it.
    brain = _Brain({"AAA": (100.0, 120.0, 90.0, 105.0)})
    opened = brain.intel_book.open_position(
        "S1", "AAA", 100.0, 95.0, 110.0, "2026-09-04", 5,
    )
    assert opened is not None

    result = settle_paper_session(brain, "2026-09-04")
    assert result["status"] == "EOD_SETTLED"
    assert result["positions_closed"] == []
    assert ("S1", "AAA") in brain.intel_book.open
    assert brain.intel_book.open[("S1", "AAA")].bars_held == 0
    assert brain.intel_book.open[("S1", "AAA")].last_marked_session == ""


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


class _NoOwnerOms:
    def list_orders(self):
        return []


class _Pipeline:
    oms = _NoOwnerOms()


def _adapter_brain(bars):
    from execution.paper_book_adapter import InstitutionalPaperBookAdapter

    brain = _Brain(bars)
    brain.intel_book = InstitutionalPaperBookAdapter(
        brain.intel_book, pipeline=_Pipeline(), runtime_state=brain.runtime_state,
    )
    return brain


def test_eod_settlement_through_production_institutional_adapter():
    # Production wraps intel_book in InstitutionalPaperBookAdapter (providers.current_regime).
    brain = _adapter_brain({"AAA": (101.0, 94.0, 99.0)})
    assert brain.intel_book.open_position("S1", "AAA", 100.0, 95.0, 110.0, "2026-09-03", 5)

    result = settle_paper_session(brain, "2026-09-04")
    assert result["status"] == "EOD_SETTLED"
    assert [row["exit_reason"] for row in result["positions_closed"]] == ["STOP"]


def test_institutional_adapter_keeps_entry_session_bar_out_of_exits():
    brain = _adapter_brain({"AAA": (100.0, 120.0, 90.0, 105.0)})
    assert brain.intel_book.open_position("S1", "AAA", 100.0, 95.0, 110.0, "2026-09-04", 5)

    result = settle_paper_session(brain, "2026-09-04")
    assert result["status"] == "EOD_SETTLED"
    assert result["positions_closed"] == []
    assert ("S1", "AAA") in brain.intel_book.open


class _DayBrain(_Brain):
    """Exact-session bars per day; a missing day is a holiday (no bhavcopy)."""

    def __init__(self, by_day):
        super().__init__({})
        self.bars_fn = lambda day: dict(by_day.get(str(day)[:10], {}))


def test_stop_touched_only_on_a_missed_session_closes_on_that_session():
    """Owner's Mac 16-18 Nov 2026: sessions skipped by an outage were never marked."""
    brain = _DayBrain({
        "2026-09-02": {"AAA": (100.0, 101.0, 99.0, 100.5)},
        "2026-09-03": {"AAA": (99.0, 100.0, 94.0, 99.5)},   # breaches the 95 stop, recovers
        "2026-09-04": {"AAA": (100.0, 103.0, 99.0, 102.0)},
        "2026-09-07": {"AAA": (102.0, 104.0, 101.0, 103.0)},
    })
    assert brain.intel_book.open_position("S1", "AAA", 100.0, 95.0, 110.0, "2026-09-01", 20) is not None
    # The host was down 2-4 Sep; the first settlement after recovery is for 7 Sep.
    out = settle_paper_session(brain, "2026-09-07")
    assert out["status"] == "EOD_SETTLED"
    assert len(out["positions_closed"]) == 1
    trade = out["positions_closed"][0]
    assert trade["exit_reason"] == "STOP"
    assert str(trade.get("exit_date") or "")[:10] == "2026-09-03"
    assert out["missed_sessions_marked"] == ["2026-09-02", "2026-09-03", "2026-09-04"]
    assert not brain.intel_book.open


def test_missed_sessions_age_the_position_and_holidays_are_skipped():
    brain = _DayBrain({
        "2026-09-02": {"AAA": (100.0, 101.0, 99.0, 100.5)},
        # 2026-09-03: exchange holiday, no bhavcopy
        "2026-09-04": {"AAA": (100.0, 101.0, 99.0, 100.5)},
        "2026-09-07": {"AAA": (100.0, 101.0, 99.0, 100.5)},
    })
    brain.intel_book.open_position("S1", "AAA", 100.0, 95.0, 110.0, "2026-09-01", 20)
    out = settle_paper_session(brain, "2026-09-07")
    assert out["missed_sessions_marked"] == ["2026-09-02", "2026-09-04"]
    pos = next(iter(brain.intel_book.open.values()))
    assert pos.bars_held == 3          # 2 Sep, 4 Sep, 7 Sep — not the holiday
    assert pos.last_marked_session == "2026-09-07"
    again = settle_paper_session(brain, "2026-09-07")
    assert again["missed_sessions_marked"] == []
    assert next(iter(brain.intel_book.open.values())).bars_held == 3


def test_a_session_older_than_the_last_mark_is_never_applied():
    book = PaperBook(capital=100_000.0)
    book.open_position("S1", "AAA", 100.0, 95.0, 110.0, "2026-09-01", 20)
    book.mark({"AAA": (100.0, 101.0, 99.0, 100.5)}, "2026-09-07", allow_entry_session=False)
    closed = book.mark({"AAA": (99.0, 100.0, 90.0, 92.0)}, "2026-09-03", allow_entry_session=False)
    assert closed == []
    pos = next(iter(book.open.values()))
    assert pos.bars_held == 1 and pos.last_marked_session == "2026-09-07"
