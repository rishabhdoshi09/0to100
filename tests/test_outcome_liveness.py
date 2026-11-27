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


def test_failed_missed_exit_does_not_advance_past_the_stop(monkeypatch):
    brain = _DayBrain({
        "2026-09-02": {"AAA": (100, 101, 99, 100)},
        "2026-09-03": {"AAA": (99, 100, 94, 99)},
        "2026-09-04": {"AAA": (100, 101, 99, 100)},
    })
    book = brain.intel_book
    pos = book.open_position("S1", "AAA", 100, 95, 110, "2026-09-01", 20)
    assert pos is not None
    mark = book.mark

    def interrupted(bars, day, **kwargs):
        if day == "2026-09-03":
            raise RuntimeError("synthetic OMS interruption")
        return mark(bars, day, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(book, "mark", interrupted)
        incomplete = settle_paper_session(brain, "2026-09-04")
    assert incomplete["status"] == "SETTLEMENT_INCOMPLETE"
    assert "2026-09-03" in incomplete["error"]
    assert pos.last_marked_session == "2026-09-02"
    replayed = settle_paper_session(brain, "2026-09-04")
    assert replayed["status"] == "EOD_SETTLED"
    assert [row["exit_date"] for row in replayed["positions_closed"]] == ["2026-09-03"]


def test_verified_special_session_blocks_later_mark_until_archive_arrives():
    data = {"2026-09-07": {"AAA": (100, 101, 99, 100)}}
    brain = _DayBrain(data)
    book = brain.intel_book
    pos = book.open_position("S1", "AAA", 100, 95, 110, "2026-09-03", 20)
    assert pos is not None
    # This is a synthetic verified calendar: Friday is closed, Saturday trades.
    def required(day):
        return day.isoformat() in {"2026-09-05", "2026-09-07"}

    incomplete = settle_paper_session(brain, "2026-09-07", session_required=required)
    assert incomplete["status"] == "SETTLEMENT_INCOMPLETE"
    assert "2026-09-05" in incomplete["error"]
    assert not pos.last_marked_session and not book.closed
    data["2026-09-05"] = {"AAA": (100, 101, 94, 99)}
    result = settle_paper_session(brain, "2026-09-07", session_required=required)
    assert result["status"] == "EOD_SETTLED"
    assert [row["exit_date"] for row in result["positions_closed"]] == ["2026-09-05"]
    assert result["missed_sessions_marked"] == ["2026-09-05"]


def test_verified_calendar_rejects_nonminimum_invalid_cursor():
    brain = _DayBrain({"2026-09-04": {"AAA": (100, 101, 99, 100)}})
    book = brain.intel_book
    good = book.open_position("S1", "AAA", 100, 95, 110, "2026-09-02", 20)
    bad = book.open_position("S2", "BBB", 100, 95, 110, "2026-09-03", 20)
    assert good is not None and bad is not None
    bad.last_marked_session = "9999-13-01"
    result = settle_paper_session(brain, "2026-09-04", session_required=lambda day: True)
    assert result["status"] == "SETTLEMENT_INCOMPLETE"
    assert "Invalid paper session cursor" in result["error"]
    assert not good.last_marked_session


def test_eod_production_singleton_refuses_to_mark_without_protected_adapter(monkeypatch):
    from research.auto_research import providers, scheduler
    from research.autonomy.outcome_liveness import _resolve_outcomes_light

    brain = _DayBrain({"2026-09-04": {"AAA": (100, 101, 94, 100)}})
    book = brain.intel_book
    assert book.open_position("S1", "AAA", 100, 95, 110, "2026-09-03", 20)
    monkeypatch.setattr(scheduler, "_BRAIN", brain)
    monkeypatch.setattr(scheduler, "get_brain", lambda: brain)
    monkeypatch.setattr(providers, "ensure_production_paper_pipeline", lambda: False)
    result = _resolve_outcomes_light(None, "2026-09-04")
    assert result["status"] == "PAPER_PIPELINE_UNAVAILABLE"
    assert book.open and not book.closed


def test_eod_failed_book_save_is_not_reported_as_complete(monkeypatch):
    brain = _DayBrain({"2026-09-04": {"AAA": (100, 101, 94, 100)}})
    assert brain.intel_book.open_position("S1", "AAA", 100, 95, 110, "2026-09-03", 20)
    with monkeypatch.context() as patch:
        patch.setattr(brain, "_save_intel_book", lambda: False)
        result = settle_paper_session(brain, "2026-09-04")
    assert result["status"] == "SETTLEMENT_INCOMPLETE"
    assert "paper book save returned failure" in result["error"]
    assert result["outcomes_recorded"] == []
    recovered = settle_paper_session(brain, "2026-09-04")
    assert recovered["status"] == "EOD_SETTLED" and recovered["outcomes_recorded"] == ["S1"]
    assert len(brain.intel_book.closed) == 1


def test_partial_audit_projection_replays_after_book_and_event_store_restart(tmp_path, monkeypatch):
    import json
    from research.intelligence.event_store import EventStore

    brain = _DayBrain({"2026-09-04": {"AAA": (100, 101, 94, 100)}})
    book = brain.intel_book
    assert book.open_position("S1", "AAA", 100, 95, 110, "2026-09-03", 20)
    path = tmp_path / "events.jsonl"
    brain.event_store = EventStore(path)
    saved = {}
    monkeypatch.setattr(brain, "_save_intel_book", lambda: saved.update(json.loads(json.dumps(book.snapshot()))))
    append = brain.event_store.append

    def interrupted(record):
        if len(brain.event_store) == 1:
            raise OSError("audit store interrupted")
        return append(record)

    monkeypatch.setattr(brain.event_store, "append", interrupted)
    incomplete = settle_paper_session(brain, "2026-09-04")
    assert incomplete["status"] == "SETTLEMENT_INCOMPLETE"
    assert len(brain.event_store) == 1
    restored = PaperBook()
    assert restored.restore(saved, strict=True)
    assert restored.pending_eod_audit == [("2026-09-04", 0)]

    restarted = _DayBrain({})
    restarted.intel_book = restored
    restarted.event_store = EventStore(path)
    completed = settle_paper_session(restarted, "2026-09-05")
    assert completed["status"] == "EOD_SETTLED" and completed["outcomes_recorded"] == ["S1"]
    assert len(restarted.event_store) == 4 and len(restored.closed) == 1
    assert restored.realized_pnl == book.realized_pnl
    assert settle_paper_session(restarted, "2026-09-05")["outcomes_recorded"] == []
    assert len(restarted.event_store) == 4
    # No subsequent book save has removed the durable cursor yet. The
    # already-written event IDs must still suppress a replay after restart.
    assert restored.restore(saved, strict=True)
    restarted.event_store = EventStore(path)
    assert settle_paper_session(restarted, "2026-09-06")["status"] == "EOD_SETTLED"
    assert len(restarted.event_store) == 4 and len(restored.closed) == 1


def test_verified_special_dispatch_is_archive_only_and_pins_settlement_day(monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    from research.autonomy import jobs, schedules as sch
    from research.autonomy.supervisor import Supervisor

    special = datetime(2026, 11, 8, 20, 5)
    later = datetime(2026, 11, 9, 20, 5)
    state = {"ready": False}
    monkeypatch.setattr(sch, "verified_special_cash_session", lambda day, holidays: day == special.date())
    monkeypatch.setattr(sch, "special_archive_available", lambda day: state["ready"] and day == special.date())
    enqueued = []
    deps = SimpleNamespace(holidays=lambda: (), now_ist=lambda: later)
    supervisor = SimpleNamespace(
        deps=deps, _release_stale_official_blocks=lambda: None, _ensure_us_market_pipeline=lambda now: None,
        jobs=SimpleNamespace(enqueue=lambda kind, **details: enqueued.append((kind, details))),
    )
    Supervisor.enqueue_due(supervisor, special.replace(hour=14))
    assert not enqueued
    Supervisor.enqueue_due(supervisor, special)
    assert enqueued[-1] == (sch.BHAVCOPY_UPDATE, {"idempotency_key": "bhavcopy_update:2026-11-08:special:2026-11-08T20"})
    Supervisor._enqueue_special_settlement(supervisor, special.date(), later)
    assert enqueued[-1][1]["idempotency_key"] == "bhavcopy_update:2026-11-08:special:2026-11-09T20"
    state["ready"] = True
    Supervisor.enqueue_due(supervisor, special)
    assert enqueued[-1] == (sch.OUTCOME_RESOLUTION, {"idempotency_key": "forward_outcome:2026-11-08:special", "critical": True})

    settled = []
    deps.resolve_outcomes = lambda day, failures: settled.append(day) or {"status": "EOD_SETTLED"}
    monkeypatch.setattr("product.autonomous_loop.settle_official_outcomes", lambda day: {})
    ctx = jobs._Ctx(deps, job=SimpleNamespace(idempotency_key="forward_outcome:2026-11-08:special"))
    result = _run_outcome_resolution_light(ctx)
    assert result.status == JS.SUCCEEDED and settled == ["2026-11-08"]
    monkeypatch.setattr(sch, "verified_special_cash_session", lambda day, holidays: False)
    assert _run_outcome_resolution_light(ctx).status == JS.BLOCKED
    assert settled == ["2026-11-08"]


def test_special_archive_job_fetches_exact_day_and_requeues_blocked_settlement(monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    from data import bhavcopy_store
    from research.autonomy import jobs, schedules as sch
    from research.autonomy.supervisor import Supervisor

    now = datetime(2026, 11, 9, 1)
    day = datetime(2026, 11, 8).date()
    monkeypatch.setattr(sch, "verified_special_cash_session", lambda candidate, holidays: candidate == day)
    fetched = []
    def fetch(candidate, *, retries, verified_special):
        fetched.append((candidate, retries, verified_special))
        return True

    monkeypatch.setattr(bhavcopy_store, "_download_day", fetch)
    deps = SimpleNamespace(now_ist=lambda: now, holidays=lambda: ())
    job = SimpleNamespace(idempotency_key="bhavcopy_update:2026-11-08:special:2026-11-09T01")
    assert jobs.run_bhavcopy_update(jobs._Ctx(deps, job=job)).status == JS.SUCCEEDED
    assert fetched == [(day, 0, True)]

    pending = SimpleNamespace(status=JS.BLOCKED, job_id="special", blocked_on=jobs.DEP_OUTCOME_DATA)
    requeued = []
    supervisor = SimpleNamespace(jobs=SimpleNamespace(enqueue=lambda *a, **k: pending, requeue=requeued.append))
    monkeypatch.setattr(sch, "special_archive_available", lambda candidate: candidate == day)
    Supervisor._enqueue_special_settlement(supervisor, day, now)
    assert requeued == ["special"]


def test_eod_preserves_unrecovered_book(monkeypatch):
    brain = _DayBrain({"2026-09-04": {"AAA": (100, 101, 94, 100)}})
    position = brain.intel_book.open_position("S1", "AAA", 100, 95, 110, "2026-09-03", 20)
    assert position is not None
    brain.intel_book_recovery_error = "book checksum mismatch"

    def unexpected_save():
        raise AssertionError("EOD must preserve the existing persisted book")

    monkeypatch.setattr(brain, "_save_intel_book", unexpected_save)
    result = settle_paper_session(brain, "2026-09-04")
    assert result["status"] == "PAPER_BOOK_RECOVERY_FAILED"
    assert position.bars_held == 0 and not brain.intel_book.closed


def test_a_session_older_than_the_last_mark_is_never_applied():
    book = PaperBook(capital=100_000.0)
    book.open_position("S1", "AAA", 100.0, 95.0, 110.0, "2026-09-01", 20)
    book.mark({"AAA": (100.0, 101.0, 99.0, 100.5)}, "2026-09-07", allow_entry_session=False)
    closed = book.mark({"AAA": (99.0, 100.0, 90.0, 92.0)}, "2026-09-03", allow_entry_session=False)
    assert closed == []
    pos = next(iter(book.open.values()))
    assert pos.bars_held == 1 and pos.last_marked_session == "2026-09-07"


def _eod_ctx():
    class Deps:
        def now_ist(self):
            from datetime import datetime
            from zoneinfo import ZoneInfo
            return datetime(2026, 9, 7, 19, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

        def holidays(self):
            return set()

        def official_history(self):
            return {"current": True, "available_session": "2026-09-07"}

        def resolve_outcomes(self, *_args, **_kwargs):
            return {"status": "EOD_SETTLED", "positions_closed": [], "outcomes_recorded": []}

    class Ctx:
        deps = Deps()
        active_failures = set()

    return Ctx()


def test_production_outcome_handler_settles_signal_log(monkeypatch):
    """Root cause of the 2026-07-30 -> 2026-11-19 BUY-outcome stall.

    install_outcome_liveness replaces run_outcome_resolution, and only the
    replaced handler called update_outcomes, so no surfaced BUY was settled.
    """
    from research.autonomy import jobs as J
    from research.autonomy import outcome_liveness as OL
    from research.autonomy import schedules as SCH

    # install_outcome_liveness mutates module globals; restore them after.
    monkeypatch.setattr(J, "run_outcome_resolution", J.run_outcome_resolution)
    monkeypatch.setitem(J.HANDLERS, SCH.OUTCOME_RESOLUTION, J.HANDLERS[SCH.OUTCOME_RESOLUTION])
    monkeypatch.setattr(J.Deps, "resolve_outcomes", J.Deps.resolve_outcomes)
    monkeypatch.setattr(OL, "_INSTALLED", False)
    OL.install_outcome_liveness()
    handler = J.HANDLERS[SCH.OUTCOME_RESOLUTION]
    assert handler is _run_outcome_resolution_light
    calls = []
    monkeypatch.setattr("product.autonomous_loop.settle_official_outcomes",
                        lambda *_a, **_k: {"n_settled": 0})
    monkeypatch.setattr("data.bhavcopy_runtime.ensure_loaded", lambda **_k: {"ready": True})
    monkeypatch.setattr("data.bhavcopy_store.is_ready", lambda: True)
    monkeypatch.setattr(
        "core.signal_outcome_tracker.update_outcomes",
        lambda lookback_days=30: calls.append(lookback_days) or
        {"written": 7, "stuck_past_horizon": 0, "fatal_error": None, "write_error": None},
    )
    result = handler(_eod_ctx())
    assert result.status == JS.SUCCEEDED
    assert calls == [J.OUTCOME_LOOKBACK_DAYS]
    assert "signal_log 7 settled/0 stuck" in result.summary
    assert result.metadata["signal_outcomes"]["written"] == 7


def test_empty_official_store_is_reported_not_treated_as_open_buys(monkeypatch):
    from research.autonomy import jobs as J

    called = []
    monkeypatch.setattr("product.autonomous_loop.settle_official_outcomes",
                        lambda *_a, **_k: {"n_settled": 0})
    monkeypatch.setattr("data.bhavcopy_runtime.ensure_loaded", lambda **_k: {"ready": False})
    monkeypatch.setattr("data.bhavcopy_store.is_ready", lambda: False)
    monkeypatch.setattr("core.signal_outcome_tracker.update_outcomes",
                        lambda **_k: called.append(1) or {})
    result = _run_outcome_resolution_light(_eod_ctx())
    assert result.status == JS.SUCCEEDED          # paper settlement is the job's authority
    assert "official store not loaded" in result.summary
    assert result.metadata["signal_outcomes_error"] == "OFFICIAL_STORE_NOT_LOADED"
    assert called == []


def test_bars_held_counts_sessions_since_entry_not_marks_applied():
    """RPEL: entered 11 Nov, marked 12, 13, 18 -> bars_held 3 though 5 sessions passed."""
    from research.autonomy.outcome_liveness import reconcile_bars_held

    by_day = {d: {"AAA": (100.0, 101.0, 99.0, 100.5)} for d in
              ("2026-11-12", "2026-11-13", "2026-11-16", "2026-11-17", "2026-11-18")}
    book = PaperBook(capital=100_000.0)
    book.open_position("S1", "AAA", 100.0, 95.0, 110.0, "2026-11-11", 20)
    for d in ("2026-11-12", "2026-11-13", "2026-11-18"):        # the outage skipped 16-17
        book.mark(by_day[d], d, allow_entry_session=False)
    pos = next(iter(book.open.values()))
    assert pos.bars_held == 3
    fixes = reconcile_bars_held(book, lambda d: by_day.get(d, {}))
    assert fixes == [{"symbol": "AAA", "from": 3, "to": 5}]
    assert pos.bars_held == 5
    assert reconcile_bars_held(book, lambda d: by_day.get(d, {})) == []   # idempotent


def test_a_suspended_day_without_a_bar_is_not_a_session_for_that_symbol():
    from research.autonomy.outcome_liveness import reconcile_bars_held

    by_day = {"2026-11-12": {"AAA": (1, 1, 1, 1)}, "2026-11-13": {"BBB": (1, 1, 1, 1)},
              "2026-11-16": {"AAA": (1, 1, 1, 1)}}
    book = PaperBook(capital=100_000.0)
    book.open_position("S1", "AAA", 100.0, 95.0, 110.0, "2026-11-11", 20)
    pos = next(iter(book.open.values()))
    pos.last_marked_session, pos.bars_held = "2026-11-16", 1
    reconcile_bars_held(book, lambda d: by_day.get(d, {}))
    assert pos.bars_held == 2                                    # 12 and 16; AAA did not trade on 13
