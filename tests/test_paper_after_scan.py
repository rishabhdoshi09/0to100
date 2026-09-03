"""Successful shared scan enqueues the existing paper cycle."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy.supervisor import Supervisor


IST = ZoneInfo("Asia/Kolkata")


class _Jobs:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, job_type, **kwargs):
        self.enqueued.append((job_type, kwargs))
        return SimpleNamespace(status=JS.PENDING, job_type=job_type)


def test_successful_intraday_scan_enqueues_paper_cycle():
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 10, 45, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_paper_after_scan(SimpleNamespace(job_type=SCH.MARKET_SCAN))
    assert jobs.enqueued
    assert jobs.enqueued[0][0] == SCH.PAPER_CYCLE
    assert jobs.enqueued[0][1]["critical"] is True


def test_data_refresh_success_does_not_enqueue_paper():
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 10, 45, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_paper_after_scan(SimpleNamespace(job_type=SCH.DATA_REFRESH))
    assert jobs.enqueued == []


def test_paper_does_not_enqueue_before_entry_window():
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 8, 10, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_paper_after_scan(SimpleNamespace(job_type=SCH.MARKET_SCAN))
    assert jobs.enqueued == []


def test_off_session_after_close_enqueues_outcome_not_scan():
    jobs = _Jobs()
    jobs.cancel_superseded_pending = lambda *_a, **_k: None
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 23, 50, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_daily_foundation = lambda *_a, **_k: None
    supervisor.enqueue_due()
    types = [job_type for job_type, _kwargs in jobs.enqueued]
    assert SCH.OUTCOME_RESOLUTION in types
    assert SCH.MARKET_SCAN not in types
    assert SCH.PAPER_CYCLE not in types


def test_intraday_enqueue_does_not_start_post_market_grind():
    jobs = _Jobs()
    jobs.cancel_superseded_pending = lambda *_a, **_k: None
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 14, 10, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_daily_foundation = lambda *_a, **_k: None
    supervisor.enqueue_due()
    types = [job_type for job_type, _kwargs in jobs.enqueued]
    assert SCH.OUTCOME_RESOLUTION not in types


def test_blocked_data_ready_outcome_is_requeued():
    class Jobs:
        def __init__(self):
            self.requeued = []
            self.unblocked = []

        def enqueue(self, job_type, **kwargs):
            return SimpleNamespace(
                job_id="out-1", job_type=job_type, status=JS.BLOCKED,
                blocked_on="DATA_READY",
            )

        def requeue(self, job_id):
            self.requeued.append(job_id)

        def get(self, job_id):
            return SimpleNamespace(job_id=job_id, status=JS.PENDING, blocked_on="")

        def unblock_dependency(self, dep):
            self.unblocked.append(dep)

    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = Jobs()
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 3, 0, 5, tzinfo=IST),
        holidays=lambda: set(),
    )
    supervisor._enqueue_post_market_grind(session_date="2026-09-02")
    assert supervisor.jobs.requeued == ["out-1"]


def test_overnight_after_midnight_settles_previous_session():
    jobs = _Jobs()
    jobs.cancel_superseded_pending = lambda *_a, **_k: None
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 3, 0, 5, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_daily_foundation = lambda *_a, **_k: None
    supervisor.enqueue_due()
    outcomes = [kwargs for job_type, kwargs in jobs.enqueued if job_type == SCH.OUTCOME_RESOLUTION]
    assert outcomes
    assert outcomes[0]["idempotency_key"] == SCH.outcome_key("2026-09-02")
    types = [job_type for job_type, _kwargs in jobs.enqueued]
    assert SCH.MARKET_SCAN not in types


def test_eod_without_kite_snapshot_still_enqueues_official_work():
    jobs = _Jobs()
    jobs.cancel_superseded_pending = lambda *_a, **_k: None

    def enqueue(job_type, **kwargs):
        jobs.enqueued.append((job_type, kwargs))
        status = JS.BLOCKED if job_type == SCH.DATA_REFRESH else JS.SUCCEEDED
        return SimpleNamespace(status=status, job_type=job_type)

    jobs.enqueue = enqueue
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 18, 20, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: None,
    )
    supervisor._enqueue_daily_foundation = lambda *_a, **_k: None
    supervisor._enqueue_post_market_grind = Supervisor._enqueue_post_market_grind.__get__(supervisor)
    supervisor.enqueue_due()
    types = [job_type for job_type, _kwargs in jobs.enqueued]
    assert SCH.BHAVCOPY_UPDATE in types
    assert SCH.MARKET_SCAN in types
    assert SCH.OUTCOME_RESOLUTION in types
    assert SCH.DATA_REFRESH in types


def test_successful_data_refresh_enqueues_current_scan_slot():
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 10, 45, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_scan_after_refresh(SimpleNamespace(job_type=SCH.DATA_REFRESH))
    assert jobs.enqueued
    assert jobs.enqueued[0][0] == SCH.MARKET_SCAN
