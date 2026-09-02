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
