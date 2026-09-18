"""Automatic scheduler is one data->scan->paper transaction and then stops."""
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
    supervisor._enqueue_paper_after_scan(SimpleNamespace(
        job_type=SCH.MARKET_SCAN,
        idempotency_key=SCH.snapshot_scan_key("snap-1"),
        input_snapshot_id="snap-1",
    ))
    assert jobs.enqueued
    assert jobs.enqueued[0][0] == SCH.PAPER_CYCLE
    assert jobs.enqueued[0][1]["critical"] is True


def test_manual_scan_does_not_auto_enqueue_paper():
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 10, 45, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_paper_after_scan(SimpleNamespace(
        job_type=SCH.MARKET_SCAN,
        idempotency_key="manual:scan:snap-1:control-1",
        input_snapshot_id="snap-1",
    ))
    assert jobs.enqueued == []


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
    supervisor._enqueue_paper_after_scan(SimpleNamespace(
        job_type=SCH.MARKET_SCAN,
        idempotency_key=SCH.snapshot_scan_key("snap-1"),
        input_snapshot_id="snap-1",
    ))
    assert jobs.enqueued == []


def test_off_session_after_close_enqueues_historical_paper(monkeypatch):
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 23, 50, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_post_market_grind = lambda *_a, **_k: None
    monkeypatch.setattr(
        "product.historical_paper_loop.pending_stage",
        lambda: {"phase": "IDLE", "batch_id": ""},
    )
    monkeypatch.setattr(
        "product.historical_paper_loop.peek_next_batch",
        lambda: {"available": True, "batch_id": "hist-1"},
    )
    supervisor.enqueue_due()
    assert jobs.enqueued == [
        (
            SCH.HISTORICAL_PAPER_CYCLE,
            {
                "idempotency_key": SCH.historical_paper_key("hist-1"),
                "input_snapshot_id": "hist-1",
            },
        )
    ]


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


def test_blocked_data_ready_outcome_is_requeued(monkeypatch):
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
    monkeypatch.setattr(
        "product.readiness.official_history",
        lambda: {"current": True, "available_session": "2026-09-02", "latest_date": "2026-09-02"},
    )
    supervisor._enqueue_post_market_grind(session_date="2026-09-02")
    assert supervisor.jobs.requeued == ["out-1"]


def test_overnight_after_midnight_continues_historical_learning(monkeypatch):
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 3, 0, 5, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_post_market_grind = lambda *_a, **_k: None
    monkeypatch.setattr(
        "product.historical_paper_loop.pending_stage",
        lambda: {"phase": "AWAITING_LEARNING", "batch_id": "hist-2"},
    )
    supervisor.enqueue_due()
    assert jobs.enqueued == [
        (
            SCH.LEARNING_CYCLE,
            {
                "idempotency_key": SCH.historical_learning_key("hist-2"),
                "input_snapshot_id": "hist-2",
            },
        )
    ]


def test_eod_refreshes_official_data_without_starting_live_scan_pipeline(monkeypatch):
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 18, 20, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: None,
    )
    supervisor._enqueue_post_market_grind = lambda *_a, **_k: None
    monkeypatch.setattr(
        "product.historical_paper_loop.pending_stage",
        lambda: {"phase": "IDLE", "batch_id": ""},
    )
    monkeypatch.setattr(
        "product.historical_paper_loop.peek_next_batch",
        lambda: {"available": False, "reason": "caught_up"},
    )
    supervisor.enqueue_due()
    types = [job_type for job_type, _kwargs in jobs.enqueued]
    assert types == [SCH.BHAVCOPY_UPDATE, SCH.DATA_REFRESH]
    assert SCH.MARKET_SCAN not in types
    assert SCH.PAPER_CYCLE not in types


def test_successful_data_refresh_enqueues_current_scan_slot():
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 10, 45, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_scan_after_refresh(SimpleNamespace(
        job_type=SCH.DATA_REFRESH,
        idempotency_key=SCH.data_refresh_key("2026-09-01"),
        output_snapshot_id="snap-1",
    ))
    assert jobs.enqueued
    assert jobs.enqueued[0][0] == SCH.MARKET_SCAN


def test_manual_data_refresh_does_not_auto_enqueue_scan():
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 1, 10, 45, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_scan_after_refresh(SimpleNamespace(
        job_type=SCH.DATA_REFRESH,
        idempotency_key="manual:data:2026-09-01",
        output_snapshot_id="snap-1",
    ))
    assert jobs.enqueued == []



def test_restart_reconciles_finished_historical_poll_before_learning(monkeypatch):
    class Jobs(_Jobs):
        def __init__(self):
            super().__init__()
            self.completed = []
            self.poll = SimpleNamespace(
                job_id="hist-poll-1",
                status=JS.PENDING,
                job_type=SCH.HISTORICAL_PAPER_CYCLE,
                idempotency_key=SCH.historical_paper_key("hist-restart"),
            )

        def find_by_type_and_key(self, job_type, key):
            assert job_type == SCH.HISTORICAL_PAPER_CYCLE
            assert key == SCH.historical_paper_key("hist-restart")
            return self.poll

        def complete(self, job_id, status, **kwargs):
            self.completed.append((job_id, status, kwargs))

    jobs = Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 3, 0, 5, tzinfo=IST),
        holidays=lambda: set(),
        active_snapshot_id=lambda: "snap-1",
    )
    supervisor._enqueue_post_market_grind = lambda *_a, **_k: None
    monkeypatch.setattr(
        "product.historical_paper_loop.pending_stage",
        lambda: {"phase": "AWAITING_LEARNING", "batch_id": "hist-restart"},
    )

    supervisor.enqueue_due()

    assert jobs.completed
    assert jobs.completed[0][0] == "hist-poll-1"
    assert jobs.completed[0][1] == JS.SKIPPED_IDEMPOTENT
    assert jobs.enqueued[-1] == (
        SCH.LEARNING_CYCLE,
        {
            "idempotency_key": SCH.historical_learning_key("hist-restart"),
            "input_snapshot_id": "hist-restart",
        },
    )



def test_forward_settlement_waits_for_required_official_session(monkeypatch):
    jobs = _Jobs()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.jobs = jobs
    supervisor.deps = SimpleNamespace(
        now_ist=lambda: datetime(2026, 9, 18, 15, 45, tzinfo=IST),
        holidays=lambda: set(),
    )
    monkeypatch.setattr(
        "product.readiness.official_history",
        lambda: {"current": False, "available_session": "2026-09-17", "latest_date": "2026-09-17"},
    )
    supervisor._enqueue_post_market_grind(session_date="2026-09-18")
    assert jobs.enqueued == []
