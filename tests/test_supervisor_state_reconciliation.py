"""Regression coverage for truthful supervisor readiness transitions."""
from __future__ import annotations

from datetime import datetime

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST
from research.autonomy.supervisor import Supervisor


_NOW = datetime(2026, 9, 17, 23, 5)


class _Deps:
    def now_ist(self):
        return _NOW

    def holidays(self):
        return set()

    def session_valid(self):
        return True

    def active_snapshot_id(self):
        return "snap-current"

    def news_health(self):
        return {"running": True, "error": ""}


def _sup(tmp_path):
    return Supervisor(tmp_path / "auto", deps=_Deps())


def test_successful_auth_probe_does_not_downgrade_data_ready(tmp_path):
    sup = _sup(tmp_path)
    assert sup.start() is True
    sup._transition(ST.DATA_READY, "fixture", "data is already accepted", "test")

    job = sup.jobs.enqueue(SCH.AUTH_HEALTH, idempotency_key="auth:test", critical=True)
    leased = sup.jobs.lease_due(sup.owner)
    assert leased is not None and leased.job_id == job.job_id
    sup._execute(leased)

    assert sup.jobs.get(job.job_id).status == JS.SUCCEEDED
    assert sup.state.state == ST.DATA_READY
    sup.shutdown()


def test_idle_tick_reconciles_latched_refreshing_to_ready(tmp_path):
    sup = _sup(tmp_path)
    assert sup.start() is True
    sup._transition(ST.DATA_REFRESHING, "fixture", "simulated stale activity hint", "test")
    sup.enqueue_due = lambda *_args, **_kwargs: None

    assert sup.tick(_NOW) is None
    assert sup.state.state == ST.DATA_READY
    assert sup.state.reason_code == "idle_reconcile"
    sup.shutdown()


def test_idle_reconcile_keeps_due_data_refresh_active(tmp_path):
    sup = _sup(tmp_path)
    assert sup.start() is True
    sup._transition(ST.DATA_REFRESHING, "fixture", "refresh is genuinely queued", "test")
    sup.jobs.enqueue(
        SCH.DATA_REFRESH,
        idempotency_key="data:due",
        scheduled_for=sup.clock() - 1.0,
        critical=True,
    )

    sup._reconcile_idle_state()
    assert sup.state.state == ST.DATA_REFRESHING
    sup.shutdown()


def test_future_scheduled_data_refresh_does_not_latch_refreshing(tmp_path):
    sup = _sup(tmp_path)
    assert sup.start() is True
    sup._transition(ST.DATA_REFRESHING, "fixture", "refresh is genuinely queued", "test")
    sup.jobs.enqueue(
        SCH.DATA_REFRESH,
        idempotency_key="data:future",
        scheduled_for=sup.clock() + 3600.0,
        critical=True,
    )

    sup._reconcile_idle_state()
    assert sup.state.state == ST.DATA_READY
    sup.shutdown()


def test_restart_retains_persisted_data_ready(tmp_path):
    first = _sup(tmp_path)
    assert first.start() is True
    first._transition(ST.DATA_READY, "fixture", "data is already accepted", "test")
    first.shutdown()

    second = _sup(tmp_path)
    assert second.start() is True
    assert second.state.state == ST.DATA_READY
    assert second.state.reason_code == "owner_resume"
    second.shutdown()


def test_idle_tick_reconciles_starting_to_ready(tmp_path):
    sup = _sup(tmp_path)
    assert sup.start() is True
    assert sup.state.state == ST.STARTING
    sup.enqueue_due = lambda *_args, **_kwargs: None
    assert sup.tick(_NOW) is None
    assert sup.state.state == ST.DATA_READY
    assert sup.state.reason_code == "idle_reconcile"
    sup.shutdown()


def test_idle_reconcile_blocks_when_snapshot_is_stale(tmp_path):
    sup = _sup(tmp_path)
    assert sup.start() is True
    sup.failures.add(H.SNAPSHOT_STALE)
    sup._save_failures()
    sup._transition(ST.DATA_REFRESHING, "fixture", "refresh activity ended", "test")

    sup._reconcile_idle_state()
    assert sup.state.state == ST.DATA_BLOCKED
    assert sup.state.reason_code == "idle_reconcile"
    sup.shutdown()


def test_publication_grace_retires_old_duplicate_data_refresh_intents(tmp_path, monkeypatch):
    sup = _sup(tmp_path)
    assert sup.start() is True
    monkeypatch.setattr(
        "product.readiness.official_history",
        lambda: {
            "usable_for_scan": True,
            "current": False,
            "publication_pending": True,
            "reason_code": "HISTORY_PUBLICATION_PENDING",
        },
    )
    old_a = sup.jobs.enqueue(
        SCH.DATA_REFRESH,
        idempotency_key="data_refresh:2026-09-16",
        critical=True,
    )
    old_b = sup.jobs.enqueue(
        SCH.DATA_REFRESH,
        idempotency_key="data_refresh:2026-09-16:eod",
        critical=True,
    )

    sup._retire_obsolete_data_refresh_work()

    assert sup.jobs.get(old_a.job_id).status == JS.CANCELLED
    assert sup.jobs.get(old_b.job_id).status == JS.CANCELLED
    sup.shutdown()


def test_genuinely_stale_history_preserves_exactly_one_refresh_recovery_intent(tmp_path, monkeypatch):
    sup = _sup(tmp_path)
    assert sup.start() is True
    monkeypatch.setattr(
        "product.readiness.official_history",
        lambda: {
            "usable_for_scan": False,
            "current": False,
            "publication_pending": False,
            "reason_code": "HISTORY_STALE",
        },
    )
    old_a = sup.jobs.enqueue(
        SCH.DATA_REFRESH,
        idempotency_key="data_refresh:2026-09-15",
        scheduled_for=sup.clock() - 30,
        critical=True,
    )
    old_b = sup.jobs.enqueue(
        SCH.DATA_REFRESH,
        idempotency_key="data_refresh:2026-09-16:eod",
        scheduled_for=sup.clock() - 10,
        critical=True,
    )

    sup._retire_obsolete_data_refresh_work()

    states = {
        old_a.job_id: sup.jobs.get(old_a.job_id).status,
        old_b.job_id: sup.jobs.get(old_b.job_id).status,
    }
    assert list(states.values()).count(JS.PENDING) == 1
    assert list(states.values()).count(JS.CANCELLED) == 1
    assert states[old_b.job_id] == JS.PENDING
    sup.shutdown()



def test_idle_reconcile_clears_latched_researching_to_observing(tmp_path):
    sup = _sup(tmp_path)
    assert sup.start() is True
    sup._transition(ST.RESEARCHING, "fixture", "research already ended", "test")

    sup._reconcile_idle_state()

    assert sup.state.state == ST.OBSERVING
    assert sup.state.reason_code == "activity_reconcile"
    assert "no due or running research job remains" in sup.state.explanation
    sup.shutdown()


def test_idle_reconcile_keeps_researching_when_research_job_is_running(tmp_path):
    sup = _sup(tmp_path)
    assert sup.start() is True
    sup._transition(ST.RESEARCHING, "fixture", "research is active", "test")
    job = sup.jobs.enqueue(
        SCH.RESEARCH_CYCLE,
        idempotency_key="hist_research:b1",
        scheduled_for=sup.clock() - 1.0,
    )
    leased = sup.jobs.lease_due(sup.owner)
    assert leased is not None and leased.job_id == job.job_id

    sup._reconcile_idle_state()

    assert sup.state.state == ST.RESEARCHING
    sup.shutdown()


def test_data_refresh_takes_activity_state_after_research_finishes(tmp_path):
    sup = _sup(tmp_path)
    assert sup.start() is True
    sup._transition(ST.RESEARCHING, "fixture", "research label is stale", "test")
    sup.jobs.enqueue(
        SCH.DATA_REFRESH,
        idempotency_key="data_refresh:due",
        scheduled_for=sup.clock() - 1.0,
        critical=True,
    )

    sup._reconcile_idle_state()

    assert sup.state.state == ST.DATA_REFRESHING
    assert sup.state.reason_code == "activity_reconcile"
    sup.shutdown()


def test_closed_market_exhausted_history_queues_one_bounded_research_replan(tmp_path, monkeypatch):
    sup = _sup(tmp_path)
    assert sup.start() is True
    sup._ensure_startup_trade_discovery = lambda: None
    sup._enqueue_post_market_grind = lambda *_args, **_kwargs: None
    sup._ensure_decision_simulation_authority = lambda: True
    sup._resource_budget = lambda: {"historical_replay_allowed": True}

    monkeypatch.setattr(
        "product.historical_paper_loop.pending_stage",
        lambda: {
            "phase": "IDLE",
            "batch_id": "",
            "processed_sessions": ["2026-01-05", "2026-01-06"],
            "thesis_hash": "thesis-a",
            "last_error": "",
        },
    )
    monkeypatch.setattr(
        "product.historical_paper_loop.peek_next_batch",
        lambda: {
            "available": False,
            "reason": "historical_backlog_caught_up",
            "processed_sessions": 2,
            "eligible_sessions": 2,
        },
    )

    request_state = {"open": True}
    request = {
        "request_id": "evreq-test",
        "status": "OPEN",
        "allowed_lanes": ["HISTORICAL_REPLAY"],
        "current_samples": 5,
        "target_samples": 30,
    }
    monkeypatch.setattr(
        "research.autonomy.evidence_acquisition.open_request_for_lane",
        lambda _lane: dict(request) if request_state["open"] else {},
    )
    plateaued = []

    def mark_exhausted(req, **kwargs):
        plateaued.append((dict(req), dict(kwargs)))
        request_state["open"] = False
        return {"status": "PLATEAUED"}

    monkeypatch.setattr(
        "research.autonomy.evidence_progress.mark_historical_source_exhausted",
        mark_exhausted,
    )

    sup.enqueue_due(_NOW)
    replans = [
        job for job in sup.jobs.list(limit=100)
        if job.job_type == SCH.RESEARCH_CYCLE
        and str(job.idempotency_key or "").startswith("research_replan:")
    ]
    assert len(replans) == 1
    assert replans[0].status == JS.PENDING
    assert plateaued and plateaued[0][0]["request_id"] == "evreq-test"

    # Upgrade cleanup must preserve the new durable replan identity.
    sup._retire_legacy_recurring_work()
    assert sup.jobs.get(replans[0].job_id).status == JS.PENDING

    # Same evidence state is idempotent: no duplicate research loop.
    sup.enqueue_due(_NOW)
    replans2 = [
        job for job in sup.jobs.list(limit=100)
        if job.job_type == SCH.RESEARCH_CYCLE
        and str(job.idempotency_key or "").startswith("research_replan:")
    ]
    assert len(replans2) == 2
    # The request transitioned OPEN -> PLATEAUED, so exactly one additional
    # replan identity without request_id is allowed. Further unchanged ticks
    # must reuse it rather than grow the queue.
    sup.enqueue_due(_NOW)
    replans3 = [
        job for job in sup.jobs.list(limit=100)
        if job.job_type == SCH.RESEARCH_CYCLE
        and str(job.idempotency_key or "").startswith("research_replan:")
    ]
    assert len(replans3) == 2
    sup.shutdown()


def test_failed_historical_phase_surfaces_incident_and_queues_replan(tmp_path, monkeypatch):
    sup = _sup(tmp_path)
    assert sup.start() is True
    sup._ensure_startup_trade_discovery = lambda: None
    sup._enqueue_post_market_grind = lambda *_args, **_kwargs: None
    sup._ensure_decision_simulation_authority = lambda: True
    sup._resource_budget = lambda: {"historical_replay_allowed": True}
    monkeypatch.setattr(
        "product.historical_paper_loop.pending_stage",
        lambda: {
            "phase": "FAILED",
            "batch_id": "b-failed",
            "processed_sessions": [],
            "thesis_hash": "thesis-a",
            "last_error": "worker crashed",
        },
    )
    incidents = []
    sup._incident = lambda code, message, job=None: incidents.append((code, message)) or {}

    sup.enqueue_due(_NOW)

    assert any(code == "HISTORICAL_PIPELINE_FAILED" for code, _ in incidents)
    replans = [
        job for job in sup.jobs.list(limit=100)
        if job.job_type == SCH.RESEARCH_CYCLE
        and str(job.idempotency_key or "").startswith("research_replan:")
    ]
    assert len(replans) == 1
    sup.shutdown()
