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
