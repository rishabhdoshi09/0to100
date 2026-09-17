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
