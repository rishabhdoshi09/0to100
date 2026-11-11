"""A daily Zerodha login after the morning auth check must revive the session.

Owner's Mac 2026-11-11: auth:2026-11-11 blocked on CREDENTIAL_UPDATE at 09:30
(no login yet); nothing emits CREDENTIAL_UPDATE, so after the 09:33 login the
day's data job and every scan->paper slot stayed blocked and no paper cycle ran.
"""
from __future__ import annotations

from types import SimpleNamespace

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy.supervisor import Supervisor


def _store(tmp_path, clock):
    return JS.JobStore(tmp_path / "jobs.db", clock=lambda: clock[0])


def _blocked_auth(store, key, dependency="CREDENTIAL_UPDATE"):
    job = store.enqueue(SCH.AUTH_HEALTH, idempotency_key=key, critical=True)
    store.block(job.job_id, dependency=dependency, reason="TokenException")
    return job


def _revive(store, date):
    stub = SimpleNamespace(jobs=store, AUTH_RETRY_S=Supervisor.AUTH_RETRY_S)
    return Supervisor._revive_blocked_auth(stub, date)


def test_todays_blocked_auth_is_requeued_after_the_retry_interval(tmp_path):
    clock = [1000.0]
    store = _store(tmp_path, clock)
    job = _blocked_auth(store, "auth:2026-11-11")
    clock[0] += 60
    assert _revive(store, "2026-11-11") is False             # too soon
    clock[0] += Supervisor.AUTH_RETRY_S
    assert _revive(store, "2026-11-11") is True
    assert store.get(job.job_id).status == JS.PENDING
    assert store.lease_due("t") is not None                  # runnable again


def test_other_days_and_other_dependencies_are_untouched(tmp_path):
    clock = [1000.0]
    store = _store(tmp_path, clock)
    old = _blocked_auth(store, "auth:2026-09-16")
    other = _blocked_auth(store, "auth:2026-11-11", dependency="SOMETHING_ELSE")
    clock[0] += 10_000
    assert _revive(store, "2026-11-11") is False
    assert store.get(old.job_id).status == JS.BLOCKED
    assert store.get(other.job_id).status == JS.BLOCKED


def test_succeeded_or_missing_auth_is_left_alone(tmp_path):
    clock = [1000.0]
    store = _store(tmp_path, clock)
    assert _revive(store, "2026-11-11") is False
    job = store.enqueue(SCH.AUTH_HEALTH, idempotency_key="auth:2026-11-11", critical=True)
    store.complete(job.job_id, JS.SUCCEEDED, result_summary="AUTH_READY")
    clock[0] += 10_000
    assert _revive(store, "2026-11-11") is False
