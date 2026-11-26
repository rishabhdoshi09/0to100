"""Cycle 11: a login after the close releases the jobs blocked on AUTH_READY.

Owner's Mac, 26 Nov: the desk was down through the session (storage), the owner logged in to
Kite at 19:30 IST. The manual data refresh ran, but the scheduled EOD Kite refresh
(data_refresh:<day>:eod) and the index warm-up (index_warmup:<day>) stayed BLOCKED on
AUTH_READY: only the open-market branch of enqueue_due schedules the AUTH_HEALTH probe whose
success releases that dependency. The index store (Nifty, VIX, sector regime context) stayed
at 23 Nov overnight. Now a closed-market tick with a recent job blocked on AUTH_READY
schedules the day's (idempotent) probe; the existing revival rules bound its retries.
"""
from __future__ import annotations

import inspect
from datetime import datetime

from research.autonomy import jobs as JOBS
from research.autonomy import schedules as SCH
from research.autonomy.supervisor import Supervisor
from tests.test_autonomy_console_runtime import _Deps


class _Evening(_Deps):
    def now_ist(self):
        return datetime(2026, 7, 31, 19, 30)


def _sup(tmp_path):
    sup = Supervisor(tmp_path / "auto", deps=_Evening())
    assert sup.start()
    return sup


def test_a_recent_job_blocked_on_auth_gets_a_closed_market_probe(tmp_path):
    sup = _sup(tmp_path)
    try:
        job = sup.jobs.enqueue(SCH.DATA_REFRESH, idempotency_key="data_refresh:2026-07-31:eod")
        sup.jobs.block(job.job_id, dependency=JOBS.DEP_AUTH, reason="auth required before data refresh")
        sup._ensure_closed_market_auth(sup.deps.now_ist())
        probe = sup.jobs.find_by_type_and_key(SCH.AUTH_HEALTH, "auth:2026-07-31")
        assert probe is not None and probe.status == "PENDING"
        sup._ensure_closed_market_auth(sup.deps.now_ist())             # idempotent
        assert len([j for j in sup.jobs.list(job_type=SCH.AUTH_HEALTH)]) == 1
    finally:
        sup.shutdown()


def test_nothing_blocked_on_auth_schedules_no_probe(tmp_path):
    sup = _sup(tmp_path)
    try:
        sup._ensure_closed_market_auth(sup.deps.now_ist())
        assert sup.jobs.find_by_type_and_key(SCH.AUTH_HEALTH, "auth:2026-07-31") is None
    finally:
        sup.shutdown()


def test_an_old_block_schedules_no_probe(tmp_path):
    sup = _sup(tmp_path)
    try:
        job = sup.jobs.enqueue(SCH.DATA_REFRESH, idempotency_key="data_refresh:2026-07-01:eod")
        sup.jobs.block(job.job_id, dependency=JOBS.DEP_AUTH, reason="auth required")
        sup.jobs._db.execute("UPDATE jobs SET blocked_at=blocked_at-5*86400 WHERE job_id=?", (job.job_id,))
        sup.jobs._db.commit()
        sup._ensure_closed_market_auth(sup.deps.now_ist())
        assert sup.jobs.find_by_type_and_key(SCH.AUTH_HEALTH, "auth:2026-07-31") is None
    finally:
        sup.shutdown()


def test_the_closed_market_branch_runs_the_probe_check():
    src = inspect.getsource(Supervisor.enqueue_due)
    closed = src[src.index("# Closed market starts by finding"):]
    assert "self._ensure_closed_market_auth(now_ist)" in closed
