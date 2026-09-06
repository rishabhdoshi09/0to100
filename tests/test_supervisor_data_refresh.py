"""Supervisor tick consumes DATA_REFRESH and only then clears snapshot_stale."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy.supervisor import Supervisor

from tests.test_autonomy import FakeDeps, _NOW


def _durable_failures(root: Path) -> set[str]:
    return set(json.loads((root / "failures.json").read_text(encoding="utf-8")))


def _durable_status_failures(root: Path) -> set[str]:
    payload = json.loads((root / "status.json").read_text(encoding="utf-8"))
    return set(payload.get("active_failures") or [])


class _RefreshDeps(FakeDeps):
    def refresh_instruments(self):
        return {"rows": 1, "fno_underlyings": 0}

    def update_bhavcopy(self):
        return {"ready": True}

    def warmup_index(self):
        return {"ok": True}

    def news_health(self):
        return {"running": True, "error": ""}

    def refresh_news(self):
        return {"articles": 0}


def _tick_until_data_refresh(sup: Supervisor, *, limit: int = 20):
    seen = None
    for _ in range(limit):
        job = sup.tick(_NOW)
        if job is None:
            continue
        if job.job_type == SCH.DATA_REFRESH:
            seen = sup.jobs.get(job.job_id)
            if seen.status in {JS.SUCCEEDED, JS.BLOCKED, JS.PERMANENT_FAILED, JS.RETRYABLE_FAILED}:
                return seen
    return seen


def test_successful_data_refresh_clears_snapshot_stale_on_supervisor_tick(tmp_path):
    root = tmp_path / "auto"
    sup = Supervisor(root, deps=_RefreshDeps(data_ok=True))
    assert sup.start() is True
    sup.failures.add(H.SNAPSHOT_STALE)
    sup._save_failures()
    sup.heartbeat()

    before = _durable_failures(root)
    assert H.SNAPSHOT_STALE in before
    assert H.SNAPSHOT_STALE in _durable_status_failures(root)

    completed = _tick_until_data_refresh(sup)
    assert completed is not None
    assert completed.job_type == SCH.DATA_REFRESH
    assert completed.status == JS.SUCCEEDED

    after_failures = _durable_failures(root)
    after_status = _durable_status_failures(root)
    assert H.SNAPSHOT_STALE not in after_failures
    assert H.SNAPSHOT_STALE not in after_status
    reopened = Supervisor(root, deps=_RefreshDeps(data_ok=True))
    assert H.SNAPSHOT_STALE not in reopened.failures
    sup.shutdown()
    reopened.shutdown()


def test_failed_data_refresh_does_not_clear_snapshot_stale(tmp_path):
    root = tmp_path / "auto"
    sup = Supervisor(root, deps=_RefreshDeps(data_ok=False, authed=True))
    assert sup.start() is True
    sup.failures.add(H.SNAPSHOT_STALE)
    sup._save_failures()
    sup.heartbeat()
    assert H.SNAPSHOT_STALE in _durable_failures(root)

    completed = _tick_until_data_refresh(sup)
    assert completed is not None
    assert completed.job_type == SCH.DATA_REFRESH
    assert completed.status != JS.SUCCEEDED

    assert H.SNAPSHOT_STALE in _durable_failures(root)
    assert H.SNAPSHOT_STALE in _durable_status_failures(root)
    assert H.SNAPSHOT_STALE in sup.failures
    sup.shutdown()
