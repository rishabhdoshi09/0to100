"""A successful scan on current official NSE history must retire snapshot-stale for paper."""
from __future__ import annotations

from datetime import datetime

from research.autonomy import health as H
from research.autonomy import job_store as JS
from research.autonomy import jobs as JOBS


class _Deps:
    def active_snapshot_id(self):
        return None

    def official_history(self):
        return {"current": True, "source": "official_nse", "available_session": "2026-09-11"}

    def live_market_ready(self):
        raise AssertionError("current official history must avoid broker/live probing")

    def run_scan(self):
        return {"summary": {"with_any_setup": 1, "momentum": 1}}

    def now_ist(self):
        return datetime(2026, 9, 11, 10, 0)

    def holidays(self):
        return set()


def test_successful_current_official_scan_clears_snapshot_stale():
    result = JOBS.run_market_scan(JOBS._Ctx(_Deps(), active_failures={H.SNAPSHOT_STALE}))

    assert result.status == JS.SUCCEEDED
    assert H.SNAPSHOT_STALE in result.clears
