"""A polling data-refresh job must not starve the market scan."""
from __future__ import annotations

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH


def test_lease_prefers_scan_over_in_progress_data_refresh(tmp_path):
    clk = [100.0]
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: clk[0])
    refresh = store.enqueue(SCH.DATA_REFRESH, idempotency_key="data:1", critical=True)
    store.enqueue(SCH.MARKET_SCAN, idempotency_key="scan:1")
    store.reschedule_retry(
        refresh.job_id,
        when=clk[0] - 1,
        error_code="DATA_REFRESH_IN_PROGRESS",
        error_message="still running",
    )
    leased = store.lease_due("owner")
    assert leased is not None
    assert leased.job_type == SCH.MARKET_SCAN


def test_lease_still_polls_refresh_when_it_is_the_only_due_job(tmp_path):
    clk = [50.0]
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: clk[0])
    refresh = store.enqueue(SCH.DATA_REFRESH, idempotency_key="data:only", critical=True)
    store.reschedule_retry(
        refresh.job_id,
        when=clk[0] - 1,
        error_code="DATA_REFRESH_IN_PROGRESS",
        error_message="still running",
    )
    leased = store.lease_due("owner")
    assert leased is not None
    assert leased.job_type == SCH.DATA_REFRESH
