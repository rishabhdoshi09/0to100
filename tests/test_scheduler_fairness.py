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



def test_historical_poll_does_not_starve_learning_job(tmp_path):
    clk = [75.0]
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: clk[0])
    historical = store.enqueue(
        SCH.HISTORICAL_PAPER_CYCLE,
        idempotency_key="hist_paper:batch-1",
    )
    store.reschedule_retry(
        historical.job_id,
        when=clk[0] - 1,
        error_code="HISTORICAL_PAPER_IN_PROGRESS",
        error_message="historical worker still running",
    )
    store.enqueue(
        SCH.LEARNING_CYCLE,
        idempotency_key="forward_learning:2026-09-18",
    )
    leased = store.lease_due("owner")
    assert leased is not None
    assert leased.job_type == SCH.LEARNING_CYCLE
