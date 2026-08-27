from __future__ import annotations

from datetime import date
import pickle

import pandas as pd

from data import bhavcopy_runtime
from data import bhavcopy_store
from research.autonomy import job_store as JS


def test_api_process_loads_persisted_bhavcopy_cache(tmp_path, monkeypatch):
    cache = tmp_path / "store_cache.pkl"
    frame = pd.DataFrame(
        [{"open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0, "volume": 1000.0}],
        index=[pd.Timestamp("2026-07-30")],
    )
    with cache.open("wb") as handle:
        pickle.dump(
            {"store": {"TEST": frame}, "last_day": date(2026, 7, 30), "sessions": 120},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    monkeypatch.setattr(bhavcopy_store, "_BHAV_DIR", tmp_path)
    monkeypatch.setattr(bhavcopy_store, "_PKL", cache)
    monkeypatch.setattr(bhavcopy_store, "_store", {})
    monkeypatch.setattr(bhavcopy_store, "_store_last_day", None)
    monkeypatch.setattr(bhavcopy_store, "_store_sessions", 0)

    state = bhavcopy_runtime.status(load_cache=True)

    assert state["ready"] is True
    assert state["symbols"] == 1
    assert state["sessions"] == 120
    assert state["latest_date"] == "2026-07-30"
    assert bhavcopy_runtime.get_ohlcv("TEST") is not None


def test_overdue_critical_uses_latest_recurring_intent(tmp_path):
    now = 20_000.0
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: now)
    try:
        store.enqueue("auth_health", scheduled_for=1_000.0, idempotency_key="auth:old", critical=True)
        store.enqueue("auth_health", scheduled_for=19_000.0, idempotency_key="auth:new", critical=True)
        store.enqueue(
            "paper_cycle",
            scheduled_for=1_000.0,
            idempotency_key="paper_cycle:snapshot:2026-07-31:intraday-1000",
            critical=True,
        )

        assert store.overdue_critical(grace_seconds=3_600.0) == []
    finally:
        store.close()


def test_running_job_type_suppresses_older_pending_overdue_row(tmp_path):
    now = 20_000.0
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: now)
    try:
        running = store.enqueue(
            "data_refresh",
            scheduled_for=1_000.0,
            idempotency_key="data:running",
            critical=True,
        )
        leased = store.lease_due("supervisor")
        assert leased is not None and leased.job_id == running.job_id
        store.enqueue(
            "data_refresh",
            scheduled_for=2_000.0,
            idempotency_key="data:pending",
            critical=True,
        )

        assert store.overdue_critical(grace_seconds=3_600.0) == []
    finally:
        store.close()


def test_genuinely_current_overdue_critical_job_is_reported(tmp_path):
    now = 20_000.0
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: now)
    try:
        expected = store.enqueue(
            "data_refresh",
            scheduled_for=1_000.0,
            idempotency_key="data:overdue",
            critical=True,
        )

        overdue = store.overdue_critical(grace_seconds=3_600.0)
        assert [job.job_id for job in overdue] == [expected.job_id]
    finally:
        store.close()


def test_reenqueue_redates_stale_pending_critical_job(tmp_path):
    clk = [1_000.0]
    store = JS.JobStore(tmp_path / "jobs.db", clock=lambda: clk[0])
    try:
        first = store.enqueue(
            "data_refresh",
            scheduled_for=1_000.0,
            idempotency_key="data_refresh:2026-08-27",
            critical=True,
        )
        clk[0] = 20_000.0
        again = store.enqueue(
            "data_refresh",
            idempotency_key="data_refresh:2026-08-27",
            critical=True,
        )
        assert again.job_id == first.job_id
        assert again.scheduled_for == 20_000.0
        assert store.overdue_critical(grace_seconds=3_600.0) == []
    finally:
        store.close()
