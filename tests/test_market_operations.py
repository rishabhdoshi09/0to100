from __future__ import annotations

from pathlib import Path

from operations.store import OperationStore, PENDING, RUNNING, SUCCEEDED


def test_operation_queue_deduplicates_active_clicks(tmp_path: Path):
    store = OperationStore(tmp_path / "ops.db")
    first, created = store.enqueue("MARKET_SCAN", lane="market_scan")
    assert created is True

    second, created = store.enqueue("MARKET_SCAN", lane="market_scan")
    assert created is False
    assert second["operation_id"] == first["operation_id"]


def test_operation_progress_is_durable(tmp_path: Path):
    store = OperationStore(tmp_path / "ops.db")
    queued, _ = store.enqueue("NEWS_REFRESH", lane="news")
    assert queued["status"] == PENDING

    leased = store.lease_next("news", worker_pid=123)
    assert leased is not None and leased["status"] == RUNNING

    store.progress(
        leased["operation_id"],
        stage="FETCHING_SOURCES",
        message="working",
        current=2,
        total=4,
    )
    active = store.get(leased["operation_id"])
    assert active is not None
    assert active["progress_pct"] == 50.0

    store.finish(
        leased["operation_id"],
        status=SUCCEEDED,
        message="done",
        result={"articles": 12},
    )
    final = store.get(leased["operation_id"])
    assert final is not None
    assert final["status"] == SUCCEEDED
    assert final["progress_current"] == 4
    assert final["result"]["articles"] == 12


def test_worker_restart_requeues_orphans(tmp_path: Path):
    store = OperationStore(tmp_path / "ops.db")
    queued, _ = store.enqueue("FNO_REFRESH", lane="data")
    assert store.lease_next("data", worker_pid=123) is not None

    assert store.recover_orphans() == 1
    recovered = store.get(queued["operation_id"])
    assert recovered is not None
    assert recovered["status"] == PENDING
