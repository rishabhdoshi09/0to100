"""Market-ops worker must come ONLINE before heavy bootstrap work."""
from __future__ import annotations

from pathlib import Path

from operations.store import OperationStore, PENDING


def test_enqueue_refreshes_pending_message_on_reclick(tmp_path: Path):
    store = OperationStore(tmp_path / "ops.db")
    first, created = store.enqueue(
        "MARKET_SCAN",
        lane="market_scan",
        message="Queued and waiting for the dedicated market-operations worker",
    )
    assert created is True
    second, created = store.enqueue(
        "MARKET_SCAN",
        lane="market_scan",
        requested_by="terminal",
        message="Queued, but market-ops worker is OFFLINE — restart stack",
    )
    assert created is False
    assert second["operation_id"] == first["operation_id"]
    assert second["status"] == PENDING
    assert "OFFLINE" in second["message"]


def test_bootstrap_does_not_require_full_pkl_load(tmp_path: Path, monkeypatch):
    from data import bhavcopy_runtime
    from operations import market_ops as MO

    monkeypatch.setattr(MO, "ROOT", tmp_path)
    monkeypatch.setattr(MO, "LOCK_PATH", tmp_path / "market_ops" / "worker.lock")
    calls = {"load_cache": []}

    def status(*, load_cache: bool = False):
        calls["load_cache"].append(load_cache)
        return {
            "ready": False,
            "sessions": 0,
            "symbols": 0,
            "cache_exists": True,
            "csv_files": 200,
        }

    monkeypatch.setattr(bhavcopy_runtime, "status", status)
    worker = MO.MarketOperationsWorker(OperationStore(tmp_path / "jobs.db"))
    queued = worker._bootstrap()
    assert True not in calls["load_cache"], "bootstrap must not block on full pkl load"
    # History present via cache/csv → no DATA_PREPARE; other product inputs still due.
    assert MO.DATA_PREPARE not in queued
    assert MO.MARKET_SCAN in queued


def test_dead_lock_can_be_reclaimed(tmp_path: Path):
    from operations.market_ops import SingleWorkerLock

    lock_path = tmp_path / "worker.lock"
    lock_path.write_text("999999", encoding="utf-8")  # almost certainly dead
    lock = SingleWorkerLock(lock_path)
    assert lock.reclaim_if_dead() is True
    assert lock.acquire() is True
    lock.release()
