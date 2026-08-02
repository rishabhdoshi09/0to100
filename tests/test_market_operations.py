from __future__ import annotations

import os
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


def test_staleness_uses_persisted_file_age(tmp_path: Path):
    from operations.market_ops import _stale

    artifact = tmp_path / "artifact.json"
    assert _stale(artifact, 60, now=1_000) is True
    artifact.write_text("{}", encoding="utf-8")
    os.utime(artifact, (970, 970))
    assert _stale(artifact, 60, now=1_000) is False
    os.utime(artifact, (900, 900))
    assert _stale(artifact, 60, now=1_000) is True


def test_bootstrap_queues_missing_product_inputs_without_network(tmp_path: Path, monkeypatch):
    from data import bhavcopy_runtime
    from operations import market_ops as MO

    monkeypatch.setattr(MO, "ROOT", tmp_path)
    monkeypatch.setattr(MO, "LOCK_PATH", tmp_path / "market_ops" / "worker.lock")
    monkeypatch.setattr(
        bhavcopy_runtime,
        "status",
        lambda load_cache=False: {"ready": True, "sessions": 500, "symbols": 3000},
    )
    monkeypatch.setattr(
        "data.us_history_store.status",
        lambda: {"ready": True, "symbols": 200, "latest_date": "2026-08-01"},
    )
    # Fresh US scan artifact under patched ROOT → US_MARKET_SCAN not due.
    us_scan = tmp_path / "logs" / "product" / "latest_us_scan.json"
    us_scan.parent.mkdir(parents=True, exist_ok=True)
    us_scan.write_text("{}", encoding="utf-8")

    store = OperationStore(tmp_path / "market_ops" / "jobs.db")
    worker = MO.MarketOperationsWorker(store)
    queued = set(worker._bootstrap())

    assert queued == {MO.FNO_REFRESH, MO.NEWS_REFRESH, MO.MARKET_SCAN, MO.LONG_TERM_SCAN}
    # A restart/click must reuse the pending work rather than duplicate it.
    assert worker._bootstrap() == []
    assert len(store.active()) == 4
