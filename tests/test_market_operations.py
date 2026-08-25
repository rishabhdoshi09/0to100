from __future__ import annotations

import os
import time
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

    store = OperationStore(tmp_path / "market_ops" / "jobs.db")
    worker = MO.MarketOperationsWorker(store)
    queued = set(worker._bootstrap())

    assert queued == {MO.FNO_REFRESH}
    # A restart/click must reuse the pending work rather than duplicate it.
    assert worker._bootstrap() == []
    assert len(store.active()) == 1


def test_bootstrap_skips_market_scan_when_momentum_artifact_is_fresh(tmp_path: Path, monkeypatch):
    from data import bhavcopy_runtime
    from operations import market_ops as MO

    monkeypatch.setattr(MO, "ROOT", tmp_path)
    monkeypatch.setattr(MO, "LOCK_PATH", tmp_path / "market_ops" / "worker.lock")
    monkeypatch.setattr(
        bhavcopy_runtime,
        "status",
        lambda load_cache=False: {"ready": True, "sessions": 500, "symbols": 3000},
    )
    product = tmp_path / "logs" / "product"
    product.mkdir(parents=True)
    (product / "fno_universe.json").write_text("{}", encoding="utf-8")
    scan_path = product / "latest_momentum_scan.json"
    scan_path.write_text('{"schema_version": 1, "records": []}', encoding="utf-8")
    now = time.time()
    os.utime(scan_path, (now - 60, now - 60))

    store = OperationStore(tmp_path / "market_ops" / "jobs.db")
    worker = MO.MarketOperationsWorker(store)
    queued = set(worker._bootstrap())
    assert MO.MARKET_SCAN not in queued
    assert queued == {MO.LONG_TERM_REFRESH}
