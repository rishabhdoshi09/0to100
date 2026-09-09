"""MARKET_SCAN durable success is stored in the operation SQLite, not just returned."""
from __future__ import annotations

import os
from pathlib import Path

from operations.market_ops import LANES, MARKET_SCAN, MarketOperationsWorker
from operations.store import SUCCEEDED, OperationStore

from tests.test_startup_stale_scan import CURRENT


def test_market_scan_durable_success_survives_store_reload(tmp_path: Path, monkeypatch):
    db = tmp_path / "ops.db"
    store = OperationStore(db)
    worker = MarketOperationsWorker(store=store)
    monkeypatch.setattr(
        worker,
        "_require_current_history",
        lambda _op: {**CURRENT, "current": True, "available_session": "2026-09-01"},
    )
    monkeypatch.setattr("scan.bulk_fetcher.adopt_ready_store", lambda overlay_live=True: 500)
    monkeypatch.setattr("data.nse_universe.get_nse_universe", lambda: ["AAA"])

    class FakeReport:
        ok = True
        status = "OK"
        payload = {"summary": {"with_any_setup": 1}, "records": [{"symbol": "AAA"}]}

    monkeypatch.setattr("scan.market_scan_service.run_whole_market_scan", lambda **_k: FakeReport())
    monkeypatch.setattr(worker, "_notify_scan_telegram", lambda _p: {"sent": False})
    monkeypatch.setattr("product.desk_pipeline.advance_desk_pipeline", lambda *_a, **_k: {})
    monkeypatch.setattr("product.autonomous_loop.advance_loop", lambda **_k: {})

    op, created = store.enqueue(MARKET_SCAN, lane=LANES[MARKET_SCAN], requested_by="terminal", priority=100)
    assert created is True
    leased = store.lease_next(LANES[MARKET_SCAN], worker_pid=os.getpid())
    assert leased is not None
    assert leased["status"] != SUCCEEDED

    result = worker._execute(leased)
    store.finish(
        leased["operation_id"],
        status=SUCCEEDED,
        message="MARKET_SCAN completed in 0.1s",
        result=result,
    )

    row = store.get(leased["operation_id"])
    assert row["kind"] == MARKET_SCAN
    assert row["status"] == SUCCEEDED
    assert row["operation_id"] == op["operation_id"]
    assert row["finished_at"]
    assert int(result["records"]) == 1

    reopened = OperationStore(db)
    restored = reopened.get(op["operation_id"])
    assert restored is not None
    assert restored["kind"] == MARKET_SCAN
    assert restored["status"] == SUCCEEDED
    assert restored["operation_id"] == op["operation_id"]
    assert restored["requested_at"]
    assert restored["started_at"]
    assert restored["finished_at"]
    latest = reopened.latest(MARKET_SCAN)
    assert latest["status"] == SUCCEEDED
    assert latest["operation_id"] == op["operation_id"]
