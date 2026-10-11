"""Operation progress polling never reads a huge completed scan payload."""
from __future__ import annotations

from fastapi.testclient import TestClient
import pytest


def test_operation_store_polling_ignores_large_result(tmp_path, monkeypatch):
    from operations.store import OperationStore, SUCCEEDED

    store = OperationStore(tmp_path / "operations.sqlite3")
    # Seed directly to emulate a legacy (pre-compaction) large result.
    with store._connect() as con:
        con.execute(
            "INSERT INTO operations "
            "(operation_id,kind,lane,status,requested_by,requested_at,updated_at,"
            "stage,message,progress_current,progress_total,result_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("op-large", "MARKET_SCAN", "scan", SUCCEEDED, "test", 1.0, 2.0,
             "SAVING", "Completed", 2334, 2334, "x" * 2_000_000),
        )
    monkeypatch.setattr(
        OperationStore, "_decode", staticmethod(
            lambda row: (_ for _ in ()).throw(AssertionError("decoded full operation blob"))
        ),
    )
    row = store.get_summary("op-large")
    assert row is not None
    assert row["status"] == SUCCEEDED
    assert row["progress_pct"] == 100.0
    assert "result" not in row
    assert "result_json" not in row
    assert store.get_summary("missing") is None


def test_http_operation_status_route_uses_summary_not_full(monkeypatch):
    import terminal_api as core
    import operations.store as operations_store

    class FakeStore:
        def __init__(self, _path):
            pass
        def get_summary(self, op):
            return {"operation_id": op, "status": "RUNNING", "stage": "SCANNING"}
        def get(self, op):
            raise AssertionError("HTTP polling unexpectedly loaded full blob")

    monkeypatch.setattr(operations_store, "OperationStore", FakeStore)
    with TestClient(core.app) as client:
        reply = client.get("/api/operations/op-123/status")
        assert reply.status_code == 200
        assert reply.json()["stage"] == "SCANNING"


def test_http_unknown_operation_remains_404(monkeypatch):
    import terminal_api as core
    import operations.store as operations_store

    class FakeStore:
        def __init__(self, _path):
            pass
        def get_summary(self, _op):
            return None

    monkeypatch.setattr(operations_store, "OperationStore", FakeStore)
    with TestClient(core.app) as client:
        assert client.get("/api/operations/missing/status").status_code == 404
