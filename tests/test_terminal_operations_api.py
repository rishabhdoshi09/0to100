"""Terminal API operation endpoints used by the live scan runner."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import terminal_api as api


@pytest.fixture()
def client(tmp_path: Path, monkeypatch) -> TestClient:
    jobs_db = tmp_path / "jobs.db"
    monkeypatch.setattr(api, "OPS_DB", str(jobs_db))
    monkeypatch.setattr(api, "_ensure_ops_worker", lambda *a, **k: {"running": True})
    return TestClient(api.app)


def test_operation_status_endpoint_returns_durable_record(client: TestClient):
    from operations.market_ops import LANES
    from operations.store import OperationStore

    store = OperationStore(api.OPS_DB)
    record, created = store.enqueue("MARKET_SCAN", lane=LANES["MARKET_SCAN"], requested_by="test")
    assert created

    response = client.get(f"/api/operations/{record['operation_id']}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["operation_id"] == record["operation_id"]
    assert payload["kind"] == "MARKET_SCAN"
    assert payload["status"] in {"PENDING", "RUNNING"}


def test_scan_control_enqueues_without_waiting_for_worker(tmp_path: Path, monkeypatch):
    import time
    from fastapi.testclient import TestClient

    import terminal_api as api
    from operations.store import OperationStore

    jobs_db = tmp_path / "jobs.db"
    monkeypatch.setattr(api, "OPS_DB", str(jobs_db))
    waited: list[bool] = []

    def ensure(*, wait: bool = True):
        waited.append(wait)
        return {"running": True}

    monkeypatch.setattr(api, "_ensure_ops_worker", ensure)
    client = TestClient(api.app)
    started = time.monotonic()
    response = client.post("/api/controls/RUN_SCAN_NOW")
    elapsed = time.monotonic() - started
    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["operation_id"]
    assert False in waited
    assert elapsed < 1.0
    store = OperationStore(jobs_db)
    assert any(item.get("kind") == "MARKET_SCAN" for item in store.active())


def test_scan_control_stays_accepted_when_ensure_raises_but_lock_is_owned(tmp_path, monkeypatch):
    import os

    from fastapi.testclient import TestClient

    import terminal_api as api

    jobs_db = tmp_path / "jobs.db"
    monkeypatch.setattr(api, "OPS_DB", str(jobs_db))
    monkeypatch.setattr(api, "OPS_ROOT", tmp_path)
    (tmp_path / "worker.lock").write_text(str(os.getpid()), encoding="utf-8")

    def ensure(*, wait: bool = True):
        raise RuntimeError(
            "Market operations worker did not become ready after bounded recovery; "
            "the command was not silently accepted. The launcher watchdog owns recovery "
            "after this attempt. Check System Health for the worker blocker."
        )

    monkeypatch.setattr(api, "_ensure_ops_worker", ensure)
    client = TestClient(api.app)
    response = client.post("/api/controls/RUN_SCAN_NOW")
    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["operation_id"]
    assert payload.get("worker_recovering") is True


def test_scan_control_deduplicates_active_market_scan(client: TestClient):
    first = client.post("/api/controls/RUN_SCAN_NOW")
    second = client.post("/api/controls/RUN_SCAN_NOW")
    assert first.status_code == 200
    assert second.status_code == 200
    first_id = first.json()["operation_id"]
    second_id = second.json()["operation_id"]
    assert first_id == second_id
    assert second.json().get("created") is False


def test_operations_payload_lists_active_operations(client: TestClient):
    client.post("/api/controls/RUN_SCAN_NOW")
    response = client.get("/api/operations")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("available") is True
    active = payload.get("active", [])
    assert any(item.get("kind") == "MARKET_SCAN" for item in active)
