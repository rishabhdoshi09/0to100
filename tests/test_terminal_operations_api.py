"""Terminal API operation endpoints used by the live scan runner."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import terminal_api as api


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    jobs_db = tmp_path / "jobs.db"
    api.OPS_DB = str(jobs_db)
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
