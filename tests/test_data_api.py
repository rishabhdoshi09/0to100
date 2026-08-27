"""API tests for data platform routes."""
from __future__ import annotations

from fastapi.testclient import TestClient

import terminal_product_api as tpa


def test_data_providers_endpoint():
    client = TestClient(tpa.app)
    response = client.get("/api/data/providers")
    assert response.status_code == 200
    body = response.json()
    assert "providers" in body and len(body["providers"]) >= 3


def test_data_coverage_endpoint():
    client = TestClient(tpa.app)
    response = client.get("/api/data/coverage?symbol=RELIANCE")
    assert response.status_code == 200
    body = response.json()
    assert body.get("symbol") == "RELIANCE"
    assert "coverage" in body


def test_data_ratios_endpoint():
    client = TestClient(tpa.app)
    response = client.get("/api/data/ratios/RELIANCE")
    assert response.status_code == 200
    body = response.json()
    assert body.get("symbol") == "RELIANCE"
    assert isinstance(body.get("ratios"), list)


def test_data_job_run_endpoint(monkeypatch):
    client = TestClient(tpa.app)
    monkeypatch.setattr(
        "data_platform.jobs.run_job",
        lambda job_id: {"ok": True, "job_id": job_id, "note": "test"},
    )
    response = client.post("/api/data/jobs/coverage_audit/run")
    assert response.status_code == 200
    assert response.json()["ok"] is True
