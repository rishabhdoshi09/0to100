"""Product bootstrap queues real data lanes on demand and at API start."""
from __future__ import annotations

from pathlib import Path

import terminal_product_api as tpa
from operations.market_ops import DATA_PREPARE, LONG_TERM_REFRESH, MARKET_SCAN


def test_queue_product_bootstrap_enqueues_first_due_step_only(tmp_path: Path, monkeypatch):
    store_path = tmp_path / "jobs.db"
    monkeypatch.setattr(tpa.core, "OPS_DB", store_path)
    monkeypatch.setattr(tpa.core, "_ensure_ops_worker", lambda: {"running": True})
    monkeypatch.setattr("product.desk_pipeline.prices_kind_due", lambda: DATA_PREPARE)
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: False)
    monkeypatch.setattr("product.desk_pipeline.long_term_is_fresh", lambda: False)
    monkeypatch.setattr("product.desk_pipeline.news_is_fresh", lambda: False)
    monkeypatch.setattr("product.desk_pipeline.acquire_is_fresh", lambda: True)

    payload = tpa.queue_product_bootstrap(requested_by="api_startup")
    assert payload["accepted"] is True
    assert payload["sequential"] is True
    assert payload["scan_reused"] is False
    assert payload["queued_kind"] == DATA_PREPARE
    kinds = {item["kind"] for item in payload["operations"]}
    assert kinds == {DATA_PREPARE}
    assert all(item["created"] for item in payload["operations"])
    assert all(item["status"] == "PENDING" for item in payload["operations"])

    again = tpa.queue_product_bootstrap(requested_by="api_startup")
    assert again["queued_kind"] is None
    assert all(item["created"] is False for item in again["operations"])


def test_queue_product_bootstrap_skips_fresh_market_scan(tmp_path: Path, monkeypatch):
    store_path = tmp_path / "jobs.db"
    monkeypatch.setattr(tpa.core, "OPS_DB", store_path)
    monkeypatch.setattr(tpa.core, "_ensure_ops_worker", lambda: {"running": True})
    monkeypatch.setattr("product.desk_pipeline.prices_kind_due", lambda: None)
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: True)
    monkeypatch.setattr("product.desk_pipeline.long_term_is_fresh", lambda: False)
    monkeypatch.setattr("product.desk_pipeline.news_is_fresh", lambda: False)
    monkeypatch.setattr("product.desk_pipeline.acquire_is_fresh", lambda: True)

    payload = tpa.queue_product_bootstrap(requested_by="api_startup")
    kinds = {item["kind"] for item in payload["operations"]}
    assert MARKET_SCAN not in kinds
    assert kinds == {LONG_TERM_REFRESH}
    assert payload["queued_kind"] == LONG_TERM_REFRESH
    assert payload["scan_reused"] is True
    assert payload["sequential"] is True


def test_desk_pipeline_get_does_not_enqueue(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient
    from operations.store import OperationStore

    store_path = tmp_path / "jobs.db"
    monkeypatch.setattr(tpa.core, "OPS_DB", store_path)
    monkeypatch.setattr(tpa.core, "_ensure_ops_worker", lambda: {"running": True})
    monkeypatch.setattr("product.desk_pipeline.prices_kind_due", lambda: DATA_PREPARE)
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: False)
    monkeypatch.setattr("product.desk_pipeline.long_term_is_fresh", lambda: False)
    monkeypatch.setattr("product.desk_pipeline.news_is_fresh", lambda: False)
    monkeypatch.setattr("product.desk_pipeline.acquire_is_fresh", lambda: True)

    client = TestClient(tpa.app)
    response = client.get("/api/desk-pipeline")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sequential"] is True
    assert body["queued_kind"] is None
    assert body["steps"][0]["id"] == "prices"
    assert OperationStore(store_path).active() == []


def test_startup_prepare_skips_under_pytest(monkeypatch):
    called: list[dict] = []
    monkeypatch.setattr(tpa, "queue_product_bootstrap", lambda **kwargs: called.append(kwargs) or {})
    tpa._startup_prepare_product()
    assert called == []


def test_startup_prepare_queues_outside_pytest(monkeypatch):
    called: list[dict] = []
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("QUANTTERM_SKIP_STARTUP_BOOTSTRAP", raising=False)
    monkeypatch.setattr(tpa, "queue_product_bootstrap", lambda **kwargs: called.append(kwargs) or {"accepted": True})
    tpa._startup_prepare_product()
    assert called == [{"requested_by": "api_startup"}]


def test_startup_prepare_honours_skip_flag(monkeypatch):
    called: list[dict] = []
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("QUANTTERM_SKIP_STARTUP_BOOTSTRAP", "1")
    monkeypatch.setattr(tpa, "queue_product_bootstrap", lambda **kwargs: called.append(kwargs) or {})
    tpa._startup_prepare_product()
    assert called == []


def test_scan_artifact_is_fresh_uses_scanned_at(tmp_path: Path):
    from datetime import datetime, timedelta, timezone

    from product.scan_store import save_scan, scan_artifact_is_fresh

    path = tmp_path / "latest_momentum_scan.json"
    now = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    save_scan({"schema_version": 1, "scanned_at": now.isoformat(), "records": []}, path)
    assert scan_artifact_is_fresh(path, max_age_s=6 * 3600, now=now) is True
    save_scan(
        {"schema_version": 1, "scanned_at": (now - timedelta(hours=8)).isoformat(), "records": []},
        path,
    )
    assert scan_artifact_is_fresh(path, max_age_s=6 * 3600, now=now) is False
    assert scan_artifact_is_fresh(tmp_path / "missing.json", max_age_s=6 * 3600, now=now) is False
