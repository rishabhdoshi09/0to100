"""Product bootstrap queues real data lanes on demand and at API start."""
from __future__ import annotations

from pathlib import Path

import terminal_product_api as tpa
from operations.market_ops import DATA_PREPARE, LONG_TERM_REFRESH, MARKET_SCAN, NEWS_REFRESH


def test_queue_product_bootstrap_enqueues_four_lanes(tmp_path: Path, monkeypatch):
    store_path = tmp_path / "jobs.db"
    monkeypatch.setattr(tpa.core, "OPS_DB", store_path)
    monkeypatch.setattr(tpa.core, "_ensure_ops_worker", lambda: {"running": True})

    payload = tpa.queue_product_bootstrap(requested_by="api_startup")
    assert payload["accepted"] is True
    kinds = {item["kind"] for item in payload["operations"]}
    assert kinds == {DATA_PREPARE, NEWS_REFRESH, MARKET_SCAN, LONG_TERM_REFRESH}
    assert all(item["created"] for item in payload["operations"])
    assert all(item["status"] == "PENDING" for item in payload["operations"])

    again = tpa.queue_product_bootstrap(requested_by="api_startup")
    assert all(item["created"] is False for item in again["operations"])


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
