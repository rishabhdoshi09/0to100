"""API tests for the read-only /api/trade-plan/{symbol} endpoint (deterministic)."""
from __future__ import annotations

from fastapi.testclient import TestClient

import terminal_product_api as api


def _client(monkeypatch, *, records, health="Healthy", capital=100_000.0):
    monkeypatch.setattr(api.core, "_scan_payload", lambda: {"records": records})
    monkeypatch.setattr(api.core, "_market_payload", lambda: {"health": health})
    monkeypatch.setattr(api.core, "_json_file", lambda *a, **k: {"capital": capital})
    return TestClient(api.app)


def test_trade_plan_endpoint_returns_risk_first_plan(monkeypatch):
    rec = {"symbol": "ACME", "entry": 100.0, "stop": 95.0, "target": 115.0}
    r = _client(monkeypatch, records=[rec]).get("/api/trade-plan/ACME")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] and body["symbol"] == "ACME"
    assert body["qty"] >= 1 and body["rupee_risk"] > 0
    assert body["reward_risk"] == 3.0 and body["invalidation_pct"] == 5.0
    assert body["market_risk_factor"] == 1.0            # healthy tape → full risk


def test_trade_plan_endpoint_throttles_in_weak_tape(monkeypatch):
    rec = {"symbol": "ACME", "entry": 100.0, "stop": 95.0, "target": 115.0}
    body = _client(monkeypatch, records=[rec], health="Weak").get("/api/trade-plan/ACME").json()
    assert body["market_risk_factor"] == 0.5
    assert body["suggested_risk_pct"] == 0.005 and "throttled" in body["summary"].lower()


def test_trade_plan_endpoint_missing_symbol_is_honest(monkeypatch):
    body = _client(monkeypatch, records=[]).get("/api/trade-plan/NOPE").json()
    assert body["available"] is False and "no current scan setup" in body["message"].lower()


def test_trade_plan_endpoint_missing_stop_is_honest(monkeypatch):
    rec = {"symbol": "ACME", "entry": 100.0, "stop": 0.0, "target": 115.0}
    body = _client(monkeypatch, records=[rec]).get("/api/trade-plan/ACME").json()
    assert body["available"] is False and "no entry/stop" in body["message"].lower()


def test_portfolio_risk_source_has_no_streamlit():
    from pathlib import Path
    source = Path("risk/portfolio_risk.py").read_text(encoding="utf-8")
    assert "streamlit" not in source


def test_trade_plan_endpoint_does_not_import_streamlit(monkeypatch):
    import sys
    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    rec = {"symbol": "ACME", "entry": 100.0, "stop": 95.0, "target": 115.0}
    r = _client(monkeypatch, records=[rec]).get("/api/trade-plan/ACME")
    assert r.status_code == 200
    assert r.json()["available"] is True
    assert "streamlit" not in sys.modules
