"""Tests for the FastAPI cockpit (no live cycle, scheduler disabled)."""
import os

# disable autostart of APScheduler in tests
os.environ["SQ_AUTOSTART_SCHEDULER"] = "false"
os.environ["SQ_DB_PATH"] = "/tmp/sq_ai_test.db"
# fresh DB for each run
if os.path.exists(os.environ["SQ_DB_PATH"]):
    os.remove(os.environ["SQ_DB_PATH"])

from fastapi.testclient import TestClient   # noqa: E402

from sq_ai.api.app import app               # noqa: E402


def test_health():
    with TestClient(app) as cl:
        r = cl.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_portfolio_cold_start():
    with TestClient(app) as cl:
        r = cl.get("/api/portfolio")
        assert r.status_code == 200
        body = r.json()
        for k in ("cash", "equity", "positions"):
            assert k in body


def test_signals_empty():
    with TestClient(app) as cl:
        r = cl.get("/api/signals/latest")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


def test_manual_trade_buy_then_sell():
    with TestClient(app) as cl:
        r = cl.post("/api/trade", json={
            "symbol": "RELIANCE.NS", "action": "BUY",
            "qty": 5, "price": 2500, "stop": 2400, "target": 2800,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        # now close it
        r2 = cl.post("/api/trade", json={
            "symbol": "RELIANCE.NS", "action": "SELL",
            "qty": 5, "price": 2600,
        })
        assert r2.status_code == 200, r2.text
        assert r2.json()["pnl"] == (2600 - 2500) * 5


def test_unknown_action_rejected():
    with TestClient(app) as cl:
        r = cl.post("/api/trade", json={
            "symbol": "X.NS", "action": "TELEPORT",
            "qty": 1, "price": 1,
        })
        assert r.status_code == 400


def test_screener_endpoint_returns_list():
    with TestClient(app) as cl:
        r = cl.get("/api/screener")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
