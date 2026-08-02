"""Pre-trade cockpit: GO / CAUTION / NO_GO composition (never places orders)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from product.pre_trade import CAUTION, GO, NO_GO, build_pre_trade


def _base_plan(**overrides):
    plan = {
        "available": True,
        "tradeable": True,
        "symbol": "ACME",
        "entry": 100.0,
        "stop": 95.0,
        "target": 115.0,
        "qty": 200,
        "rupee_risk": 1000.0,
        "heat_verdict": "OK",
        "market_health": "healthy",
        "market_risk_factor": 1.0,
        "correlation_status": "independent",
        "correlated_with": [],
        "cost_drag_r": 0.08,
        "summary": "200 sh · risk ₹1,000",
        "capital": 100_000.0,
    }
    plan.update(overrides)
    return plan


def test_pre_trade_go_when_clean(monkeypatch):
    import scan.signal_backtest as sb

    monkeypatch.setattr(sb, "load_report", lambda: None)
    monkeypatch.setattr(sb, "report_is_actionable", lambda _r=None: False)
    body = build_pre_trade(symbol="acme", plan=_base_plan())
    assert body["verdict"] == GO
    assert body["tradeable"] is True
    assert body["places_orders"] is False
    assert body["blockers"] == []
    assert "not a signal" in body["honesty"].lower() or "not a signal" in body["meaning"].lower()


def test_pre_trade_blocks_proven_loser_edge(monkeypatch):
    import scan.signal_backtest as sb

    monkeypatch.setattr(sb, "load_report", lambda: {"signals": {"X": {}}, "universe": {"run": 800}})
    monkeypatch.setattr(sb, "report_is_actionable", lambda _r=None: True)
    monkeypatch.setattr(sb, "universe_evidence_note", lambda _r=None: "800 stocks")
    body = build_pre_trade(
        symbol="LOSE",
        plan=_base_plan(),
        scan_record={"symbol": "LOSE", "edge_r": -0.2, "signals": ["PRE_BREAKOUT"], "verdict": "BUY"},
    )
    assert body["verdict"] == NO_GO
    assert body["measured_edge_r"] == -0.2
    assert any("loser" in b.lower() or "-0.20" in b or "-0.2" in b for b in body["blockers"])


def test_pre_trade_no_go_without_plan():
    body = build_pre_trade(
        symbol="ACME",
        plan={"available": False, "message": "No current scan setup for ACME."},
    )
    assert body["verdict"] == NO_GO
    assert body["tradeable"] is False
    assert any("scan" in b.lower() or "plan" in b.lower() or "setup" in b.lower() for b in body["blockers"])


def test_pre_trade_no_go_on_book_danger():
    body = build_pre_trade(symbol="ACME", plan=_base_plan(heat_verdict="DANGER"))
    assert body["verdict"] == NO_GO
    assert any("open-risk" in b.lower() or "breach" in b.lower() for b in body["blockers"])


def test_pre_trade_caution_on_weak_tape_and_correlation():
    body = build_pre_trade(
        symbol="ACME",
        plan=_base_plan(
            market_health="weak",
            market_risk_factor=0.5,
            correlation_status="adds_to_bet",
            correlated_with=["HDFCBANK"],
            heat_verdict="CAUTION",
        ),
    )
    assert body["verdict"] == CAUTION
    assert len(body["warnings"]) >= 2


def test_pre_trade_hard_data_gap_is_blocker():
    body = build_pre_trade(
        symbol="ACME",
        plan=_base_plan(),
        readiness={
            "retail_research_checklist": {
                "gaps": [
                    {
                        "key": "corporate_actions",
                        "label": "Corporate actions ledger",
                        "status": "MISSING",
                        "next_action": "Run ca-ingest",
                    }
                ]
            },
            "lanes": [],
        },
    )
    assert body["verdict"] == NO_GO
    assert any("corporate actions" in b.lower() for b in body["blockers"])


def test_pre_trade_endpoint_composes(monkeypatch):
    import terminal_product_api as api

    monkeypatch.setattr(
        api.core,
        "_scan_payload",
        lambda: {"records": [{"symbol": "ACME", "entry": 100.0, "stop": 95.0, "target": 115.0}]},
    )
    monkeypatch.setattr(api.core, "_market_payload", lambda: {"health": "Healthy"})
    monkeypatch.setattr(api.core, "_json_file", lambda *a, **k: {"capital": 100_000.0})
    monkeypatch.setattr(api.core, "_paper_payload", lambda: {"open_positions": [], "capital": 100_000.0})
    monkeypatch.setattr(
        api,
        "product_readiness",
        lambda: {"lanes": [], "retail_research_checklist": {"gaps": []}},
    )
    monkeypatch.setattr(api, "book_correlation", lambda: {"n_positions": 0, "n_bets": 0})

    client = TestClient(api.app)
    r = client.get("/api/pre-trade/ACME")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "ACME"
    assert body["verdict"] in {GO, CAUTION, NO_GO}
    assert body["places_orders"] is False
    assert "plan" in body and body["plan"]["available"] is True
