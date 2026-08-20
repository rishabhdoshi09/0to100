"""Trade desk — Ready queue, backtest lab, paper→live. ALPHA/BETA only."""
from __future__ import annotations

from product.trade_desk import (
    arm_paper_only,
    build_backtest_lab,
    build_live_journey,
    build_ready_queue,
)


def _buy(symbol: str, **extra):
    row = {
        "symbol": symbol,
        "company": f"{symbol} Co",
        "verdict": "BUY",
        "status": "Ready to trade",
        "price": 100.0,
        "entry": 100.0,
        "stop": 92.0,
        "target": 118.0,
        "score": 78.0,
        "signals": ["MOMENTUM"],
        "categories": ["Momentum"],
        "high_conviction": True,
        "breakout_conviction": 72.0,
        "avg_vol20": 800000,
        "chase_risk": False,
        "edge_r": 0.18,
        "ev_pct": 2.4,
        "ev_lb_pct": 1.6,
        "ev_n": 80,
        "ev_conf": "HIGH",
        "p_win": 58.0,
        "sector": "Chemicals",
    }
    row.update(extra)
    return row


def test_ready_queue_empty_is_honest():
    out = build_ready_queue(scan={"records": []}, market={})
    assert out["empty"] is True
    assert out["prime"] == []
    assert out["places_orders"] is False
    assert out["live_locked"] is True
    assert any("scan" in w.lower() for w in out["empty_why"])


def test_prime_survives_money_gates():
    scan = {"records": [_buy("ALPHA")], "scanned_at": "2026-08-20", "universe_size": 1}
    out = build_ready_queue(scan=scan, market={"health": "healthy"})
    assert [c["symbol"] for c in out["prime"]] == ["ALPHA"]
    assert out["prime"][0]["entry"] == 100.0
    assert out["prime"][0]["stop"] == 92.0
    assert out["empty"] is False


def test_negative_edge_never_makes_ready():
    scan = {"records": [_buy("BETA", edge_r=-0.4, ev_lb_pct=-0.8, ev_pct=-0.2)]}
    out = build_ready_queue(scan=scan, market={})
    assert out["prime"] == []
    assert all(c["symbol"] != "BETA" for c in out["actionable"])
    assert any(r["symbol"] == "BETA" for r in out["rejected_sample"])


def test_chase_is_not_ready():
    scan = {"records": [_buy("ALPHA", chase_risk=True, status="Wait for pullback")]}
    out = build_ready_queue(scan=scan, market={})
    assert out["prime"] == []
    assert out["actionable"] == []


def test_missing_stop_is_not_a_ticket():
    scan = {"records": [_buy("ALPHA", stop=0, entry=100)]}
    out = build_ready_queue(scan=scan, market={})
    assert out["prime"] == []
    assert out["actionable"] == []


def test_does_not_invent_ev_on_thin_sample():
    scan = {"records": [_buy("ALPHA", ev_pct=None, ev_lb_pct=None, ev_n=8, p_win=None)]}
    out = build_ready_queue(scan=scan, market={})
    cards = out["prime"] + out["actionable"]
    assert cards, "plan can still surface without an EV claim"
    for card in cards:
        assert "ev_pct" not in card or card.get("ev_n", 0) >= 30


def test_lab_use_cases_and_no_orders():
    report = {
        "generated_at": "2026-08-20",
        "symbols": 400,
        "universe": {"run": 400, "available": 400, "truncated": False},
        "signals": {
            "momentum": {
                "trades": 120, "closed": 120, "win_rate": 55.0,
                "expectancy_r": 0.22, "verdict": "PROVEN",
            },
            "nr7": {
                "trades": 80, "closed": 80, "win_rate": 41.0,
                "expectancy_r": -0.15, "verdict": "LOSER",
            },
        },
    }
    playbook = {
        "regime": "TRENDING",
        "best": [{"signal": "momentum", "expectancy_r": 0.22, "trades": 90, "basis": "regime"}],
        "avoid": ["nr7"],
    }
    lab = build_backtest_lab(report=report, status={"running": False, "has_report": True}, playbook=playbook)
    assert lab["actionable"] is True
    assert lab["places_orders"] is False
    assert lab["live_locked"] is True
    ids = [u["id"] for u in lab["use_cases"]]
    assert ids == ["trust_scanner", "regime_lean", "avoid_losers", "paper_loop"]
    assert lab["playbook"]["best"][0]["signal"] == "momentum"
    assert "nr7" in lab["playbook"]["avoid"]
    assert lab["loser_n"] == 1


def test_lab_missing_report_is_not_actionable():
    lab = build_backtest_lab(report={}, status={"running": False}, playbook={})
    assert lab["actionable"] is False
    trust = lab["use_cases"][0]
    assert trust["status"] in {"MISSING", "PARTIAL"}


def test_journey_never_unlocks_live_click(monkeypatch):
    monkeypatch.setattr(
        "product.trade_desk.live_arm_allowed",
        lambda **_k: (True, "LIVE unlocked — every ladder gate cleared."),
    )
    journey = build_live_journey(
        data={"bhavcopy": {"ready": True, "sessions": 400}},
        scan={"available": True, "scanned_at": "2026-08-20"},
        paper={"available": True},
        autonomy={"available": True},
        env={"QT_LIVE_ENABLED": "1"},
        paper_closed=300,
        ladder={
            "rung": {"id": "LIVE", "label": "LIVE"},
            "paper_closed": 300,
            "paper_e4_n": 300,
            "alpha": {"level": 4, "label": "E4"},
            "live_unlocked": True,
            "live_blockers": [],
            "subsystems": [{"id": "tape", "status": "READY"}],
        },
        autopilot={"armed": True, "mode": "PAPER", "allocation": 25000, "open_trades": []},
        report_card={
            "verdict": "READY_CANDIDATE",
            "verdict_reason": "earned",
            "stats": {"n": 80, "expectancy_r": 0.2, "profit_factor": 1.5},
        },
        diagnose={"headline": "armed", "blockers": []},
        lab={"actionable": True, "evidence_note": "400 stocks", "running": False},
    )
    assert journey["live_unlocked"] is False
    assert journey["places_orders"] is False
    live_step = next(s for s in journey["steps"] if s["id"] == "live_lock")
    assert live_step["status"] == "LOCKED"
    assert "ARM LIVE" in (live_step["detail"] + live_step["next_action"])


def test_arm_paper_refuses_to_leave_paper_mode(tmp_path, monkeypatch):
    import execution.autopilot as ap

    monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "autopilot.json")
    monkeypatch.setattr(ap, "_state", {}, raising=False)
    ap._state = {}
    monkeypatch.setattr(ap, "_notify", lambda *_a, **_k: None)
    monkeypatch.setattr(ap, "_log_activity", lambda *_a, **_k: None)
    monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
    out = arm_paper_only(allocation=25_000)
    assert out["ok"] is True
    assert out["mode"] == "PAPER"
    assert out["armed"] is True
    assert out["live_locked"] is True
    status = ap.get_status()
    assert status["mode"] == "PAPER"
    assert status["armed"] is True


def test_arm_paper_http_rejects_live_mode(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import execution.autopilot as ap
    import terminal_product_api as api

    monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "autopilot.json")
    monkeypatch.setattr(ap, "_state", {}, raising=False)
    ap._state = {}
    client = TestClient(api.app)
    r = client.post("/api/autopilot/arm-paper", json={"mode": "LIVE", "allocation": 25000})
    assert r.status_code == 400
    assert "LIVE" in (r.json().get("detail") or "")
