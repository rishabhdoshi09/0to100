"""Trade desk — Ready queue, backtest lab, paper→live. ALPHA/BETA only."""
from __future__ import annotations

from product.trade_desk import (
    arm_paper_only,
    build_backtest_lab,
    build_live_journey,
    build_ready_queue,
    feed_paper_classroom,
    ticket_quality,
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


def _queue(scan, market=None, workspace=None):
    return build_ready_queue(scan=scan, market=market or {}, workspace=workspace or {})


def test_ready_queue_empty_is_honest():
    out = _queue({"records": []})
    assert out["empty"] is True
    assert out["prime"] == []
    assert out["stage2"] == []
    assert out["places_orders"] is False
    assert out["live_locked"] is True
    assert any("scan" in w.lower() for w in out["empty_why"])


def test_prime_survives_money_gates():
    scan = {"records": [_buy("ALPHA")], "scanned_at": "2026-08-20", "universe_size": 1}
    out = _queue(scan, {"health": "healthy"})
    assert [c["symbol"] for c in out["prime"]] == ["ALPHA"]
    assert out["prime"][0]["entry"] == 100.0
    assert out["prime"][0]["stop"] == 92.0
    assert out["empty"] is False


def test_negative_edge_never_makes_ready():
    scan = {"records": [_buy("BETA", edge_r=-0.4, ev_lb_pct=-0.8, ev_pct=-0.2)]}
    out = _queue(scan)
    assert out["prime"] == []
    assert all(c["symbol"] != "BETA" for c in out["actionable"])
    assert all(c["symbol"] != "BETA" for c in out["stage2"])
    assert any(r["symbol"] == "BETA" for r in out["rejected_sample"])


def test_chase_is_not_ready():
    scan = {"records": [_buy("ALPHA", chase_risk=True, status="Wait for pullback")]}
    out = _queue(scan)
    assert out["prime"] == []
    assert out["actionable"] == []
    assert out["stage2"] == []


def test_missing_stop_is_not_a_ticket():
    scan = {"records": [_buy("ALPHA", stop=0, entry=100)]}
    out = _queue(scan)
    assert out["prime"] == []
    assert out["actionable"] == []
    assert out["stage2"] == []


def test_does_not_invent_ev_on_thin_sample():
    scan = {"records": [_buy("ALPHA", ev_pct=None, ev_lb_pct=None, ev_n=8, p_win=None)]}
    out = _queue(scan)
    cards = out["prime"] + out["actionable"] + out["stage2"]
    assert cards, "plan can still surface without an EV claim"
    for card in cards:
        assert "ev_pct" not in card or card.get("ev_n", 0) >= 30


def test_pattern_prebreakout_is_a_ticket():
    """Real scans are Pattern/PreBreakout — those used to vanish behind a Momentum veto."""
    row = _buy(
        "ALPHA",
        signals=["TRIANGLE", "CUP_HANDLE", "ACCUMULATION", "POCKET_PIVOT"],
        categories=["Pattern", "PreBreakout"],
        high_conviction=False,
        breakout_conviction=0,
        edge_r=None,
        ev_pct=None,
        ev_lb_pct=None,
        ev_n=0,
        p_win=None,
        rsi=58,
        volume_ratio=1.4,
    )
    out = _queue({"records": [row]})
    assert out["empty"] is False
    assert out["prime"] == []
    assert [c["symbol"] for c in out["actionable"]] == ["ALPHA"]
    assert out["actionable"][0]["atq"] > 0
    assert "ev_pct" not in out["actionable"][0]


def test_blowoff_rsi_is_not_ready():
    out = _queue({"records": [_buy("ALPHA", rsi=83, high_conviction=False, breakout_conviction=0)]})
    assert out["prime"] == []
    assert out["actionable"] == []


def test_sepa_stage2_overlay_from_cache_not_rescan():
    scan = {
        "records": [
            _buy(
                "ALPHA",
                signals=["TRIANGLE"],
                categories=["Pattern"],
                high_conviction=False,
                breakout_conviction=0,
                edge_r=None,
                ev_pct=None,
                ev_lb_pct=None,
                ev_n=0,
                score=60,
                rsi=55,
                volume_ratio=1.5,
            )
        ]
    }
    workspace = {
        "categories": [
            {
                "id": "best_setups",
                "cards": [
                    {
                        "symbol": "ALPHA",
                        "company": "Alpha Co",
                        "sepa_score": 88,
                        "sepa_passed": 7,
                        "sepa_total": 7,
                        "sepa_headline": "Stage 2 SEPA",
                        "stage_label": "STAGE 2",
                        "entry": 100.0,
                        "stop": 92.0,
                        "target": 118.0,
                    }
                ],
            }
        ]
    }
    out = _queue(scan, workspace=workspace)
    assert [c["symbol"] for c in out["stage2"]] == ["ALPHA"]
    assert out["stage2"][0]["sepa_score"] == 88
    assert out["stage2"][0]["lane"] == "stage2"
    assert all(c["symbol"] != "ALPHA" for c in out["actionable"])


def test_atq_ranks_higher_sepa_ahead_of_weaker_template():
    a = _buy(
        "ALPHA",
        signals=["TRIANGLE"],
        categories=["Pattern"],
        high_conviction=False,
        breakout_conviction=0,
        edge_r=None,
        ev_pct=None,
        ev_lb_pct=None,
        ev_n=0,
        score=50,
        sepa_score=90,
        volume_ratio=1.8,
        rsi=55,
    )
    b = _buy(
        "BETA",
        signals=["TRIANGLE"],
        categories=["Pattern"],
        high_conviction=False,
        breakout_conviction=0,
        edge_r=None,
        ev_pct=None,
        ev_lb_pct=None,
        ev_n=0,
        score=50,
        sepa_score=40,
        volume_ratio=0.6,
        rsi=70,
    )
    out = _queue({"records": [a, b]})
    assert [c["symbol"] for c in out["actionable"]] == ["ALPHA", "BETA"]
    assert out["actionable"][0]["atq"] > out["actionable"][1]["atq"]


def test_ticket_quality_is_zero_on_blowoff_and_rewards_sepa():
    base = {
        "entry": 100, "stop": 92, "target": 118, "score": 70,
        "volume_ratio": 1.5, "rsi": 55,
    }
    assert ticket_quality({**base, "sepa_score": 90}) > ticket_quality({**base, "sepa_score": 40})
    assert ticket_quality({**base, "rsi": 83}) == 0.0
    missing_ev = ticket_quality(base)
    assert missing_ev > 0
    assert ticket_quality({**base, "edge_r": -0.4}) == 0.0


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
    lesson = lab["lesson"]
    assert lesson["title"] == "What is a backtest?"
    assert "never place an order" in lesson["plain"].lower() or "never places" in lesson["plain"].lower() or "never place" in lesson["plain"]
    assert len(lesson["steps"]) == 4
    board = lab["scoreboard"]
    assert [s["signal"] for s in board["keep"]] == ["momentum"]
    assert [s["signal"] for s in board["skip"]] == ["nr7"]
    assert lab["signals"][0]["kid_label"] == "Passed"
    assert any(s["kid_label"] == "Failed" for s in lab["signals"])
    assert lab["lesson"]["cta"] == "Run the practice test"
    assert lab["schema_version"] >= 3
    assert [n["id"] for n in lab["loop"]] == ["practice", "teach", "paper", "recos"]
    assert lab["learning"]["demote_only"] is True
    assert "momentum" in lab["learning"]["keep"]
    assert "nr7" in lab["learning"]["skip"]
    assert lab["classroom"]["live_locked"] is True
    assert lab["pulse"]["tone"] in {"idle", "ready", "wait", "live", "run"}


def test_lab_missing_report_is_not_actionable():
    lab = build_backtest_lab(report={}, status={"running": False}, playbook={})
    assert lab["actionable"] is False
    trust = lab["use_cases"][0]
    assert trust["status"] in {"MISSING", "PARTIAL"}
    assert "never spends money" in lab["scoreboard"]["headline"]
    assert lab["lesson"]["cta"] == "Run the practice test"
    assert lab["places_orders"] is False
    assert lab["live_locked"] is True
    assert lab["pulse"]["tone"] in {"idle", "ready", "wait", "live", "run"}
    assert lab["classroom"]["armed"] in {True, False}


def test_lab_running_progress_is_honest():
    lab = build_backtest_lab(
        report={},
        status={"running": True, "progress": 12, "total": 80, "current": "RELIANCE"},
        playbook={},
    )
    assert lab["running"] is True
    assert lab["current"] == "RELIANCE"
    assert lab["pulse"]["tone"] == "run"
    assert "RELIANCE" in lab["lesson"]["now"]
    assert lab["loop"][0]["state"] == "RUN"


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
    monkeypatch.setattr("product.trade_desk.feed_paper_classroom", lambda: {
        "ok": True, "fed": 0, "message": "no scan", "live_locked": True,
    })
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


def test_feed_paper_classroom_noop_when_disarmed(tmp_path, monkeypatch):
    import execution.autopilot as ap

    monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "autopilot.json")
    monkeypatch.setattr(ap, "_state", {}, raising=False)
    ap._state = {}
    out = feed_paper_classroom()
    assert out["ok"] is False
    assert out["fed"] == 0
    assert out["live_locked"] is True
    assert "Arm paper" in out["message"]


def test_feed_paper_classroom_sends_ready_rows(tmp_path, monkeypatch):
    import execution.autopilot as ap

    monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "autopilot.json")
    monkeypatch.setattr(ap, "_state", {}, raising=False)
    ap._state = {}
    ap.set_config(mode="PAPER", allocation=25_000)
    monkeypatch.setattr(ap, "_notify", lambda *_a, **_k: None)
    monkeypatch.setattr(ap, "_log_activity", lambda *_a, **_k: None)
    monkeypatch.setattr(ap, "start_book_monitor", lambda: None)
    ap.arm("")
    fed = []
    monkeypatch.setattr(ap, "on_setups", lambda rows: fed.append(list(rows)))
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda: {
            "scanned_at": "2026-08-21",
            "records": [
                {
                    "symbol": "ALPHA",
                    "verdict": "BUY",
                    "status": "Ready to trade",
                    "entry": 100,
                    "stop": 92,
                    "score": 70,
                    "volume_ratio": 1.4,
                    "rsi": 55,
                    "chase_risk": False,
                }
            ],
        },
    )
    monkeypatch.setattr(ap, "diagnose_silence", lambda: {
        "headline": "armed", "in_window": True, "blockers": [], "considered_today": 1,
    })
    out = feed_paper_classroom()
    assert out["ok"] is True
    assert out["fed"] >= 1
    assert out["live_locked"] is True
    assert fed and any(r.get("symbol") == "ALPHA" for r in fed[0])


def test_ready_queue_exposes_lab_applied():
    out = _queue({"records": []})
    assert "lab_applied" in out
    assert out["lab_applied"]["plain"]


def test_paper_feed_http_requires_arm(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import execution.autopilot as ap
    import terminal_product_api as api

    monkeypatch.setattr(ap, "_STATE_FILE", tmp_path / "autopilot.json")
    monkeypatch.setattr(ap, "_state", {}, raising=False)
    ap._state = {}
    client = TestClient(api.app)
    r = client.post("/api/trade-desk/paper-feed")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["live_locked"] is True
