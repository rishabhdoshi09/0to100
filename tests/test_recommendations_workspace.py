"""Tests for Reco-style recommendations / market-reports projections."""
from __future__ import annotations

from product.recommendations_workspace import (
    build_market_reports_workspace,
    build_recommendations_workspace,
    card_from_row,
    primary_scan_category,
    risk_tier,
    upside_metrics,
)
from product.radar_workspace import enrich_scan_row


def test_upside_and_risk_from_real_fields_only():
    ups = upside_metrics({"price": 115, "entry": 100, "target": 130})
    assert ups["upside_from_entry_pct"] == 15.0
    assert ups["upside_to_target_pct"] == 13.0
    assert upside_metrics({"price": 0, "entry": 0})["upside_from_entry_pct"] is None
    assert risk_tier({"chase_risk": True}) == "High"
    assert risk_tier({"rsi": 50, "chase_risk": False}) == "Low"


def test_categories_project_from_scan_and_long_term_without_invention():
    scan = {
        "scanned_at": "2026-08-15T10:00:00+00:00",
        "records_status": "CURRENT_DAY",
        "same_ist_day": True,
        "records": [
            {
                "symbol": "TRENDY", "company": "Trendy Co", "score": 70,
                "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                "chase_risk": False, "price": 110, "entry": 100, "target": 125,
                "stop": 95, "rsi": 58, "volume_ratio": 1.5, "avg_vol20": 1e6,
                "momentum_5d": 5, "above_sma50": True, "reasons": ["RS leadership"],
            },
            {
                "symbol": "SNIPE", "company": "Snipe Ltd", "score": 80,
                "signals": ["PRE_BREAKOUT", "BREAKOUT_52W"], "verdict": "BUY",
                "status": "Watch for breakout", "categories": ["PreBreakout"],
                "pivot_distance_pct": 1.0, "chase_risk": False,
                "price": 200, "entry": 201, "target": 230, "stop": 190,
                "rsi": 55, "volume_ratio": 1.8, "avg_vol20": 2e6,
                "breakout_grade": "A", "breakout_conviction": 85,
                "reasons": ["Clean pivot"],
            },
            {
                "symbol": "RECOV", "company": "Recovery Inc", "score": 60,
                "signals": ["DOUBLE_BOTTOM", "ACCUMULATION"], "verdict": "WATCH",
                "status": "Watch", "chase_risk": False,
                "price": 50, "entry": 48, "target": 60, "stop": 45,
                "rsi": 45, "volume_ratio": 1.2, "avg_vol20": 5e5,
                "reasons": ["Base forming"],
            },
            {
                "symbol": "CHASE", "company": "Chase Co", "score": 90,
                "signals": ["MOMENTUM"], "verdict": "WATCH", "status": "Wait for pullback",
                "chase_risk": True, "price": 300, "entry": 280, "target": 340,
                "rsi": 80, "volume_ratio": 2.0, "avg_vol20": 1e6,
            },
            {
                # Coil ≠ recovery — must not land in Recovery Setups.
                "symbol": "COIL", "company": "Coil Ltd", "score": 65,
                "signals": ["CUP_HANDLE", "POCKET_PIVOT", "NR7_COIL"], "verdict": "WATCH",
                "status": "Watch", "chase_risk": False,
                "price": 90, "entry": 88, "target": 100, "rsi": 50,
                "volume_ratio": 1.1, "avg_vol20": 8e5,
            },
        ],
    }
    long_term = {
        "scanned_at": "2026-08-15T09:00:00+00:00",
        "records": [
            {
                "symbol": "QUAL", "company": "Quality Ltd",
                "classification": "QUALITY_COMPOUNDER",
                "fundamental_coverage": 0.8, "fundamental_score": 75,
                "combined_score": 82, "price": 500, "entry": 480, "target": 600,
                "quality_factors": ["ROCE", "Low debt"], "risk_flags": [],
            },
            {
                "symbol": "THIN", "classification": "QUALITY_COMPOUNDER",
                "fundamental_coverage": 0.2, "combined_score": 70,
            },
        ],
    }
    payload = build_recommendations_workspace(
        scan_payload=scan, long_term_payload=long_term, refresh_technicals=False,
    )
    by_id = {c["id"]: c for c in payload["categories"]}
    assert payload["schema_version"] == 3
    assert by_id["wealth_builders"]["count"] == 1
    assert by_id["wealth_builders"]["cards"][0]["symbol"] == "QUAL"
    assert "coverage" in (by_id["wealth_builders"]["cards"][0].get("qualify_reason") or "").lower()
    assert "QUALITY COMPOUNDER" not in " ".join(by_id["wealth_builders"]["cards"][0].get("why_now") or [])
    assert by_id["super_trends"]["count"] >= 1
    assert any(c["symbol"] == "TRENDY" for c in by_id["super_trends"]["cards"])
    assert not any(c["symbol"] == "CHASE" for c in by_id["super_trends"]["cards"])
    assert any(c["symbol"] == "SNIPE" for c in by_id["momentum_breakouts"]["cards"])
    assert any(c["symbol"] == "RECOV" for c in by_id["recovery_setups"]["cards"])
    assert not any(c["symbol"] == "COIL" for c in by_id["recovery_setups"]["cards"])
    card = card_from_row(scan["records"][0], category_id="super_trends", category_label="Super Trends")
    assert card["action_badge"] == "Buy"
    assert card["upside_from_entry_pct"] == 10.0
    assert card["opportunity_label"] == "OPPORTUNITY"
    assert card["buy_zone_low"] == 100.0
    assert card["buy_zone_high"] == 100.0
    assert card["stop"] == 95.0
    assert card["expected_payoff"] == "Unproven"
    assert card["evidence"] == "Thin"
    assert "Momentum improving" in card["why_now"]
    assert any("95" in item for item in card["what_changes_mind"])
    assert "lifecycle" in payload
    assert payload["disclaimer"]
    assert payload["desk"]["market_support"] == "Unmeasured"


def test_scan_categories_are_exclusive_primary():
    """A symbol must not appear in two scan buckets."""
    scan = {
        "scanned_at": "2026-08-15T10:00:00+00:00",
        "records": [
            {
                # Would match breakout AND could look trendy — breakout wins.
                "symbol": "BOTH", "score": 85,
                "signals": ["PRE_BREAKOUT", "MOMENTUM", "BREAKOUT_52W"],
                "verdict": "BUY", "status": "Watch for breakout",
                "categories": ["PreBreakout"], "pivot_distance_pct": 0.8,
                "chase_risk": False, "rsi": 55, "volume_ratio": 1.5, "avg_vol20": 1e6,
                "breakout_grade": "B", "above_sma50": True, "momentum_5d": 4,
            },
            {
                # Recovery + momentum signal — recovery before super-trend.
                "symbol": "TURN", "score": 62,
                "signals": ["DOUBLE_BOTTOM", "MOMENTUM"], "verdict": "WATCH",
                "status": "Watch", "chase_risk": False, "rsi": 48,
                "volume_ratio": 1.2, "avg_vol20": 5e5, "above_sma50": True,
                "momentum_5d": 3,
            },
        ],
    }
    payload = build_recommendations_workspace(
        scan_payload=scan, long_term_payload={"records": []}, refresh_technicals=False,
    )
    by_id = {c["id"]: c for c in payload["categories"]}
    scan_syms = []
    for cat_id in ("super_trends", "momentum_breakouts", "recovery_setups"):
        scan_syms.extend(c["symbol"] for c in by_id[cat_id]["cards"])
    assert len(scan_syms) == len(set(scan_syms))
    assert any(c["symbol"] == "BOTH" for c in by_id["momentum_breakouts"]["cards"])
    assert not any(c["symbol"] == "BOTH" for c in by_id["super_trends"]["cards"])
    assert any(c["symbol"] == "TURN" for c in by_id["recovery_setups"]["cards"])
    assert not any(c["symbol"] == "TURN" for c in by_id["super_trends"]["cards"])


def test_cup_handle_alone_is_not_recovery():
    row = enrich_scan_row({
        "symbol": "CUP", "signals": ["CUP_HANDLE"], "score": 70,
        "chase_risk": False, "rsi": 50, "volume_ratio": 1.2, "avg_vol20": 1e6,
        "verdict": "WATCH", "status": "Watch",
    })
    assert primary_scan_category(row) is None or primary_scan_category(row)[0] != "recovery_setups"


def test_negative_momentum_not_super_trend():
    row = enrich_scan_row({
        "symbol": "FADE", "signals": ["MOMENTUM"], "score": 90,
        "verdict": "BUY", "status": "Ready to trade", "chase_risk": False,
        "rsi": 58, "volume_ratio": 1.4, "avg_vol20": 1e6,
        "momentum_5d": -1.5, "above_sma50": True,
    })
    assigned = primary_scan_category(row)
    assert assigned is None or assigned[0] != "super_trends"


def test_wealth_empty_explains_needs_fundamentals():
    payload = build_recommendations_workspace(
        scan_payload={"records": [], "scanned_at": ""},
        long_term_payload={
            "records": [
                {
                    "symbol": "X", "classification": "NEEDS_FUNDAMENTALS",
                    "fundamental_coverage": 0.0, "combined_score": 45,
                },
            ],
        },
        refresh_technicals=False,
    )
    wealth = next(c for c in payload["categories"] if c["id"] == "wealth_builders")
    assert wealth["count"] == 0
    assert "NEEDS_FUNDAMENTALS" in wealth["empty_detail"]


def test_empty_recovery_is_honest_empty():
    payload = build_recommendations_workspace(
        scan_payload={"records": [], "scanned_at": ""},
        long_term_payload={"records": []},
        refresh_technicals=False,
    )
    recovery = next(c for c in payload["categories"] if c["id"] == "recovery_setups")
    assert recovery["count"] == 0
    assert recovery["cards"] == []
    assert "double-bottom" in recovery["empty_detail"].lower() or "accumulation" in recovery["empty_detail"].lower()


def test_market_reports_lists_today_pulse(tmp_path, monkeypatch):
    import product.recommendations_workspace as rw

    monkeypatch.setattr(rw, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        "reports.street_pulse.build_pulse",
        lambda: {
            "date": "15 August 2026",
            "takeaways": ["Nifty steady", "Breadth mixed"],
            "gainers": [], "losers": [], "breakouts_today": [],
        },
    )
    payload = build_market_reports_workspace(persist_today=True)
    assert payload["reports"]
    assert payload["reports"][0]["is_new"] is True
    assert payload["reports"][0]["title"] == "Market Pulse"
    assert (tmp_path / list(tmp_path.glob("market_pulse_*.json"))[0]).exists()
    assert "Nifty" in payload["reports"][0]["summary"]
    assert payload["reports"][0]["is_new"] is True
    assert payload.get("as_of_ist")


def test_old_market_report_is_not_marked_today(tmp_path, monkeypatch):
    import json
    import product.recommendations_workspace as rw

    monkeypatch.setattr(rw, "REPORTS_DIR", tmp_path)
    old = tmp_path / "market_pulse_2026-01-01.json"
    old.write_text(json.dumps({
        "id": "market_pulse_2026-01-01",
        "title": "Market Pulse",
        "kind": "market_pulse",
        "date": "2026-01-01",
        "created_at": "2026-01-01T00:00:00+00:00",
        "pulse": {"takeaways": ["Ancient tape"], "as_of_ist": "2026-01-01"},
    }), encoding="utf-8")
    monkeypatch.setattr(
        "reports.street_pulse.build_pulse",
        lambda: {
            "date": "16 August 2026",
            "as_of_ist": "2026-08-16",
            "takeaways": ["Fresh Nifty"],
            "gainers": [], "losers": [], "breakouts_today": [],
        },
    )
    monkeypatch.setattr(rw, "_ist_day", lambda: "2026-08-16")
    payload = build_market_reports_workspace(persist_today=True)
    assert payload["reports"][0]["is_new"] is True
    assert payload["reports"][0]["date"] == "2026-08-16"
    assert any(r["date"] == "2026-01-01" and r["is_new"] is False for r in payload["reports"])
    assert "Ancient" not in payload["reports"][0]["summary"]


def test_pulse_uses_ist_day_and_breakout_keys(monkeypatch):
    from reports import street_pulse as sp

    monkeypatch.setattr(sp, "_ist_today_label", lambda: "16 August 2026")
    monkeypatch.setattr(sp, "_ist_today_iso", lambda: "2026-08-16")
    monkeypatch.setattr(sp, "_scan_rows_latest", lambda: (
        [{"symbol": "AAA", "signals": ["BREAKOUT_52W"], "breakout_grade": "A",
          "change_pct": 4, "volume_ratio": 2.2, "reasons": ["break"],
          "categories": [], "entry": 10, "pivot_distance_pct": 0.4}],
        1,
    ))
    monkeypatch.setattr(sp, "_movers_from_bhav", lambda: ([], []))
    monkeypatch.setattr(sp, "_market_snapshot", lambda: {"indices": [], "commentary": ""})
    monkeypatch.setattr(sp, "_losing_momentum", lambda: None)
    monkeypatch.setattr(sp, "_headlines", lambda: [])
    pulse = sp.build_pulse()
    assert pulse["as_of_ist"] == "2026-08-16"
    assert pulse["date"] == "16 August 2026"
    assert pulse["breakouts_today"][0]["symbol"] == "AAA"


def test_prior_day_auto_scan_is_not_used_as_today(monkeypatch):
    from reports import street_pulse as sp

    monkeypatch.setattr(
        "scan.auto_scan.get_results",
        lambda: ([{"symbol": "OLD", "signals": ["BREAKOUT_52W"]}], 1, 1_700_000_000.0, "ready"),
    )
    monkeypatch.setattr(sp, "_ts_is_ist_today", lambda ts: False)
    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda: {"same_ist_day": False, "records": [{"symbol": "STALE"}], "universe_size": 1},
    )
    rows, universe = sp._scan_rows_latest()
    assert rows == []
    assert universe == 0


def test_report_item_carries_session_movers():
    from product.recommendations_workspace import _report_item

    item = _report_item({
        "id": "market_pulse_2026-08-16",
        "title": "Market Pulse",
        "kind": "market_pulse",
        "date": "2026-08-16",
        "pulse": {
            "as_of_ist": "2026-08-16",
            "takeaways": ["NIFTY ▲ 0.40%"],
            "gainers": [{"symbol": "AAA", "price": 10, "chg_pct": 4.2}],
            "losers": [{"symbol": "BBB", "price": 8, "chg_pct": -3.1}],
            "snapshot": {"indices": [{"name": "NIFTY 50", "price": 25000, "chg_pct": 0.4}], "commentary": ""},
            "breakouts_today": [{"symbol": "CCC"}],
        },
    }, today="2026-08-16")
    assert item["is_new"] is True
    assert item["gainers"][0]["symbol"] == "AAA"
    assert item["losers"][0]["chg_pct"] == -3.1
    assert item["snapshot"]["indices"][0]["name"] == "NIFTY 50"
    assert item["breakouts_today"] == ["CCC"]


def test_missing_scan_entry_uses_current_price():
    payload = build_recommendations_workspace(
        scan_payload={
            "scanned_at": "2026-08-16T04:00:00+00:00",
            "records": [{
                "symbol": "TRENDY", "company": "Trendy Co", "score": 70,
                "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                "chase_risk": False, "price": 200, "rsi": 58, "volume_ratio": 1.5,
                "avg_vol20": 1e6, "momentum_5d": 5, "above_sma50": True,
            }],
        },
        long_term_payload={"records": []},
        refresh_technicals=False,
    )
    trends = next(c for c in payload["categories"] if c["id"] == "super_trends")
    assert trends["cards"]
    card = trends["cards"][0]
    assert card["entry"] == 200.0
    assert card["target"] == 220.0
    assert card["stop"] == 190.0
    assert card["buy_zone_low"] is not None


def test_missing_target_keeps_real_entry():
    payload = build_recommendations_workspace(
        scan_payload={
            "scanned_at": "2026-08-16T04:00:00+00:00",
            "records": [{
                "symbol": "TRENDY", "company": "Trendy Co", "score": 70,
                "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                "chase_risk": False, "price": 200, "entry": 200, "stop": 190,
                "rsi": 58, "volume_ratio": 1.5, "avg_vol20": 1e6,
                "momentum_5d": 5, "above_sma50": True,
            }],
        },
        long_term_payload={"records": []},
        refresh_technicals=False,
    )
    trends = next(c for c in payload["categories"] if c["id"] == "super_trends")
    card = trends["cards"][0]
    assert card["entry"] == 200.0
    assert card["stop"] == 190.0
    assert card["target"] == 220.0


def test_existing_scan_entry_is_not_overwritten():
    payload = build_recommendations_workspace(
        scan_payload={
            "scanned_at": "2026-08-16T04:00:00+00:00",
            "records": [{
                "symbol": "TRENDY", "company": "Trendy Co", "score": 70,
                "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                "chase_risk": False, "price": 210, "entry": 200, "target": 240,
                "stop": 190, "rsi": 58, "volume_ratio": 1.5,
                "avg_vol20": 1e6, "momentum_5d": 5, "above_sma50": True,
            }],
        },
        long_term_payload={"records": []},
        refresh_technicals=False,
    )
    trends = next(c for c in payload["categories"] if c["id"] == "super_trends")
    card = trends["cards"][0]
    assert card["entry"] == 200.0
    assert card["target"] == 240.0
    assert card["stop"] == 190.0


def test_buy_zone_requires_real_entry_and_uses_atr_when_present():
    from product.decision_card import buy_zone, expected_payoff, evidence_strength

    assert buy_zone({"price": 110}) == (None, None)
    lo, hi = buy_zone({"entry": 100, "atr_pct": 4, "price": 100, "stop": 90})
    assert lo == 98.0  # 100 - 0.5*4
    assert hi == 102.0
    # Zone never drops through the stop.
    lo2, hi2 = buy_zone({"entry": 100, "atr_pct": 40, "price": 100, "stop": 99})
    assert lo2 >= 99
    assert hi2 >= lo2
    label, detail = expected_payoff({"price": 100, "entry": 100, "stop": 95})
    assert label == "Unproven"
    assert "30" in detail
    assert expected_payoff({"ev_n": 80, "ev_lb_pct": 1.4, "ev_conf": "HIGH"})[0] == "Positive"
    assert expected_payoff({"ev_n": 80, "ev_lb_pct": -0.8, "ev_conf": "MEDIUM"})[0] == "Negative"
    assert expected_payoff({"ev_n": 12, "ev_lb_pct": 9.9, "ev_conf": "HIGH"})[0] == "Unproven"
    assert evidence_strength({"ev_n": 80, "ev_conf": "HIGH"}) == "Strong"
    assert evidence_strength({"ev_n": 80, "ev_conf": "MEDIUM"}) == "Moderate"
    assert evidence_strength({"score": 90, "breakout_grade": "A"}) == "Moderate"
    assert evidence_strength({"score": 90}) == "Thin"


def test_decision_card_keeps_quant_machinery_behind_evidence_panel():
    from product.decision_card import decision_surface

    surface = decision_surface(
        {
            "symbol": "ABC",
            "signals": ["MOMENTUM", "GOLDEN_CROSS"],
            "reasons": ["RS leadership"],
            "price": 1200, "entry": 1180, "target": 1310, "stop": 1135,
            "atr_pct": 2.0, "volume_ratio": 1.6, "above_sma50": True,
            "sector": "Banks", "rsi": 58, "score": 72,
            "ev_n": 80, "ev_lb_pct": 1.2, "ev_pct": 1.8, "p_win": 61.0,
            "ev_conf": "HIGH",
        },
        category_id="super_trends",
        action_badge="Buy",
        qualify_reason="Trend momentum",
        market_ctx={
            "market_support": "Positive",
            "market_support_detail": "HEALTHY breadth",
            "strategy_health": "Normal",
            "strategy_health_detail": "Live expectancy +0.20R",
            "signal_health": {
                "MOMENTUM": {"health": "Normal", "n": 90, "expectancy_r": 0.22},
            },
        },
    )
    assert surface["opportunity_label"] == "OPPORTUNITY"
    assert surface["buy_zone_low"] == 1168.0  # 1180 - 0.5*(1200*0.02)
    assert surface["buy_zone_high"] == 1192.0
    assert surface["stop"] == 1135.0
    assert surface["expected_payoff"] == "Positive"
    assert surface["evidence"] == "Strong"
    assert surface["strategy_health"] == "Normal"
    assert surface["market_support"] == "Positive"
    assert surface["horizon"] == "3–9 months"
    assert "Momentum improving" in surface["why_now"]
    assert "Volume confirmation" in surface["why_now"]
    assert any("1,135" in item or "1135" in item for item in surface["what_changes_mind"])
    assert any("Banks" in item for item in surface["what_changes_mind"])
    panel = surface["evidence_panel"]
    assert panel["sample_size"] == 80
    assert panel["ev_lb_pct"] == 1.2
    assert "MOMENTUM" in panel["signals"]
    assert "30" in panel["provenance"]


def test_desk_context_breadth_gate_and_healthy_tape():
    from product.decision_card import build_desk_context

    thin = build_desk_context([{"change_pct": 1, "above_sma50": True}] * 40)
    assert thin["market_support"] == "Unmeasured"
    healthy_rows = [
        {"change_pct": 1.2, "above_sma50": True, "above_sma200": True}
        for _ in range(320)
    ]
    healthy = build_desk_context(healthy_rows)
    assert healthy["market_support"] == "Positive"
