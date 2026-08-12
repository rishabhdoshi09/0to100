"""Tests for market radar lane projections."""
from __future__ import annotations

from product.radar_workspace import (
    classify_breakout_state,
    classify_momentum_state,
    build_radar_home,
    enrich_scan_row,
)


def test_confirmed_breakout_requires_buy_and_ready_status():
    row = {
        "symbol": "TEST",
        "signals": ["BREAKOUT_52W", "MOMENTUM"],
        "verdict": "BUY",
        "status": "Ready to trade",
        "chase_risk": False,
        "score": 80,
        "volume_ratio": 1.5,
    }
    assert classify_breakout_state(row) == "confirmed_breakout"


def test_extended_breakout_when_chase_risk():
    row = {
        "symbol": "TEST",
        "signals": ["BREAKOUT_52W"],
        "verdict": "BUY",
        "status": "Wait for pullback",
        "chase_risk": True,
    }
    assert classify_breakout_state(row) == "extended_after_breakout"


def test_momentum_extended_state():
    row = {"signals": ["MOMENTUM"], "chase_risk": True, "score": 70, "status": "Watch"}
    assert classify_momentum_state(row) == "strong_but_extended"


def test_radar_home_rejects_quality_without_fundamental_coverage():
    long_term = {
        "scanned_at": "2026-08-01T00:00:00+00:00",
        "records": [
            {"symbol": "THIN", "classification": "QUALITY_COMPOUNDER", "combined_score": 88, "fundamental_coverage": 0.2},
        ],
    }
    market = {"health": "Healthy", "breadth": "60% adv", "trade_stance": "Open", "leaders": [], "laggards": []}
    payload = build_radar_home(scan_payload={"records": []}, long_term_payload=long_term, market=market)
    assert payload["counts"]["long_term_picks"] == 0


def test_radar_home_builds_three_lanes():
    scan = {
        "scanned_at": "2026-08-01T00:00:00+00:00",
        "universe_size": 100,
        "records": [
            {
                "symbol": "AAA",
                "score": 90,
                "verdict": "BUY",
                "status": "Ready to trade",
                "signals": ["MOMENTUM", "BREAKOUT_52W"],
                "chase_risk": False,
                "volume_ratio": 1.2,
                "reasons": ["Strong volume"],
                "breakout_grade": "B",
                "breakout_conviction": 55,
            },
            {
                "symbol": "BEST",
                "score": 78,
                "verdict": "BUY",
                "status": "Ready to trade",
                "signals": ["BREAKOUT_52W"],
                "chase_risk": False,
                "volume_ratio": 2.0,
                "reasons": ["A-grade break"],
                "breakout_grade": "A",
                "breakout_conviction": 82,
            },
            {
                "symbol": "BBB",
                "score": 50,
                "signals": ["MOMENTUM"],
                "chase_risk": True,
                "status": "Wait for pullback",
            },
        ],
    }
    long_term = {
        "scanned_at": "2026-08-01T00:00:00+00:00",
        "records": [
            {"symbol": "QUAL", "classification": "QUALITY_COMPOUNDER", "combined_score": 88, "fundamental_coverage": 0.8},
            {
                "symbol": "BEST",
                "classification": "QUALITY_COMPOUNDER",
                "combined_score": 86,
                "fundamental_coverage": 0.9,
                "fundamental_score": 80,
            },
        ],
    }
    market = {"health": "Healthy", "breadth": "60% adv", "trade_stance": "Open", "leaders": [], "laggards": []}
    payload = build_radar_home(scan_payload=scan, long_term_payload=long_term, market=market)
    assert payload["counts"]["momentum"] >= 1
    assert payload["counts"]["breakouts"] >= 1
    assert payload["counts"]["long_term_picks"] == 2
    assert payload["lanes"]["momentum"][0]["symbol"] == "AAA"
    # Among confirmed breakouts, BEST wins on grade/conviction + fundamentals
    assert payload["best_breakout"]["symbol"] == "BEST"
    assert payload["lanes"]["breakouts"][0]["symbol"] == "BEST"
    assert payload["lanes"]["breakouts"][0]["fundamental_score"] == 80


def test_breakout_quality_prefers_grade_and_fundamentals():
    from product.radar_workspace import breakout_quality_score
    weak = {"score": 90, "breakout_grade": "", "breakout_conviction": 40, "edge_r": 0}
    strong = {
        "score": 75, "breakout_grade": "A", "breakout_conviction": 80, "edge_r": 0.2,
        "fundamental_score": 78, "fundamental_coverage": 0.8,
        "classification": "QUALITY_COMPOUNDER",
    }
    assert breakout_quality_score(strong) > breakout_quality_score(weak)


def test_enrich_scan_row_never_fakes_daily_change_label():
    row = enrich_scan_row({"symbol": "X", "momentum_5d": 4.2, "signals": ["MOMENTUM"]}, scanned_at="2026-01-01")
    assert row["change_5d_pct"] == 4.2
    assert "sector" in row
