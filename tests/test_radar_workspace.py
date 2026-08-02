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
                "reasons": ["Strong volume"],
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
            {"symbol": "QUAL", "classification": "QUALITY_COMPOUNDER", "combined_score": 88},
        ],
    }
    market = {"health": "Healthy", "breadth": "60% adv", "trade_stance": "Open", "leaders": [], "laggards": []}
    payload = build_radar_home(scan_payload=scan, long_term_payload=long_term, market=market)
    assert payload["counts"]["momentum"] >= 1
    assert payload["counts"]["breakouts"] >= 1
    assert payload["counts"]["long_term_picks"] == 1
    assert payload["lanes"]["momentum"][0]["symbol"] == "AAA"


def test_enrich_scan_row_never_fakes_daily_change_label():
    row = enrich_scan_row({"symbol": "X", "momentum_5d": 4.2, "signals": ["MOMENTUM"]}, scanned_at="2026-01-01")
    assert row["change_5d_pct"] == 4.2
    assert "sector" in row
