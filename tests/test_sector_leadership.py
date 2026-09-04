"""Sector leadership is a ranking proxy, never a hard-gate bypass."""
from __future__ import annotations

from product.reco_ensemble import TIER_GOOD, TIER_WATCH, sort_key
from product.sector_leadership import attach_to_row, board_from_rows


def test_leading_sector_scores_higher_than_laggard():
    rows = [
        {"symbol": "HDFCBANK", "sector": "Banks", "above_sma50": True, "rs_percentile": 82, "volume_ratio": 1.5, "status": "Ready to trade", "verdict": "BUY"},
        {"symbol": "ICICIBANK", "sector": "Banks", "above_sma50": True, "rs_percentile": 78, "volume_ratio": 1.4, "status": "Ready to trade"},
        {"symbol": "TATASTEEL", "sector": "Metals", "above_sma50": False, "rs_percentile": 22, "volume_ratio": 0.6, "status": "Wait for pullback"},
        {"symbol": "JSWSTEEL", "sector": "Metals", "above_sma50": False, "rs_percentile": 18, "volume_ratio": 0.5},
    ]
    board = board_from_rows(rows, leaders=["Banks"], laggards=["Metals"])
    assert board["banks"]["score"] > board["metals"]["score"]
    assert board["banks"]["label"] in {"Sector Leadership", "Sector Money-Flow Proxy", "Sector Participation"}
    assert board["banks"]["not_institutional_cashflow"] is True


def test_missing_sector_stays_missing():
    attached = attach_to_row({"symbol": "XYZ"}, {})
    assert attached["sector_leadership_score"] is None
    assert attached["sector_leadership_label"] == ""


def test_sector_can_break_near_tie_but_not_tier():
    weak_sector = {
        "symbol": "AAA",
        "reco_tier": TIER_GOOD,
        "family_confirms": 2,
        "quality_score": 85,
        "sector_leadership_score": 20,
    }
    strong_sector = {
        "symbol": "BBB",
        "reco_tier": TIER_GOOD,
        "family_confirms": 2,
        "quality_score": 84,
        "sector_leadership_score": 86,
    }
    ordered = sorted([weak_sector, strong_sector], key=sort_key)
    assert ordered[0]["symbol"] == "BBB"
    watch = {**strong_sector, "symbol": "CCC", "reco_tier": TIER_WATCH}
    ordered2 = sorted([weak_sector, watch], key=sort_key)
    assert ordered2[0]["symbol"] == "AAA"
