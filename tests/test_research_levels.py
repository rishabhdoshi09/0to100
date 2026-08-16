"""Research buy / stop / target — no invented prices, scanner geometry."""
from __future__ import annotations

from product.radar_workspace import enrich_long_term_row
from product.recommendations_workspace import card_from_row
from product.research_levels import attach_research_levels, research_levels


def test_vol_pct_builds_2x_4x_atr_plan():
    levels = research_levels({"price": 1000.0, "vol_pct": 2.0})
    assert levels["entry"] == 1000.0
    assert levels["stop"] == 960.0          # 2 × 20
    assert levels["target"] == 1080.0       # 4 × 20
    assert levels["upside_from_buy_pct"] == 8.0
    assert levels["levels_source"] == "vol_pct"


def test_explicit_atr_wins_over_vol_pct():
    levels = research_levels({"price": 200.0, "atr": 5.0, "vol_pct": 9.0})
    assert levels["stop"] == 190.0
    assert levels["target"] == 220.0
    assert levels["levels_source"] == "atr"


def test_complete_scan_plan_is_kept():
    levels = research_levels({"price": 110, "entry": 100, "stop": 95, "target": 130})
    assert levels["entry"] == 100
    assert levels["stop"] == 95
    assert levels["target"] == 130
    assert levels["levels_source"] == "scan"
    assert levels["upside_from_buy_pct"] == 30.0


def test_missing_stop_is_filled_without_dropping_target():
    levels = research_levels({"entry": 100, "target": 110})
    assert levels["entry"] == 100
    assert levels["target"] == 110
    assert levels["stop"] == 95.0
    assert levels["upside_from_buy_pct"] == 10.0


def test_no_price_no_entry_stays_empty():
    levels = research_levels({"symbol": "NONE"})
    assert levels["entry"] is None
    assert levels["stop"] is None
    assert levels["target"] is None
    assert levels["upside_from_buy_pct"] is None


def test_enrich_long_term_row_attaches_levels():
    row = enrich_long_term_row({
        "symbol": "BSE",
        "classification": "QUALITY_COMPOUNDER",
        "price": 3447.0,
        "vol_pct": 2.0,
        "fundamental_coverage": 0.8,
    })
    assert row["entry"] == 3447.0
    assert row["stop"] < row["entry"] < row["target"]
    assert row["upside_from_buy_pct"] > 0


def test_wealth_card_without_scan_plan_gets_stop_and_upside():
    card = card_from_row(
        {
            "symbol": "OFSS",
            "classification": "QUALITY_COMPOUNDER",
            "price": 11828.0,
            "vol_pct": 1.8,
            "fundamental_coverage": 0.8,
            "combined_score": 80,
        },
        category_id="wealth_builders",
        category_label="Wealth Builders",
    )
    assert card["stop"] is not None and card["stop"] < card["entry"]
    assert card["target"] is not None and card["target"] > card["entry"]
    assert card["upside_from_buy_pct"] > 0
    assert "2×ATR stop" in card["evidence_tags"]


def test_attach_does_not_invent_a_symbol_price():
    out = attach_research_levels({"symbol": "EMPTY"})
    assert out.get("entry") in (None, 0)
    assert out.get("stop") in (None, 0)
    assert out.get("target") in (None, 0)
