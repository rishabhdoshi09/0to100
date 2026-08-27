"""Mixture-of-experts Recommendations: families, theses, honest empties."""
from __future__ import annotations

import pytest

from product.reco_ensemble import attach_expert_layer
from product.reco_methods import allows_buy
from product.recommendations_workspace import build_recommendations_workspace, card_from_row
from research.reco_ensemble.experiment import run_comparison


@pytest.fixture(autouse=True)
def _isolate_case_db(tmp_path, monkeypatch):
    import product.case_memory as cm
    monkeypatch.setattr(cm, "CASES_DB", tmp_path / "cases.db")


def _layer(row: dict) -> dict:
    return attach_expert_layer([row])[0]


def test_sepa_only_is_watch_not_high_conviction():
    row = _layer({
        "symbol": "JUSTSEPA",
        "signals": ["MOMENTUM"],
        "verdict": "WATCH",
        "status": "Watch",
        "chase_risk": False,
        "sepa_score": 100,
        "volume_ratio": 0,
        "rsi": 50,
    })
    sepa = next(e for e in row["experts"] if e["id"] == "sepa")
    assert sepa["status"] == "pass"
    assert row["family_confirms"] == 1
    assert row["reco_tier"] == "watch"
    assert allows_buy(row) is False
    card = card_from_row(row, category_id="super_trends", category_label="Super Trends")
    assert card["action_badge"] == "Watch"
    assert card["reco_tier"] == "watch"


def test_momentum_plus_quality_recommends_without_sepa():
    row = _layer({
        "symbol": "MOMQ",
        "signals": ["MOMENTUM"],
        "verdict": "BUY",
        "status": "Ready to trade",
        "chase_risk": False,
        "price": 120,
        "entry": 118,
        "stop": 110,
        "target": 140,
        "rsi": 55,
        "volume_ratio": 1.4,
        "avg_vol20": 1e6,
        "momentum_5d": 9.0,
        "classification": "QUALITY_COMPOUNDER",
        "fundamental_coverage": 0.82,
        "fundamental_score": 76,
    })
    sepa = next(e for e in row["experts"] if e["id"] == "sepa")
    mom_q = next(e for e in row["experts"] if e["id"] == "mom_quality")
    assert sepa["status"] == "unknown"
    assert mom_q["status"] == "pass"
    assert row["family_confirms"] >= 2
    assert row["primary_thesis"] == "Momentum + Quality"
    assert row["reco_tier"] in {"good_setup", "high_conviction"}
    assert allows_buy(row) is True
    card = card_from_row(row, category_id="super_trends", category_label="Super Trends")
    assert card["action_badge"] == "Buy"


def test_tape_rs_breakout_are_two_families_not_three():
    row = _layer({
        "symbol": "CORR",
        "signals": ["BREAKOUT_52W", "MOMENTUM"],
        "verdict": "BUY",
        "status": "Ready to trade",
        "breakout_grade": "A",
        "volume_ratio": 1.8,
        "avg_vol20": 2e6,
        "rsi": 55,
        "chase_risk": False,
        "momentum_5d": 6.0,
        "rs_percentile": 88,
        "price": 200,
        "entry": 198,
        "stop": 190,
        "target": 230,
    })
    families = {f["id"]: f["status"] for f in row["families"]}
    assert families.get("structure") == "pass"
    assert families.get("price_leadership") == "pass"
    # Volume of the breakout bar is not a third independent family.
    assert families.get("participation") != "pass"
    assert row["family_confirms"] == 2
    assert "Price Leadership" in row["family_line"]
    assert "Structure" in row["family_line"]


def test_earnings_without_sequential_prints_is_not_invented():
    row = _layer({
        "symbol": "CAGR",
        "chase_risk": False,
        "fundamentals": {"sales_growth_3y": 18, "profit_growth_3y": 22},
        "sales_growth_3y": 18,
        "profit_growth_3y": 22,
    })
    earnings = next(e for e in row["experts"] if e["id"] == "earnings")
    assert earnings["status"] == "neutral"
    assert earnings["eligible"] is False
    assert any("not sequential" in x.lower() or "cagr" in x.lower() for x in earnings["evidence"])
    assert "consensus" not in " ".join(earnings["evidence"]).lower()


def test_earnings_sequential_acceleration_can_propose():
    row = _layer({
        "symbol": "ACCEL",
        "chase_risk": False,
        "sales_growth_qoq": 12.0,
        "pat_growth_qoq": 18.0,
    })
    earnings = next(e for e in row["experts"] if e["id"] == "earnings")
    assert earnings["status"] == "pass"
    assert earnings["eligible"] is True


def test_empty_high_conviction_is_success():
    payload = build_recommendations_workspace(
        scan_payload={
            "scanned_at": "2026-08-26T04:00:00+00:00",
            "records": [{
                "symbol": "LONELY", "company": "Lonely Co", "score": 70,
                "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                "chase_risk": False, "price": 10, "rsi": 58, "volume_ratio": 1.5,
                "avg_vol20": 1e6, "momentum_5d": 8,
            }],
        },
        long_term_payload={"records": []},
        refresh_technicals=False,
    )
    assert payload["ensemble"]["empty_high_conviction"] is True
    assert payload["ensemble"]["high_conviction_count"] == 0
    assert "NO HIGH-CONVICTION" in payload["ensemble"]["empty_line"]
    assert "Checked" in (payload["ensemble"].get("empty_detail") or "")
    trends = next(c for c in payload["categories"] if c["id"] == "super_trends")
    assert trends["cards"]
    assert trends["cards"][0]["action_badge"] == "Watch"


def test_extended_leader_is_watch_even_with_quality():
    row = _layer({
        "symbol": "CHASEQ",
        "signals": ["MOMENTUM", "BREAKOUT_52W"],
        "verdict": "WATCH",
        "status": "Wait for pullback",
        "breakout_grade": "A",
        "chase_risk": True,
        "rsi": 80,
        "volume_ratio": 2.0,
        "momentum_5d": 12,
        "classification": "QUALITY_COMPOUNDER",
        "fundamental_coverage": 0.9,
        "fundamental_score": 80,
        "rs_percentile": 92,
    })
    assert row["entry_state"] == "extended"
    assert row["reco_tier"] == "watch"
    assert allows_buy(row) is False


def test_breakout_quality_and_sector_is_high_conviction():
    row = _layer({
        "symbol": "HICONV",
        "signals": ["BREAKOUT_52W", "MOMENTUM"],
        "verdict": "BUY",
        "status": "Ready to trade",
        "breakout_grade": "A",
        "volume_ratio": 1.8,
        "avg_vol20": 2e6,
        "rsi": 55,
        "chase_risk": False,
        "momentum_5d": 8.0,
        "classification": "QUALITY_COMPOUNDER",
        "fundamental_coverage": 0.8,
        "fundamental_score": 74,
        "sector": "Banks",
        "sector_leader": True,
        "price": 200,
        "entry": 198,
        "stop": 190,
        "target": 230,
    })
    assert row["family_confirms"] >= 3
    assert row["reco_tier"] == "high_conviction"
    assert allows_buy(row) is True
    """SEPA + breakout + accumulation is still tape. Funds unknown → not High Conviction."""
    row = _layer({
        "symbol": "BEPLISH",
        "signals": ["PRE_BREAKOUT", "MOMENTUM", "ACCUMULATION", "POCKET_PIVOT"],
        "verdict": "BUY",
        "status": "Ready to trade",
        "breakout_grade": "A",
        "volume_ratio": 3.0,
        "avg_vol20": 2e6,
        "rsi": 55,
        "chase_risk": False,
        "momentum_5d": 6.0,
        "rs_percentile": 81,
        "sepa_score": 100,
        "price": 130,
        "entry": 128,
        "stop": 120,
        "target": 150,
    })
    assert row["family_confirms"] >= 3
    assert row["reco_tier"] == "good_setup"
    assert allows_buy(row) is True
    quality = next(e for e in row["experts"] if e["id"] == "quality")
    assert quality["status"] == "unknown"


def test_historical_comparison_is_blocked_without_inventing_results():
    result = run_comparison()
    assert result["blocked"] is True
    assert result["verdict"] == "INCONCLUSIVE"
    assert "promoted" in result["note"].lower() or "not promoted" in result["note"].lower()
    assert result["spec"]["llm_not_in_money_path"] is True
    for name in result["variants"]:
        assert result["variants"][name]["status"] in {"not_run", "registered_not_run"}


def test_workspace_exposes_ensemble_and_no_score_soup_claim():
    payload = build_recommendations_workspace(
        scan_payload={"records": [], "scanned_at": ""},
        long_term_payload={"records": []},
    )
    assert payload["schema_version"] == 4
    assert payload["ensemble"]["empty_high_conviction"] is True
    assert "families" in payload["methods_note"].lower()
    assert "SEPA" in payload["methods_note"]
    assert "indicator" in payload["methods_note"].lower() or "soup" not in payload["cmp_note"].lower()
