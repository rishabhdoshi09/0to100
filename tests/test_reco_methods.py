"""Mixture-of-experts Reco layer: families, not indicator soup."""
from __future__ import annotations

import pytest

from product.reco_methods import (
    MIN_CONFIRMS_FOR_BUY,
    allows_buy,
    attach_method_scores,
    attach_research_overlays,
    score_methods,
    sort_key,
)
from product.recommendations_workspace import build_recommendations_workspace, card_from_row


@pytest.fixture(autouse=True)
def _isolate_case_db(tmp_path, monkeypatch):
    import product.case_memory as cm
    monkeypatch.setattr(cm, "CASES_DB", tmp_path / "cases.db")


def test_sepa_alone_is_not_a_buy():
    row = attach_method_scores({
        "symbol": "SEPAONLY",
        "signals": ["BREAKOUT_52W"],
        "verdict": "BUY",
        "status": "Ready to trade",
        "breakout_grade": "A",
        "volume_ratio": 1.8,
        "avg_vol20": 1e6,
        "rsi": 55,
        "chase_risk": False,
        "sepa_score": 100,
        "price": 130,
        "entry": 128,
        "stop": 120,
        "target": 150,
    })
    # Tape + SEPA = two methods. That's the intended Buy path.
    assert row["method_confirms"] >= MIN_CONFIRMS_FOR_BUY
    sepa_only = attach_method_scores({
        "symbol": "JUSTSEPA",
        "signals": ["MOMENTUM"],
        "verdict": "WATCH",
        "status": "Watch",
        "chase_risk": False,
        "sepa_score": 100,
        "volume_ratio": 0,
        "rsi": 50,
    })
    assert any(m["id"] == "sepa" and m["status"] == "pass" for m in sepa_only["methods"])
    assert sepa_only["method_confirms"] == 1
    assert allows_buy(sepa_only) is False
    card = card_from_row(sepa_only, category_id="super_trends", category_label="Super Trends")
    assert card["action_badge"] == "Watch"


def test_tape_without_second_method_is_watch():
    card = card_from_row(
        {
            "symbol": "TAPEONLY",
            "signals": ["BREAKOUT_52W"],
            "verdict": "BUY",
            "status": "Ready to trade",
            "breakout_grade": "A",
            "volume_ratio": 1.6,
            "avg_vol20": 2e6,
            "rsi": 55,
            "chase_risk": False,
            "price": 200,
            "entry": 198,
            "stop": 190,
            "target": 230,
        },
        category_id="momentum_breakouts",
        category_label="Momentum Breakouts",
    )
    assert card["action_badge"] == "Watch"
    assert card["method_confirms"] == 1
    assert card["entry"] == 198.0  # stored plan is kept; not invented


def test_tape_plus_funds_is_buy():
    card = card_from_row(
        {
            "symbol": "QUALITY",
            "signals": ["BREAKOUT_52W"],
            "verdict": "BUY",
            "status": "Ready to trade",
            "breakout_grade": "A",
            "volume_ratio": 1.6,
            "avg_vol20": 2e6,
            "rsi": 55,
            "chase_risk": False,
            "classification": "QUALITY_COMPOUNDER",
            "fundamental_coverage": 0.8,
            "fundamental_score": 74,
            "price": 200,
            "entry": 198,
            "stop": 190,
            "target": 230,
        },
        category_id="momentum_breakouts",
        category_label="Momentum Breakouts",
    )
    assert card["action_badge"] == "Buy"
    assert card["method_confirms"] >= 2
    labels = {m["label"] for m in card["methods"] if m["status"] == "pass"}
    assert "Tape" in labels
    assert "Funds" in labels


def test_missing_methods_stay_unknown_not_pass():
    panel = score_methods({"symbol": "EMPTY", "signals": ["MOMENTUM"], "chase_risk": False})
    by_id = {m["id"]: m for m in panel["methods"]}
    for key in ("sepa", "funds", "rs", "ev", "trend"):
        assert by_id[key]["status"] == "unknown"
        assert by_id[key]["points"] is None
    assert panel["method_confirms"] == 0
    assert "invent" not in panel["method_line"].lower()


def test_known_negative_ev_fails_not_unknown():
    panel = score_methods({
        "symbol": "LEAK", "chase_risk": False, "ev_n": 80, "ev_lb_pct": -1.2,
    })
    ev = next(m for m in panel["methods"] if m["id"] == "ev")
    assert ev["status"] == "fail"


def test_thin_ev_sample_is_unknown():
    panel = score_methods({
        "symbol": "THIN", "chase_risk": False, "ev_n": 12, "ev_lb_pct": 9.9,
    })
    ev = next(m for m in panel["methods"] if m["id"] == "ev")
    assert ev["status"] == "unknown"


def test_overlays_merge_long_term_funds_onto_scan_row():
    scan, lt = attach_research_overlays(
        [{"symbol": "QUAL", "signals": ["MOMENTUM"], "volume_ratio": 1.2, "chase_risk": False, "rsi": 55}],
        [{
            "symbol": "QUAL",
            "classification": "QUALITY_COMPOUNDER",
            "fundamental_coverage": 0.8,
            "fundamental_score": 70,
        }],
        scanned_at="2026-08-26T00:00:00+00:00",
    )
    assert scan[0]["classification"] == "QUALITY_COMPOUNDER"
    funds = next(m for m in scan[0]["methods"] if m["id"] == "funds")
    assert funds["status"] == "pass"


def test_more_confirms_rank_ahead_of_sepa_heavy_single():
    weak = attach_method_scores({
        "symbol": "AAA", "sepa_score": 100, "chase_risk": False, "score": 99,
    })
    strong = attach_method_scores({
        "symbol": "BBB",
        "chase_risk": False,
        "score": 60,
        "volume_ratio": 1.8,
        "avg_vol20": 1e6,
        "status": "Ready to trade",
        "verdict": "BUY",
        "rsi": 58,
        "above_sma50": True,
        "above_sma200": True,
        "classification": "GARP_CANDIDATE",
        "fundamental_coverage": 0.7,
        "fundamental_score": 65,
        "rs_percentile": 82,
    })
    assert strong["method_confirms"] > weak["method_confirms"]
    ordered = sorted([weak, strong], key=sort_key)
    assert ordered[0]["symbol"] == "BBB"


def test_workspace_ranks_quality_ahead_of_raw_score():
    payload = build_recommendations_workspace(
        scan_payload={
            "scanned_at": "2026-08-26T04:00:00+00:00",
            "records": [
                {
                    "symbol": "HYPE", "company": "Hype Co", "score": 99,
                    "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                    "chase_risk": False, "price": 10, "rsi": 58, "volume_ratio": 1.5,
                    "avg_vol20": 1e6, "momentum_5d": 8,
                },
                {
                    "symbol": "SOLID", "company": "Solid Co", "score": 61,
                    "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                    "chase_risk": False, "price": 20, "rsi": 55, "volume_ratio": 1.4,
                    "avg_vol20": 1e6, "momentum_5d": 4, "above_sma50": True, "above_sma200": True,
                    "classification": "QUALITY_COMPOUNDER",
                    "fundamental_coverage": 0.82, "fundamental_score": 77,
                    "rs_percentile": 88,
                },
            ],
        },
        long_term_payload={"records": []},
        refresh_technicals=False,
    )
    trends = next(c for c in payload["categories"] if c["id"] == "super_trends")
    assert trends["cards"][0]["symbol"] == "SOLID"
    assert trends["cards"][0]["action_badge"] == "Buy"
    assert payload["methods_note"]
    assert "SEPA" in payload["methods_note"]
