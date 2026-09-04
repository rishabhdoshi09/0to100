"""Invalid tickers and failed lookups are not AVOID judgments."""
from __future__ import annotations

from product import decision_taxonomy as T
from product.decision_committee import evaluate_committee, evaluate_many
from product.judgment_census import build_census
from product.scorecards import build_scorecards, reason_scorecards


def test_garbage_ticker_is_invalid_symbol_not_avoid():
    rec = evaluate_committee({"symbol": "NOTAREALTICKERZZZ"}, broker_ok=False, load_research=False)
    assert rec.decision == T.NO_JUDGMENT
    assert rec.reason_code == T.INVALID_SYMBOL
    assert rec.decision != T.AVOID
    assert rec.reason_code != T.LOW_QUALITY_SETUP
    payload = rec.as_dict()
    assert T.is_non_judgment(payload["decision"], payload["reason_code"])
    assert not T.is_judgment_row(payload)


def test_missing_symbol_is_invalid_not_avoid():
    rec = evaluate_committee({"symbol": ""}, broker_ok=False, load_research=False)
    assert rec.decision == T.NO_JUDGMENT
    assert rec.reason_code == T.INVALID_SYMBOL


def test_data_unavailable_card_is_not_avoid():
    rec = evaluate_committee(
        {"symbol": "RELIANCE", "status": "DATA_UNAVAILABLE", "reason": "no official bars"},
        broker_ok=False,
        load_research=False,
    )
    assert rec.decision == T.NO_JUDGMENT
    assert rec.reason_code == T.DATA_UNAVAILABLE
    assert rec.decision != T.AVOID


def test_analysis_error_is_not_avoid(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr("product.decision_committee._evaluate_committee_body", boom)
    rec = evaluate_committee(
        {"symbol": "TCS", "reco_tier": "high_conviction", "entry_state": "ready", "entry": 1, "stop": 0.9, "target": 1.2},
        broker_ok=False,
        load_research=False,
    )
    assert rec.decision == T.NO_JUDGMENT
    assert rec.reason_code == T.ANALYSIS_ERROR


def test_non_judgment_excluded_from_avoid_scorecards_and_census():
    rows = [
        {"symbol": "TCS", "decision": "AVOID", "reason_code": "LOW_QUALITY_SETUP", "methods_buy": []},
        {
            "symbol": "NOTAREALTICKERZZZ",
            "decision": T.NO_JUDGMENT,
            "reason_code": T.INVALID_SYMBOL,
            "methods_buy": [],
        },
    ]
    cards = build_scorecards(rows)
    assert cards["n_rows"] == 1
    assert cards["n_excluded_non_judgment"] == 1
    reasons = reason_scorecards(rows)
    codes = {item["reason_code"] for item in reasons["reasons"]}
    assert T.INVALID_SYMBOL not in codes
    assert "LOW_QUALITY_SETUP" in codes
    census = build_census(
        scan={"records": [], "coverage": {}},
        reco={"categories": []},
        committee=rows,
        session="2026-09-01",
        scan_run_id="scan-1",
        generated_at="2026-09-01T10:00:00+00:00",
    )
    assert census["judgments"]["AVOID"] == 1
    assert census["judgments"]["NO_JUDGMENT"] == 1
    many = evaluate_many([{"symbol": "NOTAREALTICKERZZZ"}], load_research=False)
    assert many[0].decision == T.NO_JUDGMENT
    assert all(item.decision != T.AVOID for item in many)
