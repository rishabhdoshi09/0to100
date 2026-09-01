from __future__ import annotations

from product.strategy_catalog import (
    UNVERIFIED,
    decorate_card,
    ensemble_identity,
    fundamental_disagreement,
    method_identity,
    production_registry,
    related_signal_calibration,
)


def test_ensemble_identity_is_stable_and_unverified():
    first = ensemble_identity()
    second = ensemble_identity()
    assert first["strategy_id"] == "QT_RECO_ENSEMBLE"
    assert first["rules_hash"] == second["rules_hash"]
    assert first["backtest_parity"] == UNVERIFIED
    assert "UNVERIFIED" in first["backtest_parity_detail"]


def test_method_identity_does_not_borrow_paper_strategies():
    tape = method_identity("tape")
    assert tape["strategy_id"] == "QT_METHOD_TAPE"
    assert tape["backtest_parity"] == UNVERIFIED
    registry = production_registry()
    assert registry["role"] == "production_recommendations"
    assert all(row["backtest_parity"] == UNVERIFIED for row in registry["methods"])


def test_decorate_card_keeps_funds_from_overriding_tape():
    card = decorate_card({
        "symbol": "KAYNES",
        "conflicts": [],
        "methods": [
            {"id": "tape", "label": "Tape", "status": "pass", "points": 90, "detail": "grade A"},
            {"id": "funds", "label": "Funds", "status": "fail", "points": 0, "detail": "AVOID"},
            {"id": "sepa", "label": "SEPA", "status": "pass", "points": 55, "detail": "overlay"},
        ],
    })
    assert card["production_strategy"]["strategy_id"] == "QT_RECO_ENSEMBLE"
    assert card["backtest_parity"] == UNVERIFIED
    assert "Technical structure passed" in card["fundamental_disagreement"]
    assert card["methods"][0]["strategy_id"] == "QT_METHOD_TAPE"


def test_missing_funds_are_unknown_not_failed():
    text = fundamental_disagreement({
        "methods": [
            {"id": "tape", "status": "pass"},
            {"id": "funds", "status": "unknown"},
        ],
    })
    assert "missing" in text.lower()


def test_signal_calibration_without_file_stays_missing(tmp_path):
    payload = related_signal_calibration(tmp_path / "missing.json")
    assert payload["available"] is False
    assert payload["parity"] == UNVERIFIED
