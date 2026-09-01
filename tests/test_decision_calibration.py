"""Phase 6 — DecisionCalibrationEngine."""

from __future__ import annotations

import pytest

from product.decision_calibration import DecisionCalibrationEngine


def test_confidence_bucket_outcomes_update(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    for i in range(20):
        eng.record(
            predicted_confidence="high_conviction",
            realized_win=i % 2 == 0,
            setup="VCP",
            regime="RISK_ON",
            sector="IT",
            decision_as_of="2026-01-01",
            outcome_as_of="2026-02-01",
        )
    s = eng.summary(bucket="high_conviction")
    assert s["status"] == "MEASURED"
    assert s["sample_size"] == 20
    assert 0.4 <= s["actual_hit_rate"] <= 0.6
    assert s["affects_production"] is False
    assert s["rename_tier"] is False


def test_small_sample_stays_insufficient(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    eng.record(
        predicted_confidence="high_conviction",
        realized_win=True,
        decision_as_of="2026-01-01",
        outcome_as_of="2026-02-01",
    )
    s = eng.summary(bucket="high_conviction")
    assert s["status"] == "INSUFFICIENT_EVIDENCE"
    assert s["sample_size"] == 1
    assert s["overconfidence"] is False
    assert s["affects_production"] is False


def test_overconfidence_detected(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    for _ in range(20):
        eng.record(
            predicted_confidence="high_conviction",
            realized_win=False,
            predicted_p=0.80,
            decision_as_of="2026-01-01",
            outcome_as_of="2026-03-01",
        )
    s = eng.summary(bucket="high_conviction")
    assert s["overconfidence"] is True
    assert s["actual_hit_rate"] == 0.0
    assert s["rename_tier"] is False


def test_no_production_behavior_changes_from_one_observation(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    row = eng.record(
        predicted_confidence="good_setup",
        realized_win=False,
        decision_as_of="2026-01-01",
        outcome_as_of="2026-01-20",
    )
    assert row["production_changed"] is False
    assert eng.store["affects_production"] is False
    assert eng.summary(bucket="good_setup")["affects_production"] is False


def test_calibration_evidence_is_pit_safe(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    with pytest.raises(ValueError):
        eng.record(
            predicted_confidence="high_conviction",
            realized_win=True,
            decision_as_of="2026-06-01",
            outcome_as_of="2026-01-01",
        )
    eng.record(
        predicted_confidence="high_conviction",
        realized_win=True,
        decision_as_of="2026-01-01",
        outcome_as_of="2026-06-01",
    )
    assert eng.store["observations"][0]["decision_as_of"] == "2026-01-01"
