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
    assert s["probability_status"] == "NO_EXPLICIT_PROBABILITIES"
    assert s["probability_sample_size"] == 0
    assert s["expected_p"] is None
    assert s["brier"] is None
    assert s["overconfidence"] is False


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
    assert s["probability_status"] == "MEASURED"
    assert s["probability_sample_size"] == 20
    assert s["probability_actual_hit_rate"] == 0.0
    assert s["expected_p"] == 0.8
    assert s["brier"] == 0.64
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


def test_probability_calibration_has_independent_sample_floor(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    for i in range(20):
        eng.record(
            predicted_confidence="good_setup",
            realized_win=i % 2 == 0,
            predicted_p=0.70 if i < 10 else None,
            decision_as_of="2026-01-01",
            outcome_as_of="2026-02-01",
        )

    s = eng.summary(bucket="good_setup")
    assert s["status"] == "MEASURED"
    assert s["sample_size"] == 20
    assert s["actual_hit_rate"] == 0.5
    assert s["confidence_interval"][0] < 0.5 < s["confidence_interval"][1]
    assert s["probability_sample_size"] == 10
    assert s["probability_status"] == "INSUFFICIENT_PROBABILITY_EVIDENCE"
    assert s["expected_p"] is None
    assert s["brier"] is None
    assert s["calibration_gap"] is None
    assert s["overconfidence"] is False


def test_dossier_reports_probability_coverage_without_inventing_scores(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    for i in range(20):
        eng.record(
            predicted_confidence="high_conviction",
            realized_win=i < 12,
            predicted_p=0.65 if i < 5 else None,
            decision_as_of="2026-01-01",
            outcome_as_of="2026-02-01",
        )

    d = eng.dossier()
    assert d["settled_observations"] == 20
    assert d["explicit_probability_observations"] == 5
    assert d["probability_coverage"] == 0.25
    assert d["overall"]["status"] == "MEASURED"
    assert d["overall"]["probability_status"] == "INSUFFICIENT_PROBABILITY_EVIDENCE"
    assert d["overall"]["brier"] is None
    assert d["affects_production"] is False
    assert d["live_locked"] is True


def test_invalid_probability_is_not_silently_used(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    for i in range(20):
        eng.record(
            predicted_confidence="watch",
            realized_win=i % 2 == 0,
            predicted_p=1.2,
            decision_as_of="2026-01-01",
            outcome_as_of="2026-02-01",
        )

    s = eng.summary(bucket="watch")
    assert s["status"] == "MEASURED"
    assert s["probability_status"] == "NO_EXPLICIT_PROBABILITIES"
    assert s["expected_p"] is None
    assert s["brier"] is None
