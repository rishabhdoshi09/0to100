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
    assert s["calibration_actionable"] is True
    assert s["calibration_direction"] == "OVERCONFIDENT"
    assert s["calibration_adjustment"] > 0
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
    assert all(row["predicted_p"] is None for row in eng.store["observations"])


def test_probability_difference_inside_wilson_uncertainty_is_not_actionable(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    for i in range(20):
        eng.record(
            predicted_confidence="good_setup",
            realized_win=i % 2 == 0,
            predicted_p=0.55,
            decision_as_of="2026-01-01",
            outcome_as_of="2026-02-01",
        )

    s = eng.summary(bucket="good_setup")
    assert s["probability_status"] == "MEASURED"
    assert s["actual_hit_rate"] == 0.5
    assert s["expected_p"] == 0.55
    assert s["probability_confidence_interval"][0] < 0.55 < s["probability_confidence_interval"][1]
    assert s["calibration_gap"] == 0.05
    assert s["calibration_actionable"] is False
    assert s["calibration_direction"] == "NONE"
    assert s["calibration_adjustment"] == 0.0
    assert s["overconfidence"] is False
    assert s["underconfidence"] is False


def test_probability_drift_detects_recent_forecast_degradation(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    # Prior window: highly accurate explicit probabilities.
    for i in range(20):
        eng.record(
            predicted_confidence="high_conviction",
            realized_win=True,
            predicted_p=0.90,
            decision_as_of=f"2026-01-{(i % 9) + 1:02d}",
            outcome_as_of=f"2026-02-{(i % 9) + 1:02d}",
        )
    # Recent window: same confidence, systematically wrong.
    for i in range(20):
        eng.record(
            predicted_confidence="high_conviction",
            realized_win=False,
            predicted_p=0.90,
            decision_as_of=f"2026-03-{(i % 9) + 1:02d}",
            outcome_as_of=f"2026-04-{(i % 9) + 1:02d}",
        )

    drift = eng.probability_drift(bucket="high_conviction")
    assert drift["status"] == "DEGRADING"
    assert drift["recent_n"] == 20
    assert drift["baseline_n"] == 20
    assert drift["recent_brier"] == 0.81
    assert drift["baseline_brier"] == 0.01
    assert drift["brier_delta"] == 0.8
    assert drift["delta_confidence_interval"][0] > 0
    assert drift["degradation_detected"] is True
    assert drift["improvement_detected"] is False
    assert drift["affects_production"] is False
    assert drift["live_locked"] is True


def test_probability_drift_refuses_small_history(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    for i in range(25):
        eng.record(
            predicted_confidence="good_setup",
            realized_win=i % 2 == 0,
            predicted_p=0.60,
            decision_as_of="2026-01-01",
            outcome_as_of=f"2026-02-{(i % 9) + 1:02d}",
        )

    drift = eng.probability_drift()
    assert drift["status"] == "INSUFFICIENT_PROBABILITY_HISTORY"
    assert drift["explicit_probability_observations"] == 25
    assert drift["degradation_detected"] is False
    assert drift["affects_production"] is False


def test_probability_drift_stable_when_windows_are_indistinguishable(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    for i in range(40):
        eng.record(
            predicted_confidence="good_setup",
            realized_win=i % 2 == 0,
            predicted_p=0.50,
            decision_as_of="2026-01-01",
            outcome_as_of=f"2026-{2 + (i // 20):02d}-{(i % 9) + 1:02d}",
        )

    drift = eng.probability_drift()
    assert drift["status"] == "STABLE_WITHIN_UNCERTAINTY"
    assert drift["brier_delta"] == 0.0
    assert drift["degradation_detected"] is False
    assert drift["improvement_detected"] is False


def test_probability_calibration_is_scoped_to_prediction_source(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    # Old formula: confident and systematically wrong.
    for i in range(20):
        eng.record(
            predicted_confidence="high_conviction",
            realized_win=False,
            predicted_p=0.80,
            prediction_source="evidence_v1",
            decision_as_of="2026-01-01",
            outcome_as_of=f"2026-02-{(i % 9) + 1:02d}",
        )
    # New formula: confident and correct.
    for i in range(20):
        eng.record(
            predicted_confidence="high_conviction",
            realized_win=True,
            predicted_p=0.80,
            prediction_source="evidence_v2",
            decision_as_of="2026-03-01",
            outcome_as_of=f"2026-04-{(i % 9) + 1:02d}",
        )

    old = eng.summary(prediction_source="evidence_v1")
    new = eng.summary(prediction_source="evidence_v2")
    assert old["probability_status"] == "MEASURED"
    assert old["probability_actual_hit_rate"] == 0.0
    assert old["calibration_direction"] == "OVERCONFIDENT"
    assert new["probability_status"] == "MEASURED"
    assert new["probability_actual_hit_rate"] == 1.0
    assert new["calibration_direction"] == "UNDERCONFIDENT"
    assert old["prediction_source"] == "evidence_v1"
    assert new["prediction_source"] == "evidence_v2"

    dossier = eng.dossier()
    assert dossier["prediction_sources"] == ["evidence_v1", "evidence_v2"]
    assert dossier["source_summaries"]["evidence_v1"]["probability_actual_hit_rate"] == 0.0
    assert dossier["source_summaries"]["evidence_v2"]["probability_actual_hit_rate"] == 1.0
    assert dossier["unversioned_probability_observations"] == 0


def test_unversioned_probability_rows_stay_visible_but_do_not_enter_version_scope(tmp_path):
    eng = DecisionCalibrationEngine(tmp_path / "c.json")
    for _ in range(20):
        eng.record(
            predicted_confidence="watch",
            realized_win=True,
            predicted_p=0.55,
            decision_as_of="2026-01-01",
            outcome_as_of="2026-02-01",
        )

    scoped = eng.summary(prediction_source="evidence_v2")
    dossier = eng.dossier()
    assert scoped["sample_size"] == 0
    assert scoped["probability_status"] == "NO_EXPLICIT_PROBABILITIES"
    assert dossier["explicit_probability_observations"] == 20
    assert dossier["unversioned_probability_observations"] == 20
    assert dossier["prediction_sources"] == []
