from __future__ import annotations

from datetime import datetime, timedelta, timezone

from product.decision import Decision
from product import evidence_intelligence as EI
from research import feature_store as FS


def _decision(i: int, when: datetime, *, rs: float = 90.0, state: str = "WAIT") -> Decision:
    return Decision(
        symbol=f"T{i:03d}",
        state=state,
        setup="VCP_BREAKOUT",
        generated_at=when.isoformat(),
        score=80.0,
        market_state="TRENDING_BULL",
        sector_state="STRONG",
        technical_evidence={
            "rsi": 60.0 + (i % 5),
            "atr_pct": 2.4,
            "rs_percentile": rs,
            "quality_score": 82.0,
            "volume_ratio": 1.6,
            "pct_from_pivot": 1.0,
            "breadth_pct_above_50dma": 62.0,
            "sector_strength": 1.2,
            "index_trend": "UP",
            "correlation_regime": "NORMAL",
        },
        entry=100.0,
        stop=95.0,
        target=110.0,
        evidence_class="PAPER_FORWARD",
    )


def test_decision_prediction_freeze_and_outcome_immutability(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DB_PATH", tmp_path / "features.db")
    d = _decision(1, datetime(2026, 1, 2, tzinfo=timezone.utc))

    frozen = EI.freeze_decision(
        d,
        evidence={
            "calibrated_p_positive_R": 0.63,
            "p_positive_R": 0.66,
            "historical_confidence": 78.0,
            "decision_confidence": 72.0,
            "formula_version": "evidence_v1",
            "challenger_shadow": {
                "p_positive_R": 0.59,
                "model_version": "clf_test",
                "status": "SHADOW_CANDIDATE",
            },
        },
    )
    assert frozen["status"] == "frozen"

    row = FS.get_observation(f"decision::{d.decision_id}")
    assert row is not None
    assert row["meta"]["predicted_p"] == 0.63
    assert row["meta"]["challenger_predicted_p"] == 0.59
    assert row["outcome"] is None

    first = EI.settle_decision(
        d.decision_id,
        1.25,
        evidence_class="PAPER_FORWARD",
        resolved_at="2026-01-10T00:00:00+00:00",
    )
    assert first["status"] == "settled"
    same = EI.settle_decision(
        d.decision_id,
        1.25,
        evidence_class="PAPER_FORWARD",
    )
    assert same["status"] == "exists"
    conflict = EI.settle_decision(
        d.decision_id,
        -1.0,
        evidence_class="PAPER_FORWARD",
    )
    assert conflict["status"] == "conflict"

    row = FS.get_observation(f"decision::{d.decision_id}")
    assert row["outcome_meta"]["evidence_class"] == "PAPER_FORWARD"
    assert row["outcome_meta"]["not_pnl"] is False


def test_historical_evidence_is_strictly_point_in_time_and_sample_aware(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DB_PATH", tmp_path / "features.db")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for i in range(30):
        d = _decision(i, base + timedelta(days=i), rs=86.0 + (i % 8))
        EI.freeze_decision(d)
        r = 1.2 if i % 5 else -1.0
        lane = "PAPER_FORWARD" if i < 15 else "FORWARD_COUNTERFACTUAL"
        EI.settle_decision(
            d.decision_id,
            r,
            evidence_class=lane,
            not_pnl=lane == "FORWARD_COUNTERFACTUAL",
        )

    query_time = base + timedelta(days=45)
    query = _decision(200, query_time, rs=92.0)

    future = _decision(999, base + timedelta(days=60), rs=92.0)
    EI.freeze_decision(future)
    EI.settle_decision(
        future.decision_id,
        -10.0,
        evidence_class="PAPER_FORWARD",
    )

    evidence = EI.evidence_read(query)
    assert evidence["raw_n"] == 30
    assert evidence["effective_n"] > 20
    assert evidence["research_positive_R_estimate"] is not None
    assert evidence["p_positive_R"] is None
    assert evidence["is_win_probability"] is False
    assert evidence["probability_evidence_scope"] == "INSUFFICIENT_REAL_FORWARD_PAPER"
    assert evidence["forward_probability_raw_n"] == 15
    assert evidence["historical_confidence"] > 0
    assert evidence["calibration_applied"] is False
    assert evidence["calibration_contract_version"] == "explicit_probability_only_v2"
    assert evidence["calibrated_p_positive_R"] is None
    assert evidence["evidence_lane_counts"]["PAPER_FORWARD"] == 15
    assert evidence["evidence_lane_counts"]["FORWARD_COUNTERFACTUAL"] == 15
    assert all(a["symbol"] != "T999" for a in evidence["nearest_analogs"])
    assert evidence["affects_selection"] is False
    assert evidence["live_locked"] is True


def test_evidence_probability_is_not_adjusted_by_non_probability_hit_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DB_PATH", tmp_path / "features.db")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(30):
        d = _decision(i, base + timedelta(days=i), rs=88.0 + (i % 5))
        EI.freeze_decision(d)
        EI.settle_decision(
            d.decision_id,
            1.0 if i % 2 == 0 else -1.0,
            evidence_class="PAPER_FORWARD",
        )

    from product import decision_calibration as DC

    def tier_only_summary(self, **_kwargs):
        return {
            "status": "MEASURED",
            "sample_size": 50,
            "actual_hit_rate": 0.20,
            "probability_status": "NO_EXPLICIT_PROBABILITIES",
            "probability_sample_size": 0,
            "expected_p": None,
            "calibration_gap": None,
        }

    monkeypatch.setattr(DC.DecisionCalibrationEngine, "summary", tier_only_summary)
    evidence = EI.evidence_read(_decision(200, base + timedelta(days=45), rs=91.0))

    assert evidence["research_positive_R_estimate"] is not None
    assert evidence["p_positive_R"] is not None
    assert evidence["probability_evidence_scope"] == "REAL_FORWARD_PAPER"
    assert evidence["forward_probability_effective_n"] >= EI.CLAIM_MIN_EFFECTIVE_N
    assert evidence["calibration_applied"] is False
    assert evidence["calibrated_p_positive_R"] == evidence["p_positive_R"]


def test_historical_and_counterfactual_evidence_never_masquerade_as_forward_probability(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DB_PATH", tmp_path / "features.db")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(36):
        d = _decision(i, base + timedelta(days=i), rs=87.0 + (i % 6))
        EI.freeze_decision(d)
        EI.settle_decision(
            d.decision_id,
            1.0 if i % 3 else -0.8,
            evidence_class="HISTORICAL_REPLAY" if i < 18 else "FORWARD_COUNTERFACTUAL",
            not_pnl=True,
        )

    evidence = EI.evidence_read(_decision(300, base + timedelta(days=50), rs=91.0))

    assert evidence["effective_n"] >= EI.CLAIM_MIN_EFFECTIVE_N
    assert evidence["research_positive_R_estimate"] is not None
    assert evidence["forward_probability_raw_n"] == 0
    assert evidence["forward_probability_effective_n"] == 0.0
    assert evidence["p_positive_R"] is None
    assert evidence["calibrated_p_positive_R"] is None
    assert evidence["is_win_probability"] is False
    assert evidence["probability_evidence_scope"] == "INSUFFICIENT_REAL_FORWARD_PAPER"
    assert evidence["probability_contract_version"] == "real_forward_only_v2"
    assert evidence["historical_confidence"] > 0
    assert evidence["affects_selection"] is False
    assert evidence["live_locked"] is True


def test_real_forward_sample_can_support_probability_without_counterfactual_substitution(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DB_PATH", tmp_path / "features.db")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(32):
        d = _decision(i, base + timedelta(days=i), rs=88.0 + (i % 4))
        EI.freeze_decision(d)
        if i < 26:
            lane = "PAPER_FORWARD"
            not_pnl = False
        else:
            lane = "FORWARD_COUNTERFACTUAL"
            not_pnl = True
        EI.settle_decision(
            d.decision_id,
            1.0 if i % 4 else -1.0,
            evidence_class=lane,
            not_pnl=not_pnl,
        )

    evidence = EI.evidence_read(_decision(400, base + timedelta(days=50), rs=90.0))

    assert evidence["forward_probability_raw_n"] == 26
    assert evidence["forward_probability_effective_n"] >= EI.CLAIM_MIN_EFFECTIVE_N
    assert evidence["p_positive_R"] is not None
    assert evidence["p_positive_R_lower_95"] is not None
    assert evidence["p_positive_R_upper_95"] is not None
    assert evidence["is_win_probability"] is True
    assert evidence["probability_evidence_scope"] == "REAL_FORWARD_PAPER"
    assert evidence["evidence_lane_counts"]["FORWARD_COUNTERFACTUAL"] == 6
