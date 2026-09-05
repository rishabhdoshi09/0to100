from __future__ import annotations

from product.autonomous_evolution import (
    _aggregate_splits,
    _split_plan,
    summarize_report,
)
from product.evidence_confidence import confidence_from_policies


def _buy(setup: str, r_value: float) -> dict:
    return {
        "symbol": "TEST",
        "decision": "BUY",
        "setup": setup,
        "pit_grade": "PIT_STRONG",
        "pit": {"comparable_to_forward": True},
        "outcome_status": "MATURED",
        "r_multiple": r_value,
    }


def _historical_policy(setup: str = "VCP", score: float = 72.0) -> dict:
    return {
        "policy_id": f"HIST_SETUP::{setup}",
        "dimension": "setup",
        "bucket": setup,
        "sample_size": 30,
        "expectancy_R": 0.45,
        "expectancy_difference_R": 0.45,
        "production_status": "ELIGIBLE",
        "confidence": "REPRODUCED_BACKTEST",
        "affects_selection": True,
        "historical_reproduced_positive": True,
        "historical_confidence_score": score,
        "splits_tested": 3,
        "positive_splits": 3,
    }


def test_reproduced_history_requires_multiple_positive_splits(monkeypatch):
    monkeypatch.setenv("QT_EVOLUTION_MIN_HIST_N", "8")
    monkeypatch.setenv("QT_EVOLUTION_MIN_POSITIVE_SPLITS", "2")
    monkeypatch.setenv("QT_EVOLUTION_MIN_MEAN_R", "0.15")

    reports = [
        {"decisions": [_buy("VCP", 0.8), _buy("VCP", 0.6), _buy("VCP", 0.4)]},
        {"decisions": [_buy("VCP", 0.7), _buy("VCP", 0.5), _buy("VCP", 0.3)]},
        {"decisions": [_buy("VCP", 0.9), _buy("VCP", 0.2), _buy("VCP", 0.4)]},
    ]
    summaries = [summarize_report(report, split_id=f"s{i}") for i, report in enumerate(reports)]
    setups, _ = _aggregate_splits(summaries)

    row = setups["VCP"]
    assert row["n"] == 9
    assert row["tested_splits"] == 3
    assert row["positive_splits"] == 3
    assert row["reproduced"] is True
    assert 0 < row["historical_confidence_score"] <= 79


def test_one_good_backtest_slice_is_not_reproduction(monkeypatch):
    monkeypatch.setenv("QT_EVOLUTION_MIN_HIST_N", "3")
    monkeypatch.setenv("QT_EVOLUTION_MIN_POSITIVE_SPLITS", "2")
    monkeypatch.setenv("QT_EVOLUTION_MIN_MEAN_R", "0.10")

    summaries = [
        summarize_report({"decisions": [_buy("BREAKOUT", 1.0)] * 3}, split_id="positive"),
        summarize_report({"decisions": [_buy("BREAKOUT", -0.6)] * 3}, split_id="negative"),
    ]
    setups, _ = _aggregate_splits(summaries)

    assert setups["BREAKOUT"]["positive_splits"] == 1
    assert setups["BREAKOUT"]["reproduced"] is False
    assert setups["BREAKOUT"]["historical_confidence_score"] <= 49


def test_forward_paper_can_strengthen_or_decay_historical_confidence():
    historical = _historical_policy()
    positive_forward = {
        "policy_id": "SETUP::VCP",
        "dimension": "setup",
        "bucket": "VCP",
        "sample_size": 20,
        "expectancy_R": 0.40,
        "expectancy_difference_R": 0.40,
        "evidence_source": "paper_forward_taken_execution_adjusted",
        "affects_selection": True,
    }
    negative_forward = {
        **positive_forward,
        "sample_size": 10,
        "expectancy_R": -0.50,
        "expectancy_difference_R": -0.50,
    }

    base = confidence_from_policies({"setup_label": "VCP"}, [historical])
    strengthened = confidence_from_policies(
        {"setup_label": "VCP"}, [historical, positive_forward]
    )
    decayed = confidence_from_policies(
        {"setup_label": "VCP"}, [historical, negative_forward]
    )

    assert base["confidence_stage"] == "HISTORICAL_BASE"
    assert strengthened["evidence_confidence_score"] > base["evidence_confidence_score"]
    assert strengthened["confidence_stage"] == "FORWARD_CONFIRMED"
    assert decayed["evidence_confidence_score"] < base["evidence_confidence_score"]
    assert decayed["confidence_stage"] == "FORWARD_DECAYED"
    assert decayed["paper_eligible"] is False
    assert decayed["live_locked"] is True


def test_positive_gross_only_paper_does_not_fake_confidence_boost():
    historical = _historical_policy(score=70.0)
    gross_only = {
        "policy_id": "SETUP::VCP",
        "sample_size": 15,
        "expectancy_R": 0.60,
        "expectancy_difference_R": 0.60,
        "evidence_source": "paper_forward_taken_gross_only",
        "affects_selection": False,
    }

    result = confidence_from_policies({"setup_label": "VCP"}, [historical, gross_only])

    assert result["evidence_confidence_score"] == 70.0
    assert result["forward_n"] == 0
    assert result["forward_observed_n"] == 15
    assert result["confidence_stage"] == "FORWARD_EVIDENCE_UNTRUSTED"
    assert result["forward_trusted_positive"] is False


def test_production_history_gate_blocks_until_bootstrap_completes(monkeypatch):
    import product.autonomous_evolution as evolution
    from product.evidence_policy_engine import _historical_gate

    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {
            "required": True,
            "status": "RUNNING",
            "analysis_complete": False,
            "paper_ready_setups": 0,
        },
    )
    monkeypatch.setattr(evolution, "ensure_started_async", lambda: {"status": "RUNNING"})

    gate = _historical_gate({"setup_label": "VCP"}, [], enabled=True)

    assert gate["paper_eligible"] is False
    assert gate["bootstrap_complete"] is False
    assert gate["confidence_stage"] == "HISTORICAL_BOOTSTRAP"
    assert gate["live_locked"] is True


def test_production_history_gate_releases_only_reproduced_setup(monkeypatch):
    import product.autonomous_evolution as evolution
    from product.evidence_policy_engine import _historical_gate

    monkeypatch.setattr(
        evolution,
        "bootstrap_status",
        lambda: {
            "required": True,
            "status": "SUCCEEDED",
            "analysis_complete": True,
            "paper_ready_setups": 1,
        },
    )

    gate = _historical_gate(
        {"setup_label": "VCP"},
        [_historical_policy()],
        enabled=True,
    )
    unknown = _historical_gate(
        {"setup_label": "UNSEEN"},
        [_historical_policy()],
        enabled=True,
    )

    assert gate["paper_eligible"] is True
    assert gate["historical_ready"] is True
    assert gate["bootstrap_complete"] is True
    assert unknown["paper_eligible"] is False
    assert unknown["historical_ready"] is False


def test_split_plan_is_disjoint_and_leaves_forward_outcome_buffer(monkeypatch):
    monkeypatch.setenv("QT_EVOLUTION_SPLITS", "3")
    monkeypatch.setenv("QT_EVOLUTION_SESSIONS_PER_SPLIT", "4")
    monkeypatch.setenv("QT_EVOLUTION_OUTCOME_BUFFER", "5")
    sessions = [f"2026-01-{day:02d}" for day in range(1, 25)]

    plan = _split_plan(sessions)

    assert len(plan) == 3
    flattened = [day for split in plan for day in split["sessions"]]
    assert len(flattened) == len(set(flattened)) == 12
    assert max(flattened) < sessions[-5]
