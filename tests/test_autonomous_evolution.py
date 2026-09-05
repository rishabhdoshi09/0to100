from __future__ import annotations

from product.autonomous_evolution import (
    _aggregate_splits,
    _split_plan,
    confidence_from_policies,
    summarize_report,
)


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
    historical = {
        "policy_id": "HIST_SETUP::VCP",
        "dimension": "setup",
        "bucket": "VCP",
        "sample_size": 30,
        "expectancy_R": 0.45,
        "historical_reproduced_positive": True,
        "historical_confidence_score": 72.0,
        "splits_tested": 3,
        "positive_splits": 3,
    }
    positive_forward = {
        "policy_id": "SETUP::VCP",
        "dimension": "setup",
        "bucket": "VCP",
        "sample_size": 20,
        "expectancy_R": 0.40,
        "expectancy_difference_R": 0.40,
        "evidence_source": "paper_forward_taken_execution_adjusted",
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
