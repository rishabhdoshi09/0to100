from research.autonomy.replay_stopping import assess_replay_need, coverage_gap


def test_coverage_gap_is_deterministic_and_only_uses_observed_regimes():
    assert coverage_gap({"BULL_TREND": 1, "BEAR": 3}, ["BEAR", "BULL_TREND"]) == {
        "BEAR": 0,
        "BULL_TREND": 2,
    }


def test_continue_when_known_regime_gap_can_be_reduced():
    out = assess_replay_need(
        coverage={"BULL_TREND": 1, "BEAR": 3},
        remaining_regimes=["BULL_TREND", "BULL_TREND", "BEAR"],
        remaining_total=3,
    )
    assert out["decision"] == "CONTINUE"
    assert out["reason"] == "coverage_gap_remains"
    assert out["expected_information_gain"] == 2
    assert out["evidence_lane"] == "HISTORICAL_REPLAY"
    assert out["outcome_blind"] is True
    assert out["live_money_unchanged"] is True


def test_stop_when_known_coverage_is_plateaued():
    out = assess_replay_need(
        coverage={"BULL_TREND": 4, "BEAR": 3},
        remaining_regimes=["BULL_TREND", "BEAR"],
        remaining_total=2,
    )
    assert out["decision"] == "STOP"
    assert out["reason"] == "coverage_plateau"
    assert out["expected_information_gain"] == 0


def test_missing_pit_context_fails_closed_to_continue_not_false_plateau():
    out = assess_replay_need(
        coverage={"BULL_TREND": 5},
        remaining_regimes=[],
        remaining_unknown=6,
        remaining_total=8,
    )
    assert out["decision"] == "CONTINUE"
    assert out["reason"] == "pit_context_too_incomplete_to_prove_plateau"


def test_empty_backlog_stops_without_inventing_evidence():
    out = assess_replay_need(coverage={}, remaining_regimes=[], remaining_total=0)
    assert out["decision"] == "STOP"
    assert out["reason"] == "historical_backlog_caught_up"
