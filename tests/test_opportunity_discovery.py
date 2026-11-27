import pandas as pd
import pytest

from research.opportunity_discovery import _path, _rules, _summary, _trade
from research.calibration import discovery_exit_calibration


def test_common_entry_and_portfolio_cash_use_same_costs():
    sessions = pd.bdate_range("2025-01-06", periods=10)
    flat = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.}, index=sessions)
    trade = _trade(flat, sessions, .0032)
    assert trade[0] == pytest.approx(-.0032)
    assert trade[2] == "expiry"
    assert _path([trade], 10)[-1] == pytest.approx(1 - .0032/5)
    missing = _trade(flat.iloc[1:], sessions, .0032)
    assert missing == (None, [1.0] * 10, "missing_entry")
    unresolved = _trade(flat.iloc[:1], sessions, .0032)
    assert unresolved[0] is None and unresolved[2] == "censored_after_entry"
    assert unresolved[1][0] == pytest.approx(1 - .0032)
    assert unresolved[1][1:] == [0.] * 9


def test_selection_uses_rank_regime_and_market_relative_return():
    common = dict(score=20, r30=.02, uptrend=True, reversion=0., sector_edge=0., momentum=.01, turnover=8)
    rows = {
        "AAA": {**common, "signals": {"MOMENTUM"}},
        "BBB": {**common, "score": 30, "r30": .05, "signals": {"PULLBACK_SUPPORT", "POCKET_PIVOT"}},
    }
    selected = _rules(rows, False, .03)
    assert selected["scanner"] == ["BBB", "AAA"]
    assert selected["relative_strength"] == ["BBB"]
    assert selected["scanner_risk_on"] == []
    assert selected["scanner_confluence"] == ["BBB"]


def test_exit_forecast_uses_only_earlier_cycles():
    def row(ordinal, outcome):
        return dict(ordinal=ordinal, opportunities=1, trades=[(outcome, .01)],
                    path=[1.002] * 10, random_path=[1.] * 10, market_path=[1.] * 10)

    report = _summary([row(59, "target"), row(69, "stop")], 1, 10)
    assert report["early_exit_forecast"] == pytest.approx({"target": .6, "stop": .2, "expiry": .2})
    assert report["later_exit_rate"] == {"target": 0., "stop": 1., "expiry": 0.}
    assert report["later_brier"] == pytest.approx(.6**2 + .8**2 + .2**2)


def test_discovery_calibration_compares_forecasts_on_same_resolved_fills():
    first = dict(day="2025-01-02", ordinal=59, trades=[("target", .04), ("missing_entry", None)])
    second = dict(day="2025-01-16", ordinal=69, trades=[("stop", -.03), ("censored_after_entry", None)])
    random_first = dict(day=first["day"], ordinal=59,
                        trades=[("stop", -.03), ("stop", -.03), ("target", .04)])
    result = discovery_exit_calibration([first, second], [random_first, second], training_cycles=1)
    assert result["strategy_forecast"]["probabilities"] == pytest.approx(dict(target=.6, stop=.2, expiry=.2))
    assert result["strategy_forecast"]["brier"] == pytest.approx(1.04)
    assert result["strategy_forecast"]["log_loss"] > result["reference_forecast"]["log_loss"]
    assert result["reference_training_resolved_fills"] == 2
    assert result["resolved_fills"] == result["resolved_cycles"] == 1
    assert result["evaluation_counts"] == {"stop": 1, "censored_after_entry": 1}
    with pytest.raises(ValueError, match="strictly ordered"):
        discovery_exit_calibration([second, first], [random_first, second], training_cycles=1)
    overlapping = dict(second, ordinal=68)
    with pytest.raises(ValueError, match="non-overlapping"):
        discovery_exit_calibration([first, overlapping], [random_first, overlapping], training_cycles=1)


def test_discovery_calibration_with_no_resolved_training_stays_unmeasured():
    rows = [dict(day="2025-01-02", ordinal=59, trades=[("missing_entry", None)]),
            dict(day="2025-01-16", ordinal=69, trades=[("expiry", .01)])]
    result = discovery_exit_calibration(rows, rows, training_cycles=1)
    assert result["status"] == "UNMEASURED"
    assert "strategy_forecast" not in result
