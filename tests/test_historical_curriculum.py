from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.autonomy import historical_curriculum as HC


def _trend_frame(periods=230, future_crash=False):
    dates = pd.bdate_range("2025-10-01", periods=periods)
    close = np.linspace(100.0, 220.0, periods)
    if future_crash:
        close[-1] = 40.0
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
        },
        index=dates,
    )


def test_classifier_discards_future_rows_even_from_untrusted_loader():
    frame = _trend_frame()
    as_of = frame.index[-2].strftime("%Y-%m-%d")
    future = frame.copy()
    future.iloc[-1, future.columns.get_loc("Close")] = 1.0
    future.iloc[-1, future.columns.get_loc("High")] = 2.0
    future.iloc[-1, future.columns.get_loc("Low")] = 0.5

    state = HC.classify_session(as_of, index_fn=lambda _ticker, _day: future)

    assert state.available is True
    assert state.regime == "BULL_TREND"
    assert state.history_rows == len(frame) - 1
    assert state.close == pytest.approx(float(frame["Close"].iloc[-2]), abs=1e-6)


def test_classifier_refuses_shallow_history():
    frame = _trend_frame(periods=80)
    day = frame.index[-1].strftime("%Y-%m-%d")
    state = HC.classify_session(day, index_fn=lambda _ticker, _day: frame)
    assert state.available is False
    assert state.history_rows == 80
    assert "need 200" in state.reason


def test_curriculum_prioritizes_underrepresented_regimes():
    regimes = {
        "2026-01-01": "BULL_TREND",
        "2026-01-02": "BULL_TREND",
        "2026-01-03": "BULL_TREND",
        "2026-01-04": "BEAR",
        "2026-01-05": "BEAR",
        "2026-01-06": "CHOPPY",
    }

    def classify(day):
        return HC.HistoricalMarketState(day, True, regime=regimes[day], history_rows=220)

    out = HC.select_regime_balanced_sessions(
        list(regimes),
        processed_sessions=["2026-01-01", "2026-01-02"],
        batch_size=3,
        classifier=classify,
    )

    assert out["selection_policy"] == "ACTIVE_REGIME_COVERAGE"
    assert out["outcome_blind_selection"] is True
    assert out["sessions"][0] == "2026-01-04"
    assert "2026-01-06" in out["sessions"]
    assert out["coverage_before"] == {"BULL_TREND": 2}
    assert out["coverage_after"]["BEAR"] >= 1
    assert out["coverage_after"]["CHOPPY"] >= 1


def test_curriculum_falls_back_truthfully_when_regime_unavailable():
    days = ["2026-01-01", "2026-01-02", "2026-01-03"]
    unavailable = lambda day: HC.HistoricalMarketState(day, False, reason="no cache")

    out = HC.select_regime_balanced_sessions(
        days, processed_sessions=[], batch_size=2, classifier=unavailable
    )

    assert out["sessions"] == days[:2]
    assert out["selection_policy"] == "DURABLE_CURSOR"
    assert out["known_regime_sessions_selected"] == 0
    assert out["unknown_regime_sessions_selected"] == 2
