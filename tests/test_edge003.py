"""EDGE-003: frozen knobs, PIT SMA, next-open fills, no FEATURE-002 / BUY edits."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from research.edge001.calendar import cost_fraction, holding_window
from research.edge003.analyze import classify
from research.edge003.constants import (
    DSR_N_TRIALS,
    PRIMARY_BOOK,
    PRIMARY_CADENCE,
    PRIMARY_SIGNAL,
    SLOPE_LOOKBACK,
    SMA_WINDOW,
)
from research.edge003.study import ew_one_way_turnover
from research.edge003.trend import dist_above_sma, sma_at, trend_flag


def test_primary_knobs_frozen():
    assert PRIMARY_SIGNAL == "T1_PRICE_GT_SMA200_AND_SMA200_RISING"
    assert PRIMARY_BOOK == "all_qualifiers_equal_weight"
    assert PRIMARY_CADENCE == "monthly"
    assert SMA_WINDOW == 200
    assert SLOPE_LOOKBACK == 21
    assert DSR_N_TRIALS == 16


def test_sma_and_flag_are_point_in_time():
    close = np.linspace(80.0, 180.0, 280)
    j = 250
    a = trend_flag(close, j, 200, 21, True)
    leaked = np.append(close, [10_000.0])
    b = trend_flag(leaked, j, 200, 21, True)
    assert a is not None and a == b
    assert sma_at(close, j, 200) == sma_at(leaked, j, 200)
    later = trend_flag(leaked, j + 1, 200, 21, True)
    assert later != a or sma_at(leaked, j + 1, 200) != sma_at(close, j, 200)


def test_trend_flag_fail_closed():
    assert trend_flag(np.array([10.0, 11.0]), 1, 200, 21, True) is None
    bad = np.ones(230)
    bad[100] = 0.0
    assert trend_flag(bad, 229, 200, 21, True) is None
    assert sma_at(np.ones(10), 9, 200) is None


def test_t1_requires_price_above_rising_sma():
    # Rising SMA, last close below SMA → False
    close = np.concatenate([np.full(180, 100.0), np.linspace(100.0, 140.0, 50)])
    j = len(close) - 1
    close[j] = 50.0
    assert trend_flag(close, j, 200, 21, True) is False
    # Rising SMA, last close above SMA → True
    close[j] = 200.0
    assert trend_flag(close, j, 200, 21, True) is True
    # Flat/down SMA, price above → False when slope required; True without
    down = np.linspace(200.0, 100.0, 230)
    k = len(down) - 1
    down[k] = 150.0
    assert trend_flag(down, k, 200, 21, True) is False
    assert trend_flag(down, k, 200, 21, False) is True


def test_dist_above_sma():
    close = np.full(220, 100.0)
    close[-1] = 110.0
    d = dist_above_sma(close, 219, 200)
    assert d is not None and abs(d - (110.0 / (199 * 100.0 + 110.0) * 200.0 - 1.0)) < 1e-9


def test_next_open_not_same_close():
    sessions = [date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 28), date(2020, 3, 2)]
    idx = {d: i for i, d in enumerate(sessions)}
    window = holding_window(sessions, date(2020, 1, 31), date(2020, 2, 28), idx)
    assert window == (date(2020, 2, 3), date(2020, 3, 2))


def test_variable_n_turnover_and_cost_units():
    assert ew_one_way_turnover([], ["A", "B"]) == 1.0
    # Full replace of a 2-name book: one-way = 1.0
    assert abs(ew_one_way_turnover(["A", "B"], ["C", "D"]) - 1.0) < 1e-12
    # Keep both names, same N: 0
    assert abs(ew_one_way_turnover(["A", "B"], ["A", "B"])) < 1e-12
    # 0.32% RT × 50% one-way = 0.16% of NAV
    assert abs(cost_fraction(0.5, 0.32) - 0.0016) < 1e-12


def test_classify_rejects_marketwide_zero_excess():
    stats = {
        "primary": {
            "cagr_net": 0.25, "cagr_gross": 0.26, "ew_cagr": 0.256,
            "nifty_cagr": 0.21, "excess_cagr_ew": 0.0, "excess_cagr_nifty": 0.04,
            "calmar": 0.6, "mean_qualifier_share": 0.96,
            "by_year_net": {"2021": 0.4, "2022": 0.1, "2023": 0.2},
        },
        "blocks": {
            "development": {"n": 28, "excess_cagr_ew": 0.005, "excess_cagr_nifty": 0.04},
            "validation": {"n": 24, "excess_cagr_ew": -0.01, "excess_cagr_nifty": 0.02},
            "confirmation": {"n": 18, "excess_cagr_ew": 0.0, "excess_cagr_nifty": 0.01},
        },
        "inclusion": {"t1_minus_ext1": -0.002, "mean_t1_share": 0.96},
        "formula_excess_ew": {"T1": 0.0, "T2": 0.0, "T1_TOP20": -0.02},
    }
    d = classify(stats)
    assert d["label"] == "REJECT"
    assert "qualifier_is_the_market" in d["failures"]
    assert d["live_trading_authorised"] is False
    assert d["feature002_change_authorised"] is False


def test_classify_only_top20_is_modify():
    stats = {
        "primary": {
            "cagr_net": 0.20, "cagr_gross": 0.21, "ew_cagr": 0.25,
            "nifty_cagr": 0.21, "excess_cagr_ew": -0.05, "excess_cagr_nifty": -0.01,
            "calmar": 0.4, "mean_qualifier_share": 0.55,
            "by_year_net": {"2021": 0.3, "2022": -0.1, "2023": 0.2},
        },
        "blocks": {
            "development": {"n": 28, "excess_cagr_ew": -0.02, "excess_cagr_nifty": 0.0},
            "validation": {"n": 24, "excess_cagr_ew": -0.04, "excess_cagr_nifty": -0.02},
            "confirmation": {"n": 18, "excess_cagr_ew": -0.06, "excess_cagr_nifty": -0.03},
        },
        "inclusion": {"t1_minus_ext1": 0.001, "mean_t1_share": 0.55},
        "formula_excess_ew": {"T1": -0.05, "T2": -0.04, "T1_TOP20": 0.08},
    }
    d = classify(stats)
    assert d["label"] == "MODIFY HYPOTHESIS"
    assert "only_top20_distance_works" in d["failures"]
    assert d["live_trading_authorised"] is False


def test_classify_promising_when_later_excess_and_real_filter():
    stats = {
        "primary": {
            "cagr_net": 0.30, "cagr_gross": 0.31, "ew_cagr": 0.25,
            "nifty_cagr": 0.21, "excess_cagr_ew": 0.05, "excess_cagr_nifty": 0.09,
            "calmar": 0.8, "mean_qualifier_share": 0.48,
            "by_year_net": {"2021": 0.4, "2022": 0.05, "2023": 0.25, "2025": 0.10},
        },
        "blocks": {
            "development": {"n": 28, "excess_cagr_ew": 0.04, "excess_cagr_nifty": 0.08},
            "validation": {"n": 24, "excess_cagr_ew": 0.06, "excess_cagr_nifty": 0.10},
            "confirmation": {"n": 18, "excess_cagr_ew": 0.03, "excess_cagr_nifty": 0.05},
        },
        "inclusion": {"t1_minus_ext1": 0.004, "mean_t1_share": 0.48},
        "formula_excess_ew": {"T1": 0.05, "T2": 0.03, "T1_TOP20": 0.02},
    }
    d = classify(stats)
    assert d["label"] == "PROMISING — FORWARD VALIDATION WARRANTED"
    assert d["live_trading_authorised"] is False


def test_edge003_does_not_import_live_execution():
    text = (Path(__file__).resolve().parents[1] / "research" / "edge003" / "study.py").read_text()
    assert "place_trade" not in text
    assert "observe_production_scan" not in text
    assert "telegram" not in text.lower()


def test_feature002_and_production_paths_untouched():
    import subprocess

    from research.feature002.constants import (
        FEATURE_SET_VERSION,
        FORWARD_START_DATE,
        R3_FORMULA,
        R3_RS_WEIGHT,
        R3_TREND_WEIGHT,
    )

    root = Path(__file__).resolve().parents[1]
    paths = [
        "research/feature002",
        "scan/auto_scan.py",
        "scan/unified_scanner.py",
        "execution/",
        "alerts/telegram_actions.py",
    ]
    refs = (
        "origin/cursor/feature-002-shadow-rank-942f",
        "cursor/feature-002-shadow-rank-942f",
    )
    for ref in refs:
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=root, capture_output=True, text=True,
        )
        if probe.returncode != 0:
            continue
        diff = subprocess.check_output(["git", "diff", ref, "--", *paths], cwd=root)
        assert diff == b"", diff.decode()[:1000]
        break

    assert FEATURE_SET_VERSION == "feature-002.v1"
    assert FORWARD_START_DATE == "2026-07-24"
    assert abs(R3_RS_WEIGHT - 0.67) < 1e-12
    assert abs(R3_TREND_WEIGHT - 0.33) < 1e-12
    assert "within_set_pctl(rs_percentile)" in R3_FORMULA
    auto = (root / "scan" / "auto_scan.py").read_text()
    assert "observe_production_scan" in auto
    assert "feature002_shadow_skip" in auto
