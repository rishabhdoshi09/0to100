"""EDGE-004: frozen knobs, PIT 21d return, no FEATURE-002 / BUY edits."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from research.edge001.calendar import holding_window
from research.edge001.momentum import incl_momentum, skip_momentum
from research.edge004.analyze import classify
from research.edge004.constants import DSR_N_TRIALS, PRIMARY_N, PRIMARY_RANKER, R0_SKIP, R1_LOOKBACK


def test_primary_knobs_frozen():
    assert PRIMARY_RANKER == "R1_21_INCLUSIVE_LOSERS"
    assert PRIMARY_N == 20
    assert R1_LOOKBACK == 21
    assert R0_SKIP == 5
    assert DSR_N_TRIALS == 24


def test_r1_is_inclusive_not_skip():
    close = np.arange(1.0, 40.0)
    j = 30
    r1 = incl_momentum(close, j, 21)
    r0 = skip_momentum(close, j, 21, 5)
    assert r1 == close[j] / close[j - 21] - 1.0
    assert r0 == close[j - 5] / close[j - 21] - 1.0
    assert r1 != r0


def test_momentum_is_point_in_time():
    close = np.linspace(80.0, 180.0, 80)
    j = 50
    a = incl_momentum(close, j, 21)
    leaked = np.append(close, [10_000.0])
    b = incl_momentum(leaked, j, 21)
    assert a == b
    later = incl_momentum(leaked, j + 1, 21)
    assert later != a


def test_fail_closed_on_short_history():
    assert incl_momentum(np.array([10.0, 11.0]), 1, 21) is None
    bad = np.zeros(30)
    bad[-1] = 12.0
    assert incl_momentum(bad, 29, 21) is None


def test_next_open_not_same_close():
    sessions = [date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 28), date(2020, 3, 2)]
    idx = {d: i for i, d in enumerate(sessions)}
    window = holding_window(sessions, date(2020, 1, 31), date(2020, 2, 28), idx)
    assert window == (date(2020, 2, 3), date(2020, 3, 2))


def test_classify_rejects_no_slope_no_excess():
    stats = {
        "primary": {
            "cagr_net": 0.10, "cagr_gross": 0.12, "ew_cagr": 0.25,
            "nifty_cagr": 0.21, "excess_cagr_ew": -0.15, "excess_cagr_nifty": -0.11,
            "calmar": 0.2, "by_year_net": {"2021": 0.2, "2022": -0.1, "2023": 0.1},
        },
        "deciles": {"R1": {"spearman": -0.4, "d10_minus_d1": -0.02}},
        "blocks": {
            "development": {"n": 28, "excess_cagr_ew": -0.10, "excess_cagr_nifty": -0.08},
            "validation": {"n": 24, "excess_cagr_ew": -0.12, "excess_cagr_nifty": -0.10},
            "confirmation": {"n": 18, "excess_cagr_ew": -0.18, "excess_cagr_nifty": -0.14},
        },
        "formula_excess_ew": {"R1": -0.15, "R2": -0.12, "WIN": -0.04},
        "inference": {"excess_ew": {"ci": {"excludes_zero": False}}, "harness_excess_ew": {"verdict": "REJECT"}},
    }
    d = classify(stats)
    assert d["label"] == "REJECT"
    assert d["live_trading_authorised"] is False


def test_classify_only_winners_is_modify():
    stats = {
        "primary": {
            "cagr_net": 0.10, "cagr_gross": 0.12, "ew_cagr": 0.25,
            "nifty_cagr": 0.21, "excess_cagr_ew": -0.15, "excess_cagr_nifty": -0.11,
            "calmar": 0.3, "by_year_net": {"2021": 0.2, "2022": -0.05, "2023": 0.1},
        },
        "deciles": {"R1": {"spearman": 0.1, "d10_minus_d1": -0.01}},
        "blocks": {
            "development": {"n": 28, "excess_cagr_ew": -0.08, "excess_cagr_nifty": -0.04},
            "validation": {"n": 24, "excess_cagr_ew": -0.12, "excess_cagr_nifty": -0.08},
            "confirmation": {"n": 18, "excess_cagr_ew": -0.16, "excess_cagr_nifty": -0.10},
        },
        "formula_excess_ew": {"R1": -0.15, "R2": -0.10, "WIN": 0.05},
        "inference": {"excess_ew": {"ci": {"excludes_zero": False}}, "harness_excess_ew": {"verdict": "REJECT"}},
    }
    d = classify(stats)
    assert d["label"] == "MODIFY HYPOTHESIS"
    assert "only_winners_work" in d["failures"]
    assert d["live_trading_authorised"] is False


def test_classify_robustness_bar_blocks_promising():
    """EDGE-003 lesson: small positive excess + CI includes 0 is RESEARCH-ONLY."""
    stats = {
        "primary": {
            "cagr_net": 0.27, "cagr_gross": 0.28, "ew_cagr": 0.256,
            "nifty_cagr": 0.21, "excess_cagr_ew": 0.014, "excess_cagr_nifty": 0.06,
            "calmar": 1.0, "by_year_net": {"2021": 0.4, "2022": 0.05, "2023": 0.25, "2025": 0.02},
        },
        "deciles": {"R1": {"spearman": 0.55, "d10_minus_d1": 0.01}},
        "blocks": {
            "development": {"n": 28, "excess_cagr_ew": 0.02, "excess_cagr_nifty": 0.06},
            "validation": {"n": 24, "excess_cagr_ew": 0.02, "excess_cagr_nifty": 0.05},
            "confirmation": {"n": 18, "excess_cagr_ew": 0.005, "excess_cagr_nifty": 0.01},
        },
        "formula_excess_ew": {"R1": 0.014, "R2": 0.01, "WIN": -0.02},
        "inference": {
            "excess_ew": {"ci": {"excludes_zero": False}},
            "harness_excess_ew": {"verdict": "INCONCLUSIVE"},
        },
    }
    d = classify(stats)
    assert d["label"] == "RESEARCH-ONLY"
    assert d["live_trading_authorised"] is False


def test_edge004_does_not_import_live_execution():
    text = (Path(__file__).resolve().parents[1] / "research" / "edge004" / "study.py").read_text()
    assert "place_trade" not in text
    assert "observe_production_scan" not in text
    assert "telegram" not in text.lower()


def test_feature002_and_production_paths_untouched():
    import subprocess

    from research.feature002.constants import FEATURE_SET_VERSION, FORWARD_START_DATE, R3_FORMULA

    root = Path(__file__).resolve().parents[1]
    paths = ["research/feature002", "scan/auto_scan.py", "scan/unified_scanner.py", "execution/", "alerts/telegram_actions.py"]
    refs = ("origin/cursor/feature-002-shadow-rank-942f", "cursor/feature-002-shadow-rank-942f")
    for ref in refs:
        probe = subprocess.run(["git", "rev-parse", "--verify", ref], cwd=root, capture_output=True, text=True)
        if probe.returncode != 0:
            continue
        diff = subprocess.check_output(["git", "diff", ref, "--", *paths], cwd=root)
        assert diff == b"", diff.decode()[:1000]
        break
    assert FEATURE_SET_VERSION == "feature-002.v1"
    assert FORWARD_START_DATE == "2026-07-24"
    assert "within_set_pctl(rs_percentile)" in R3_FORMULA
