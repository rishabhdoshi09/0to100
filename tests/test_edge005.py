"""EDGE-005: frozen knobs, PIT 52w high, no FEATURE-002 / BUY edits."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from research.edge001.calendar import holding_window
from research.edge005.analyze import classify
from research.edge005.constants import DSR_N_TRIALS, P1_LOOKBACK, PRIMARY_N, PRIMARY_RANKER
from research.edge005.proximity import proximity_to_high


def test_primary_knobs_frozen():
    assert PRIMARY_RANKER == "P1_252_NEAR_HIGH"
    assert PRIMARY_N == 20
    assert P1_LOOKBACK == 252
    assert DSR_N_TRIALS == 20


def test_proximity_is_point_in_time():
    close = np.concatenate([np.linspace(50.0, 150.0, 200), np.linspace(149.0, 100.0, 80)])
    j = 260
    a = proximity_to_high(close, j, 252)
    leaked = np.append(close, [10_000.0])
    b = proximity_to_high(leaked, j, 252)
    assert a is not None and a == b
    later = proximity_to_high(leaked, j + 1, 252)
    assert later != a


def test_proximity_fail_closed():
    assert proximity_to_high(np.array([10.0, 11.0]), 1, 252) is None
    bad = np.ones(260)
    bad[100] = 0.0
    assert proximity_to_high(bad, 259, 252) is None


def test_proximity_at_high_is_one():
    close = np.linspace(80.0, 180.0, 260)
    p = proximity_to_high(close, 259, 252)
    assert p is not None and abs(p - 1.0) < 1e-12
    close[259] = close[258] * 0.5
    p2 = proximity_to_high(close, 259, 252)
    assert p2 is not None and p2 < 0.6


def test_next_open_not_same_close():
    sessions = [date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 28), date(2020, 3, 2)]
    idx = {d: i for i, d in enumerate(sessions)}
    window = holding_window(sessions, date(2020, 1, 31), date(2020, 2, 28), idx)
    assert window == (date(2020, 2, 3), date(2020, 3, 2))


def test_classify_rejects_no_slope_no_excess():
    stats = {
        "primary": {
            "cagr_net": 0.12, "cagr_gross": 0.14, "ew_cagr": 0.25,
            "nifty_cagr": 0.21, "excess_cagr_ew": -0.13, "excess_cagr_nifty": -0.09,
            "calmar": 0.3, "by_year_net": {"2021": 0.2, "2022": -0.05, "2023": 0.1},
        },
        "deciles": {"P1": {"spearman": -0.2, "d10_minus_d1": -0.01}},
        "blocks": {
            "development": {"n": 28, "excess_cagr_ew": -0.10, "excess_cagr_nifty": -0.06},
            "validation": {"n": 24, "excess_cagr_ew": -0.12, "excess_cagr_nifty": -0.08},
            "confirmation": {"n": 18, "excess_cagr_ew": -0.14, "excess_cagr_nifty": -0.10},
        },
        "formula_excess_ew": {"P1": -0.13, "P3": -0.10, "LAG": -0.08},
        "inference": {"excess_ew": {"ci": {"excludes_zero": False}}, "harness_excess_ew": {"verdict": "REJECT"}},
    }
    d = classify(stats)
    assert d["label"] == "REJECT"
    assert d["live_trading_authorised"] is False


def test_classify_robustness_bar_blocks_promising():
    stats = {
        "primary": {
            "cagr_net": 0.27, "cagr_gross": 0.28, "ew_cagr": 0.256,
            "nifty_cagr": 0.21, "excess_cagr_ew": 0.014, "excess_cagr_nifty": 0.06,
            "calmar": 1.0, "by_year_net": {"2021": 0.4, "2022": 0.05, "2023": 0.25, "2025": 0.02},
        },
        "deciles": {"P1": {"spearman": 0.55, "d10_minus_d1": 0.01}},
        "blocks": {
            "development": {"n": 28, "excess_cagr_ew": 0.02, "excess_cagr_nifty": 0.06},
            "validation": {"n": 24, "excess_cagr_ew": 0.02, "excess_cagr_nifty": 0.05},
            "confirmation": {"n": 18, "excess_cagr_ew": 0.004, "excess_cagr_nifty": 0.01},
        },
        "formula_excess_ew": {"P1": 0.014, "P3": 0.01, "LAG": -0.04},
        "inference": {
            "excess_ew": {"ci": {"excludes_zero": False}},
            "harness_excess_ew": {"verdict": "INCONCLUSIVE"},
        },
    }
    d = classify(stats)
    assert d["label"] == "RESEARCH-ONLY"
    assert d["live_trading_authorised"] is False


def test_edge005_does_not_import_live_execution():
    text = (Path(__file__).resolve().parents[1] / "research" / "edge005" / "study.py").read_text()
    assert "place_trade" not in text
    assert "observe_production_scan" not in text
    assert "telegram" not in text.lower()


def test_feature002_and_production_paths_untouched():
    import subprocess
    from research.feature002.constants import FEATURE_SET_VERSION, FORWARD_START_DATE

    root = Path(__file__).resolve().parents[1]
    paths = ["research/feature002", "scan/auto_scan.py", "execution/", "alerts/telegram_actions.py"]
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
