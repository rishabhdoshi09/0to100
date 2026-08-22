"""EDGE-001: frozen knobs, PIT momentum, next-open fills, no FEATURE-002 / BUY edits."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from research.edge001.analyze import classify
from research.edge001.calendar import cost_fraction, holding_window, month_ends, next_session
from research.edge001.constants import (
    DSR_N_TRIALS,
    PRIMARY_CADENCE,
    PRIMARY_N,
    PRIMARY_RANKER,
    SKIP,
)
from research.edge001.momentum import incl_momentum, skip_momentum


def test_primary_knobs_are_frozen():
    assert PRIMARY_RANKER == "M1_12_1"
    assert PRIMARY_N == 20
    assert PRIMARY_CADENCE == "monthly"
    assert SKIP == 21
    assert DSR_N_TRIALS == 64


def test_skip_momentum_is_12_1_not_12_0():
    close = np.arange(1.0, 301.0)
    j = 260
    m1 = skip_momentum(close, j, 252, 21)
    naive = incl_momentum(close, j, 252)
    assert m1 == close[j - 21] / close[j - 252] - 1.0
    assert naive == close[j] / close[j - 252] - 1.0
    assert m1 != naive


def test_momentum_is_point_in_time():
    close = np.linspace(80.0, 180.0, 280)
    j = 260
    a = skip_momentum(close, j, 252, 21)
    leaked = np.append(close, [10_000.0])
    b = skip_momentum(leaked, j, 252, 21)
    assert a == b
    later = skip_momentum(leaked, j + 1, 252, 21)
    assert later != a


def test_fail_closed_on_short_history():
    close = np.array([10.0, 11.0, 12.0])
    assert skip_momentum(close, 2, 252, 21) is None
    assert incl_momentum(close, 2, 252) is None
    bad = np.array([0.0] * 260 + [12.0])
    assert skip_momentum(bad, 260, 252, 21) is None


def test_next_open_not_same_close():
    sessions = [date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 28), date(2020, 3, 2)]
    idx = {d: i for i, d in enumerate(sessions)}
    assert next_session(sessions, date(2020, 1, 31), idx) == date(2020, 2, 3)
    window = holding_window(sessions, date(2020, 1, 31), date(2020, 2, 28), idx)
    assert window == (date(2020, 2, 3), date(2020, 3, 2))
    assert window[0] != date(2020, 1, 31)
    assert holding_window(sessions, date(2020, 2, 28), None, idx) is None


def test_month_end_is_last_session():
    sessions = [date(2021, 1, 28), date(2021, 1, 29), date(2021, 2, 1)]
    assert month_ends(sessions) == [date(2021, 1, 29), date(2021, 2, 1)]


def test_cost_uses_percent_points():
    # 50% one-way * 0.32% RT = 0.16% of NAV, not 16%.
    assert abs(cost_fraction(0.5, 0.32) - 0.0016) < 1e-12


def test_classify_rejects_unordered_and_no_excess():
    stats = {
        "primary": {
            "cagr_net": -0.02, "cagr_gross": 0.01, "ew_cagr": 0.08,
            "nifty_cagr": 0.10, "excess_cagr_ew": -0.10, "excess_cagr_nifty": -0.12,
            "calmar": 0.05, "by_year_net": {"2021": 0.02, "2022": -0.01, "2023": 0.01},
        },
        "deciles": {"M1": {"spearman": -0.2, "d10_only": True}},
        "blocks": {
            "development": {"n": 20, "excess_cagr_ew": 0.01},
            "validation": {"n": 12, "excess_cagr_ew": -0.05, "excess_cagr_nifty": -0.06},
            "confirmation": {"n": 12, "excess_cagr_ew": -0.08, "excess_cagr_nifty": -0.09},
        },
        "h3": {"excess_cagr_ew": -0.04},
        "formula_excess_ew": {"M1": -0.1, "M2": -0.1, "M3": -0.1},
    }
    d = classify(stats)
    assert d["label"] == "REJECT"
    assert d["live_trading_authorised"] is False
    assert d["feature002_change_authorised"] is False


def test_feature002_and_production_paths_untouched():
    import subprocess

    root = Path(__file__).resolve().parents[1]
    diff = subprocess.check_output(
        [
            "git", "diff", "54a423f", "--",
            "research/feature002",
            "scan/auto_scan.py",
            "scan/unified_scanner.py",
            "execution/",
            "alerts/telegram_actions.py",
        ],
        cwd=root,
    )
    assert diff == b"", diff.decode()[:1000]


def test_stale_last_print_is_not_live():
    from research.edge001.study import live_on_session
    from research.sepa.universe_pit import FastInvestable

    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    dead = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1_000_000},
        index=idx,
    )
    live = dead.copy()
    live.index = pd.date_range("2024-01-01", periods=300, freq="B")
    fast = FastInvestable({"DEAD": dead, "LIVE": live})
    as_of = date(2026, 6, 15)
    assert live_on_session(fast, "DEAD", as_of) is False
    assert live_on_session(fast, "LIVE", as_of) is False
    last = live.index[-1].date()
    assert live_on_session(fast, "LIVE", last) is True


def test_edge001_does_not_import_live_execution():
    text = (Path(__file__).resolve().parents[1] / "research" / "edge001" / "study.py").read_text()
    assert "place_trade" not in text
    assert "telegram" not in text.lower()
    assert "observe_production_scan" not in text
