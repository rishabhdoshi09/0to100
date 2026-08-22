"""EDGE-002: frozen knobs, PIT vol, no FEATURE-002 / BUY edits."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from research.edge001.calendar import holding_window
from research.edge002.constants import DSR_N_TRIALS, PRIMARY_N, PRIMARY_RANKER, V1_LOOKBACK
from research.edge002.vol import realized_vol


def test_primary_knobs_frozen():
    assert PRIMARY_RANKER == "V1_126_REALIZED_VOL"
    assert PRIMARY_N == 20
    assert V1_LOOKBACK == 126
    assert DSR_N_TRIALS == 48


def test_realized_vol_is_point_in_time():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    j = 150
    a = realized_vol(close, j, 126)
    leaked = np.append(close, [close[-1] * 3.0])
    b = realized_vol(leaked, j, 126)
    assert a is not None and a == b
    later = realized_vol(leaked, j + 1, 126)
    assert later != a


def test_realized_vol_fail_closed():
    assert realized_vol(np.array([10.0, 11.0]), 1, 126) is None
    bad = np.ones(130)
    bad[100] = 0.0
    assert realized_vol(bad, 129, 126) is None


def test_next_open_not_same_close():
    sessions = [date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 28), date(2020, 3, 2)]
    idx = {d: i for i, d in enumerate(sessions)}
    window = holding_window(sessions, date(2020, 1, 31), date(2020, 2, 28), idx)
    assert window == (date(2020, 2, 3), date(2020, 3, 2))


def test_edge002_does_not_import_live_execution():
    text = (Path(__file__).resolve().parents[1] / "research" / "edge002" / "study.py").read_text()
    assert "place_trade" not in text
    assert "observe_production_scan" not in text
    assert "telegram" not in text.lower()
