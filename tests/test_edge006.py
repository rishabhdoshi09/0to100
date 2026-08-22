"""EDGE-006: frozen knobs, PIT ADV, no FEATURE-002 / BUY edits."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from research.edge001.calendar import holding_window
from research.edge006.analyze import classify
from research.edge006.constants import DSR_N_TRIALS, PRIMARY_N, PRIMARY_RANKER


def test_primary_knobs_frozen():
    assert PRIMARY_RANKER == "L1_20D_HIGH_ADV"
    assert PRIMARY_N == 20
    assert DSR_N_TRIALS == 12


def test_next_open_not_same_close():
    sessions = [date(2020, 1, 31), date(2020, 2, 3), date(2020, 2, 28), date(2020, 3, 2)]
    idx = {d: i for i, d in enumerate(sessions)}
    assert holding_window(sessions, date(2020, 1, 31), date(2020, 2, 28), idx) == (date(2020, 2, 3), date(2020, 3, 2))


def test_classify_rejects_no_excess_no_slope():
    stats = {
        "primary": {"cagr_net": 0.10, "cagr_gross": 0.12, "ew_cagr": 0.25, "nifty_cagr": 0.21,
                    "excess_cagr_ew": -0.15, "excess_cagr_nifty": -0.11, "by_year_net": {"2021": 0.1, "2022": 0.05, "2023": 0.08}},
        "deciles": {"L1": {"spearman": -0.3, "d10_minus_d1": -0.01}},
        "blocks": {
            "development": {"n": 28, "excess_cagr_ew": -0.1, "excess_cagr_nifty": -0.08},
            "validation": {"n": 24, "excess_cagr_ew": -0.12, "excess_cagr_nifty": -0.09},
            "confirmation": {"n": 18, "excess_cagr_ew": -0.14, "excess_cagr_nifty": -0.10},
        },
        "formula_excess_ew": {"L1": -0.15, "L0": -0.10},
        "inference": {"excess_ew": {"ci": {"excludes_zero": False}}, "harness_excess_ew": {"verdict": "REJECT"}},
    }
    d = classify(stats)
    assert d["label"] == "REJECT"
    assert d["live_trading_authorised"] is False


def test_edge006_does_not_import_live_execution():
    text = (Path(__file__).resolve().parents[1] / "research" / "edge006" / "study.py").read_text()
    assert "place_trade" not in text
    assert "observe_production_scan" not in text
