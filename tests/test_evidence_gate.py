"""Invariant 6 on the canonical scanner: a BUY needs a positive measured edge.

Owner's Mac 2026-11-15: the walk-forward backtest measured 13 of 17 signals
as net losers, yet the live scanner kept issuing BUYs. Calibration only
rescaled scores relative to each other; the combo-edge veto existed only in
the legacy auto_scan loop; and any 52-week breakout / high-tight flag became a
BUY regardless of score or evidence.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from scan import unified_scanner as US


def _snap(edges):
    return {"snapshot_id": "test", "multipliers": {}, "signal_registry": {}, "backtest_edges": edges}


def _breakout_frame():
    close = np.linspace(100, 200, 260)
    close[-3] = close[-4] - 2.0
    close[-8] = close[-9] - 2.0
    close[-12] = close[-13] - 2.0
    close[-1] = 212
    vol = np.full(260, 2e6)
    vol[-1] = 5e6
    end = pd.offsets.BDay().rollback(pd.Timestamp(date.today()))
    idx = pd.date_range(end=end, periods=260, freq="B")
    return pd.DataFrame({"close": close, "high": close * 1.01, "low": close * 0.99, "volume": vol}, index=idx)


# ── verdict rule ──────────────────────────────────────────────────────────────

def test_without_any_evidence_the_legacy_rule_stands():
    assert US._evidence_verdict(40.0, ["BREAKOUT_52W"]) == ("BUY", "")
    assert US._evidence_verdict(80.0, ["MOMENTUM", "CUP_HANDLE"]) == ("BUY", "")
    assert US._evidence_verdict(40.0, ["MOMENTUM", "CUP_HANDLE"]) == ("WATCH", "")


def test_52_week_shortcut_needs_its_own_positive_evidence():
    unproven = {"MOMENTUM": 0.05}                        # 52W has < 30 trades: absent
    assert US._evidence_verdict(40.0, ["BREAKOUT_52W"], signal_edges=unproven)[0] == "WATCH"
    proven = {"BREAKOUT_52W": 0.20}
    assert US._evidence_verdict(40.0, ["BREAKOUT_52W"], signal_edges=proven)[0] == "BUY"
    losing = {"HIGH_TIGHT_FLAG": -0.30}
    assert US._evidence_verdict(40.0, ["HIGH_TIGHT_FLAG"], signal_edges=losing)[0] == "WATCH"


def test_buy_needs_a_positive_measured_combo_edge():
    edges = {"FLAT_BASE": -0.83, "NR7_COIL": -0.45, "VOL_SQUEEZE": 0.08, "MOMENTUM": -0.03}
    verdict, why = US._evidence_verdict(90.0, ["FLAT_BASE", "NR7_COIL"], signal_edges=edges)
    assert verdict == "WATCH" and "-0.64R" in why
    # Breakeven (NEUTRAL, within noise of zero) is not an edge either: the
    # two survivors of the 2026-11-12 scan measured +0.003R.
    verdict, why = US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"], signal_edges=edges)
    assert verdict == "WATCH" and "+0.03R" in why and "+0.05R bar" in why
    positive = {"VOL_SQUEEZE": 0.08, "DELIVERY_SPIKE": 0.04}
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "DELIVERY_SPIKE"], signal_edges=positive) == ("BUY", "")
    assert US._MIN_BUY_EDGE_R == 0.05


def test_unevidenced_combo_makes_no_claim_either_way():
    edges = {"FLAT_BASE": -0.83}
    assert US._evidence_verdict(90.0, ["MOMENTUM", "CUP_HANDLE"], signal_edges=edges) == ("BUY", "")


def test_quality_veto_still_outranks_everything():
    edges = {"BREAKOUT_52W": 0.5}
    assert US._evidence_verdict(99.0, ["BREAKOUT_52W", "MOMENTUM"], breakout_watch_veto=True,
                                signal_edges=edges) == ("WATCH", "")


def test_floor_is_tunable(monkeypatch):
    monkeypatch.setattr(US, "_MIN_BUY_EDGE_R", 0.10)
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"],
                                signal_edges={"VOL_SQUEEZE": 0.08, "MOMENTUM": 0.06})[0] == "WATCH"
    monkeypatch.setattr(US, "_MIN_BUY_EDGE_R", 0.0)
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "MOMENTUM"],
                                signal_edges={"VOL_SQUEEZE": 0.08, "MOMENTUM": -0.03})[0] == "BUY"


def test_backtest_edges_keep_only_evidenced_signals():
    report = {"signals": {"A": {"trades": 30, "expectancy_r": -0.2},
                          "B": {"trades": 29, "expectancy_r": 0.9},
                          "C": {"trades": "x", "expectancy_r": 1}}}
    assert US.backtest_edges(report) == {"A": {"mean": -0.2, "lower": None, "n": 30}}
    assert US.backtest_edges(None) == {}
    assert US.combo_edge_r(["A", "B"], {"A": -0.2}) == -0.2
    assert US.combo_edge_r(["B"], {"A": -0.2}) is None
    precise = {"signals": {"A": {"trades": 40, "expectancy_r": 0.1, "expectancy_r_precise": 0.0812,
                                 "lower_r": -0.01}}}
    assert US.backtest_edges(precise) == {"A": {"mean": 0.0812, "lower": -0.01, "n": 40}}


# ── scanner integration ───────────────────────────────────────────────────────

def test_scanner_turns_a_losing_buy_into_an_explained_watch():
    losing = {"BREAKOUT_52W": 0.2, "VOL_SQUEEZE": -0.5, "POCKET_PIVOT": -0.5}
    r = US.UnifiedScanner(calibration_snapshot=_snap(losing))._analyze("X", _breakout_frame())
    assert r.signals[:3] == ["BREAKOUT_52W", "VOL_SQUEEZE", "POCKET_PIVOT"]
    assert r.verdict == "WATCH" and r.edge_gated is True and r.edge_r == pytest.approx(-0.267, abs=1e-3)
    assert r.reasons[-1].startswith("⚖ Measured edge")
    assert not r.reasons[0].startswith("⚖")             # reasons[0] stays the safety slot


def test_scanner_keeps_an_evidenced_positive_buy():
    edges = {"VOL_SQUEEZE": 0.12, "POCKET_PIVOT": 0.0}
    r = US.UnifiedScanner(calibration_snapshot=_snap(edges))._analyze("X", _breakout_frame())
    assert r.verdict == "BUY" and r.edge_gated is False and r.edge_r == pytest.approx(0.06)


def test_replay_and_backtest_scanners_measure_raw_setups():
    losing = {"BREAKOUT_52W": -0.5, "VOL_SQUEEZE": -0.5, "POCKET_PIVOT": -0.5}
    r = US.UnifiedScanner(calibration_snapshot=_snap(losing), evidence_gate=False)._analyze(
        "X", _breakout_frame())
    assert r.verdict == "BUY" and r.edge_r is None and r.edge_gated is False


def test_backtest_and_historical_replay_construct_without_the_gate(monkeypatch):
    import product.historical_replay as HR
    import scan.signal_backtest as SB
    import data.bhavcopy_store as store

    seen = []

    class Spy:
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("evidence_gate"))

        def _analyze(self, *_a, **_k):
            return None

    monkeypatch.setattr(US, "UnifiedScanner", Spy)
    monkeypatch.setattr(store, "store_symbols", lambda: [])
    SB.run_backtest()
    HR.scan_session("2026-11-13", [], ohlcv_fn=lambda s: None)
    assert seen == [False, False]


def test_snapshot_freezes_the_gate_evidence(tmp_path):
    from scan.calibration_snapshot import build_snapshot

    snap = build_snapshot(
        backtest_report={"signals": {"VOL_SQUEEZE": {"trades": 545, "expectancy_r": 0.08},
                                     "HIGH_TIGHT_FLAG": {"trades": 19, "expectancy_r": 0.33}}},
        live_profile={}, data_identity="t", thesis_hash="t",
    )
    assert snap["backtest_edges"] == {"VOL_SQUEEZE": {"mean": 0.08, "lower": None, "n": 545}}
    assert snap["backtest_combo_edges"] == {}


def test_scan_record_carries_the_edge_fields():
    from product.scan_store import _record

    r = US.UnifiedScanner(calibration_snapshot=_snap({"VOL_SQUEEZE": -0.5, "POCKET_PIVOT": -0.5}))._analyze(
        "X", _breakout_frame())
    row = _record(r, {}, set())
    assert row["verdict"] == "WATCH" and row["status"] != "Ready to trade"
    assert row["edge_gated"] is True and row["edge_r"] == pytest.approx(-0.5)


# ── combinations and confidence (2026-11-15, second pass) ────────────────────

def test_a_positive_average_that_noise_could_explain_is_not_proven():
    edges = {"VOL_SQUEEZE": {"mean": 0.08, "lower": -0.03, "n": 545},
             "DELIVERY_SPIKE": {"mean": 0.09, "lower": 0.01, "n": 239}}   # lower bounds average -0.01
    verdict, why = US._evidence_verdict(90.0, ["VOL_SQUEEZE", "DELIVERY_SPIKE"], signal_edges=edges)
    assert verdict == "WATCH" and "95% lower bound" in why and "could be noise" in why
    proven = {"VOL_SQUEEZE": {"mean": 0.12, "lower": 0.03, "n": 545},
              "DELIVERY_SPIKE": {"mean": 0.09, "lower": 0.02, "n": 239}}
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "DELIVERY_SPIKE"], signal_edges=proven) == ("BUY", "")


def test_the_exact_combination_outranks_the_average_of_its_signals():
    edges = {"VOL_SQUEEZE": {"mean": 0.12, "lower": 0.03, "n": 545},
             "POCKET_PIVOT": {"mean": 0.10, "lower": 0.02, "n": 1385}}
    combos = {"POCKET_PIVOT|VOL_SQUEEZE": {"mean": -0.20, "lower": -0.35, "n": 64}}
    verdict, why = US._evidence_verdict(90.0, ["VOL_SQUEEZE", "POCKET_PIVOT"],
                                        signal_edges=edges, combo_edges=combos)
    assert verdict == "WATCH" and "this exact combination" in why
    ev = US.setup_evidence(["POCKET_PIVOT", "VOL_SQUEEZE"], edges, combos)
    assert ev["source"] == "combination" and ev["mean"] == -0.20
    # Without the combination's own record the signals' average decides.
    assert US._evidence_verdict(90.0, ["VOL_SQUEEZE", "POCKET_PIVOT"], signal_edges=edges) == ("BUY", "")


def test_the_52_week_shortcut_needs_proof_not_just_a_positive_average():
    noisy = {"BREAKOUT_52W": {"mean": 0.20, "lower": -0.05, "n": 40}}
    assert US._evidence_verdict(40.0, ["BREAKOUT_52W"], signal_edges=noisy)[0] == "WATCH"
    proven = {"BREAKOUT_52W": {"mean": 0.20, "lower": 0.05, "n": 40}}
    assert US._evidence_verdict(40.0, ["BREAKOUT_52W"], signal_edges=proven)[0] == "BUY"


def test_scanner_reports_the_evidence_it_judged_on():
    combos = {"BREAKOUT_52W|POCKET_PIVOT|VOL_SQUEEZE": {"mean": 0.30, "lower": 0.10, "n": 45}}
    edges = {"VOL_SQUEEZE": {"mean": -0.5, "lower": -0.6, "n": 545}}
    snap = {**_snap(edges), "backtest_combo_edges": combos}
    r = US.UnifiedScanner(calibration_snapshot=snap)._analyze("X", _breakout_frame())
    assert r.verdict == "BUY" and r.edge_source == "combination"
    assert r.edge_r == pytest.approx(0.30) and r.edge_lower_r == pytest.approx(0.10)

