"""Pre-declared exit-construction experiment in the weekly walk-forward (research only)."""

from __future__ import annotations

import pandas as pd

import scan.signal_backtest as SB


def _fwd(highs, lows, closes, opens=None):
    n = len(highs)
    return pd.DataFrame({"open": opens or [closes[0]] * n, "high": highs, "low": lows, "close": closes},
                        index=pd.bdate_range("2026-01-05", periods=n))


def test_trail_turns_a_drifting_winner_into_a_scratch_only_in_trail_variants():
    # Fills at 100, touches +2.5% (arms the trail), dips back to 99.9, then drifts to 104.
    highs = [100.5, 102.5, 101.0] + [104.0] * 17
    lows = [99.5, 100.5, 99.8] + [103.0] * 17
    closes = [100.2, 102.0, 100.4] + [103.8] * 17
    opens = [99.9, 100.3, 101.5] + [103.5] * 17           # no gap through the trailed stop
    got = {n: (o, r) for n, o, r in SB.exit_variant_results(100.0, 94.0, 112.0, _fwd(highs, lows, closes, opens), 10)}
    assert got["V0_prod_be_h"][0] == "SCRATCH"            # stopped at entry after arming
    assert got["V1_no_trail_h"][0] == "FLAT" and got["V1_no_trail_h"][1] > 0.5
    assert got["V2_be_2h"][0] == "SCRATCH"
    assert got["V3_no_trail_2h"][1] > 0.5


def test_no_fill_is_not_a_trade_in_any_variant():
    highs = [99.0] * 20
    out = SB.exit_variant_results(100.0, 94.0, 112.0, _fwd(highs, [98.0] * 20, [98.5] * 20), 10)
    assert out == []


def test_report_is_research_only_with_cluster_bounds_and_bh():
    stats = {}
    for i in range(40):
        day = f"2025-{1 + i // 20:02d}-{i % 20 + 1:02d}"
        for name, r in (("V0_prod_be_h", -0.2), ("V3_no_trail_2h", 0.4 if i % 2 else 0.2)):
            st = stats.setdefault(name, {"trades": 0, "wins": 0, "scratches": 0, "r_sum": 0.0, "clusters": {}})
            st["trades"] += 1
            st["wins"] += r > 0
            st["r_sum"] += r
            SB._add_cluster(st["clusters"], day, r)
    rep = SB.exit_variant_report(stats)
    assert rep["status"] == "RESEARCH_ONLY"
    v3 = rep["variants"]["V3_no_trail_2h"]
    assert v3["clusters"] == 40 and v3["lower_r"] is not None
    assert "bh_significant" in v3 and "p_two_sided" in v3
    assert rep["variants"]["V0_prod_be_h"]["expectancy_r_precise"] < 0


def _hist(n=300, drift=0.0, vol=1.0, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.01 * vol, n))
    return pd.DataFrame({"open": close, "high": close * (1 + 0.01 * vol), "low": close * (1 - 0.01 * vol), "close": close},
                        index=pd.bdate_range("2025-01-01", periods=n))


def test_confirmation_features_are_pit_and_oriented_higher_is_better():
    calm, wild = SB.confirmation_features(_hist(vol=0.5)), SB.confirmation_features(_hist(vol=3.0))
    assert calm["low_atr"] > wild["low_atr"]                 # lower ATR scores higher
    assert SB.confirmation_features(_hist(n=40)) is None      # too little history: no value invented


def test_confirmation_excludes_the_discovery_window_and_needs_bh():
    rows = []
    for d in range(30):
        day = f"2025-{1 + d // 28:02d}-{d % 28 + 1:02d}"
        for k in range(6):
            rows.append({"day": day, "r": float(k), "low_atr": float(k), "near_high": float(-k), "not_extended": 0.1 * k})
    for k in range(6):   # discovery-window rows must not count
        rows.append({"day": "2026-06-15", "r": float(k), "low_atr": float(-k), "near_high": 0.0, "not_extended": 0.0})
    rep = SB.confirmation_report(rows)
    assert rep["status"] == "RESEARCH_ONLY" and rep["days_used"] == 30
    assert rep["features"]["low_atr"]["verdict"] == "CONFIRMED"
    assert rep["features"]["near_high"]["verdict"] == "CONTRADICTED"
    assert rep["features"]["composite"]["mean_ic"] > 0


def test_confirmation_with_too_few_days_is_insufficient():
    rows = [{"day": "2025-01-02", "r": float(k), "low_atr": float(k), "near_high": 0.0, "not_extended": 0.0} for k in range(6)]
    assert SB.confirmation_report(rows)["features"]["low_atr"]["verdict"] == "INSUFFICIENT EVIDENCE"
