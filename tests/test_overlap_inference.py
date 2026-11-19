"""Dependence-aware inference: correlated or duplicated evidence cannot confirm.

Null data (true mean zero) with the structures the product actually has:
10-session outcome windows that overlap, slow regimes on top, evidence copied
into a later window, and rows duplicated inside a session. A false
confirmation is a two-sided p < 0.05; at a nominal 5% the robust test must
stay close to 5% (Monte Carlo slack below), while the superseded block
t-test and the day-clustered error are shown to exceed it.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

from research import overlap_inference as OI

H = 10


def _overlap_series(rng, n, phi=0.0):
    z = rng.normal(size=n + H + 300)
    if phi:
        for i in range(1, len(z)):
            z[i] = phi * z[i - 1] + math.sqrt(1 - phi * phi) * z[i]
    z = z[300:]
    c = np.concatenate([[0.0], np.cumsum(z)])
    window = (c[H + 1:n + H + 1] - c[1:n + 1]) / H        # mean of z over s+1..s+H
    return window + 0.3 * rng.normal(size=n) / math.sqrt(H)


def _cells(values, ordinals=None):
    ordinals = range(len(values)) if ordinals is None else ordinals
    return {int(o): [float(v), 1] for o, v in zip(ordinals, values)}


def _rates(gen, sims, seed):
    rng = np.random.default_rng(seed)
    robust = naive = clustered = 0
    for _ in range(sims):
        cells = gen(rng)
        rep = OI.overlap_robust_test(cells, H)
        robust += rep["p_two_sided"] is not None and rep["p_two_sided"] < 0.05
        p_old = OI.naive_block_t_test(cells, H)
        naive += p_old is not None and p_old < 0.05
        z = abs(rep["mean"] / rep["se_day_clustered"])
        clustered += math.erfc(z / math.sqrt(2)) < 0.05
    return robust / sims, naive / sims, clustered / sims


SCENARIOS = {
    "overlapping_windows": lambda rng: _cells(_overlap_series(rng, 300)),
    "overlap_plus_slow_regime": lambda rng: _cells(_overlap_series(rng, 300, phi=0.8)),
    "overlap_plus_very_slow_regime": lambda rng: _cells(_overlap_series(rng, 300, phi=0.9)),
    "block_copied_into_next_window": lambda rng: _cells(np.concatenate(
        [np.concatenate([b, b]) for b in np.split(_overlap_series(rng, 150), 15)])),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_null_evidence_with_dependence_rarely_confirms(name):
    robust, naive, clustered = _rates(SCENARIOS[name], sims=400, seed=zlib.crc32(name.encode()))
    assert robust <= 0.08, (name, robust)                    # nominal 5%, Monte Carlo slack
    assert clustered > 0.20, (name, clustered)               # day clustering ignores the overlap


def test_the_superseded_block_test_falsely_confirms_where_the_robust_one_does_not():
    robust, naive, _ = _rates(SCENARIOS["block_copied_into_next_window"], sims=400, seed=5)
    assert naive > 0.15 and robust < naive / 2
    robust, naive, _ = _rates(SCENARIOS["overlap_plus_very_slow_regime"], sims=400, seed=6)
    assert naive > 0.10 and robust < naive


def test_duplicating_rows_inside_a_session_changes_nothing():
    rng = np.random.default_rng(1)
    base = _overlap_series(rng, 260) + 0.05
    once = OI.overlap_robust_test(_cells(base), H)
    copied = OI.overlap_robust_test({o: [5 * s, 5 * c] for o, (s, c) in _cells(base).items()}, H)
    for key in ("mean", "se", "t", "p_two_sided", "lower_95_one_sided", "df"):
        assert copied[key] == pytest.approx(once[key]), key


def test_repeating_a_session_on_the_next_session_does_not_double_the_evidence():
    rng = np.random.default_rng(2)
    base = rng.normal(0.08, 1.0, size=150)
    honest = OI.overlap_robust_test(_cells(base, ordinals=range(0, 300, 2)), H)
    doubled = OI.overlap_robust_test(_cells(np.repeat(base, 2)), H)
    assert doubled["se"] >= 0.95 * honest["se"]
    assert doubled["p_two_sided"] >= 0.9 * honest["p_two_sided"]


def test_real_effect_with_enough_windows_is_still_detected():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(100):
        x = _overlap_series(rng, 400)
        rep = OI.overlap_robust_test(_cells(x + 0.6 * x.std()), H)
        hits += rep["status"] == OI.OK and rep["p_two_sided"] < 0.05 and rep["mean"] > 0
    assert hits >= 80


def test_thin_or_degenerate_evidence_is_never_a_verdict():
    rep = OI.overlap_robust_test(_cells(np.full(400, 0.2)), H)
    assert rep["status"] == OI.INSUFFICIENT and rep["reason"] == "DEGENERATE_ZERO_DISPERSION"
    assert rep["p_two_sided"] is None
    rng = np.random.default_rng(4)
    rep = OI.overlap_robust_test(_cells(rng.normal(1.0, 0.1, size=190)), H)      # 19 blocks
    assert rep["independent_sessions"] == 19 and rep["status"] == OI.INSUFFICIENT
    assert rep["lower_95_one_sided"] is not None                                  # still a valid bound
    packed = {i // 6: [1.0, 1] for i in range(300)}                               # many rows, 50 sessions
    assert OI.overlap_robust_test(packed, H)["independent_sessions"] == 5


def test_standard_error_is_never_below_the_day_clustered_one():
    rng = np.random.default_rng(7)
    for _ in range(50):
        rep = OI.overlap_robust_test(_cells(rng.normal(size=250)), H)
        assert rep["se"] >= rep["se_day_clustered"]


def test_residual_dependence_beyond_the_overlap_is_measured_and_widens():
    rng = np.random.default_rng(8)
    plain = OI.overlap_robust_test(_cells(_overlap_series(rng, 400)), H)
    slow = OI.overlap_robust_test(_cells(_overlap_series(rng, 400, phi=0.9)), H)
    assert slow["residual_dependence"]["examined"] and slow["residual_dependence"]["lag_h_autocorr"] > 0.2
    assert slow["residual_dependence"]["variance_widening"] > plain["residual_dependence"]["variance_widening"]


def test_cells_from_clusters_merges_days_and_drops_unindexed_days():
    cells = OI.cells_from_clusters({"2026-11-02": [1.0, 2], "2026-11-03": [3.0, 1], "x": [9.0, 9]},
                                   {"2026-11-02": 5, "2026-11-03": 5})
    assert cells == {5: [4.0, 3]}


# ── adversarial nulls (2026-11-20 review; full study: research/overlap_stress.py) ──

@pytest.mark.parametrize("name", ["heavy_tails_t3", "cross_sectional_concentration",
                                  "missing_sessions_40pct_plus_month_gap", "duplicated_outcomes_lag1to3",
                                  "overlapping_event_windows", "exactly_20_blocks", "regime_persistence_ar0.9",
                                  "volatility_regime_switches", "nonstationary_variance",
                                  "symbol_dependence_rank_ic"])
def test_adversarial_nulls_stay_near_nominal(name):
    from research import overlap_stress as ST
    seed, gen = ST.SCENARIOS[name]
    out = ST.run_scenario(gen, seed, 300)
    assert out["verdict_eligible"] > 0
    assert out["false_confirmation_rate"] <= 0.08, out
    assert out["false_confirmation_rate"] <= out["superseded_block_t_rate"], out


def test_selecting_the_best_of_twenty_null_features_is_controlled_by_bh():
    from research import overlap_stress as ST
    out = ST.run_selection(201, 120)
    assert out["any_raw_p_below_0.05"] > 0.25           # picking the best raw p "finds" something
    assert out["any_bh_rejection"] <= 0.10              # BH across the family does not


def test_student_t_is_more_conservative_than_a_normal_reference():
    from research import overlap_stress as ST
    seed, gen = ST.SCENARIOS["exactly_20_blocks"]
    out = ST.run_scenario(gen, seed, 300)
    assert out["false_confirmation_rate"] <= out["same_se_normal_reference_rate"]


def test_repeated_tuning_needs_the_family_correction():
    from research import overlap_stress as ST
    out = ST.run_tuning(301, 120)
    assert out["holm"] <= out["best_of_variants_raw"] and out["holm"] <= 0.06


def test_effective_df_shrinks_with_residual_dependence():
    rng = np.random.default_rng(12)
    slow = OI.overlap_robust_test(_cells(_overlap_series(rng, 400, phi=0.9)), H)
    assert slow["df"] < slow["df_raw"] and slow["residual_dependence"]["variance_widening"] > 1.0

