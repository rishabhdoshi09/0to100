"""
Deterministic, network-free tests for the Institutional Momentum Breakout research
framework (EXP-006). No wall-clock, no network, no machine-timezone dependence —
all series are synthetic with fixed ISO dates and fixed shapes.

Covers: point-in-time primitives, prior-upmove/leadership, base detection, breakout
quality, structural stop + gap handling, sector strength, valuation CONTEXT (flag
never auto-rejects; future/stale/missing handled), event deduplication,
reproducibility, and regression protection (the framework touches NO execution path).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pytest

from research.momentum_breakout import primary_config, BarSeries, consider, scan_symbol
from research.momentum_breakout import experiment as EXP
from research.momentum_breakout import pit as P
from research.momentum_breakout import pit_safety as PS
from research.momentum_breakout import features as F
from research.momentum_breakout import observation as OBS
from research.momentum_breakout.observation import ELIGIBLE, REJECTED
from research.momentum_breakout import detector as D


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic synthetic series builder (leader → contracting base → breakout)
# ══════════════════════════════════════════════════════════════════════════════

def build_series(symbol="TESTCO", exchange="NSE", *, leader=True,
                 prior_len=340, base_len=60, base_depth=0.08, contract=True,
                 continuation=120, p0=100.0, resistance_mult=1.6,
                 breakout=True, breakout_close_mult=1.03, breakout_gap=0.0,
                 vol_base=1e6, vol_dryup=0.4, vol_breakout=3.0,
                 bench_slope=0.0002, intrabar=0.008):
    R = p0 * resistance_mult
    N = prior_len + base_len + ((1 + continuation) if breakout else 0)
    close = np.zeros(N)
    if leader:
        close[:prior_len] = p0 * (R / p0) ** np.linspace(0, 1, prior_len)
    else:                                   # long downtrend into R (not a leader)
        hi = R * 2.0
        close[:prior_len] = hi * (R / hi) ** np.linspace(0, 1, prior_len)
    for k in range(base_len):
        i = prior_len + k
        amp = base_depth * (1.0 - (0.7 * k / base_len if contract else 0.0))
        center = R * (1 - base_depth / 2)
        close[i] = min(center + (amp / 2) * R * np.sin(k / 3.5), R * 0.995)
    if breakout:
        b = prior_len + base_len
        close[b] = R * breakout_close_mult
        for k in range(1, continuation + 1):
            close[b + k] = close[b + k - 1] * 1.004
    high = close * (1 + intrabar)
    low = close * (1 - intrabar)
    open_ = np.empty(N); open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + 0.001)
    for k in range(base_len):               # base highs stay under the pivot
        high[prior_len + k] = min(high[prior_len + k], R * 0.99)
    if breakout:
        b = prior_len + base_len
        high[b] = max(R * 1.035, close[b] * (1 + intrabar))
        open_[b] = close[b - 1] * (1 + breakout_gap)
    vol = np.full(N, vol_base)
    vol[prior_len:prior_len + base_len] = vol_base * vol_dryup
    if breakout:
        vol[prior_len + base_len] = vol_base * vol_breakout
    d0 = dt.date(2018, 1, 1)
    dates = [(d0 + dt.timedelta(days=i)).isoformat() for i in range(N)]
    bench = 100.0 * (1 + bench_slope) ** np.arange(N)
    return BarSeries(symbol, exchange, dates, open_, high, low, close, vol, bench), \
        prior_len + base_len


STRONG_SECTOR = {"sector_rs_pct": 5.0, "breadth_pct_above_50dma": 70.0,
                 "membership_pit": True, "turnover_cr": 50.0}
WEAK_SECTOR = {"sector_rs_pct": -3.0, "breadth_pct_above_50dma": 30.0,
               "membership_pit": True, "turnover_cr": 50.0}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Point-in-time primitives
# ══════════════════════════════════════════════════════════════════════════════

class TestPITPrimitives:
    def test_window_never_reads_future(self):
        a = np.arange(10.0)
        w = P.window(a, 5, 3)
        assert list(w) == [3.0, 4.0, 5.0]        # ends at i=5, no bar 6+
        with pytest.raises(P.FutureLeak):
            P.window(a, 20, 3)                    # out of range fails closed

    def test_assert_no_future_read_guard(self):
        P.assert_no_future_read(5, 5)             # reading the obs bar is fine
        with pytest.raises(P.FutureLeak):
            P.assert_no_future_read(5, 6)

    def test_rel_strength_is_pure_and_past_only(self):
        # appending FUTURE bars must not change a past relative-strength value
        close = np.linspace(100, 200, 300)
        bench = np.linspace(100, 150, 300)
        rs_short = P.rel_strength_vs_benchmark(close, bench, 250, 126)
        close2 = np.concatenate([close, close[-1] * np.ones(50)])
        bench2 = np.concatenate([bench, bench[-1] * np.ones(50)])
        rs_long = P.rel_strength_vs_benchmark(close2, bench2, 250, 126)
        assert rs_short == rs_long
        # matches manual computation from only bars i-lb and i
        man = (close[250] / close[124] - 1) - (bench[250] / bench[124] - 1)
        assert rs_short == pytest.approx(man * 100)

    def test_atr_sma_warmup_is_nan(self):
        c = np.arange(1, 20.0)
        assert np.isnan(P.sma(c, 3, 10))          # not enough bars
        assert np.isnan(P.atr(c, c, c, 2, 14))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Prior upmove & leadership
# ══════════════════════════════════════════════════════════════════════════════

class TestPriorUpmove:
    def test_genuine_leader_qualifies(self):
        s, sig = build_series(leader=True)
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert obs is not None and obs.eligibility == ELIGIBLE
        assert obs.prior_upmove["rel_to_bench_pct"] > 0
        assert obs.prior_upmove["above_200dma"] is True
        assert obs.component_scores["leadership"] > 40

    def test_downtrend_stock_not_a_leader_even_with_a_breakout(self):
        # one breakout bar out of a long downtrend must NOT qualify as a leader
        s, sig = build_series(leader=False)
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        # either no valid base at all, or rejected for weak prior leadership/trend
        if obs is not None:
            assert obs.eligibility == REJECTED
            assert (D.R_WEAK_PRIOR_RS in obs.rejection_reasons
                    or D.R_NOT_ABOVE_TREND in obs.rejection_reasons)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Base detection
# ══════════════════════════════════════════════════════════════════════════════

class TestBaseDetection:
    def test_long_contracting_base_detected(self):
        s, sig = build_series(base_len=60, contract=True)
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert obs is not None and obs.eligibility == ELIGIBLE
        assert obs.base_duration >= primary_config().base_min_len

    def test_deep_unstable_base_rejected(self):
        # a base deeper than max_base_depth_pct is not a constructive base → no candidate
        s, sig = build_series(base_depth=0.60)
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert obs is None            # _detect_base refuses an over-deep base

    def test_future_bars_do_not_alter_an_earlier_base(self):
        s_short, sig = build_series(continuation=5)
        s_long, sig2 = build_series(continuation=200)
        assert sig == sig2
        o1 = consider(s_short, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        o2 = consider(s_long, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert o1 is not None and o2 is not None
        assert o1.base_start_date == o2.base_start_date
        assert o1.pivot == o2.pivot
        assert o1.event_id() == o2.event_id()       # identical past → identical event

    def test_base_and_pivot_ids_reproducible(self):
        s1, sig = build_series()
        s2, _ = build_series()
        o1 = consider(s1, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        o2 = consider(s2, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert o1.event_id() == o2.event_id()
        assert o1.as_dict() == o2.as_dict()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Breakout quality
# ══════════════════════════════════════════════════════════════════════════════

class TestBreakout:
    def test_confirmed_close_qualifies(self):
        s, sig = build_series(breakout_close_mult=1.03)
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert obs is not None and obs.breakout_quality["confirmed_close"] is True

    def test_intraday_high_above_pivot_without_confirmed_close_does_not_qualify(self):
        # breakout bar pierces the pivot intraday (high) but closes back below it
        s, sig = build_series(breakout_close_mult=0.985)   # close clearly below pivot
        R = 160.0
        s.high[sig] = R * 1.00                              # high pokes above the ~0.99R pivot
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert obs is None            # no confirmed close → not a candidate

    def test_next_bar_entry_prevents_same_period_leakage(self):
        s, sig = build_series()
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        t = EXP.simulate_trade(s, sig, obs.structural_stop, primary_config())
        assert t is not None
        assert t.entry_date == s.dates[sig + 1]        # entry is the NEXT bar
        # entry price is anchored to the next bar's open, not the signal close
        assert t.entry_price != round(float(s.close[sig]), 4)

    def test_overextended_breakout_is_recorded_and_rejected(self):
        # a breakout that closes many ATR above the pivot is a chase
        s, sig = build_series(breakout_close_mult=1.60)   # far above pivot
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert obs is not None
        assert obs.breakout_quality["overextended"] is True
        assert D.R_OVEREXTENDED in obs.rejection_reasons
        assert obs.eligibility == REJECTED


# ══════════════════════════════════════════════════════════════════════════════
# 5. Structural stop & initial risk
# ══════════════════════════════════════════════════════════════════════════════

class TestStructuralStop:
    def test_stop_uses_only_signal_time_info(self):
        s_short, sig = build_series(continuation=5)
        s_long, _ = build_series(continuation=200)
        o1 = consider(s_short, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        o2 = consider(s_long, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert o1.structural_stop == o2.structural_stop
        assert o1.initial_risk_pct == o2.initial_risk_pct

    def test_risk_pct_is_deterministic(self):
        s, sig = build_series()
        r1 = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR).initial_risk_pct
        r2 = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR).initial_risk_pct
        assert r1 == r2 and r1 > 0

    def test_excessive_structural_risk_is_rejected(self):
        s, sig = build_series()
        tight = replace(primary_config(), max_initial_risk_pct=0.1)  # anything real exceeds
        obs = consider(s, sig, tight, sector_ctx=STRONG_SECTOR)
        assert obs is not None and D.R_RISK_TOO_HIGH in obs.rejection_reasons

    def test_gap_through_stop_not_filled_at_stop_price(self):
        s, sig = build_series(continuation=40)
        cfg = primary_config()
        obs = consider(s, sig, cfg, sector_ctx=STRONG_SECTOR)
        stop = obs.structural_stop
        # force a hard gap-down BELOW the stop two bars after entry
        gi = sig + 3
        s.open[gi] = stop * 0.90
        s.high[gi] = stop * 0.92
        s.low[gi] = stop * 0.88
        s.close[gi] = stop * 0.89
        t = EXP.simulate_trade(s, sig, stop, cfg)
        assert t is not None and t.exit_reason == "gap_stop"
        # filled at the (worse) gap open, strictly below the stop — never at the stop
        assert t.exit_price < stop


# ══════════════════════════════════════════════════════════════════════════════
# 6. Sector strength
# ══════════════════════════════════════════════════════════════════════════════

class TestSectorStrength:
    def test_strong_sector_candidate_qualifies(self):
        s, sig = build_series()
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert obs.eligibility == ELIGIBLE

    def test_same_setup_in_weak_sector_is_rejected(self):
        s, sig = build_series()
        obs = consider(s, sig, primary_config(), sector_ctx=WEAK_SECTOR)
        assert D.R_WEAK_SECTOR in obs.rejection_reasons
        assert obs.eligibility == REJECTED
        # the identical price setup DID qualify with a strong sector — sector is the swing
        strong = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert strong.eligibility == ELIGIBLE

    def test_missing_sector_membership_is_surfaced_as_a_limitation(self):
        s, sig = build_series()
        no_pit = {"sector_rs_pct": 5.0, "breadth_pct_above_50dma": 70.0,
                  "membership_pit": False, "turnover_cr": 50.0}
        obs = consider(s, sig, primary_config(), sector_ctx=no_pit)
        assert OBS.FLAG_SECTOR_MEMBERSHIP_NOT_PIT in obs.data_quality_flags


# ══════════════════════════════════════════════════════════════════════════════
# 7. Valuation CONTEXT (never a primary reject; PIT-safe)
# ══════════════════════════════════════════════════════════════════════════════

class TestValuationContext:
    def test_extreme_valuation_flags_but_does_not_reject(self):
        s, sig = build_series()
        val = {"available_ts": "2000-01-01", "pe": 150.0, "price_to_sales": 40.0}
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR,
                       valuation_record=val)
        assert OBS.FLAG_EXTREME_PE in obs.data_quality_flags
        assert OBS.FLAG_HIGH_EXPECTATION_RISK in obs.data_quality_flags
        assert obs.eligibility == ELIGIBLE            # momentum candidate NOT rejected

    def test_stale_valuation_flagged(self):
        s, sig = build_series()
        val = {"available_ts": "2000-01-01", "pe": 20.0, "age_days": 500}
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR,
                       valuation_record=val)
        assert OBS.FLAG_VALUATION_STALE in obs.data_quality_flags

    def test_future_fundamentals_are_rejected_not_used(self):
        s, sig = build_series()
        future_ts = (dt.date(2018, 1, 1) + dt.timedelta(days=sig + 100)).isoformat()
        val = {"available_ts": future_ts, "pe": 150.0}   # published AFTER the signal
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR,
                       valuation_record=val)
        assert OBS.FLAG_VALUATION_UNAVAILABLE in obs.data_quality_flags
        assert obs.valuation.get("available") is False
        assert obs.valuation.get("pe") is None          # NOT forward-filled

    def test_missing_fundamentals_do_not_become_zero(self):
        s, sig = build_series()
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR,
                       valuation_record=None)
        assert OBS.FLAG_VALUATION_UNAVAILABLE in obs.data_quality_flags
        assert "pe" not in obs.valuation or obs.valuation.get("pe") is None


# ══════════════════════════════════════════════════════════════════════════════
# 8. Event deduplication
# ══════════════════════════════════════════════════════════════════════════════

class TestDeduplication:
    def test_one_breakout_event_one_observation(self):
        s, sig = build_series(continuation=120)
        events = scan_symbol(s, primary_config(), sector_ctx_fn=lambda ss, i: STRONG_SECTOR)
        assert len(events) == 1
        assert events[0].candidate_date == s.dates[sig]

    def test_consecutive_closes_above_pivot_do_not_duplicate(self):
        # the whole continuation stays above the pivot; still exactly one event
        s, sig = build_series(continuation=120)
        events = scan_symbol(s, primary_config(), sector_ctx_fn=lambda ss, i: STRONG_SECTOR)
        cds = [e.candidate_date for e in events]
        assert len(cds) == len(set(cds)) == 1

    def test_equivalent_detectors_do_not_double_count_via_registry(self):
        reg = PS.EventRegistry()
        s, sig = build_series()
        obs = consider(s, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        assert reg.register(obs.event_id()) is True     # first detector counts it
        assert reg.register(obs.event_id()) is False     # second (equivalent) does not
        assert len(reg) == 1

    def test_a_genuine_new_base_creates_a_new_event(self):
        # two separate breakouts (a fresh base after a pullback) → two distinct events
        s1, sig = build_series()
        o1 = consider(s1, sig, primary_config(), sector_ctx=STRONG_SECTOR)
        # a different base window / pivot must yield a different event id
        s2, sig2 = build_series(resistance_mult=1.9)
        o2 = consider(s2, sig2, primary_config(), sector_ctx=STRONG_SECTOR)
        assert o1.event_id() != o2.event_id()


# ══════════════════════════════════════════════════════════════════════════════
# 9. Reproducibility & provenance
# ══════════════════════════════════════════════════════════════════════════════

class TestReproducibility:
    def test_same_inputs_produce_identical_observations(self):
        s1, sig = build_series()
        s2, _ = build_series()
        prov = {"experiment_id": EXP.EXPERIMENT_ID, "dataset_snapshot_id": "snap1",
                "code_commit": "abc123", "config_hash": primary_config().config_hash()}
        o1 = consider(s1, sig, primary_config(), sector_ctx=STRONG_SECTOR, provenance=prov)
        o2 = consider(s2, sig, primary_config(), sector_ctx=STRONG_SECTOR, provenance=prov)
        assert o1.to_json() == o2.to_json()

    def test_material_config_change_alters_the_hash(self):
        base = primary_config()
        changed = replace(base, max_initial_risk_pct=base.max_initial_risk_pct + 1.0)
        assert base.config_hash() != changed.config_hash()

    def test_config_hash_is_stable_across_runs(self):
        assert primary_config().config_hash() == primary_config().config_hash()

    def test_experiment_spec_is_pre_registered_exp006(self):
        sp = EXP.spec(primary_config())
        assert sp["experiment_id"] == "EXP-006"
        assert sp["primary_exit"] == EXP.PRIMARY_EXIT
        assert sp["no_post_result_optimisation"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 10. Experiment plumbing → existing evidence gate
# ══════════════════════════════════════════════════════════════════════════════

class TestExperimentEvaluation:
    def _trades(self, n, mean_R):
        out = []
        for k in range(n):
            r = mean_R + (0.1 if k % 2 == 0 else -0.1)
            out.append(EXP.SimTrade(
                symbol="X", entry_date="2019-01-01", exit_date="2019-02-01",
                entry_price=100.0, exit_price=100.0 + r, stop_price=99.0,
                holding_period=20, gross_R=r, net_R=r, exit_reason="trend_break",
                mae_R=-0.3, mfe_R=1.0, benchmark_return=0.0))
        return out

    def test_evaluate_returns_a_harness_verdict(self):
        res = EXP.evaluate_trades(self._trades(40, 0.4), n_trials=len(EXP.EXIT_VARIANTS),
                                  require_alpha=False)
        assert res["verdict"] in ("PROMOTE", "REJECT", "UNDERPOWERED", "INCONCLUSIVE")
        assert res["n_trades"] == 40
        assert "profit_factor" in res and "expectancy_R" in res

    def test_small_sample_is_underpowered_not_a_claim(self):
        res = EXP.evaluate_trades(self._trades(5, 0.4), require_alpha=False)
        assert res["verdict"] == "UNDERPOWERED"

    def test_ablation_configs_are_distinct_pre_registered(self):
        abl = EXP.ablation_configs(primary_config())
        assert set(abl) >= {"prior_only", "prior_plus_base", "full_framework"}
        # relaxed ablations differ from the full config (different hashes)
        assert abl["prior_only"].config_hash() != abl["full_framework"].config_hash()


# ══════════════════════════════════════════════════════════════════════════════
# 11. Regression protection — the framework touches NO execution path
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionIsolation:
    def test_framework_imports_no_execution_or_telegram(self):
        # check ACTUAL imports/calls (not descriptive prose in docstrings)
        import inspect
        import research.momentum_breakout as pkg
        from research.momentum_breakout import (pit, config, observation, features,
                                                pit_safety, detector, experiment)
        forbidden = ("import execution", "from execution", "import alerts",
                     "from alerts", ".place_trade(", "import data.kite_client",
                     "us_autopilot", "fo_executor")
        for mod in (pkg, pit, config, observation, features, pit_safety, detector,
                    experiment):
            # strip the module docstring so prose describing what we DON'T do can't trip it
            src = inspect.getsource(mod)
            code = src.split('"""', 2)[-1] if src.count('"""') >= 2 else src
            for pat in forbidden:
                assert pat not in code, f"{mod.__name__} references {pat}"

    def test_paper_autopilot_and_live_lock_unchanged(self, monkeypatch):
        # the money-safety invariants from the prior milestone still hold
        import execution.autopilot as ap
        monkeypatch.delenv("QT_LIVE_ENABLED", raising=False)
        assert ap._live_enabled() is False              # LIVE migration-locked by default

    def test_telegram_order_path_still_paper_only(self, monkeypatch):
        import alerts.telegram_actions as ta
        import inspect
        src = inspect.getsource(ta)
        assert src.count("place_trade(") == 1 and "paper=True" in src
