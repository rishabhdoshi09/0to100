"""
Research Operating System — harness tests.

The harness exists to STOP the system believing noise. So these tests are
adversarial: a fair coin must never promote, a search-selected winner must be
deflated away, pure-noise batches must survive FDR and the Reality Check, and
the statistical primitives must match known textbook values.
"""
import math

import numpy as np
import pytest

from research import harness as H


class TestExpectancyStats:
    def test_known_values(self):
        r = [1.0, -1.0, 2.0, -1.0, 1.0, 0.0, 1.0, -1.0]
        s = H.expectancy_stats(r)
        assert s["n"] == 8
        assert s["mean_r"] == pytest.approx(np.mean(r))
        assert s["std_r"] == pytest.approx(np.std(r, ddof=1))
        # kurtosis is NON-excess (normal == 3), as PSR/DSR expect
        assert s["kurtosis"] > 0

    def test_clear_edge_is_significant(self):
        rng = np.random.default_rng(1)
        s = H.expectancy_stats(rng.normal(0.4, 1.0, 300))
        assert s["mean_r"] > 0 and s["p_value"] < 0.01     # one-sided, tiny p

    def test_empty_and_singleton(self):
        assert H.expectancy_stats([])["n"] == 0
        assert H.expectancy_stats([1.0])["std_r"] == 0.0    # no dispersion


class TestProbabilisticSharpe:
    def test_zero_sharpe_is_half(self):
        assert H.probabilistic_sharpe_ratio(0.0, 100) == pytest.approx(0.5, abs=1e-9)

    def test_monotone_in_n_and_sharpe(self):
        a = H.probabilistic_sharpe_ratio(0.3, 50)
        b = H.probabilistic_sharpe_ratio(0.3, 500)          # more data → more sure
        c = H.probabilistic_sharpe_ratio(0.6, 500)          # bigger edge → more sure
        assert a < b < c
        assert 0.0 <= a <= 1.0 and 0.0 <= c <= 1.0

    def test_fat_tails_lower_confidence(self):
        # negative skew + high kurtosis inflates the SR standard error → lower PSR
        thin = H.probabilistic_sharpe_ratio(0.3, 200, skew=0.0, kurtosis=3.0)
        fat = H.probabilistic_sharpe_ratio(0.3, 200, skew=-1.0, kurtosis=8.0)
        assert fat < thin


class TestDeflatedSharpe:
    def test_expected_max_grows_with_trials(self):
        a = H.expected_max_sharpe_null(10, 0.02)
        b = H.expected_max_sharpe_null(1000, 0.02)
        assert 0 < a < b
        assert H.expected_max_sharpe_null(1, 0.02) == 0.0    # a single trial → no bar

    def test_selection_is_deflated_away(self):
        # a Sharpe that easily passes PSR as a single trial must FAIL once we
        # admit it was the best of many trials — the whole point of the DSR.
        sr, n = 0.15, 250
        psr = H.probabilistic_sharpe_ratio(sr, n)
        d = H.deflated_sharpe_ratio(sr, n, n_trials=200, sharpe_variance=0.02)
        assert psr > 0.95                                    # looks great alone
        assert d["dsr"] < 0.10                               # crushed by deflation
        assert d["sr0_expected_max_null"] > sr               # bar exceeds the estimate

    def test_estimates_drive_variance_and_trials(self):
        rng = np.random.default_rng(3)
        ests = rng.normal(0.0, 0.14, 500)
        d = H.deflated_sharpe_ratio(0.5, 250, n_trials=1, sharpe_estimates=ests)
        assert d["n_trials"] == 500                          # measured from the array
        assert d["sharpe_variance"] == pytest.approx(np.var(ests, ddof=1), rel=1e-6)


class TestBenjaminiHochberg:
    def test_finds_real_signals_controls_false(self):
        rng = np.random.default_rng(4)
        signal = np.full(5, 1e-6)                            # 5 genuine discoveries
        noise = rng.uniform(0, 1, 95)                        # 95 nulls
        out = H.benjamini_hochberg(np.concatenate([signal, noise]), alpha=0.05)
        assert out["rejected"][:5].all()                     # all real ones caught
        assert out["n_rejected"] < 15                        # false discoveries controlled

    def test_all_null_rejects_few(self):
        rng = np.random.default_rng(5)
        out = H.benjamini_hochberg(rng.uniform(0, 1, 200), alpha=0.05)
        assert out["n_rejected"] <= 5                        # ~FDR-controlled under null

    def test_qvalues_monotone_and_bounded(self):
        out = H.benjamini_hochberg([0.001, 0.01, 0.04, 0.2, 0.5, 0.9])
        q = out["qvalues"]
        assert np.all((q >= 0) & (q <= 1))
        order = np.argsort([0.001, 0.01, 0.04, 0.2, 0.5, 0.9])
        assert np.all(np.diff(q[order]) >= -1e-12)           # non-decreasing in p

    def test_empty(self):
        out = H.benjamini_hochberg([])
        assert out["n_rejected"] == 0 and out["rejected"].size == 0


class TestRealityCheck:
    def test_pure_noise_is_not_significant(self):
        rng = np.random.default_rng(6)
        noise = rng.normal(0.0, 1.0, (200, 10))              # 10 worthless strategies
        out = H.whites_reality_check(noise, n_boot=1000, seed=6)
        assert out["reality_check_p"] > 0.10                 # best is just the luckiest

    def test_real_winner_among_noise_is_caught(self):
        rng = np.random.default_rng(7)
        perf = rng.normal(0.0, 1.0, (250, 10))
        perf[:, 3] = rng.normal(0.35, 1.0, 250)              # strategy 3 is genuinely good
        out = H.whites_reality_check(perf, n_boot=1000, seed=7)
        assert out["best_strategy"] == 3
        assert out["reality_check_p"] < 0.05                 # survives data-snooping


class TestPowerAnalysis:
    def test_known_value(self):
        # d = 0.2, one-sided alpha 5%, power 80% → ~155 (textbook)
        n = H.min_samples_for_edge(edge_r=0.2, std_r=1.0, alpha=0.05, power=0.80)
        assert 150 <= n <= 160

    def test_smaller_edge_needs_more(self):
        assert H.min_samples_for_edge(0.1, 1.0) > H.min_samples_for_edge(0.3, 1.0)

    def test_non_positive_edge_is_unreachable(self):
        assert H.min_samples_for_edge(0.0, 1.0) > 10 ** 8    # can't power a zero effect


class TestPurgedKFold:
    def test_train_test_disjoint_and_partition(self):
        splits = H.purged_kfold_indices(n=100, k=5, embargo=2, label_horizon=3)
        assert len(splits) == 5
        all_test = np.concatenate([te for _, te in splits])
        assert sorted(all_test.tolist()) == list(range(100))  # test folds partition
        for train, test in splits:
            assert set(train).isdisjoint(set(test))           # never overlap

    def test_purge_and_embargo_remove_the_right_bars(self):
        splits = H.purged_kfold_indices(n=100, k=5, embargo=2, label_horizon=3)
        # take a middle fold (indices 20..39)
        train, test = splits[1]
        ts, teend = int(test[0]), int(test[-1])
        trainset = set(train.tolist())
        # label_horizon 3 → the 2 bars immediately BEFORE the fold are purged
        assert (ts - 1) not in trainset and (ts - 2) not in trainset
        # embargo 2 → the 2 bars immediately AFTER the fold are gone
        assert (teend + 1) not in trainset and (teend + 2) not in trainset
        # a bar well before the purge window is still available
        assert (ts - 5) in trainset

    def test_degenerate_k(self):
        assert H.purged_kfold_indices(0, 5) == []
        assert len(H.purged_kfold_indices(10, 1)) == 1        # k=1 → no split


class TestEvaluateGate:
    """The full kill-criteria matrix — the gate every learning claim passes."""

    def test_fair_coin_never_promotes(self):
        rng = np.random.default_rng(8)
        for _ in range(5):
            v = H.evaluate(rng.choice([1.0, -1.0], 400)).verdict
            assert v != "PROMOTE"

    def test_planted_edge_promotes(self):
        rng = np.random.default_rng(9)
        r = H.evaluate(rng.normal(0.30, 1.0, 250))
        assert r.verdict == "PROMOTE"
        assert "edge" in r.insight.lower()                    # user-facing insight

    def test_tiny_sample_is_floored(self):
        rng = np.random.default_rng(10)
        # a gorgeous-looking edge on 8 trades must NOT promote (invariant #6)
        assert H.evaluate(rng.normal(1.0, 0.5, 8)).verdict == "UNDERPOWERED"

    def test_loser_is_rejected(self):
        rng = np.random.default_rng(11)
        assert H.evaluate(rng.normal(-0.25, 1.0, 200)).verdict == "REJECT"

    def test_hard_floor_is_the_default_not_the_power_need(self):
        # regression: the production floor (min_n) must stay 30 and must NOT be
        # collapsed into the power-based sample need. A clean loser over 200
        # trades REJECTs even when the power-need for a tiny edge exceeds 200.
        rng = np.random.default_rng(14)
        r = H.evaluate(rng.normal(-0.25, 1.0, 200))
        assert r.verdict == "REJECT"
        assert r.min_n_needed != 30                           # power need, not floor

    def test_research_floor_is_opt_in_only(self):
        # a research caller may explore below the 30 floor by passing min_n; the
        # production default (no arg) still enforces the mechanical floor.
        rng = np.random.default_rng(15)
        edge = rng.normal(0.8, 1.0, 20)                       # strong, but n<30
        assert H.evaluate(edge).verdict == "UNDERPOWERED"     # floored by default
        assert "floor" in H.evaluate(edge).insight.lower()
        # opt in to a lower research floor → the sample is now judged on merit
        assert H.evaluate(edge, min_n=15).verdict != "UNDERPOWERED"

    def test_search_winner_is_deflated_not_promoted(self):
        rng = np.random.default_rng(12)
        # a marginal positive edge, selected as the best of 400 trials → must not
        # promote once deflation is applied
        r = H.evaluate(rng.normal(0.10, 1.0, 200), n_trials=400,
                       sharpe_estimates=rng.normal(0.0, 0.12, 400))
        assert r.verdict != "PROMOTE"

    def test_every_verdict_has_a_plain_english_insight(self):
        rng = np.random.default_rng(13)
        for arr in (rng.normal(0.3, 1, 200), rng.normal(-0.3, 1, 200),
                    rng.normal(1, 0.5, 8)):
            assert len(H.evaluate(arr).insight) > 10          # kill-criterion: a sentence


from research import drift as D


class TestConceptDrift:
    """Catch edge decay before the drawdown — a change-point located by max-|t|
    split and CONFIRMED by permutation p-value, so noise never cries wolf."""

    def test_stationary_stream_is_stable(self):
        rng = np.random.default_rng(20)
        assert D.assess_drift(rng.normal(0.1, 1.0, 200), seed=1).status == "STABLE"

    def test_clear_decay_is_caught_and_located(self):
        rng = np.random.default_rng(21)
        # +0.5R for 100 trades, then -0.5R for 100 — an unmistakable decay
        stream = np.concatenate([rng.normal(0.5, 1.0, 100),
                                 rng.normal(-0.5, 1.0, 100)])
        r = D.assess_drift(stream, seed=1)
        assert r.status == "DECAYING"
        assert r.delta_r < 0
        assert 80 <= r.change_point <= 120                    # near the true break
        assert "deterioration" in r.insight
        assert r.confidence == "HIGH"                          # big, clear shift

    def test_clear_strengthening_is_caught(self):
        rng = np.random.default_rng(22)
        stream = np.concatenate([rng.normal(-0.4, 1.0, 100),
                                 rng.normal(0.5, 1.0, 100)])
        r = D.assess_drift(stream, seed=1)
        assert r.status == "STRENGTHENING" and r.delta_r > 0

    def test_recovery_after_a_decay_is_labelled(self):
        rng = np.random.default_rng(28)
        # good → bad → good again: a decay that has since rebounded. The trader
        # question is "abandon it?" — answer: no, it RECOVERED.
        stream = np.concatenate([rng.normal(0.4, 1.0, 50),
                                 rng.normal(-0.5, 1.0, 50),
                                 rng.normal(0.45, 1.0, 50)])
        r = D.assess_drift(stream, seed=1)
        assert r.status == "RECOVERING"
        assert "RECOVERED" in r.insight

    def test_risk_profile_shift_under_a_flat_mean(self):
        rng = np.random.default_rng(29)
        # mean holds ~0.1R but outcome volatility ~triples — same size, more risk
        stream = np.concatenate([rng.normal(0.1, 0.7, 70),
                                 rng.normal(0.1, 2.2, 70)])
        r = D.assess_drift(stream, seed=1)
        assert r.status == "STABLE"                            # the AVERAGE held
        assert r.risk_profile_changed is True                  # but the risk moved
        assert r.variance_ratio >= 2.0
        assert "risk profile" in r.insight.lower()

    def test_stationary_stream_no_false_risk_flag(self):
        # a plain N(0.1,1) stream must not be flagged as a risk-profile change
        # just because a permutation split happened to look big (alpha control)
        flags = sum(D.assess_drift(np.random.default_rng(s).normal(0.1, 1.0, 140),
                                   seed=1).risk_profile_changed for s in range(12))
        assert flags <= 1                                      # ~1% alpha, not 5%

    def test_confidence_tier_scales_with_evidence(self):
        assert D._confidence_tier(0.001, 50) == "HIGH"
        assert D._confidence_tier(0.02, 20) == "MEDIUM"
        assert D._confidence_tier(0.04, 8) == "LOW"            # weak p, few trades
        assert D._confidence_tier(0.001, 5) == "LOW"           # strong p, too few

    def test_too_few_outcomes_is_stable_not_a_claim(self):
        rng = np.random.default_rng(23)
        r = D.assess_drift(rng.normal(-0.5, 1.0, 12), seed=1)
        assert r.status == "STABLE" and "need" in r.insight.lower()

    def test_false_alarm_rate_is_controlled(self):
        rng = np.random.default_rng(24)
        alarms = sum(D.assess_drift(rng.normal(0.05, 1.0, 200), seed=1).status
                     != "STABLE" for _ in range(25))
        assert alarms <= 3                                    # ~alpha under the null

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(25)
        s = np.concatenate([rng.normal(0.4, 1, 90), rng.normal(-0.3, 1, 90)])
        assert D.assess_drift(s, seed=7).insight == D.assess_drift(s, seed=7).insight

    def test_page_hinkley_streaming_detector(self):
        rng = np.random.default_rng(26)
        down = np.concatenate([rng.normal(0.6, 0.5, 80), rng.normal(-0.6, 0.5, 80)])
        assert D.page_hinkley(down)["direction"] == "down"
        assert D.page_hinkley(rng.normal(0.0, 0.5, 60))["detected"] is False

    def test_max_split_locates_the_break(self):
        rng = np.random.default_rng(27)
        s = np.concatenate([rng.normal(0.6, 1, 100), rng.normal(-0.6, 1, 100)])
        cp, t = D._max_split_t(s, min_seg=15)
        assert 85 <= cp <= 115 and t < 0                      # located, sign = down

    def test_report_and_directives_fail_open(self):
        # no signal_log in the test env → must return empty lists, never raise
        assert isinstance(D.drift_report(), list)
        assert isinstance(D.drift_directives(), list)


from research import calibration as C


class TestCalibration:
    """Forecast reliability + the Confidence Ledger — do our probabilities mean
    anything, and WHEN are they trustworthy?"""

    def test_perfectly_calibrated_scores_well(self):
        rng = np.random.default_rng(30)
        p = rng.uniform(0.3, 0.9, 3000)
        y = (rng.uniform(size=3000) < p).astype(float)      # outcomes match the odds
        s = C.calibration_summary(p, y)
        assert s["ece"] < 0.05                                # low calibration error
        assert s["brier_skill"] > 0                           # beats a base-rate guess
        assert "calibrated" in s["insight"]

    def test_overconfidence_is_detected(self):
        rng = np.random.default_rng(31)
        p = np.full(500, 0.8)                                 # promises 80%
        y = (rng.uniform(size=500) < 0.5).astype(float)       # delivers 50%
        mt = C.miscalibration_test(p, y)
        assert mt["direction"] == "overconfident"
        assert mt["z"] < 0 and mt["p_value"] < 0.01
        assert C.calibration_summary(p, y)["ece"] > 0.2

    def test_brier_perfect_and_bins(self):
        assert C.brier_score([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])["brier"] == 0.0
        bins = C.reliability_bins([0.15, 0.16, 0.85, 0.88], [0, 0, 1, 1], n_bins=10)
        assert len(bins) == 2 and all(b["count"] == 2 for b in bins)   # 0.1-0.2 & 0.8-0.9

    def test_conditional_gate_flags_only_the_bad_slice(self):
        rng = np.random.default_rng(32)
        # group A calibrated, group B overconfident → only B survives FDR
        pa = rng.uniform(0.4, 0.7, 300); ya = (rng.uniform(size=300) < pa).astype(float)
        pb = np.full(300, 0.75); yb = (rng.uniform(size=300) < 0.45).astype(float)
        probs = np.concatenate([pa, pb]); succ = np.concatenate([ya, yb])
        groups = ["A"] * 300 + ["B"] * 300
        found = C.conditional_overconfidence(probs, succ, groups)
        assert [f["group"] for f in found] == ["B"]
        assert found[0]["direction"] == "overconfident"

    def test_conditional_ignores_thin_slices(self):
        rng = np.random.default_rng(33)
        p = rng.uniform(0.4, 0.7, 100); y = (rng.uniform(size=100) < 0.2).astype(float)
        groups = ["tiny"] * 100                               # < min_slice? use big min_n
        assert C.conditional_overconfidence(p, y, groups, min_n=500) == []

    def test_fail_open_io(self):
        # no decisions.db in the test env → sane empties, never raises
        assert C.calibration_report()["n"] == 0
        assert C.confidence_ledger_findings() == []
        assert C.calibration_directives() == []


from research import counterfactual as CF


class TestCounterfactualGates:
    """Which filters EARN vs COST money — the rejected-trade ledger nobody else
    has. FDR-gated so a 'costly gate' is real, not one of fourteen coin-flips."""

    def test_costing_and_earning_gates_are_classified(self):
        rng = np.random.default_rng(40)
        rejected = {
            "good_gate": rng.normal(-0.45, 1.0, 150),    # rejected losers → EARNING
            "bad_gate":  rng.normal(0.40, 1.0, 150),     # rejected winners → COSTING
            "noise_gate": rng.normal(0.0, 1.0, 150),     # breakeven → not significant
        }
        taken = rng.normal(0.15, 1.0, 300)
        found = {f["gate"]: f["verdict"] for f in CF.gate_attribution(rejected, taken)}
        assert found.get("bad_gate") == "COSTING"
        assert found.get("good_gate") == "EARNING"
        assert "noise_gate" not in found                 # correctly not flagged

    def test_thin_gate_is_ignored(self):
        rng = np.random.default_rng(41)
        rejected = {"thin": rng.normal(0.6, 1.0, 12)}    # < min_n
        assert CF.gate_attribution(rejected) == []

    def test_many_noise_gates_survive_fdr(self):
        rng = np.random.default_rng(42)
        rejected = {f"g{i}": rng.normal(0.0, 1.0, 80) for i in range(15)}
        found = CF.gate_attribution(rejected)
        assert len(found) <= 2                            # FDR controls false gates

    def test_decision_r_math_and_guards(self):
        # +6% outcome on a 3% risk (entry 100, stop 97) → +2R
        assert CF._decision_r(100.0, 97.0, 6.0) == pytest.approx(2.0)
        assert CF._decision_r(100.0, 100.0, 5.0) is None  # zero risk → invalid
        assert CF._decision_r(100.0, 105.0, 5.0) is None  # stop above entry → invalid

    def test_fail_open_io(self):
        assert CF.gate_attribution_report() == []
        assert CF.gate_directives() == []


from research import market_memory as M


class TestMarketMemory:
    """Trade-level analogs — win rate + MAE/MFE from similar historical setups.
    A decision engine (informs stop/target), not just a similarity toy."""

    def test_forward_outcome_win_with_excursions(self):
        # entry 100, stop 94 (risk 6), target 112; dips to 97 then runs to 113
        oc = M.forward_outcome(100, 94, 112, [101, 99, 105, 113],
                               [98, 97, 103, 109], [100, 98, 104, 112])
        assert oc["won"] is True and oc["r"] == pytest.approx(2.0)
        assert oc["mae"] == pytest.approx((100 - 97) / 6)     # worst dip, in R
        assert oc["mfe"] == pytest.approx((113 - 100) / 6)    # best run, in R

    def test_forward_outcome_loss_and_invalid(self):
        loss = M.forward_outcome(100, 94, 112, [101, 99], [98, 93], [100, 95])
        assert loss["won"] is False and loss["r"] == -1.0
        assert M.forward_outcome(100, 100, 112, [1], [1], [1]) is None  # zero risk

    def test_retrieval_pulls_the_near_cluster(self):
        rng = np.random.default_rng(50)
        near = rng.normal([60, 5, 3, 2, 10, 4, 1], 0.3, (60, 7))
        far = rng.normal([85, 20, 8, 5, 40, 25, 0], 0.3, (60, 7))
        corpus = np.vstack([near, far])
        outcomes = ([{"r": 0.8, "mae": 0.4, "mfe": 1.5, "hold": 6, "won": True}] * 60
                    + [{"r": -0.6, "mae": 1.2, "mfe": 0.3, "hold": 4, "won": False}] * 60)
        q = np.array([60, 5, 3, 2, 10, 4, 1], float)
        s = M.analog_summary(q, corpus, outcomes, k=30)
        assert s["win_rate"] > 0.9                             # pulled the good near-cluster
        assert s["avg_mae"] == pytest.approx(0.4, abs=0.01)

    def test_below_min_analogs_is_silent(self):
        corpus = np.random.default_rng(51).normal(0, 1, (5, 7))
        outcomes = [{"r": 1.0, "mae": 0.2, "mfe": 1.0, "hold": 3, "won": True}] * 5
        assert M.analog_summary(np.zeros(7), corpus, outcomes, min_analogs=20) == {}

    def test_mahalanobis_finds_actual_nearest(self):
        corpus = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, 0.1], [20.0, 20.0]])
        inv = M.robust_inv_cov(corpus)
        idx, _ = M.mahalanobis_knn(np.array([0.0, 0.0]), corpus, inv, k=2)
        assert set(idx.tolist()) == {0, 2}                    # the two near origin

    def test_extract_features_shape_and_short_window(self):
        close = np.linspace(80, 120, 220)
        f = M.extract_features(close, close + 1, close - 1, np.full(220, 1e6))
        assert f is not None and f.size == len(M.FEATURE_NAMES)
        assert M.extract_features(close[:30], close[:30], close[:30], None) is None

    def test_find_analogs_fail_open(self):
        import pandas as pd
        close = np.linspace(80, 120, 220)
        df = pd.DataFrame({"close": close, "high": close + 1, "low": close - 1,
                           "volume": [1e6] * 220})
        assert M.find_analogs("X", df) == {}                  # no corpus in test env


from research import registry as REG


class TestExperimentRegistry:
    """Pre-registration (can't move the goalposts) + champion/challenger (no
    live change without winning a bake-off) — the Research OS's discipline."""

    def _tmp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(REG, "_DB_PATH", tmp_path / "exp.db")

    def test_hypothesis_id_is_deterministic(self):
        a = REG.hypothesis_id("h", {"dsr": {"gte": 0.95}}, {"w": 1})
        b = REG.hypothesis_id("h", {"dsr": {"gte": 0.95}}, {"w": 1})
        c = REG.hypothesis_id("h", {"dsr": {"gte": 0.90}}, {"w": 1})
        assert a == b and a != c                              # same def → same id

    def test_meets_criteria(self):
        crit = {"dsr": {"gte": 0.95}, "n": {"gte": 50}}
        assert REG.meets_criteria({"dsr": 0.97, "n": 120}, crit) is True
        assert REG.meets_criteria({"dsr": 0.90, "n": 120}, crit) is False
        assert REG.meets_criteria({"dsr": 0.97}, crit) is False   # missing metric fails

    def test_preregistration_freezes_the_verdict(self, tmp_path, monkeypatch):
        self._tmp(tmp_path, monkeypatch)
        hid = REG.register_hypothesis("idea", {"dsr": {"gte": 0.95}}, {"w": 1})
        assert REG.register_hypothesis("idea", {"dsr": {"gte": 0.95}}, {"w": 1}) == hid
        assert REG.record_result(hid, {"dsr": 0.97})["status"] == "PROMOTED"
        hid2 = REG.register_hypothesis("bad", {"dsr": {"gte": 0.95}}, {"w": 2})
        assert REG.record_result(hid2, {"dsr": 0.60})["status"] == "REJECTED"

    def test_unregistered_result_is_refused(self, tmp_path, monkeypatch):
        self._tmp(tmp_path, monkeypatch)
        assert REG.record_result("nope", {"dsr": 0.99}).get("error") == "not_registered"

    def test_should_promote_needs_margin_and_significance(self):
        # clear, significant win → promote
        ch = [0.30, 0.32, 0.28, 0.35, 0.31]
        cp = [0.20, 0.19, 0.22, 0.18, 0.21]
        assert REG.should_promote(0.312, 0.20, margin=0.02,
                                  challenger_scores=ch, champion_scores=cp)["promote"]
        # marginal + not significant → hold
        d = REG.should_promote(0.20, 0.20, margin=0.0,
                               challenger_scores=[0.21, 0.19, 0.20, 0.22, 0.18],
                               champion_scores=[0.20] * 5)
        assert d["promote"] is False

    def test_champion_challenger_flow(self, tmp_path, monkeypatch):
        self._tmp(tmp_path, monkeypatch)
        assert REG.evaluate_challenger("scorer", "v1", 0.20)["promote"] is True
        # a significant winner takes the crown
        REG.evaluate_challenger("scorer", "v2", 0.31, margin=0.02,
                                challenger_scores=[0.30, 0.32, 0.28, 0.35, 0.31],
                                champion_scores=[0.20, 0.19, 0.22, 0.18, 0.21])
        assert REG.current_champion("scorer")["model_id"] == "v2"
        # a marginal challenger is held — champion unchanged
        REG.evaluate_challenger("scorer", "v3", 0.315, margin=0.0,
                                challenger_scores=[0.32, 0.31, 0.30],
                                champion_scores=[0.31, 0.31, 0.31])
        assert REG.current_champion("scorer")["model_id"] == "v2"


from research import edge_timeline as ET


class TestEdgeTimeline:
    """A signal's drift HISTORY is evidence — one snapshot can't tell a cyclical
    edge (recovers every time) from a dead one (never came back)."""

    def _tmp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ET, "_DB_PATH", tmp_path / "edge.db")

    # ── pure classifier ──
    def test_cyclical_needs_repeat_decay_and_recovery(self):
        events = [{"status": "DECAYING", "n": 60}, {"status": "RECOVERING", "n": 92},
                  {"status": "DECAYING", "n": 150}, {"status": "RECOVERING", "n": 178}]
        p = ET._classify_profile(events, 190)
        assert p["profile"] == "CYCLICAL"
        assert p["n_decays"] == 2 and p["n_recoveries"] == 2
        assert p["median_recovery_trades"] == 30.0            # (32 + 28) / 2

    def test_unrecovered_decay_is_dead(self):
        # decayed and never recovered across many trades
        p = ET._classify_profile([{"status": "DECAYING", "n": 50}], current_n=120)
        assert p["profile"] == "DEAD"

    def test_recent_decay_is_not_yet_dead(self):
        p = ET._classify_profile([{"status": "DECAYING", "n": 100}], current_n=115)
        assert p["profile"] == "DECAYING"                     # too soon to retire

    def test_long_uninterrupted_stable_is_durable(self):
        p = ET._classify_profile([{"status": "STABLE", "n": 40}], current_n=140)
        assert p["profile"] == "DURABLE"

    def test_no_history_is_emerging(self):
        assert ET._classify_profile([], current_n=20)["profile"] == "EMERGING"
        assert ET._classify_profile([], current_n=0)["profile"] == "UNKNOWN"

    # ── I/O: transition ledger, not a sample log ──
    def test_records_only_transitions(self, tmp_path, monkeypatch):
        self._tmp(tmp_path, monkeypatch)
        rng = np.random.default_rng(40)
        decaying = list(np.concatenate([rng.normal(0.5, 1, 60),
                                        rng.normal(-0.4, 1, 60)]))
        streams = {"breakout": decaying}
        first = ET.record_snapshot(streams=streams, now="2026-01-01T00:00:00")
        assert first and first[0]["signal"] == "breakout"
        assert first[0]["status"] == "DECAYING"
        # same state again → NO new row (transition ledger)
        again = ET.record_snapshot(streams=streams, now="2026-01-02T00:00:00")
        assert again == []
        assert len(ET.signal_history("breakout")) == 1

    def test_profile_and_report_fail_open(self):
        # no DB rows in a fresh env → never raises
        assert isinstance(ET.timeline_report(), list)
        assert isinstance(ET.timeline_directives(), list)
        assert ET.signal_profile("nonexistent", streams={})["profile"] in (
            "EMERGING", "UNKNOWN")

    def test_dead_signal_becomes_a_retire_directive(self, tmp_path, monkeypatch):
        self._tmp(tmp_path, monkeypatch)
        # seed a signal whose latest transition is a long-unrecovered decay
        c = ET._conn()
        c.execute("INSERT INTO edge_events (signal, observed_at, status, n) "
                  "VALUES ('breakout','2026-01-01T00:00:00','DECAYING',40)")
        c.commit(); c.close()
        prof = ET.signal_profile("breakout", streams={"breakout": [0.0] * 120})
        assert prof["profile"] == "DEAD"
