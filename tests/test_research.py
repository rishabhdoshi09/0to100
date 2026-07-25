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
        assert "weakening" in r.insight

    def test_clear_strengthening_is_caught(self):
        rng = np.random.default_rng(22)
        stream = np.concatenate([rng.normal(-0.4, 1.0, 100),
                                 rng.normal(0.5, 1.0, 100)])
        r = D.assess_drift(stream, seed=1)
        assert r.status == "STRENGTHENING" and r.delta_r > 0

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
