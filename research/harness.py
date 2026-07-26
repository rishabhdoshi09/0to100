"""
🔬 Research Governance Harness — the anti-overfitting gauntlet.

The single most important subsystem in the moat roadmap: with a small,
slowly-growing outcome sample, running many models WILL surface spurious
"edges" (data-mining bias / multiple-comparisons). This module is the discipline
that separates a research organisation from a curve-fit backtest. Nothing else
in the learning loop is safe without it.

It provides the standard, textbook methods — nothing exotic, everything
auditable:

  • expectancy_stats()        — honest per-trade R stats + a one-sided t-test.
  • probabilistic_sharpe_ratio() (PSR, Bailey & López de Prado 2012) — P(true
                                Sharpe > benchmark), correcting for track-record
                                length, skew and fat tails.
  • deflated_sharpe_ratio()   (DSR, Bailey & López de Prado 2014) — PSR against
                                the EXPECTED-MAXIMUM Sharpe under the null across
                                N trials. This is what kills "we tried 200 things
                                and the best had Sharpe 2".
  • benjamini_hochberg()      — False Discovery Rate control across a batch of
                                hypotheses.
  • whites_reality_check()    (White 2000) — bootstrap data-snooping test: is the
                                BEST of L strategies genuinely good, or just the
                                luckiest? Stationary bootstrap (Politis-Romano).
  • min_samples_for_edge()    — power analysis: how many trades to detect a given
                                edge at a given power. Turns the system's ≥30 rule
                                into a principled, per-effect-size gate.
  • purged_kfold_indices()    — purged & embargoed walk-forward CV (López de
                                Prado) so training never leaks across the label
                                horizon (a 5-day outcome overlaps 5 bars).
  • evaluate()                — the single gate every claim passes through →
                                a verdict (PROMOTE / REJECT / UNDERPOWERED /
                                INCONCLUSIVE) + a plain-English one-liner.

Pure functions over numpy arrays; scipy for the distributions. No I/O, fully
unit-tested against known statistical cases (a fair coin must NOT promote; a
planted edge must; pure noise must be caught by FDR and the Reality Check).
"""
from __future__ import annotations

import math
import os as _os
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from scipy.stats import t as _student_t
from scipy.stats import skew as _skew
from scipy.stats import kurtosis as _kurtosis

# Euler–Mascheroni constant, used in the expected-maximum-order-statistic term
# of the Deflated Sharpe Ratio.
_EULER_GAMMA = 0.5772156649015329

# ── Gate thresholds (env-tunable) ─────────────────────────────────────────────
# A claim PROMOTES only when the (deflated, when multi-trial) probability that
# its true Sharpe beats zero clears this. 0.95 = the conventional 5%.
_PROMOTE_P = float(_os.getenv("QT_HARNESS_PROMOTE_P", "0.95") or 0.95)
# The smallest per-trade edge (in R) we insist on being POWERED to detect. If the
# sample is too small to detect an edge this size at _POWER, the verdict is
# UNDERPOWERED — "we cannot tell", never a confident claim.
_MIN_DETECTABLE_R = float(_os.getenv("QT_HARNESS_MIN_DETECTABLE_R", "0.10") or 0.10)
_POWER = float(_os.getenv("QT_HARNESS_POWER", "0.80") or 0.80)
_ALPHA = float(_os.getenv("QT_HARNESS_ALPHA", "0.05") or 0.05)
# Hard floor: NOTHING promotes below this many outcomes, however good it looks.
# A high Sharpe on a handful of trades is usually luck, and its skew/kurtosis
# (which PSR leans on) are meaningless. This is the system's invariant #6
# ("<30 trades = no claim") made mechanical.
_HARD_MIN_N = int(_os.getenv("QT_HARNESS_MIN_N", "30") or 30)
# A SEPARATE, lower floor that RESEARCH contexts may pass to evaluate(min_n=...)
# so experiments can explore small samples WITHOUT ever lowering the production
# default (which stays a mechanical 30, invariant #6). Production callers must
# never pass this; it exists so a research sweep doesn't silently change live
# behaviour to justify itself.
RESEARCH_MIN_N = int(_os.getenv("QT_RESEARCH_MIN_SAMPLE", str(_HARD_MIN_N)) or _HARD_MIN_N)


# ══════════════════════════════════════════════════════════════════════════════
# Descriptive statistics
# ══════════════════════════════════════════════════════════════════════════════

def expectancy_stats(returns) -> dict:
    """Per-trade R statistics for a stream of realised R-multiples.

    Returns n, mean_r, std_r, per-trade sharpe (mean/std), the one-sided t-test
    of H0: mean ≤ 0 (p_value = P the edge is a fluke), and the higher moments
    (skew, kurtosis — kurtosis is NON-excess, i.e. a normal distribution = 3,
    the convention the PSR/DSR formulas expect). NaNs are dropped."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = int(r.size)
    if n == 0:
        return {"n": 0, "mean_r": 0.0, "std_r": 0.0, "sharpe": 0.0,
                "t_stat": 0.0, "p_value": 1.0, "skew": 0.0, "kurtosis": 3.0}
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean / std if std > 0 else 0.0
    if std > 0 and n > 1:
        t_stat = mean / (std / math.sqrt(n))
        p_value = float(_student_t.sf(t_stat, df=n - 1))   # one-sided P(T ≥ t)
    else:
        t_stat, p_value = 0.0, 1.0
    sk = float(_skew(r)) if n > 2 else 0.0
    ku = float(_kurtosis(r, fisher=False)) if n > 3 else 3.0   # non-excess
    return {"n": n, "mean_r": mean, "std_r": std, "sharpe": sharpe,
            "t_stat": float(t_stat), "p_value": p_value,
            "skew": sk, "kurtosis": ku}


# ══════════════════════════════════════════════════════════════════════════════
# Sharpe-ratio inference — PSR & the Deflated Sharpe Ratio
# ══════════════════════════════════════════════════════════════════════════════

def probabilistic_sharpe_ratio(sharpe: float, n: int, skew: float = 0.0,
                               kurtosis: float = 3.0,
                               sharpe_benchmark: float = 0.0) -> float:
    """PSR — the probability that the TRUE Sharpe exceeds `sharpe_benchmark`,
    given an observed `sharpe` over `n` observations with the given skew and
    (non-excess) kurtosis. Bailey & López de Prado (2012).

    Fat tails and negative skew inflate the standard error of a Sharpe estimate;
    PSR accounts for both, so it is strictly more honest than a raw t-stat."""
    if n < 2:
        return 0.0
    var_term = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe
    if var_term <= 0:
        # degenerate: the moment-adjusted variance is non-positive; fall back to
        # the sign of the edge rather than returning nonsense.
        return 1.0 if sharpe > sharpe_benchmark else 0.0
    z = (sharpe - sharpe_benchmark) * math.sqrt(n - 1) / math.sqrt(var_term)
    return float(norm.cdf(z))


def expected_max_sharpe_null(n_trials: int, sharpe_variance: float) -> float:
    """The EXPECTED MAXIMUM Sharpe under the null (true Sharpe = 0) across
    `n_trials` independent trials, each an estimate with variance
    `sharpe_variance`. This is the bar a selected strategy must clear — because
    the max of many zero-mean estimates is positive. Bailey & López de Prado
    (2014), via the expected value of the maximum of N standard normals."""
    if n_trials < 1 or sharpe_variance <= 0:
        return 0.0
    if n_trials == 1:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1.0 - _EULER_GAMMA) * z1
                                         + _EULER_GAMMA * z2)


def deflated_sharpe_ratio(sharpe: float, n: int, skew: float = 0.0,
                          kurtosis: float = 3.0, n_trials: int = 1,
                          sharpe_estimates=None,
                          sharpe_variance: float | None = None) -> dict:
    """DSR — PSR benchmarked against the expected-max Sharpe under the null over
    `n_trials`. The correct answer to "we searched many configurations and kept
    the best". Deflates the observed Sharpe by how much selection alone would
    have produced.

    Provide EITHER `sharpe_estimates` (the Sharpes of ALL trials — the variance
    and the trial count are then measured from them) OR `sharpe_variance` +
    `n_trials`. With neither, sr0 = 0 and DSR degenerates to PSR(0)."""
    if sharpe_estimates is not None:
        est = np.asarray(sharpe_estimates, dtype=float)
        est = est[~np.isnan(est)]
        if est.size > 1:
            sharpe_variance = float(est.var(ddof=1))
            n_trials = int(est.size)
    sr0 = expected_max_sharpe_null(n_trials, sharpe_variance or 0.0)
    dsr = probabilistic_sharpe_ratio(sharpe, n, skew, kurtosis,
                                     sharpe_benchmark=sr0)
    return {"dsr": dsr, "sr0_expected_max_null": sr0, "n_trials": n_trials,
            "sharpe_variance": float(sharpe_variance or 0.0)}


# ══════════════════════════════════════════════════════════════════════════════
# Multiple-testing control
# ══════════════════════════════════════════════════════════════════════════════

def benjamini_hochberg(pvalues, alpha: float = _ALPHA) -> dict:
    """Benjamini–Hochberg False Discovery Rate control across a batch of
    hypotheses. Returns a boolean `rejected` mask (in the INPUT order), the BH
    adjusted q-values, the p-value threshold, and the count. Controls the
    expected proportion of false discoveries at `alpha` — the right tool when
    testing many signal/combo/regime hypotheses at once."""
    p = np.asarray(pvalues, dtype=float)
    m = int(p.size)
    if m == 0:
        return {"rejected": np.zeros(0, dtype=bool), "qvalues": np.zeros(0),
                "threshold": 0.0, "n_rejected": 0}
    order = np.argsort(p)
    ranked = p[order]
    crit = (np.arange(1, m + 1) / m) * alpha
    below = ranked <= crit
    if below.any():
        kmax = int(np.max(np.nonzero(below)[0]))
        threshold = float(ranked[kmax])
    else:
        threshold = 0.0
    rejected = p <= threshold
    # BH adjusted q-values: enforce monotonicity from the largest p downward.
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    qvalues = np.empty(m, dtype=float)
    qvalues[order] = q_ranked
    return {"rejected": rejected, "qvalues": qvalues,
            "threshold": threshold, "n_rejected": int(rejected.sum())}


def _stationary_bootstrap_indices(T: int, mean_block: float,
                                  rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano stationary bootstrap: geometric block lengths with mean
    `mean_block`, wrapping circularly. Preserves short-range serial dependence
    (which naive i.i.d. resampling would destroy)."""
    p = 1.0 / max(1.0, mean_block)
    idx = np.empty(T, dtype=np.int64)
    t = int(rng.integers(0, T))
    for i in range(T):
        idx[i] = t
        if rng.random() < p:
            t = int(rng.integers(0, T))
        else:
            t = (t + 1) % T
    return idx


def whites_reality_check(perf_matrix, n_boot: int = 2000,
                         block: float | None = None,
                         seed: int | None = None) -> dict:
    """White's Reality Check (2000) for data-snooping. `perf_matrix` is
    shape (T periods, L strategies): each column is a strategy's per-period
    performance RELATIVE to the benchmark (e.g. excess return; 0 = matched the
    benchmark). Tests H0: the best strategy's expected out-performance ≤ 0.

    Returns the reality-check p-value (small = the best strategy is genuinely
    good, not the luckiest of L), the index of the best strategy, and its mean.
    Uses the stationary bootstrap to respect serial dependence."""
    f = np.asarray(perf_matrix, dtype=float)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    T, L = f.shape
    if T < 2:
        return {"reality_check_p": 1.0, "best_strategy": 0, "best_mean": 0.0,
                "V_stat": 0.0, "block": 1.0, "n_boot": 0}
    means = f.mean(axis=0)
    V_stat = float(np.sqrt(T) * np.max(means))
    if block is None:
        block = max(1.0, round(T ** (1.0 / 3.0)))   # standard rule of thumb
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_boot):
        idx = _stationary_bootstrap_indices(T, block, rng)
        boot_means = f[idx].mean(axis=0)
        # recenter each strategy by its own full-sample mean (White's V*)
        Vb = float(np.sqrt(T) * np.max(boot_means - means))
        if Vb >= V_stat:
            exceed += 1
    p = (exceed + 1) / (n_boot + 1)          # +1 = never report exactly 0
    best = int(np.argmax(means))
    return {"reality_check_p": float(p), "best_strategy": best,
            "best_mean": float(means[best]), "V_stat": V_stat,
            "block": float(block), "n_boot": n_boot}


# ══════════════════════════════════════════════════════════════════════════════
# Power analysis
# ══════════════════════════════════════════════════════════════════════════════

def min_samples_for_edge(edge_r: float, std_r: float, alpha: float = _ALPHA,
                         power: float = _POWER, two_sided: bool = False) -> int:
    """How many trades are needed to detect an edge of `edge_r` (per-trade R)
    given per-trade dispersion `std_r`, at significance `alpha` and `power`?
    One-sample-t approximation via the standardized effect size d = edge/std.
    Returns a large sentinel when the effect is non-positive/degenerate (you can
    never be powered to detect a zero effect)."""
    if edge_r <= 0 or std_r <= 0:
        return 10 ** 9
    d = abs(edge_r) / std_r
    z_alpha = norm.ppf(1.0 - (alpha / 2.0 if two_sided else alpha))
    z_beta = norm.ppf(power)
    n = ((z_alpha + z_beta) / d) ** 2
    return int(math.ceil(n))


# ══════════════════════════════════════════════════════════════════════════════
# Purged & embargoed cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def purged_kfold_indices(n: int, k: int = 5, embargo=0,
                         label_horizon: int = 1) -> list[tuple]:
    """Purged, embargoed k-fold splits over `n` time-ordered samples (López de
    Prado, *Advances in Financial Machine Learning*). Each sample's label is
    realised `label_horizon` bars later (e.g. a 5-day outcome), so a naive split
    leaks: a training bar right before the test block sees the future.

    For each contiguous test fold we drop from training (a) the test block, (b)
    the `label_horizon` bars immediately before it (their labels overlap the
    test period — PURGE), and (c) an EMBARGO buffer immediately after it.
    `embargo` may be an integer bar count or a fraction of n. Returns a list of
    (train_idx, test_idx) numpy arrays."""
    if n <= 0 or k <= 1:
        return [(np.arange(0), np.arange(n))] if n > 0 else []
    indices = np.arange(n)
    folds = [f for f in np.array_split(indices, k) if f.size > 0]
    emb = int(embargo) if embargo >= 1 else int(round(float(embargo) * n))
    lab = max(0, int(label_horizon) - 1)      # horizon of 1 = same-bar, no purge
    splits: list[tuple] = []
    for fold in folds:
        test_start, test_end = int(fold[0]), int(fold[-1])
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_start:test_end + 1] = False               # the test block
        purge_lo = max(0, test_start - lab)
        train_mask[purge_lo:test_start] = False                   # PURGE (before)
        emb_hi = min(n, test_end + 1 + emb)
        train_mask[test_end + 1:emb_hi] = False                   # EMBARGO (after)
        splits.append((indices[train_mask], indices[test_start:test_end + 1]))
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark-relative inference — is it ALPHA, or just BETA (market exposure)?
# ══════════════════════════════════════════════════════════════════════════════

def alpha_beta(strategy_returns, benchmark_returns) -> dict:
    """Decompose a strategy's per-trade returns into ALPHA (skill) and BETA
    (market exposure) via OLS: r_strat = α + β·r_bench + ε, where the two arrays
    are PAIRED — benchmark_returns[i] is the benchmark's return over trade i's
    holding window (same units as the strategy's, e.g. R or %).

    A strategy that only rides the market has α ≈ 0 and β ≈ 1: it looks profitable
    in a bull tape but has no edge. The committee's #1 objection ("is this alpha or
    long-beta?") is answered by the ONE-SIDED test of H0: α ≤ 0. Returns α, β,
    α's standard error / t-stat / one-sided p-value, the benchmark correlation,
    and the information ratio (α ÷ residual σ — active return per unit active
    risk). Degenerate inputs (n<3, zero benchmark variance) return a null result
    that will never pass the gate."""
    x = np.asarray(benchmark_returns, dtype=float)
    y = np.asarray(strategy_returns, dtype=float)
    m = min(x.size, y.size)
    x, y = x[:m], y[:m]
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = int(x.size)
    null = {"n": n, "alpha": 0.0, "beta": 0.0, "se_alpha": 0.0, "t_alpha": 0.0,
            "p_alpha": 1.0, "corr": 0.0, "info_ratio": 0.0, "beats_benchmark": False}
    if n < 3:
        return null
    sxx = float(np.sum((x - x.mean()) ** 2))
    if sxx <= 0:                                   # benchmark is constant → no fit
        return null
    beta, alpha = np.polyfit(x, y, 1)              # [slope, intercept]
    resid = y - (alpha + beta * x)
    dof = n - 2
    sse = float(np.sum(resid ** 2))
    s_err = math.sqrt(sse / dof) if dof > 0 else 0.0
    se_alpha = s_err * math.sqrt(1.0 / n + (x.mean() ** 2) / sxx) if s_err > 0 else 0.0
    if se_alpha > 0:
        t_alpha = float(alpha) / se_alpha
        p_alpha = float(_student_t.sf(t_alpha, df=dof))     # one-sided H0: α ≤ 0
    else:
        t_alpha, p_alpha = 0.0, (0.0 if alpha > 0 else 1.0)
    resid_sd = float(resid.std(ddof=1)) if n > 1 else 0.0
    info_ratio = float(alpha) / resid_sd if resid_sd > 0 else 0.0
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
    return {"n": n, "alpha": float(alpha), "beta": float(beta),
            "se_alpha": float(se_alpha), "t_alpha": float(t_alpha),
            "p_alpha": p_alpha, "corr": corr, "info_ratio": info_ratio,
            "beats_benchmark": bool(alpha > 0 and p_alpha < _ALPHA)}


def factor_attribution(strategy_returns, factors) -> dict:
    """Multi-factor generalisation of `alpha_beta`. A single index regression can
    still credit as "alpha" a return that is really a KNOWN systematic tilt —
    size, value, momentum, low-vol, sector, liquidity. Factor-neutral alpha is a
    strictly stronger claim: the return that survives AFTER controlling for every
    supplied exposure.

    `factors` is a dict {name: per-trade factor return} (or a 2-D array, shape
    (n, k)), each paired with the strategy's per-trade returns. Fits
    r_strat = α + Σ βᵢ·fᵢ + ε by OLS and one-sided-tests H0: α ≤ 0. Returns α, the
    per-factor βs, α's se/t/p, R² (how much of the strategy is explained by the
    factors), and `beats_benchmark` = α>0 & p<α. Degenerate inputs (n < k+2,
    singular design) return a null result that never passes the gate."""
    y = np.asarray(strategy_returns, dtype=float)
    if isinstance(factors, dict):
        names = list(factors.keys())
        cols = [np.asarray(factors[k], dtype=float) for k in names]
    else:
        F = np.asarray(factors, dtype=float)
        if F.ndim == 1:
            F = F.reshape(-1, 1)
        names = [f"f{i}" for i in range(F.shape[1])]
        cols = [F[:, i] for i in range(F.shape[1])]
    m = min([y.size] + [c.size for c in cols]) if cols else y.size
    y = y[:m]
    cols = [c[:m] for c in cols]
    k = len(cols)
    null = {"n": int(y.size), "alpha": 0.0, "betas": {}, "se_alpha": 0.0,
            "t_alpha": 0.0, "p_alpha": 1.0, "r_squared": 0.0,
            "beats_benchmark": False}
    if k == 0:
        return null
    X_raw = np.column_stack(cols)
    mask = ~(np.isnan(y) | np.isnan(X_raw).any(axis=1))
    y, X_raw = y[mask], X_raw[mask]
    n = int(y.size)
    if n < k + 2:
        return null
    X = np.column_stack([np.ones(n), X_raw])          # intercept first
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return null                                    # collinear factors
    coef = xtx_inv @ (X.T @ y)
    resid = y - X @ coef
    dof = n - (k + 1)
    if dof <= 0:
        return null
    sse = float(resid @ resid)
    sigma2 = sse / dof
    diag = np.diag(xtx_inv)
    se_alpha = math.sqrt(sigma2 * diag[0]) if sigma2 * diag[0] > 0 else 0.0
    alpha = float(coef[0])
    betas = {names[i]: float(coef[i + 1]) for i in range(k)}
    if se_alpha > 0:
        t_alpha = alpha / se_alpha
        p_alpha = float(_student_t.sf(t_alpha, df=dof))
    else:
        t_alpha, p_alpha = 0.0, (0.0 if alpha > 0 else 1.0)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 0.0
    return {"n": n, "alpha": alpha, "betas": betas, "se_alpha": float(se_alpha),
            "t_alpha": float(t_alpha), "p_alpha": p_alpha,
            "r_squared": float(r2),
            "beats_benchmark": bool(alpha > 0 and p_alpha < _ALPHA)}


# ══════════════════════════════════════════════════════════════════════════════
# Correlation-adjusted inference — trades are NOT i.i.d.
# ══════════════════════════════════════════════════════════════════════════════

def effective_sample_size(returns, max_lag: int | None = None) -> float:
    """The AUTOCORRELATION-ADJUSTED sample size. Trades taken in the same tape,
    the same sector, the same week co-move — they are not independent draws, so a
    naive `n` overstates the evidence and every t-test / CI / power calc built on
    it is optimistic. N_eff = n / (1 + 2·Σ ρ_k), summing the INITIAL positive
    autocorrelation sequence (truncated at the first non-positive lag — the
    standard initial-positive-sequence estimator). Clamped to [1, n]. For an
    i.i.d. stream N_eff ≈ n; for a strongly serially-correlated one it can be far
    smaller — which is the honest denominator."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = int(r.size)
    if n < 4:
        return float(n)
    r = r - r.mean()
    var = float(np.mean(r ** 2))
    if var <= 0:
        return float(n)
    if max_lag is None:
        max_lag = min(n - 1, max(1, int(10 * math.log10(n))))
    rho_sum = 0.0
    for k in range(1, max_lag + 1):
        rho_k = float(np.mean(r[:-k] * r[k:])) / var       # biased (÷n) estimator
        if rho_k <= 0:                                      # initial positive seq.
            break
        rho_sum += (1.0 - k / n) * rho_k
    n_eff = n / (1.0 + 2.0 * rho_sum)
    return float(min(float(n), max(1.0, n_eff)))


def block_bootstrap_mean_ci(returns, n_boot: int = 2000, block: float | None = None,
                            alpha: float = _ALPHA, seed: int | None = None) -> dict:
    """A confidence interval for mean per-trade R that RESPECTS serial dependence.
    A textbook i.i.d. CI (±1.96·σ/√n) is too tight when trades co-move; the
    stationary bootstrap (Politis–Romano) resamples geometric BLOCKS, preserving
    short-range dependence, so the interval widens honestly. Returns the mean, the
    (1−alpha) CI bounds, whether the lower bound clears zero, the block length and
    the effective sample size. `ci_lower > 0` is the correlation-aware analogue of
    'significantly positive' — the criterion the evidence gate should use, not the
    naive t-test."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = int(r.size)
    if T < 2:
        return {"mean_r": float(r.mean()) if T else 0.0, "ci_lower": 0.0,
                "ci_upper": 0.0, "excludes_zero": False, "block": 1.0,
                "n_eff": float(T), "n_boot": 0}
    if block is None:
        block = max(1.0, round(T ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _stationary_bootstrap_indices(T, block, rng)
        boot[i] = r[idx].mean()
    lo = float(np.percentile(boot, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return {"mean_r": float(r.mean()), "ci_lower": lo, "ci_upper": hi,
            "excludes_zero": bool(lo > 0.0), "block": float(block),
            "n_eff": effective_sample_size(r), "n_boot": n_boot}


# ══════════════════════════════════════════════════════════════════════════════
# The gate — every claim passes through here
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HypothesisResult:
    """The harness's verdict on one claim. `verdict` ∈
    {PROMOTE, REJECT, UNDERPOWERED, INCONCLUSIVE}."""
    verdict: str
    n: int
    mean_r: float
    sharpe: float
    p_value: float
    psr: float
    dsr: float
    n_trials: int
    min_n_needed: int
    insight: str                       # plain-English, user-facing (kill rule)
    stats: dict = field(default_factory=dict)


def evaluate(returns, n_trials: int = 1, sharpe_estimates=None,
             min_detectable_r: float = _MIN_DETECTABLE_R,
             promote_p: float = _PROMOTE_P,
             min_n: int = _HARD_MIN_N,
             benchmark_returns=None,
             require_block_ci: bool = False,
             block_ci_seed: int | None = None) -> HypothesisResult:
    """The single gate. Feed a stream of realised R-multiples for one strategy/
    signal/combo (and, if it was selected from a search, the number of trials or
    the Sharpes of every trial). Returns a verdict + a plain-English insight.

    Logic, in order (significance is judged BEFORE power — a result that already
    clears the significance bar is, by definition, powered enough for its own
    effect size; power only disambiguates NON-significant results):
      1. PROMOTE — a positive edge whose deflated probability of beating zero
         clears `promote_p`. Deflation uses the trial count / trial-Sharpes so a
         search-selected winner must beat the expected max under the null.
      2. REJECT — a non-positive point estimate (no edge, or a loser). We never
         act on a non-positive edge regardless of sample size.
      3. UNDERPOWERED — a positive-but-not-significant edge on a sample too small
         to have detected a `min_detectable_r` edge at the configured power. The
         honest "absence of evidence ≠ evidence of absence" — keep tracking.
         (This is the ≥30 rule, generalised and made principled.)
      4. INCONCLUSIVE — a positive-but-not-significant edge on a sample that WAS
         large enough to detect a meaningful edge → probably not a real one.

    Two OPTIONAL, harder gates a would-be PROMOTE must ALSO clear (both off by
    default so existing callers are unchanged):
      • `benchmark_returns` (paired per-trade benchmark R) → the edge must carry
        significant ALPHA vs the benchmark, else it is demoted to REJECT as
        long-beta (rides the market, no skill). Answers "alpha or beta?".
      • `require_block_ci` → the correlation-aware block-bootstrap CI lower bound
        for mean R must exceed zero, else demoted to INCONCLUSIVE. Because trades
        are not i.i.d., this is the honest significance test, stricter than PSR.
    """
    s = expectancy_stats(returns)
    n, mean, std, sharpe = s["n"], s["mean_r"], s["std_r"], s["sharpe"]
    psr = probabilistic_sharpe_ratio(sharpe, n, s["skew"], s["kurtosis"], 0.0)
    d = deflated_sharpe_ratio(sharpe, n, s["skew"], s["kurtosis"],
                              n_trials=n_trials, sharpe_estimates=sharpe_estimates)
    dsr = d["dsr"]
    eff_trials = d["n_trials"]
    # `need_n` = power-based sample needed to DETECT a min_detectable_r edge;
    # `min_n` = the hard floor below which no claim is made at all (production
    # default 30, tunable for research via QT_RESEARCH_MIN_SAMPLE). Two distinct
    # thresholds — never collapse them into one variable.
    need_n = min_samples_for_edge(min_detectable_r, std if std > 0 else 1.0)
    # multi-trial → judge on the DEFLATED probability; single trial → PSR.
    prob = dsr if eff_trials > 1 else psr
    n_eff = effective_sample_size(returns)

    def _pack(verdict: str, insight: str) -> HypothesisResult:
        st = dict(s); st["n_eff"] = round(n_eff, 2)
        return HypothesisResult(verdict=verdict, n=n, mean_r=round(mean, 4),
                                sharpe=round(sharpe, 4), p_value=s["p_value"],
                                psr=round(psr, 4), dsr=round(dsr, 4),
                                n_trials=eff_trials, min_n_needed=need_n,
                                insight=insight, stats=st)

    if n < min_n:
        return _pack("UNDERPOWERED",
                     f"Only {n} trades — below the {min_n}-trade floor for "
                     f"any claim (a great-looking edge on a handful of trades is "
                     f"usually luck). Keep tracking.")
    if mean > 0 and prob >= promote_p:
        # ── the two optional institutional gates, applied only at the PROMOTE edge
        if benchmark_returns is not None:
            # a dict/2-D of factors → factor-neutral alpha; a 1-D array → the
            # single-index special case.
            multi = isinstance(benchmark_returns, dict) or (
                np.asarray(benchmark_returns).ndim == 2)
            ab = (factor_attribution(returns, benchmark_returns) if multi
                  else alpha_beta(returns, benchmark_returns))
            if not ab["beats_benchmark"]:
                what = ("known factor exposures" if multi else "the benchmark")
                return _pack("REJECT",
                             f"Looks profitable ({mean:+.2f}R) but it's exposure, "
                             f"not alpha: α={ab['alpha']:+.2f} "
                             f"(p={ab['p_alpha']:.2f}) after controlling for "
                             f"{what}. No skill beyond known exposures.")
        if require_block_ci:
            ci = block_bootstrap_mean_ci(returns, seed=block_ci_seed)
            if not ci["excludes_zero"]:
                return _pack("INCONCLUSIVE",
                             f"Positive ({mean:+.2f}R) but once trade correlation is "
                             f"accounted for (N_eff≈{n_eff:.0f} of {n}) the CI lower "
                             f"bound is {ci['ci_lower']:+.2f}R — not clear of zero.")
        tail = (f" (deflated for {eff_trials} trials)" if eff_trials > 1 else "")
        return _pack("PROMOTE",
                     f"Real edge: {mean:+.2f}R over {n} trades, "
                     f"{prob*100:.0f}% confident it beats zero{tail}.")
    if mean <= 0:
        return _pack("REJECT",
                     f"No edge — {mean:+.2f}R over {n} trades. Not worth acting on.")
    if n < need_n:
        return _pack("UNDERPOWERED",
                     f"Promising ({mean:+.2f}R) but not enough evidence yet — "
                     f"{n} trades, need ~{need_n} to trust a {min_detectable_r:+.2f}R "
                     f"edge. Holding the claim.")
    return _pack("INCONCLUSIVE",
                 f"Positive ({mean:+.2f}R over {n} trades) but only "
                 f"{prob*100:.0f}% confident (need {promote_p*100:.0f}%), and the "
                 f"sample was big enough to show a real edge. Likely noise.")
