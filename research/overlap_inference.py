"""Inference for statistics whose fixed-horizon outcome windows overlap.

Every learning claim in QuantTerm is about an outcome measured over the next
``h`` NSE sessions (h = 10). A statistic recorded on session ``s`` (a day's
mean R, a day's rank IC, a challenger-minus-champion IC difference) depends
on the market over sessions s+1 .. s+h, so two sessions fewer than ``h``
apart share outcome sessions and are NOT independent. Averaging them into
consecutive non-overlapping blocks (the 2026-11-19 method) does not remove
that: the last sessions of one block share nine of ten outcome sessions with
the first sessions of the next, and slow regimes add dependence beyond the
overlap. Monte Carlo on null data (no true effect) showed the block t-test
"confirming" 10-28% of the time at a nominal 5%.

The method here:

1. Session cells. Values are keyed by NSE session ordinal; several values on
   one session (trades, rows) form one cell (sum, count). Duplicating a row
   inside a session cannot shrink the standard error (cluster-robust).
2. Offset subsamples. Split the cells by ``ordinal mod h``. Within one
   subsample any two sessions are at least ``h`` apart, so their windows do
   not overlap. The overall (count-weighted) mean is the weighted average of
   the subsample means, and by Minkowski's inequality its standard error is
   at most the weighted SUM of the subsample standard errors, however
   strongly the subsamples are correlated with each other. That bound is the
   standard error used; it needs no model of the overlap.
3. Residual dependence. Dependence that survives a gap of ``h`` sessions
   (regime persistence, copied evidence) is measured as the autocorrelation
   of cell residuals at exactly lag ``h``. Every subsample variance is
   widened by the AR(1) long-run factor (1+r)/(1-r), where r is the larger of
   rho's one-sided 95% upper bound and its bias-corrected value, so an
   uncertain rho is treated as possibly positive.
4. Degrees of freedom are the non-overlapping h-session blocks holding data,
   minus one, divided by the residual widening (effective blocks). Fewer than
   MIN_INDEPENDENT_BLOCKS raw blocks, or zero dispersion, is never a verdict.
   Blocks are counted as the largest set of mutually disjoint windows, not
   occupied bins.

ASSUMPTION (stated, not testable from one history): dependence decays.
The bound is valid when outcome dependence beyond 2h sessions is negligible
or captured by the lag-h residual check. A single shock common to the WHOLE
history (one market-wide sign that every trade shares) is one observation
however many trades it touches, and no test on one realised history can
detect it (GPT cross-review C-1). Every p-value, lower bound and family-wise
claim built on this module inherits that assumption; the "any dependence"
in the alpha-ledger guarantee refers to dependence BETWEEN looks and
hypotheses (the union bound), not to dependence inside one test's data.

Validated in tests/test_overlap_inference.py: under overlapping-window,
persistent-regime and duplicated-block null data the false-confirmation
rate stays near or below the nominal 5%, while the previous block t-test
and the day-clustered error exceed it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

#: A verdict needs at least this many non-overlapping horizon blocks.
MIN_INDEPENDENT_BLOCKS = 20
#: Lag-h residual autocorrelation is estimated only from this many pairs.
MIN_PAIRS_FOR_RESIDUAL_RHO = 10
#: Cap on the widening estimate (rho -> 1 would make the variance infinite).
MAX_RESIDUAL_RHO = 0.95
METHOD = "offset-subsample Minkowski bound + lag-h residual widening (research/overlap_inference.py)"

INSUFFICIENT = "INSUFFICIENT"
OK = "OK"


def cells_from_pairs(pairs: Iterable[tuple[int, float]]) -> dict[int, list[float]]:
    """{ordinal: [sum, count]} from (ordinal, value) pairs; same-session values merge."""
    cells: dict[int, list[float]] = {}
    for ordinal, value in pairs:
        v = float(value)
        if not math.isfinite(v):
            continue
        cell = cells.setdefault(int(ordinal), [0.0, 0])
        cell[0] += v
        cell[1] += 1
    return cells


def cells_from_clusters(clusters: Mapping[str, Sequence[float]], ordinals: Mapping[str, int]) -> dict[int, list[float]]:
    """{ordinal: [sum, count]} from day-keyed [sum, count] clusters.

    Days without an ordinal are dropped (the caller decides whether that is
    acceptable); days sharing an ordinal merge into one cell.
    """
    cells: dict[int, list[float]] = {}
    for day, row in (clusters or {}).items():
        key = str(day)[:10]
        if key not in ordinals:
            continue
        total, count = float(row[0]), int(row[1])
        if count <= 0 or not math.isfinite(total):
            continue
        cell = cells.setdefault(int(ordinals[key]), [0.0, 0])
        cell[0] += total
        cell[1] += count
    return cells


def _t_sf(t: float, df: int) -> float:
    try:
        from scipy.stats import t as student_t

        return float(student_t.sf(t, df))
    except Exception:  # pragma: no cover - scipy is a hard dependency in CI
        return 0.5 * math.erfc(t / math.sqrt(2.0))


def _t_ppf(q: float, df: int) -> float:
    try:
        from scipy.stats import t as student_t

        return float(student_t.ppf(q, df))
    except Exception:  # pragma: no cover
        return {0.95: 1.645, 0.975: 1.96}.get(q, 1.96)


def overlap_robust_test(cells: Mapping[int, Sequence[float]], horizon: int, *,
                        min_blocks: int = MIN_INDEPENDENT_BLOCKS) -> dict[str, Any]:
    """Mean of a session-indexed statistic with overlap-aware inference.

    ``cells``: {NSE session ordinal: (sum of values, count)}. Unweighted
    daily statistics are cells with count 1; trade outcomes are cells of a
    day's trades. Returns mean, standard errors, df, t, two-sided p, the
    one-sided 95% lower bound and a status. ``status`` is OK only with at
    least ``min_blocks`` independent blocks and non-zero dispersion; the
    numbers are still reported (and ``lower_95_one_sided`` is still valid and
    conservative) when the status is INSUFFICIENT for lack of blocks.
    """
    h = max(1, int(horizon))
    items = sorted((int(o), float(c[0]), int(c[1])) for o, c in (cells or {}).items() if int(c[1]) > 0)
    n = sum(c for _, _, c in items)
    g = len(items)
    out: dict[str, Any] = {
        "method": METHOD, "horizon_sessions": h, "n": n, "sessions": g,
        "independent_sessions": 0, "df": None, "mean": None, "se": None,
        "se_day_clustered": None, "variance_inflation": None, "t": None,
        "p_two_sided": None, "lower_95_one_sided": None, "ci95": None,
        "residual_dependence": None, "status": INSUFFICIENT, "reason": "",
    }
    if n <= 0 or g == 0:
        out["reason"] = "NO_DATA"
        return out
    first = items[0][0]
    # Independent blocks = the largest set of sessions whose h-session windows
    # are mutually disjoint (greedy, exact on a line). Occupied h-wide bins
    # over-count: sessions 0, 19, 20, 39, 40 ... occupy a bin each, yet no more
    # than half of their windows can be chosen without overlap (GPT
    # cross-review C-1).
    blocks = 0
    last = None
    for o, _s, _c in items:
        if last is None or o >= last + h:
            blocks += 1
            last = o
    out["occupied_bins"] = len({(o - first) // h for o, _, _ in items})
    out["independent_sessions"] = blocks
    mean = sum(s for _, s, _ in items) / n
    out["mean"] = mean
    resid = {o: s - mean * c for o, s, c in items}
    ss = sum(u * u for u in resid.values())
    scale = max(1.0, abs(mean)) * 1e-12
    if ss <= (scale * n) ** 2 or g < 2:
        out["reason"] = "DEGENERATE_ZERO_DISPERSION" if g >= 2 else "ONE_SESSION"
        return out
    small = g / (g - 1)
    se_cluster = math.sqrt(ss / (n * n) * small)
    out["se_day_clustered"] = se_cluster

    # Residual dependence that survives a gap of exactly h sessions.
    pair_sum, pairs = 0.0, 0
    for o, u in resid.items():
        v = resid.get(o + h)
        if v is not None:
            pair_sum += u * v
            pairs += 1
    rho = None
    widening = 1.0
    if pairs >= MIN_PAIRS_FOR_RESIDUAL_RHO:
        rho = (pair_sum / pairs) / (ss / g)
        # Widen with the larger of a one-sided 95% UPPER bound on rho and the
        # small-sample bias-corrected rho: the estimate itself is noisy, and
        # under strong regime persistence (AR 0.9) the point estimate alone
        # left the false-confirmation rate at 5.9% (research.overlap_stress).
        k_eff = max(2, blocks)
        candidates = [rho + 1.645 * math.sqrt(max(0.0, 1.0 - rho * rho) / pairs)]
        if rho > 0:
            candidates.append(rho + (1.0 + 3.0 * rho) / k_eff)
        r = min(MAX_RESIDUAL_RHO, max(candidates))
        if r > 0:
            widening = (1.0 + r) / (1.0 - r)
    out["residual_dependence"] = {
        "lag_h_autocorr": None if rho is None else round(rho, 4),
        "pairs": pairs,
        "flagged": bool(rho is not None and rho > 1.645 / math.sqrt(max(1, pairs))),
        "variance_widening": round(widening, 4),
        "examined": rho is not None,
    }

    # Minkowski bound over the h offset subsamples. Inside one subsample the
    # long-run variance is the larger of (a) the residual-rho widening and
    # (b) a Bartlett Newey-West term at lags h and 2h (sessions one and two
    # windows apart), never below the plain within-subsample variance.
    by_offset: dict[int, list[int]] = {}
    for o, _s, _c in items:
        by_offset.setdefault((o - first) % h, []).append(o)
    counts = {o: c for o, _s, c in items}
    se = 0.0
    for members in by_offset.values():
        n_c = sum(counts[o] for o in members)
        base = sum(resid[o] ** 2 for o in members)
        nw = base
        for lag, weight in ((h, 2.0 / 3.0), (2 * h, 1.0 / 3.0)):
            nw += 2.0 * weight * sum(resid[o] * resid[o + lag] for o in members if (o + lag) in resid)
        lrv = max(base * widening, nw, base)
        se += (n_c / n) * math.sqrt(lrv / (n_c * n_c) * small)
    se = max(se, se_cluster)
    out["se"] = se
    out["variance_inflation"] = round((se / se_cluster) ** 2, 3)
    # Effective df (Satterthwaite-style): persistence beyond the overlap means
    # the blocks carry less independent information than their count, so
    # the Student-t reference uses (blocks - 1) / widening degrees of freedom.
    # Under AR(0.9) regimes this moved the null false-confirmation rate from
    # ~4.6-5.4% to 3.1% [2.4%, 4.0%] (research.overlap_stress, seed 1103).
    df = max(1, int((blocks - 1) / widening))
    out["df"] = df
    out["df_raw"] = max(1, blocks - 1)
    t = mean / se
    out["t"] = t
    out["p_two_sided"] = min(1.0, 2.0 * _t_sf(abs(t), df))
    out["lower_95_one_sided"] = mean - _t_ppf(0.95, df) * se
    half = _t_ppf(0.975, df) * se
    out["ci95"] = [mean - half, mean + half]
    if blocks < int(min_blocks):
        out["reason"] = f"{blocks} independent blocks < {int(min_blocks)}"
        return out
    out["status"] = OK
    return out


def naive_block_t_test(cells: Mapping[int, Sequence[float]], horizon: int) -> float | None:
    """The superseded 2026-11-19 method, kept ONLY so tests can show its failure."""
    h = max(1, int(horizon))
    items = sorted((int(o), float(c[0]), int(c[1])) for o, c in cells.items() if int(c[1]) > 0)
    if not items:
        return None
    first = items[0][0]
    blocks: dict[int, list[float]] = {}
    for o, s, c in items:
        b = blocks.setdefault((o - first) // h, [0.0, 0])
        b[0] += s
        b[1] += c
    means = [b[0] / b[1] for _, b in sorted(blocks.items())]
    k = len(means)
    if k < 2:
        return None
    m = sum(means) / k
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / (k - 1))
    if sd <= 0:
        return None
    return min(1.0, 2.0 * _t_sf(abs(m / (sd / math.sqrt(k))), k - 1))
