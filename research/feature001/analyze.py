"""Explanatory FEATURE-001 stats. No VALIDATED_EDGE. No production model."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from research.feature001.constants import (
    BREAKOUT_KEYS,
    FAMILY_KEYS,
    FDR_Q,
    MIN_N,
    MOMENTUM_KEYS,
    RS_BUCKETS,
    TREND_BUCKETS,
    YEARS,
)
from research.feature001.dataset import explode_families


def _arr(xs) -> np.ndarray:
    return np.asarray([float(x) for x in xs if x is not None and x == x], dtype=float)


def _mean(xs) -> float | None:
    a = _arr(xs)
    return None if a.size == 0 else float(a.mean())


def mean_ci(xs, seed: int = 11) -> dict[str, Any]:
    a = _arr(xs)
    if a.size == 0:
        return {"n": 0, "mean": None, "ci_lower": None, "ci_upper": None}
    if a.size < 2:
        return {"n": int(a.size), "mean": float(a.mean()), "ci_lower": None, "ci_upper": None}
    try:
        from research.harness import block_bootstrap_mean_ci
        ci = block_bootstrap_mean_ci(a, n_boot=400, seed=seed)
        return {"n": int(a.size), "mean": float(a.mean()),
                "ci_lower": ci.get("ci_lower"), "ci_upper": ci.get("ci_upper"),
                "n_eff": ci.get("n_eff")}
    except Exception:
        se = float(a.std(ddof=1) / math.sqrt(a.size))
        return {"n": int(a.size), "mean": float(a.mean()),
                "ci_lower": float(a.mean() - 1.96 * se),
                "ci_upper": float(a.mean() + 1.96 * se)}


def spearman(xs, ys) -> dict[str, Any]:
    x, y = [], []
    for a, b in zip(xs, ys):
        if a is None or b is None:
            continue
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if fa != fa or fb != fb:
            continue
        x.append(fa)
        y.append(fb)
    if len(x) < 8:
        return {"n": len(x), "rho": None, "p": None}
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y)
        return {"n": len(x), "rho": float(r.statistic), "p": float(r.pvalue)}
    except Exception:
        rx = _rank(np.asarray(x))
        ry = _rank(np.asarray(y))
        rho = float(np.corrcoef(rx, ry)[0, 1])
        return {"n": len(x), "rho": rho, "p": None}


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    return ranks


def pearson(xs, ys) -> float | None:
    x, y = [], []
    for a, b in zip(xs, ys):
        if a is None or b is None:
            continue
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if fa != fa or fb != fb:
            continue
        x.append(fa)
        y.append(fb)
    if len(x) < 8:
        return None
    c = np.corrcoef(np.asarray(x), np.asarray(y))[0, 1]
    return None if c != c else float(c)


def residualize(y, x) -> list[float | None]:
    """OLS residual of y on [1, x]. Returns None where either is missing."""
    pairs = []
    idx = []
    for i, (a, b) in enumerate(zip(y, x)):
        if a is None or b is None:
            continue
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if fa != fa or fb != fb:
            continue
        pairs.append((fa, fb))
        idx.append(i)
    out: list[float | None] = [None] * len(y)
    if len(pairs) < 8:
        return out
    yy = np.asarray([p[0] for p in pairs], dtype=float)
    xx = np.asarray([p[1] for p in pairs], dtype=float)
    A = np.column_stack([np.ones(len(xx)), xx])
    try:
        beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
        resid = yy - A @ beta
        for i, r in zip(idx, resid):
            out[i] = float(r)
    except Exception:
        return out
    return out


def pf(xs) -> float | None:
    a = _arr(xs)
    if a.size == 0:
        return None
    gp = float(a[a > 0].sum())
    gl = float(-a[a < 0].sum())
    if gl <= 0:
        return None if gp <= 0 else float("inf")
    return gp / gl


def win_rate(rows: Iterable[dict]) -> float | None:
    closed = [r for r in rows if r.get("outcome") in {"WIN", "LOSS"}]
    if not closed:
        return None
    return sum(1 for r in closed if r.get("outcome") == "WIN") / len(closed)


def baseline_block(rows: list[dict[str, Any]], *, years: float | None = None) -> dict[str, Any]:
    rs = [r.get("net_r") for r in rows]
    a = _arr(rs)
    wins = [r["net_r"] for r in rows if r.get("outcome") == "WIN"]
    losses = [r["net_r"] for r in rows if r.get("outcome") == "LOSS"]
    maes = [r.get("mae_r") for r in rows]
    mfes = [r.get("mfe_r") for r in rows]
    n = len(rows)
    closed = sum(1 for r in rows if r.get("outcome") in {"WIN", "LOSS"})
    ci = mean_ci(rs)
    firsts = sorted({r.get("as_of") for r in rows if r.get("as_of")})
    span_years = None
    if firsts:
        span_years = max(0.25, (pd_days(firsts[-1], firsts[0]) / 365.25))
    yr = years if years is not None else span_years
    dd = None
    if a.size:
        eq = np.cumsum(a)
        peak = np.maximum.accumulate(eq)
        dd = float((eq - peak).min())
    return {
        "n": n,
        "closed": closed,
        "expectancy_r": ci,
        "win_rate": win_rate(rows),
        "pf": pf(rs),
        "mae": _mean(maes),
        "mfe": _mean(mfes),
        "stop_before_1r": _mean([1.0 if r.get("stop_before_1r") else 0.0 for r in rows]),
        "hit_1r": _mean([1.0 if r.get("hit_1r") else 0.0 for r in rows]),
        "hit_2r": _mean([1.0 if r.get("hit_2r") else 0.0 for r in rows]),
        "avg_winner": _mean(wins),
        "avg_loser": _mean(losses),
        "drawdown_proxy_r": dd,
        "frequency_per_year": None if not yr else n / yr,
        "bottom_decile_r": None if a.size < 10 else float(np.quantile(a, 0.10)),
    }


def pd_days(a: str, b: str) -> float:
    from datetime import date
    da = date.fromisoformat(str(a)[:10])
    db = date.fromisoformat(str(b)[:10])
    return abs((da - db).days)


def _group(rows, key_fn):
    out = defaultdict(list)
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        out[k].append(r)
    return dict(out)


def classify_family_feature(
    *,
    n: int,
    year_deltas: dict[str, float | None],
    overall_delta: float | None,
    residual_rho: float | None,
    residual_p: float | None,
    tail_improved: bool,
    rank_spread: float | None,
) -> str:
    """Prespecified policy. Evidence fills the cell; the brief's example is not a target."""
    if n < MIN_N:
        return "INSUFFICIENT_DATA"
    signed = [v for v in year_deltas.values() if v is not None]
    if len(signed) < 3:
        return "INSUFFICIENT_DATA"
    pos = sum(1 for v in signed if v > 0)
    neg = sum(1 for v in signed if v < 0)
    redundant = (
        residual_rho is not None
        and abs(residual_rho) < 0.05
        and (residual_p is None or residual_p > 0.10)
    )
    if overall_delta is not None and overall_delta < -0.05 and neg >= 4:
        return "NEGATIVE"
    if pos >= 5 and overall_delta is not None and overall_delta > 0.03:
        if redundant:
            return "REDUNDANT"
        if rank_spread is not None and rank_spread > 0:
            return "POSITIVE_RANK_FEATURE"
        return "POSITIVE_RANK_FEATURE"
    if tail_improved and (overall_delta is None or overall_delta <= 0.05):
        return "RISK_FILTER_VALUE"
    if redundant and (overall_delta is None or abs(overall_delta) <= 0.05):
        return "REDUNDANT"
    if 2 <= pos <= 4 and neg >= 2:
        return "UNSTABLE"
    if tail_improved:
        return "RISK_FILTER_VALUE"
    return "UNSTABLE"


def _delta_mean(hi: list[dict], lo: list[dict]) -> float | None:
    a, b = _mean([r.get("net_r") for r in hi]), _mean([r.get("net_r") for r in lo])
    if a is None or b is None:
        return None
    return a - b


def _tail_rate(rows: list[dict], q: float, all_r: np.ndarray) -> float | None:
    if all_r.size < 10 or not rows:
        return None
    thr = float(np.quantile(all_r, q))
    hits = sum(1 for r in rows if r.get("net_r") is not None and float(r["net_r"]) <= thr)
    return hits / len(rows)


def family_feature_study(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    if feature == "trend":
        key = lambda r: r.get("trend_bucket")
        cont = "n_structure_passed"
        strong = [r for r in rows if r.get("structure_pass") is True]
        weak = [r for r in rows if r.get("trend_bucket") == "non"]
        buckets = {b: [r for r in rows if r.get("trend_bucket") == b] for b in TREND_BUCKETS}
    else:
        key = lambda r: r.get("rs_bucket")
        cont = "rs_percentile"
        strong = [r for r in rows if r.get("rs_ge_70") is True]
        weak = [r for r in rows if r.get("rs_percentile") is not None and r["rs_percentile"] < 50]
        buckets = {b: [r for r in rows if r.get("rs_bucket") == b] for b in RS_BUCKETS}

    year_deltas = {}
    by_year = _group(rows, lambda r: r.get("year"))
    for y, yr in by_year.items():
        if feature == "trend":
            hi = [r for r in yr if r.get("structure_pass") is True or (r.get("n_structure_passed") or 0) >= 6]
            lo = [r for r in yr if (r.get("n_structure_passed") or 0) <= 3]
        else:
            hi = [r for r in yr if r.get("rs_percentile") is not None and r["rs_percentile"] >= 80]
            lo = [r for r in yr if r.get("rs_percentile") is not None and r["rs_percentile"] < 50]
        year_deltas[y] = _delta_mean(hi, lo)

    overall_delta = _delta_mean(strong, weak)
    sp = spearman([r.get(cont) for r in rows], [r.get("net_r") for r in rows])
    resid = residualize([r.get("net_r") for r in rows], [r.get("mom_score") for r in rows])
    rsp = spearman([r.get(cont) for r in rows], resid)
    all_r = _arr([r.get("net_r") for r in rows])
    tail_s = _tail_rate(strong, 0.10, all_r)
    tail_w = _tail_rate(weak, 0.10, all_r)
    tail_improved = (
        tail_s is not None and tail_w is not None and tail_s < tail_w - 0.02
    ) or (
        _mean([r.get("mae_r") for r in strong]) is not None
        and _mean([r.get("mae_r") for r in weak]) is not None
        and _mean([r.get("mae_r") for r in strong]) < _mean([r.get("mae_r") for r in weak]) - 0.05
    )
    rank_spread = overall_delta
    cls = classify_family_feature(
        n=len(rows),
        year_deltas=year_deltas,
        overall_delta=overall_delta,
        residual_rho=rsp.get("rho"),
        residual_p=rsp.get("p"),
        tail_improved=bool(tail_improved),
        rank_spread=rank_spread,
    )
    return {
        "n": len(rows),
        "baseline": baseline_block(rows),
        "buckets": {k: baseline_block(v) for k, v in buckets.items()},
        "strong_vs_weak_delta_r": overall_delta,
        "spearman": sp,
        "residual_after_mom": rsp,
        "year_deltas": year_deltas,
        "tail_rate_strong": tail_s,
        "tail_rate_weak": tail_w,
        "stop_before_1r_strong": _mean([1.0 if r.get("stop_before_1r") else 0.0 for r in strong]),
        "stop_before_1r_weak": _mean([1.0 if r.get("stop_before_1r") else 0.0 for r in weak]),
        "mae_strong": _mean([r.get("mae_r") for r in strong]),
        "mae_weak": _mean([r.get("mae_r") for r in weak]),
        "classification": cls,
        "n_strong": len(strong),
        "n_weak": len(weak),
    }


def joint_abcd(rows: list[dict[str, Any]]) -> dict[str, Any]:
    a = rows
    b = [r for r in rows if r.get("structure_pass") is True]
    c = [r for r in rows if r.get("rs_ge_70") is True]
    d = [r for r in rows if r.get("structure_pass") is True and r.get("rs_ge_70") is True]
    return {
        "A_strategy_alone": baseline_block(a),
        "B_plus_strong_trend": baseline_block(b),
        "C_plus_strong_rs": baseline_block(c),
        "D_trend_and_rs": baseline_block(d),
        "note": "Attribution only. Not a candidate production rule.",
    }


def ranking_study(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_date = _group(events, lambda r: r.get("as_of"))
    spreads = {"score": [], "trend": [], "rs": [], "score_plus_rs": []}
    topk = {"score": [], "rs": [], "trend": []}
    n_days = 0
    for _, day in by_date.items():
        if len(day) < 8:
            continue
        n_days += 1
        k = max(2, len(day) // 5)

        def spread(key, default=0.0):
            ranked = sorted(day, key=lambda r: (r.get(key) if r.get(key) is not None else default), reverse=True)
            top = ranked[:k]
            bot = ranked[-k:]
            return _delta_mean(top, bot)

        def feat(r, name):
            if name == "score":
                return r.get("score")
            if name == "trend":
                return (r.get("trend") or {}).get("n_structure_passed")
            if name == "rs":
                return (r.get("rs") or {}).get("rs_percentile")
            if name == "score_plus_rs":
                s = r.get("score")
                p = (r.get("rs") or {}).get("rs_percentile")
                if s is None or p is None:
                    return None
                return float(s) + float(p) / 10.0
            return None

        for name in spreads:
            vals = [(feat(r, name), r) for r in day]
            vals = [(v, r) for v, r in vals if v is not None]
            if len(vals) < 8:
                continue
            vals.sort(key=lambda t: t[0], reverse=True)
            top = [t[1] for t in vals[:k]]
            bot = [t[1] for t in vals[-k:]]
            dlt = _delta_mean(top, bot)
            if dlt is not None:
                spreads[name].append(dlt)
            if name in topk:
                hits = [1.0 if (t[1].get("net_r") or 0) > 0 else 0.0 for t in vals[:k]]
                topk[name].append(_mean(hits))

    out = {
        "n_rank_days": n_days,
        "top_minus_bottom": {k: mean_ci(v) for k, v in spreads.items()},
        "precision_top_quintile": {k: _mean(v) for k, v in topk.items()},
        "note": "Within-day ranks among simultaneous fires. Research only.",
    }
    # family-conditional rank: breakout-only days
    return out


def incremental_after_existing(events: list[dict[str, Any]]) -> dict[str, Any]:
    y = [e.get("net_r") for e in events]
    score = [e.get("score") for e in events]
    mom = [e.get("mom_score") for e in events]
    mom5 = [e.get("momentum_5d") for e in events]
    trend = [(e.get("trend") or {}).get("n_structure_passed") for e in events]
    rs = [(e.get("rs") or {}).get("rs_percentile") for e in events]
    resid_score = residualize(y, score)
    resid_mom = residualize(y, mom)
    return {
        "corr_rs_mom_score": pearson(rs, mom),
        "corr_rs_momentum_5d": pearson(rs, mom5),
        "corr_trend_mom_score": pearson(trend, mom),
        "corr_trend_rs": pearson(trend, rs),
        "corr_score_r": spearman(score, y),
        "corr_trend_r": spearman(trend, y),
        "corr_rs_r": spearman(rs, y),
        "trend_after_score": spearman(trend, resid_score),
        "rs_after_score": spearman(rs, resid_score),
        "trend_after_mom": spearman(trend, resid_mom),
        "rs_after_mom": spearman(rs, resid_mom),
    }


def temporal_blocks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_year = {y: [r for r in rows if r.get("year") == y] for y in YEARS}
    rolling = {}
    # 2-year overlapping blocks
    pairs = [("2020-2021", ("2020", "2021")), ("2021-2022", ("2021", "2022")),
             ("2022-2023", ("2022", "2023")), ("2023-2024", ("2023", "2024")),
             ("2024-2025", ("2024", "2025")), ("2025-2026", ("2025", "2026"))]
    for name, ys in pairs:
        chunk = [r for r in rows if r.get("year") in ys]
        rolling[name] = baseline_block(chunk)
    return {
        "by_year": {y: baseline_block(v) for y, v in by_year.items()},
        "rolling_2y": rolling,
    }


def run_analysis(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = explode_families(events)
    by_fam = _group(rows, lambda r: r.get("family"))
    baselines = {}
    trend_study = {}
    rs_study = {}
    joint = {}
    policy = {}
    for fam in FAMILY_KEYS:
        fr = by_fam.get(fam, [])
        baselines[fam] = baseline_block(fr)
        trend_study[fam] = family_feature_study(fr, "trend")
        rs_study[fam] = family_feature_study(fr, "rs")
        joint[fam] = joint_abcd(fr)
        policy[fam] = {
            "trend": trend_study[fam]["classification"],
            "rs": rs_study[fam]["classification"],
            "n": len(fr),
        }

    # category rollups (reported after family-level, never as a substitute)
    by_cat = _group(rows, lambda r: r.get("family_category"))
    cat_base = {c: baseline_block(v) for c, v in by_cat.items()}

    brk = [r for r in rows if r.get("family") in BREAKOUT_KEYS]
    mom = [r for r in rows if r.get("family") in MOMENTUM_KEYS]

    tests = []
    h1 = spearman([r.get("n_structure_passed") for r in brk], [r.get("net_r") for r in brk])
    tests.append({"id": "H1", "p": h1.get("p"), "stat": h1.get("rho"), "n": h1.get("n")})
    h2 = spearman([r.get("rs_percentile") for r in brk + mom], [r.get("net_r") for r in brk + mom])
    tests.append({"id": "H2", "p": h2.get("p"), "stat": h2.get("rho"), "n": h2.get("n")})

    strong_t = [r for r in rows if r.get("structure_pass") is True]
    weak_t = [r for r in rows if r.get("trend_bucket") == "non"]
    all_r = _arr([r.get("net_r") for r in rows])
    tail_s = _tail_rate(strong_t, 0.10, all_r)
    tail_w = _tail_rate(weak_t, 0.10, all_r)
    # two-proportion z as a simple H3 p
    h3_p = None
    if tail_s is not None and tail_w is not None and strong_t and weak_t:
        p1, p0 = tail_s, tail_w
        n1, n0 = len(strong_t), len(weak_t)
        p = (p1 * n1 + p0 * n0) / (n1 + n0)
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n0)) if 0 < p < 1 else 0.0
        if se > 0:
            z = (p1 - p0) / se
            h3_p = float(2 * (1 - _phi(abs(z))))
    tests.append({"id": "H3", "p": h3_p, "stat": None if tail_s is None or tail_w is None else tail_s - tail_w,
                  "n": len(rows)})

    inc = incremental_after_existing(events)
    tests.append({"id": "H4_trend", "p": (inc["trend_after_mom"] or {}).get("p"),
                  "stat": (inc["trend_after_mom"] or {}).get("rho"),
                  "n": (inc["trend_after_mom"] or {}).get("n")})
    tests.append({"id": "H4_rs", "p": (inc["rs_after_mom"] or {}).get("p"),
                  "stat": (inc["rs_after_mom"] or {}).get("rho"),
                  "n": (inc["rs_after_mom"] or {}).get("n")})

    # H5: heterogeneity of family-level trend deltas
    fam_deltas = [trend_study[f]["strong_vs_weak_delta_r"] for f in FAMILY_KEYS
                  if trend_study[f]["strong_vs_weak_delta_r"] is not None and trend_study[f]["n"] >= MIN_N]
    h5_stat = None
    if len(fam_deltas) >= 4:
        h5_stat = float(np.std(fam_deltas))
    tests.append({"id": "H5", "p": None, "stat": h5_stat, "n": len(fam_deltas),
                  "note": "heterogeneity = std of family trend deltas; exploratory if no p"})

    pvals = [t["p"] for t in tests if t.get("p") is not None]
    fdr = None
    try:
        from research.harness import benjamini_hochberg
        if pvals:
            raw = [t["p"] if t.get("p") is not None else 1.0 for t in tests]
            bh = benjamini_hochberg(raw, alpha=FDR_Q)
            fdr = {
                "q": FDR_Q,
                "threshold": bh.get("threshold"),
                "n_rejected": int(bh.get("n_rejected") or 0),
                "qvalues": [float(x) for x in bh.get("qvalues")],
            }
            for t, q, rej in zip(tests, bh.get("qvalues"), bh.get("rejected")):
                t["q"] = float(q)
                t["fdr_reject"] = bool(rej)
    except Exception:
        fdr = {"error": "harness_unavailable"}

    # final feature status (prespecified aggregation)
    trend_classes = [policy[f]["trend"] for f in FAMILY_KEYS if policy[f]["n"] >= MIN_N]
    rs_classes = [policy[f]["rs"] for f in FAMILY_KEYS if policy[f]["n"] >= MIN_N]
    final_trend = _final_status(trend_classes, inc.get("trend_after_mom"), "trend")
    final_rs = _final_status(rs_classes, inc.get("rs_after_mom"), "rs")

    return {
        "claim_class": "EXPLANATORY",
        "n_events": len(events),
        "n_family_rows": len(rows),
        "n_primary_hypotheses": 5,
        "n_tested": len(tests),
        "baselines": baselines,
        "category_baselines": cat_base,
        "trend_study": trend_study,
        "rs_study": rs_study,
        "joint": joint,
        "policy": policy,
        "ranking": ranking_study(events),
        "incremental": inc,
        "temporal_events": temporal_blocks(events),
        "temporal_rows": temporal_blocks(rows),
        "hypotheses": tests,
        "fdr": fdr,
        "final_trend": final_trend,
        "final_rs": final_rs,
        "h1": h1,
        "h2": h2,
    }


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _final_status(classes: list[str], after_mom: dict | None, which: str) -> str:
    if not classes:
        return "KEEP RESEARCH-ONLY"
    pos = classes.count("POSITIVE_RANK_FEATURE")
    risk = classes.count("RISK_FILTER_VALUE")
    neg = classes.count("NEGATIVE")
    uns = classes.count("UNSTABLE")
    red = classes.count("REDUNDANT")
    n = len(classes)
    rho = (after_mom or {}).get("rho")
    if neg >= max(3, n // 2) and pos == 0:
        return "RETIRE"
    if pos >= 3 and (rho is None or abs(rho) >= 0.05):
        return "FORWARD-VALIDATE AS RANK FEATURE"
    if risk >= 3 and pos <= 1:
        return "FORWARD-VALIDATE AS RISK FILTER"
    if red >= n - 2 and pos == 0:
        return "KEEP RESEARCH-ONLY"
    if uns >= n // 2:
        return "KEEP RESEARCH-ONLY"
    if pos >= 1 and which == "rs":
        return "KEEP RESEARCH-ONLY"
    return "KEEP RESEARCH-ONLY"
