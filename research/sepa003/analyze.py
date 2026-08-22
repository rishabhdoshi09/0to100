"""Explanatory stats for SEPA-003. No production model. No new OOS claim."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

from research.sepa003.constants import (
    FDR_Q,
    MIN_N,
    R2_DIR,
    RS_BUCKETS,
    WEAK_ERA_START,
    WIN_ERA_END,
)


def _arr(xs) -> np.ndarray:
    return np.asarray([float(x) for x in xs if x is not None and x == x], dtype=float)


def mean_ci(xs, seed: int = 7) -> dict[str, Any]:
    a = _arr(xs)
    if a.size < 2:
        return {"n": int(a.size), "mean": None if a.size == 0 else float(a.mean()),
                "ci_lower": None, "ci_upper": None}
    try:
        from research.harness import block_bootstrap_mean_ci
        ci = block_bootstrap_mean_ci(a, n_boot=600, seed=seed)
        return {"n": int(a.size), "mean": float(a.mean()),
                "ci_lower": ci.get("ci_lower"), "ci_upper": ci.get("ci_upper"),
                "n_eff": ci.get("n_eff")}
    except Exception:
        return {"n": int(a.size), "mean": float(a.mean()), "ci_lower": None, "ci_upper": None}


def cliffs_delta(a, b) -> float | None:
    x, y = _arr(a), _arr(b)
    if x.size == 0 or y.size == 0:
        return None
    # P(x>y) - P(x<y)
    gt = sum(float(np.sum(xi > y)) for xi in x)
    lt = sum(float(np.sum(xi < y)) for xi in x)
    return float((gt - lt) / (x.size * y.size))


def describe(xs) -> dict[str, Any]:
    a = _arr(xs)
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "q1": float(np.quantile(a, 0.25)),
        "q3": float(np.quantile(a, 0.75)),
    }


def mw_p(a, b) -> float | None:
    x, y = _arr(a), _arr(b)
    if x.size < 5 or y.size < 5:
        return None
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return None


def classify_feature(year_means: dict[str, float], overall_sign: int, n: int) -> str:
    if n < MIN_N:
        return "INSUFFICIENT_DATA"
    signed = [v for v in year_means.values() if v is not None]
    if len(signed) < 3:
        return "INSUFFICIENT_DATA"
    pos = sum(1 for v in signed if v > 0)
    neg = sum(1 for v in signed if v < 0)
    if pos == len(signed) and overall_sign > 0:
        return "ROBUST_POSITIVE"
    if neg == len(signed) and overall_sign < 0:
        return "NO_SIGNAL"
    if pos >= len(signed) - 1 and overall_sign > 0:
        return "CONTEXT_DEPENDENT"
    if pos and neg:
        return "UNSTABLE"
    return "NO_SIGNAL"


def load_r2_ladder() -> dict[str, Any]:
    path = R2_DIR / "ablation_001r2.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out = {"variants": {}, "funnel_snapshots": raw.get("funnel_snapshots"),
           "funnel_unique": raw.get("funnel_unique"),
           "diagnostics": raw.get("diagnostics")}
    for k, v in (raw.get("variants") or {}).items():
        if not isinstance(v, dict):
            continue
        out["variants"][k] = {
            "n_raw": v.get("n_raw_signal_days"),
            "n_deduped": v.get("n_deduped") or v.get("n"),
            "expectancy_r": v.get("expectancy_r"),
            "statistical_verdict": v.get("statistical_verdict"),
            "by_year": v.get("by_year"),
            "walk_forward": {
                blk: {
                    "n": (vv or {}).get("n"),
                    "expectancy_r": (vv or {}).get("expectancy_r"),
                    "statistical_verdict": (vv or {}).get("statistical_verdict"),
                }
                for blk, vv in (v.get("walk_forward") or {}).items()
                if isinstance(vv, dict)
            },
        }
    return out


def by_key(rows: Sequence[dict], key: str, outcome: str = "net_r") -> dict[str, Any]:
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        k = r.get(key)
        if k is None or k == "" or r.get(outcome) is None:
            continue
        buckets[str(k)].append(r[outcome])
    return {k: mean_ci(v) for k, v in sorted(buckets.items())}


def era_shift(rows: Sequence[dict], fields: Sequence[str]) -> dict[str, Any]:
    win = [r for r in rows if str(r.get("as_of") or r.get("entry_date") or "") <= WIN_ERA_END]
    weak = [r for r in rows if str(r.get("as_of") or r.get("entry_date") or "") >= WEAK_ERA_START]
    out = {"n_win": len(win), "n_weak": len(weak), "fields": {}}
    for f in fields:
        a = [r.get(f) for r in win]
        b = [r.get(f) for r in weak]
        out["fields"][f] = {
            "winning_era": describe(a),
            "weak_era": describe(b),
            "cliffs_delta": cliffs_delta(a, b),
            "mw_p": mw_p(a, b),
            "outcome_win": mean_ci([r.get("net_r") for r in win if r.get("net_r") is not None]),
            "outcome_weak": mean_ci([r.get("net_r") for r in weak if r.get("net_r") is not None]),
        }
    out["regime_mix_win"] = _counts(win, "regime_entry")
    out["regime_mix_weak"] = _counts(weak, "regime_entry")
    out["sector_coverage_win"] = _coverage(win)
    out["sector_coverage_weak"] = _coverage(weak)
    return out


def _counts(rows, key) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for r in rows:
        c[str(r.get(key) or "UNKNOWN")] += 1
    return dict(c)


def _coverage(rows) -> dict[str, Any]:
    n = len(rows)
    known = sum(1 for r in rows if r.get("sector") not in (None, "", "UNKNOWN"))
    return {"n": n, "mapped": known, "pct_mapped": None if not n else round(100.0 * known / n, 2)}


def quartile_outcome(rows: Sequence[dict], field: str, outcome: str = "net_r") -> dict[str, Any]:
    usable = [r for r in rows if r.get(field) is not None and r.get(outcome) is not None]
    if len(usable) < MIN_N:
        return {"n": len(usable), "class": "INSUFFICIENT_DATA"}
    vals = _arr([r[field] for r in usable])
    qs = np.quantile(vals, [0.25, 0.5, 0.75])
    bins = {"Q1_low": [], "Q2": [], "Q3": [], "Q4_high": []}
    for r in usable:
        v = float(r[field])
        if v <= qs[0]:
            bins["Q1_low"].append(r[outcome])
        elif v <= qs[1]:
            bins["Q2"].append(r[outcome])
        elif v <= qs[2]:
            bins["Q3"].append(r[outcome])
        else:
            bins["Q4_high"].append(r[outcome])
    packed = {k: mean_ci(v) for k, v in bins.items()}
    means = [packed[k]["mean"] for k in ("Q1_low", "Q2", "Q3", "Q4_high") if packed[k]["mean"] is not None]
    mono = None
    if len(means) == 4:
        mono = all(means[i] <= means[i + 1] for i in range(3)) or all(means[i] >= means[i + 1] for i in range(3))
    year_effect = {}
    for y in sorted({str(r.get("year")) for r in usable}):
        yr = [r for r in usable if str(r.get("year")) == y]
        if len(yr) < MIN_N:
            continue
        low = [r[outcome] for r in yr if float(r[field]) <= np.median(_arr([x[field] for x in yr]))]
        high = [r[outcome] for r in yr if float(r[field]) > np.median(_arr([x[field] for x in yr]))]
        year_effect[y] = (float(np.mean(high)) - float(np.mean(low))) if low and high else None
    overall = 0
    if packed["Q4_high"]["mean"] is not None and packed["Q1_low"]["mean"] is not None:
        overall = 1 if packed["Q4_high"]["mean"] > packed["Q1_low"]["mean"] else -1
    return {
        "n": len(usable),
        "quartiles": packed,
        "monotonic": mono,
        "year_high_minus_low": year_effect,
        "class": classify_feature(year_effect, overall, len(usable)),
        "q_cuts": [float(x) for x in qs],
    }


def matched_vcp(f_rows, controls, outcome="fwd_pct_20") -> dict[str, Any]:
    """Stratify on year + RS bucket + regime; VCP vs Stage-2+RS no-VCP."""
    def key(r):
        return (str(r.get("year")), str(r.get("rs_bucket")), str(r.get("regime_entry") or r.get("regime")))

    treat = [r for r in f_rows if r.get("vcp_detected") and r.get(outcome) is not None and not r.get("ca_censored")]
    ctrl = [r for r in controls if r.get("is_control") and r.get(outcome) is not None]
    by_t, by_c = defaultdict(list), defaultdict(list)
    for r in treat:
        by_t[key(r)].append(r[outcome])
    for r in ctrl:
        by_c[key(r)].append(r[outcome])
    diffs = []
    detail = []
    for k in sorted(set(by_t) & set(by_c)):
        if len(by_t[k]) < 5 or len(by_c[k]) < 5:
            continue
        d = float(np.mean(by_t[k]) - np.mean(by_c[k]))
        diffs.append(d)
        detail.append({"stratum": k, "n_vcp": len(by_t[k]), "n_ctrl": len(by_c[k]), "diff": d})
    return {
        "n_strata": len(detail),
        "mean_diff": None if not diffs else float(np.mean(diffs)),
        "n_vcp": len(treat),
        "n_ctrl": len(ctrl),
        "strata": detail[:40],
        "note": "NEW_HYPOTHESIS — not VALIDATED_EDGE",
    }


def huber_explain(rows: Sequence[dict], fields: Sequence[str], y="net_r") -> dict[str, Any]:
    usable = [r for r in rows if r.get(y) is not None]
    if len(usable) < 80:
        return {"n": len(usable), "note": "INSUFFICIENT_DATA"}
    X = []
    Y = []
    used = []
    for f in fields:
        col = [r.get(f) for r in usable]
        if sum(v is not None for v in col) < 80:
            continue
        used.append(f)
    if not used:
        return {"n": len(usable), "note": "no_fields"}
    med = {f: float(np.nanmedian(_arr([r.get(f) for r in usable]))) for f in used}
    for r in usable:
        row = []
        for f in used:
            v = r.get(f)
            row.append(med[f] if v is None else float(v))
        X.append(row)
        Y.append(float(r[y]))
    Xn = np.asarray(X, dtype=float)
    Yn = np.asarray(Y, dtype=float)
    # standardize
    mu, sd = Xn.mean(0), Xn.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (Xn - mu) / sd
    try:
        from sklearn.linear_model import HuberRegressor
        model = HuberRegressor().fit(Z, Yn)
        coef = {f: float(c) for f, c in zip(used, model.coef_)}
        return {"n": int(len(Yn)), "coef_std": coef, "intercept": float(model.intercept_),
                "note": "explanatory only; not deployed"}
    except Exception as exc:
        # OLS fallback
        try:
            beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z], Yn, rcond=None)
            coef = {f: float(c) for f, c in zip(used, beta[1:])}
            return {"n": int(len(Yn)), "coef_std": coef, "intercept": float(beta[0]),
                    "note": f"OLS fallback ({exc}); not deployed"}
        except Exception as exc2:
            return {"n": int(len(Yn)), "error": str(exc2)}


def primary_tests(f_rows, controls, g_panel, ladder) -> dict[str, Any]:
    fills = [r for r in f_rows if r.get("net_r") is not None and not r.get("ca_censored")]
    tests = []

    # H1 — cite R2.1 A vs B plus reconstructed MAE vs G
    a = ((ladder.get("variants") or {}).get("A") or {})
    b = ((ladder.get("variants") or {}).get("B") or {})
    h1_p = None
    tests.append({
        "id": "H1",
        "r2_A_expectancy": a.get("expectancy_r"),
        "r2_B_expectancy": b.get("expectancy_r"),
        "r2_A_confirmation": ((a.get("walk_forward") or {}).get("confirmation") or {}),
        "r2_B_confirmation": ((b.get("walk_forward") or {}).get("confirmation") or {}),
        "direction": "B loses less than A on pooled R; confirmation both REJECT",
        "p": h1_p,
        "label": "adverse_selection_reducer_candidate",
    })

    # H2 — RS buckets on G 20d % and F net R
    g_live = [r for r in g_panel if r.get("fwd_20d_pct") is not None and not r.get("ca_censored")]
    h2_g = {bkt: mean_ci([r["fwd_20d_pct"] for r in g_live if r.get("rs_bucket") == bkt]) for bkt in RS_BUCKETS}
    h2_f = {bkt: mean_ci([r["net_r"] for r in fills if r.get("rs_bucket") == bkt]) for bkt in RS_BUCKETS}
    # trend: 95-99 vs 50-69
    hi = [r["fwd_20d_pct"] for r in g_live if r.get("rs_bucket") == "95-99"]
    lo = [r["fwd_20d_pct"] for r in g_live if r.get("rs_bucket") == "50-69"]
    tests.append({
        "id": "H2",
        "g_buckets": h2_g,
        "f_buckets": h2_f,
        "p": mw_p(hi, lo),
        "year_g": {y: mean_ci([r["fwd_20d_pct"] for r in g_live if r.get("year") == y])
                   for y in sorted({str(r.get("year")) for r in g_live})},
    })

    # H3
    c = ((ladder.get("variants") or {}).get("C") or {})
    d = ((ladder.get("variants") or {}).get("D") or {})
    matched = matched_vcp(fills, controls)
    tests.append({
        "id": "H3",
        "r2_C": c.get("expectancy_r"),
        "r2_D": d.get("expectancy_r"),
        "r2_C_confirmation": ((c.get("walk_forward") or {}).get("confirmation") or {}),
        "r2_D_confirmation": ((d.get("walk_forward") or {}).get("confirmation") or {}),
        "matched_fwd20": matched,
        "p": None,
    })

    # H4 tightness of final contraction — lower depth should be better if claim holds
    q4 = quartile_outcome(fills, "final_contraction_pct")
    tests.append({"id": "H4", "quartile": q4, "p": q4.get("quartiles", {}).get("Q1_low", {}).get("n")})

    # H5 dry-up — lower ratio = more dry-up
    q5 = quartile_outcome(fills, "dry_up_ratio")
    tests.append({"id": "H5", "quartile": q5})

    # H6 pivot extension vs MAE
    q6 = quartile_outcome(fills, "distance_from_pivot_pct", outcome="mae_r")
    tests.append({"id": "H6", "quartile_mae": q6,
                  "fail_by_quartile": quartile_outcome(fills, "distance_from_pivot_pct", outcome="net_r")})

    # H7 regime
    by_reg = by_key(fills, "regime_entry")
    tests.append({"id": "H7", "by_regime": by_reg, "n_unknown": sum(1 for r in fills if r.get("regime_entry") in (None, "UNKNOWN"))})

    # H8 sector
    cov = _coverage(fills)
    lead = [r["net_r"] for r in fills if r.get("sector") not in (None, "UNKNOWN") and (r.get("sector_rs") or 0) >= 70]
    weak = [r["net_r"] for r in fills if r.get("sector") not in (None, "UNKNOWN") and r.get("sector_rs") is not None and r.get("sector_rs") < 40]
    tests.append({
        "id": "H8",
        "coverage": cov,
        "leading_group": mean_ci(lead),
        "weak_group": mean_ci(weak),
        "p": mw_p(lead, weak),
        "insufficient": cov.get("pct_mapped") is not None and cov["pct_mapped"] < 50,
    })

    pvals = []
    ids = []
    for t in tests:
        if t["id"] == "H2" and t.get("p") is not None:
            pvals.append(t["p"]); ids.append("H2")
        elif t["id"] == "H8" and t.get("p") is not None:
            pvals.append(t["p"]); ids.append("H8")
        elif t["id"] == "H4":
            a = (q4.get("quartiles") or {}).get("Q1_low", {})
            b = (q4.get("quartiles") or {}).get("Q4_high", {})
            pv = mw_p(
                [r["net_r"] for r in fills if r.get("final_contraction_pct") is not None
                 and r["final_contraction_pct"] <= (q4.get("q_cuts") or [np.inf])[0]],
                [r["net_r"] for r in fills if r.get("final_contraction_pct") is not None
                 and r["final_contraction_pct"] > (q4.get("q_cuts") or [0, 0, 0])[-1]],
            )
            t["p"] = pv
            if pv is not None:
                pvals.append(pv); ids.append("H4")
        elif t["id"] == "H5":
            cuts = q5.get("q_cuts") or []
            if len(cuts) == 3:
                pv = mw_p(
                    [r["net_r"] for r in fills if r.get("dry_up_ratio") is not None and r["dry_up_ratio"] <= cuts[0]],
                    [r["net_r"] for r in fills if r.get("dry_up_ratio") is not None and r["dry_up_ratio"] > cuts[-1]],
                )
                t["p"] = pv
                if pv is not None:
                    pvals.append(pv); ids.append("H5")
        elif t["id"] == "H6":
            cuts = q6.get("q_cuts") or []
            if len(cuts) == 3:
                pv = mw_p(
                    [r["mae_r"] for r in fills if r.get("distance_from_pivot_pct") is not None and r["distance_from_pivot_pct"] <= cuts[0]],
                    [r["mae_r"] for r in fills if r.get("distance_from_pivot_pct") is not None and r["distance_from_pivot_pct"] > cuts[-1]],
                )
                t["p"] = pv
                if pv is not None:
                    pvals.append(pv); ids.append("H6")
        elif t["id"] == "H7":
            bull = [r["net_r"] for r in fills if r.get("regime_entry") in ("BULL", "STRONG_BULL")]
            rest = [r["net_r"] for r in fills if r.get("regime_entry") in ("CORRECTION", "BEAR", "SIDEWAYS")]
            pv = mw_p(bull, rest)
            t["p"] = pv
            if pv is not None:
                pvals.append(pv); ids.append("H7")

    fdr = None
    if pvals:
        from research.harness import benjamini_hochberg
        fdr = benjamini_hochberg(pvals, alpha=FDR_Q)
        fdr_out = {
            "ids": ids,
            "pvalues": pvals,
            "qvalues": [float(x) for x in fdr["qvalues"]],
            "rejected": [bool(x) for x in fdr["rejected"]],
            "q": FDR_Q,
            "n_primary_tested": 8,
            "n_with_pvalue": len(pvals),
        }
    else:
        fdr_out = {"ids": [], "n_primary_tested": 8, "n_with_pvalue": 0}
    return {"tests": tests, "fdr": fdr_out, "n_fills": len(fills)}


def decay_verdict(shift: dict[str, Any], by_regime: dict[str, Any], by_year: dict[str, Any]) -> str:
    """MARKET_CHANGED / POPULATION_CHANGED / UNSTABLE_EDGE / INCONCLUSIVE."""
    fields = shift.get("fields") or {}
    mix_win = shift.get("regime_mix_win") or {}
    mix_weak = shift.get("regime_mix_weak") or {}
    n_win = sum(mix_win.values()) or 1
    n_weak = sum(mix_weak.values()) or 1
    bull_win = (mix_win.get("BULL", 0) + mix_win.get("STRONG_BULL", 0)) / n_win
    bull_weak = (mix_weak.get("BULL", 0) + mix_weak.get("STRONG_BULL", 0)) / n_weak
    pop_flags = 0
    for name in ("dry_up_ratio", "final_contraction_pct", "distance_from_pivot_pct",
                 "stop_distance_pct", "breakout_gap_pct", "rs_percentile"):
        rec = fields.get(name) or {}
        d = rec.get("cliffs_delta")
        if d is not None and abs(d) >= 0.15:
            pop_flags += 1
    year_means = [v.get("mean") for v in by_year.values() if v.get("mean") is not None]
    sign_flips = 0
    if year_means:
        signs = [1 if m > 0 else -1 for m in year_means]
        sign_flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    if abs(bull_win - bull_weak) >= 0.15 and pop_flags <= 1:
        return "MARKET_CHANGED"
    if pop_flags >= 3:
        return "POPULATION_CHANGED"
    if sign_flips >= 3:
        return "UNSTABLE_EDGE"
    if abs(bull_win - bull_weak) >= 0.08 and pop_flags >= 1:
        return "INCONCLUSIVE"
    if sign_flips >= 2:
        return "UNSTABLE_EDGE"
    return "INCONCLUSIVE"


def run_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    f_rows = [r for r in (payload.get("features") or [])]
    fills = [r for r in f_rows if r.get("net_r") is not None and not r.get("ca_censored")]
    controls = payload.get("controls") or []
    g_panel = payload.get("g_panel") or []
    ladder = load_r2_ladder()
    fields = (
        "rs_percentile", "final_contraction_pct", "dry_up_ratio", "tightness",
        "distance_from_pivot_pct", "breakout_gap_pct", "stop_distance_pct",
        "contraction_count", "base_depth_pct", "dist_52w_high_pct",
        "dist_sma50_pct", "mae_r", "mfe_r",
    )
    shift = era_shift(fills, fields)
    by_year = by_key(fills, "year")
    by_reg = by_key(fills, "regime_entry")
    primary = primary_tests(f_rows, controls, g_panel, ladder)
    model = huber_explain(
        fills,
        ["rs_percentile", "final_contraction_pct", "dry_up_ratio",
         "distance_from_pivot_pct", "stop_distance_pct", "breakout_gap_pct",
         "dist_52w_high_pct", "idx_ret20"],
    )
    coverage_year = {}
    for y in sorted({str(r.get("year")) for r in fills}):
        coverage_year[y] = _coverage([r for r in fills if str(r.get("year")) == y])
    return {
        "experiment": "SEPA-003",
        "not_validated_edge": True,
        "confirmation_already_observed": True,
        "n_reconstructed_fills": len(fills),
        "reconstructed_expectancy": mean_ci([r["net_r"] for r in fills]),
        "by_year": by_year,
        "by_regime": by_reg,
        "by_rs_bucket": by_key(fills, "rs_bucket"),
        "by_sector": by_key([r for r in fills if r.get("sector") != "UNKNOWN"], "sector"),
        "decay": shift,
        "decay_verdict": decay_verdict(shift, by_reg, by_year),
        "primary": primary,
        "quartiles": {
            "final_contraction_pct": quartile_outcome(fills, "final_contraction_pct"),
            "dry_up_ratio": quartile_outcome(fills, "dry_up_ratio"),
            "distance_from_pivot_pct": quartile_outcome(fills, "distance_from_pivot_pct"),
            "stop_distance_pct": quartile_outcome(fills, "stop_distance_pct"),
            "breakout_gap_pct": quartile_outcome(fills, "breakout_gap_pct"),
            "tightness": quartile_outcome(fills, "tightness"),
        },
        "matched": matched_vcp(fills, controls),
        "huber": model,
        "sector_coverage_by_year": coverage_year,
        "r2_ladder": ladder.get("variants"),
        "index_source": payload.get("index_source"),
        "skipped": payload.get("skipped"),
        "n_controls": len(controls),
        "n_g": len(g_panel),
    }
