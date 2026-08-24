"""
🎯 Forecast calibration + the Confidence Ledger — "do our probabilities mean
anything?"

A platform that can HONESTLY show "over the last 240 forecasts we said 70%, and
68% actually happened — 2% calibration error" is trustworthy in a market full of
curve-fit hype. Almost no retail platform can, because none of them tracked
their own predictions against outcomes. QuantTerm's decision_journal already
logs every prediction's probability AND its realised outcome — so this is
buildable now, and it is a moat competitors structurally lack.

Two layers:

  1. OVERALL calibration — reliability curve, Expected Calibration Error (ECE),
     Brier score + skill. The public, honest track record.

  2. The CONFIDENCE LEDGER — conditional over/under-confidence: "when I lean on
     delivery spikes in weak breadth, I am systematically overconfident." This
     is where the system learns WHEN its confidence is trustworthy. The danger
     is obvious and we guard it explicitly: slicing many conditions is a
     multiple-testing minefield, so every conditional claim is tested with a
     Poisson-binomial calibration test and then FDR-corrected through the
     Research OS. A finding must survive that gauntlet to be reported.

Pure functions take probabilities in [0,1] and binary outcomes in {0,1}. The
I/O layer reads decisions.db (p_win is stored 0-100 → normalised; a "win" is
outcome_pct ≥ WIN_PCT). Fail-open throughout.
"""
from __future__ import annotations

import os as _os

import numpy as np
from scipy.stats import norm

_ECE_BINS = int(_os.getenv("QT_CALIB_BINS", "10") or 10)
_MIN_SLICE_N = int(_os.getenv("QT_CALIB_MIN_SLICE", "30") or 30)
_ALPHA = float(_os.getenv("QT_CALIB_ALPHA", "0.05") or 0.05)


# ══════════════════════════════════════════════════════════════════════════════
# Overall calibration
# ══════════════════════════════════════════════════════════════════════════════

def brier_score(probs, outcomes) -> dict:
    """Brier score = mean((p − outcome)²), lower is better (0 = perfect). The
    Brier SKILL score compares it to always predicting the base rate: >0 means
    the forecasts add information, ≤0 means they don't beat a constant guess."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    n = p.size
    if n == 0:
        return {"brier": 0.0, "brier_skill": 0.0, "n": 0}
    brier = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    brier_ref = base * (1 - base)                      # Brier of the base-rate guess
    skill = 1.0 - brier / brier_ref if brier_ref > 0 else 0.0
    return {"brier": brier, "brier_skill": float(skill), "n": int(n)}


def reliability_bins(probs, outcomes, n_bins: int = _ECE_BINS) -> list[dict]:
    """Reliability curve: bin forecasts by predicted probability and, per bin,
    report the mean predicted vs the observed success frequency. Perfect
    calibration → observed ≈ predicted in every bin."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        c = int(mask.sum())
        if c == 0:
            continue
        out.append({"lo": float(lo), "hi": float(hi), "count": c,
                    "mean_pred": float(p[mask].mean()),
                    "obs_freq": float(y[mask].mean())})
    return out


def expected_calibration_error(probs, outcomes, n_bins: int = _ECE_BINS) -> float:
    """ECE = count-weighted mean |observed − predicted| across reliability bins.
    0 = perfectly calibrated; the headline honesty number."""
    p = np.asarray(probs, dtype=float)
    bins = reliability_bins(probs, outcomes, n_bins)
    n = p.size
    if n == 0 or not bins:
        return 0.0
    return float(sum(b["count"] * abs(b["obs_freq"] - b["mean_pred"])
                     for b in bins) / n)


def miscalibration_test(probs, outcomes) -> dict:
    """Is a set of forecasts significantly mis-calibrated? Poisson-binomial
    normal test: under perfect calibration the expected number of successes is
    Σpᵢ with variance Σpᵢ(1−pᵢ). z = (observed − Σpᵢ)/√variance. A negative z
    = OVERCONFIDENT (fewer wins than promised); positive = underconfident.
    Correctly handles forecasts with DIFFERENT probabilities (unlike a single-p
    binomial), which is exactly the conditional-slice case."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    n = p.size
    if n == 0:
        return {"gap": 0.0, "z": 0.0, "p_value": 1.0, "n": 0,
                "direction": "", "mean_pred": 0.0, "obs_freq": 0.0}
    expected = float(p.sum())
    variance = float(np.sum(p * (1.0 - p)))
    observed = float(y.sum())
    gap = float(p.mean() - y.mean())                   # predicted − observed
    if variance <= 0:
        z, pval = 0.0, 1.0
    else:
        z = (observed - expected) / np.sqrt(variance)
        pval = float(2.0 * norm.sf(abs(z)))            # two-sided
    direction = "overconfident" if gap > 0 else "underconfident" if gap < 0 else ""
    return {"gap": gap, "z": float(z), "p_value": pval, "n": int(n),
            "direction": direction, "mean_pred": float(p.mean()),
            "obs_freq": float(y.mean())}


def calibration_summary(probs, outcomes, n_bins: int = _ECE_BINS) -> dict:
    """The full overall read + a plain-English headline (the user-facing
    sentence the kill-criterion demands)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    n = int(p.size)
    if n == 0:
        return {"n": 0, "insight": "No resolved forecasts yet to score."}
    b = brier_score(p, y)
    ece = expected_calibration_error(p, y, n_bins)
    mt = miscalibration_test(p, y)
    mean_pred, mean_obs = float(p.mean()), float(y.mean())
    quality = ("well-calibrated" if ece < 0.05 else
               "roughly calibrated" if ece < 0.10 else "poorly calibrated")
    lean = ""
    if mt["p_value"] < 0.05 and mt["direction"]:
        lean = f", and {mt['direction']} overall"
    insight = (f"Predicted {mean_pred*100:.0f}%, actual {mean_obs*100:.0f}% — "
               f"{quality} ({ece*100:.0f}% error over {n} forecasts){lean}.")
    return {"n": n, "mean_pred": round(mean_pred, 4),
            "mean_obs": round(mean_obs, 4), "gap": round(mean_pred - mean_obs, 4),
            "ece": round(ece, 4), "brier": round(b["brier"], 4),
            "brier_skill": round(b["brier_skill"], 4),
            "z": round(mt["z"], 3), "p_value": round(mt["p_value"], 4),
            "reliability_bins": reliability_bins(p, y, n_bins),
            "insight": insight}


# ══════════════════════════════════════════════════════════════════════════════
# The Confidence Ledger — conditional over/under-confidence (harness-gated)
# ══════════════════════════════════════════════════════════════════════════════

def conditional_overconfidence(probs, outcomes, groups, min_n: int = _MIN_SLICE_N,
                               alpha: float = _ALPHA) -> list[dict]:
    """Find the CONDITIONS under which the system is mis-calibrated — e.g.
    "when regime=BEAR the forecasts are overconfident". Every slice with ≥
    `min_n` resolved forecasts is tested (miscalibration_test), then the batch
    of p-values is FDR-corrected via the Research OS so we don't manufacture a
    spurious 'overconfident on Tuesdays' from noise. Returns ONLY the slices
    that survive, worst gap first.

    `groups` is a sequence (len == len(probs)) of hashable slice labels."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    g = np.asarray(groups, dtype=object)
    if p.size == 0:
        return []
    tests = []
    for key in sorted({str(k) for k in g}):
        mask = (g.astype(str) == key)
        if int(mask.sum()) < min_n:
            continue
        t = miscalibration_test(p[mask], y[mask])
        t["group"] = key
        tests.append(t)
    if not tests:
        return []
    from research.harness import benjamini_hochberg
    bh = benjamini_hochberg([t["p_value"] for t in tests], alpha=alpha)
    findings = []
    for t, keep, q in zip(tests, bh["rejected"], bh["qvalues"]):
        if not keep or not t["direction"]:
            continue
        t["qvalue"] = float(q)
        t["insight"] = (
            f"When {t['group']}: {t['direction']} — predicted "
            f"{t['mean_pred']*100:.0f}%, actual {t['obs_freq']*100:.0f}% "
            f"over {t['n']} forecasts.")
        findings.append(t)
    return sorted(findings, key=lambda t: abs(t["gap"]), reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# I/O layer — read decisions.db, score the ledger (fail-open)
# ══════════════════════════════════════════════════════════════════════════════

def _resolved_predictions() -> dict:
    """From decisions.db: forecasts that carried a p_win AND resolved to an
    outcome. Returns aligned arrays {probs[0,1], success[0,1]} plus per-record
    slice labels (regime, source, reason). Fail-open → empty."""
    try:
        from core.decision_journal import _conn, WIN_PCT
        c = _conn()
        try:
            rows = c.execute(
                "SELECT symbol, decided_at, decision, p_win, outcome_pct, reason, source "
                "FROM decisions "
                "WHERE p_win IS NOT NULL AND outcome_pct IS NOT NULL").fetchall()
        finally:
            c.close()
    except Exception:
        return {"probs": np.zeros(0), "success": np.zeros(0),
                "reason": [], "source": []}
    from core.decision_journal import fold_opportunities, WIN_PCT
    unique = fold_opportunities([dict(r) for r in rows])
    probs, success, reason, source = [], [], [], []
    for r in unique:
        try:
            pw = float(r["p_win"])
        except Exception:
            continue
        pw = pw / 100.0 if pw > 1.0 else pw            # stored 0-100 → [0,1]
        pw = min(1.0, max(0.0, pw))
        probs.append(pw)
        success.append(1.0 if float(r["outcome_pct"]) >= WIN_PCT else 0.0)
        reason.append(str(r["reason"] or "—"))
        source.append(str(r["source"] or "—"))
    return {"probs": np.asarray(probs), "success": np.asarray(success),
            "reason": reason, "source": source}


def calibration_report() -> dict:
    """The headline forecast-reliability read from the live ledger. Fail-open."""
    d = _resolved_predictions()
    if d["probs"].size == 0:
        return {"n": 0, "insight": "No resolved forecasts yet to score."}
    return calibration_summary(d["probs"], d["success"])


def confidence_ledger_findings() -> list[dict]:
    """Harness-gated conditional over/under-confidence findings from the live
    ledger, sliced by rejection reason and source. Fail-open → []."""
    d = _resolved_predictions()
    if d["probs"].size == 0:
        return []
    out = []
    for label, groups in (("source", d["source"]), ("reason", d["reason"])):
        out.extend(conditional_overconfidence(d["probs"], d["success"], groups))
    return sorted(out, key=lambda t: abs(t["gap"]), reverse=True)[:5]


def calibration_directives(max_items: int = 2) -> list[dict]:
    """Brain-ready directives from the Confidence Ledger. Overconfidence in a
    bucket = a 'warn' (trust its odds less); underconfidence = an 'info' (room
    to lean in). Fail-open → []."""
    dirs: list[dict] = []
    for f in confidence_ledger_findings()[:max_items]:
        if f["direction"] == "overconfident":
            dirs.append({"severity": "warn",
                         "text": f"🎯 {f['insight']} Trust this bucket's odds less."})
        elif f["direction"] == "underconfident":
            dirs.append({"severity": "info",
                         "text": f"🎯 {f['insight']} Room to lean in here."})
    return dirs
