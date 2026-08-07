"""
🔬 Drift Attribution — "what CHANGED?" when a signal's edge moved.

Detecting that an edge decayed is half the job; the other half is explaining
WHY, so the fix is targeted instead of a blunt "stop trading it". When
`assess_drift` confirms a change-point in a signal's R-stream, this module splits
that signal's trades at the break and compares the BEFORE vs AFTER populations on
the dimensions the log actually records:

    • REGIME MIX — did the decay coincide with the tape rotating into a regime
      the signal hates (e.g. breakouts that only worked in TRENDING_BULL now
      firing mostly in DISTRIBUTION)? A proportion shift, reported per regime.
    • SETUP QUALITY — did post-drift trades score materially lower on the
      scanner's own quality_score (Welch t-test), i.e. we lowered our own bar?

The verdict is a plain-English driver list ("edge fell as its trades rotated
from TRENDING_BULL → DISTRIBUTION (+38pp)"), evidence-gated so we never narrate
noise. Pure comparison (`_population_diff`) is unit-tested; the I/O layer reads
the same `signal_log` as drift/live_edge and fails open to an empty read.
"""
from __future__ import annotations

import numpy as np

from research import drift as _drift

# A regime is called a DRIVER only if its share of the signal's trades moved by
# at least this much across the break — a fifth of the mix rotating is a real
# change, smaller wobbles are sampling noise.
_REGIME_DRIVER_PP = float(__import__("os").getenv("QT_ATTRIB_REGIME_PP", "0.20") or 0.20)
_QUALITY_ALPHA = 0.05          # Welch p for a real quality shift
_QUALITY_MIN_DELTA = 3.0       # and at least this many quality-score points


# ══════════════════════════════════════════════════════════════════════════════
# Pure comparison core
# ══════════════════════════════════════════════════════════════════════════════

def _proportions(labels: list[str]) -> dict[str, float]:
    labels = [l for l in labels if l]
    if not labels:
        return {}
    n = len(labels)
    out: dict[str, float] = {}
    for l in labels:
        out[l] = out.get(l, 0.0) + 1.0 / n
    return out


def _welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(delta_mean = mean(b)−mean(a), two-sided Welch p). Small-sample / zero-
    variance safe. Returns p=1.0 when a test can't be formed."""
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return (float(b.mean() - a.mean()) if a.size and b.size else 0.0), 1.0
    delta = float(b.mean() - a.mean())
    # near-constant inputs make Welch's t degenerate (scipy warns on catastrophic
    # cancellation) — no usable variance, so we can't claim significance.
    if a.std(ddof=1) < 1e-9 and b.std(ddof=1) < 1e-9:
        return delta, 1.0
    try:
        from scipy.stats import ttest_ind
        p = float(ttest_ind(b, a, equal_var=False).pvalue)
        if not np.isfinite(p):
            p = 1.0
    except Exception:
        p = 1.0
    return delta, p


def _population_diff(pre_rows: list[dict], post_rows: list[dict]) -> dict:
    """Compare BEFORE vs AFTER trade populations. Each row: {regime, quality}.
    Returns regime proportion shifts, the quality Welch test, an evidence-gated
    driver list, and a plain-English summary. Pure — no I/O."""
    pre_reg = _proportions([str(r.get("regime") or "") for r in pre_rows])
    post_reg = _proportions([str(r.get("regime") or "") for r in post_rows])
    regimes = sorted(set(pre_reg) | set(post_reg))
    shifts = []
    for rg in regimes:
        d = post_reg.get(rg, 0.0) - pre_reg.get(rg, 0.0)
        shifts.append({"regime": rg, "pre": round(pre_reg.get(rg, 0.0), 3),
                       "post": round(post_reg.get(rg, 0.0), 3), "delta": round(d, 3)})
    shifts.sort(key=lambda s: -abs(s["delta"]))

    q_pre = np.array([r["quality"] for r in pre_rows
                      if r.get("quality") is not None], float)
    q_post = np.array([r["quality"] for r in post_rows
                       if r.get("quality") is not None], float)
    q_delta, q_p = _welch(q_pre, q_post)

    drivers: list[str] = []
    # regime rotation driver: the single biggest gainer/loser past the threshold
    top = shifts[0] if shifts else None
    if top and abs(top["delta"]) >= _REGIME_DRIVER_PP:
        gain = [s for s in shifts if s["delta"] >= _REGIME_DRIVER_PP]
        loss = [s for s in shifts if s["delta"] <= -_REGIME_DRIVER_PP]
        if gain and loss:
            drivers.append(
                f"trades rotated from {loss[0]['regime']} → {gain[0]['regime']} "
                f"({gain[0]['delta']*100:+.0f}pp)")
        elif gain:
            drivers.append(f"more trades in {gain[0]['regime']} "
                           f"({gain[0]['delta']*100:+.0f}pp)")
        elif loss:
            drivers.append(f"fewer trades in {loss[0]['regime']} "
                           f"({loss[0]['delta']*100:+.0f}pp)")
    # quality driver
    if q_p < _QUALITY_ALPHA and abs(q_delta) >= _QUALITY_MIN_DELTA:
        drivers.append(f"setup quality {'fell' if q_delta < 0 else 'rose'} "
                       f"{abs(q_delta):.0f} pts (p={q_p:.03f})")

    if drivers:
        summary = "What changed: " + "; ".join(drivers) + "."
    else:
        summary = ("What changed: no clear shift in regime mix or setup quality "
                   "— the edge moved but the population looks the same "
                   "(look wider than the log records).")
    return {"regime_shifts": shifts, "quality_pre": round(float(q_pre.mean()), 1)
            if q_pre.size else None,
            "quality_post": round(float(q_post.mean()), 1) if q_post.size else None,
            "quality_delta": round(q_delta, 2), "quality_p": round(q_p, 4),
            "drivers": drivers, "summary": summary}


# ══════════════════════════════════════════════════════════════════════════════
# I/O — read the real signal_log, per-signal, with regime + quality
# ══════════════════════════════════════════════════════════════════════════════

def _signal_rows(signal: str) -> list[dict]:
    """Time-ordered [{r, regime, quality}, …] for ONE signal from signal_log.
    Uses the same per-row R formula as scan/live_edge. Fail-open → []."""
    try:
        from core.signal_outcome_tracker import _get_conn
        from scan.live_edge import _row_r
        conn = _get_conn()
        rows = conn.execute(
            """SELECT archetype, entry_price, stop_price, outcome_pct,
                      regime, quality_score
               FROM signal_log
               WHERE worked IS NOT NULL AND outcome_pct IS NOT NULL
                 AND entry_price > 0 AND stop_price > 0
               ORDER BY logged_at ASC"""
        ).fetchall()
        conn.close()
    except Exception:
        return []
    out: list[dict] = []
    for row in rows:
        if signal not in str(row["archetype"] or "").split("|"):
            continue
        r = _row_r(dict(row))
        if r is None:
            continue
        q = row["quality_score"]
        out.append({"r": r, "regime": str(row["regime"] or ""),
                    "quality": float(q) if q is not None else None})
    return out


def attribution(signal: str) -> dict:
    """For one signal: locate its drift change-point, and if confirmed, explain
    WHAT changed across the break (regime rotation / quality). Fail-open →
    {'drift': False}."""
    rows = _signal_rows(signal)
    if len(rows) < _drift._MIN_N:
        return {"signal": signal, "drift": False,
                "note": f"Only {len(rows)} outcomes — too few to attribute."}
    r_stream = [r["r"] for r in rows]
    d = _drift.assess_drift(r_stream)
    if d.status == "STABLE" or d.change_point is None:
        return {"signal": signal, "drift": False, "status": d.status,
                "note": "No confirmed change-point to attribute."}
    cp = int(d.change_point)
    diff = _population_diff(rows[:cp], rows[cp:])
    return {"signal": signal, "drift": True, "status": d.status,
            "delta_r": d.delta_r, "baseline_r": d.baseline_r,
            "recent_r": d.recent_r, "change_point": cp, "n": d.n,
            **diff}


def attribution_report(max_items: int = 5) -> list[dict]:
    """Attach a 'what changed' explanation to every signal drift_report() flags.
    Strongest drift first (drift_report's own ranking). Fail-open → []."""
    try:
        flagged = _drift.drift_report()
    except Exception:
        return []
    out: list[dict] = []
    for r in flagged[:max_items]:
        a = attribution(r["signal"])
        if a.get("drift"):
            out.append(a)
    return out
