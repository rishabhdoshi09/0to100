"""
📉 Concept-drift detection — catch edge decay before the drawdown.

Every edge decays. The system already blends "recent vs lifetime" expectancy,
but that is a lagging eyeball check — by the time the lifetime average turns,
you have bled a month. This module watches each signal's live outcome stream
(R-multiples, in time order) and answers, early and per-signal:

    "Is this signal's edge DECAYING, STRENGTHENING, or STABLE — and if it
     changed, roughly when?"

Method: a streaming change-point detector (**Page–Hinkley**, the canonical
online test named in the directive) locates a candidate shift; then the shift
is only reported if the before/after segments differ **significantly**
(Welch t-test + minimum-sample-per-segment). The change-point finds *where*;
the significance gate stops us crying drift on noise — the discipline the
Research OS demands.

The pure detector (`page_hinkley`, `assess_drift`) is unit-tested on synthetic
streams; the I/O layer reads the real `signal_log` (same R formula as
`scan/live_edge`, one language for evidence) and emits Brain-ready directives
("📉 breakout edge weakening — 3 weeks of decay, size down").
"""
from __future__ import annotations

import os as _os
from dataclasses import dataclass

import numpy as np

# ── Detector tuning (env-tunable) ─────────────────────────────────────────────
# Page-Hinkley `delta` is the per-sample tolerance (drift smaller than this is
# ignored as normal wobble); `threshold` (lambda) is the accumulated-deviation
# alarm level — higher = fewer false alarms, slower detection. Defaults are
# calibrated for R-multiple streams (std ~1): a sustained ~0.3R shift over a few
# dozen trades trips it, ordinary noise does not.
_PH_DELTA = float(_os.getenv("QT_DRIFT_PH_DELTA", "0.01") or 0.01)
_PH_LAMBDA = float(_os.getenv("QT_DRIFT_PH_LAMBDA", "8.0") or 8.0)
_MIN_N = int(_os.getenv("QT_DRIFT_MIN_N", "40") or 40)          # need a stream
_MIN_SEG = int(_os.getenv("QT_DRIFT_MIN_SEG", "15") or 15)      # per segment
_SIG_ALPHA = float(_os.getenv("QT_DRIFT_ALPHA", "0.05") or 0.05)


# ══════════════════════════════════════════════════════════════════════════════
# Pure detector — Page-Hinkley (two-sided)
# ══════════════════════════════════════════════════════════════════════════════

def page_hinkley(x, delta: float = _PH_DELTA, threshold: float = _PH_LAMBDA,
                 min_samples: int = _MIN_N) -> dict:
    """Two-sided Page-Hinkley change-point test on a time-ordered stream.

    Tracks the cumulative deviation of each sample from the running mean and
    alarms when the accumulated up- or down-drift exceeds `threshold`. Returns:
      detected      — did a shift trip the alarm?
      direction     — 'up' | 'down' | '' (strengthening vs decaying)
      change_point  — index where the drift most likely STARTED (the extremum
                      of the cumulative statistic before the alarm)
      alarm_index   — index at which the alarm tripped
      magnitude     — the PH statistic at the alarm
    Pure; no I/O. NaNs dropped."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    base = {"detected": False, "direction": "", "change_point": None,
            "alarm_index": None, "magnitude": 0.0, "n": n}
    if n < min_samples:
        return base
    mean = 0.0
    m_inc = 0.0     # cumulative deviation for INCREASE detection
    m_dec = 0.0     # cumulative deviation for DECREASE detection
    min_inc = 0.0   # running min of m_inc  → PH_up = m_inc - min_inc
    max_dec = 0.0   # running max of m_dec  → PH_down = max_dec - m_dec
    arg_min_inc = 0
    arg_max_dec = 0
    for t in range(n):
        mean += (x[t] - mean) / (t + 1)
        m_inc += (x[t] - mean - delta)
        if m_inc < min_inc:
            min_inc, arg_min_inc = m_inc, t
        ph_up = m_inc - min_inc
        m_dec += (x[t] - mean + delta)
        if m_dec > max_dec:
            max_dec, arg_max_dec = m_dec, t
        ph_down = max_dec - m_dec
        if t + 1 >= min_samples:
            if ph_down > threshold and ph_down >= ph_up:
                return {"detected": True, "direction": "down",
                        "change_point": arg_max_dec, "alarm_index": t,
                        "magnitude": float(ph_down), "n": n}
            if ph_up > threshold:
                return {"detected": True, "direction": "up",
                        "change_point": arg_min_inc, "alarm_index": t,
                        "magnitude": float(ph_up), "n": n}
    return base


# ══════════════════════════════════════════════════════════════════════════════
# High-level assessment — change-point + significance gate
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DriftResult:
    status: str            # STABLE | DECAYING | STRENGTHENING
    n: int
    baseline_r: float      # mean R before the change point
    recent_r: float        # mean R after the change point
    delta_r: float         # recent − baseline
    change_point: int | None
    n_since_change: int
    p_value: float         # Welch test of the before/after difference
    insight: str           # plain-English, user-facing


def _max_split_t(x: np.ndarray, min_seg: int) -> tuple[int, float]:
    """Scan every split point and return (best_split, signed Welch-t) for the
    split with the largest |t| between the before/after segments. Vectorised via
    prefix sums — O(n). This LOCATES a change-point far better than adaptive-mean
    Page-Hinkley, because it evaluates every candidate split directly."""
    n = x.size
    if n < 2 * min_seg:
        return -1, 0.0
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])
    cs = np.arange(min_seg, n - min_seg + 1)
    c = cs.astype(float)
    m = (n - cs).astype(float)
    mean_a = s1[cs] / c
    mean_b = (s1[n] - s1[cs]) / m
    var_a = (s2[cs] - s1[cs] ** 2 / c) / np.maximum(c - 1.0, 1.0)
    var_b = ((s2[n] - s2[cs]) - (s1[n] - s1[cs]) ** 2 / m) / np.maximum(m - 1.0, 1.0)
    se = np.sqrt(var_a / c + var_b / m)
    tvals = np.where(se > 0, (mean_b - mean_a) / se, 0.0)
    k = int(np.argmax(np.abs(tvals)))
    return int(cs[k]), float(tvals[k])


def _permutation_pvalue(x: np.ndarray, observed_abs_t: float, min_seg: int,
                        n_perm: int, seed: int | None) -> float:
    """p-value for the max-|t| change-point statistic by permutation. Shuffling
    destroys any time-ordering, so the fraction of shuffles whose best split
    matches or beats the observed statistic is the honest p — and it PROPERLY
    accounts for having scanned every split (no need for a separate multiple-
    comparisons correction)."""
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        _, t = _max_split_t(rng.permutation(x), min_seg)
        if abs(t) >= observed_abs_t:
            exceed += 1
    return (exceed + 1) / (n_perm + 1)


def assess_drift(r_stream, min_samples: int = _MIN_N, min_seg: int = _MIN_SEG,
                 alpha: float = _SIG_ALPHA, n_perm: int = 300,
                 seed: int | None = 0) -> DriftResult:
    """Locate the most likely change-point (max-|t| split), then CONFIRM it by
    permutation p-value. Absent a significant, well-separated shift → STABLE.
    The permutation test is what keeps a noisy stream from crying 'edge
    weakening' every other week: it already accounts for having looked at every
    possible split."""
    x = np.asarray(r_stream, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)

    def _stable(reason: str) -> DriftResult:
        mn = float(x.mean()) if n else 0.0
        return DriftResult("STABLE", n, round(mn, 3), round(mn, 3), 0.0,
                           None, 0, 1.0, reason)

    if n < min_samples:
        return _stable(f"Only {n} outcomes — need ~{min_samples} to judge drift.")
    cp, t = _max_split_t(x, min_seg)
    if cp < 0:
        return _stable("Edge steady — no separable change-point.")
    p = _permutation_pvalue(x, abs(t), min_seg, n_perm, seed)
    baseline, recent = float(x[:cp].mean()), float(x[cp:].mean())
    d = recent - baseline
    if p >= alpha or abs(d) < 1e-9:
        return _stable("A wobble, not a regime change (not significant).")
    status = "DECAYING" if d < 0 else "STRENGTHENING"
    verb = "weakening" if d < 0 else "strengthening"
    insight = (f"edge {verb}: {baseline:+.2f}R → {recent:+.2f}R "
               f"({d:+.2f}R) over the last {n - cp} trades (p={p:.03f}).")
    return DriftResult(status, n, round(baseline, 3), round(recent, 3),
                       round(d, 3), int(cp), int(n - cp), round(p, 4), insight)


# ══════════════════════════════════════════════════════════════════════════════
# I/O layer — read the real signal_log, per-signal, time-ordered
# ══════════════════════════════════════════════════════════════════════════════

def _signal_r_streams() -> dict[str, list[float]]:
    """{signal_key: [R, R, …]} in time order, from the outcome log. Uses the
    SAME per-row R formula as scan/live_edge (one language for evidence).
    Fail-open: any error → {}."""
    try:
        from core.signal_outcome_tracker import _get_conn
        from scan.live_edge import _row_r
        conn = _get_conn()
        rows = conn.execute(
            """SELECT archetype, entry_price, stop_price, outcome_pct
               FROM signal_log
               WHERE worked IS NOT NULL AND outcome_pct IS NOT NULL
                 AND entry_price > 0 AND stop_price > 0
               ORDER BY logged_at ASC"""
        ).fetchall()
        conn.close()
    except Exception:
        return {}
    streams: dict[str, list[float]] = {}
    for row in rows:
        r = _row_r(dict(row))
        if r is None:
            continue
        for sig in str(row["archetype"] or "").split("|"):
            sig = sig.strip()
            if sig:
                streams.setdefault(sig, []).append(r)
    return streams


def drift_report(min_n: int = _MIN_N) -> list[dict]:
    """Per-signal drift status from the live log. Returns only signals with a
    CONFIRMED drift (STABLE ones omitted), strongest change first. Fail-open."""
    out: list[dict] = []
    for sig, rs in _signal_r_streams().items():
        if len(rs) < min_n:
            continue
        d = assess_drift(rs)
        if d.status != "STABLE":
            out.append({"signal": sig, "status": d.status,
                        "delta_r": d.delta_r, "baseline_r": d.baseline_r,
                        "recent_r": d.recent_r, "n": d.n,
                        "n_since_change": d.n_since_change, "insight": d.insight})
    return sorted(out, key=lambda r: abs(r["delta_r"]), reverse=True)


def drift_directives(max_items: int = 3) -> list[dict]:
    """Brain-ready directives for signals in confirmed drift. Decay = a 'warn'
    (size down / demote); strengthening = an 'info' (lean in a little). Maps
    raw signal keys to plain labels. Fail-open (news/DB down → [])."""
    try:
        from scan.unified_scanner import SIGNAL_META
    except Exception:
        SIGNAL_META = {}
    dirs: list[dict] = []
    for r in drift_report()[:max_items]:
        label = SIGNAL_META.get(r["signal"], (r["signal"],))[0]
        if r["status"] == "DECAYING":
            dirs.append({"severity": "warn",
                         "text": f"📉 {label} {r['insight']} Size down / demote it."})
        else:
            dirs.append({"severity": "info",
                         "text": f"📈 {label} {r['insight']} Edge improving."})
    return dirs
