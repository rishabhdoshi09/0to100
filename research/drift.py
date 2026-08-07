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
    status: str            # STABLE | DECAYING | STRENGTHENING | RECOVERING
    n: int
    baseline_r: float      # mean R before the change point
    recent_r: float        # mean R after the change point
    delta_r: float         # recent − baseline
    change_point: int | None
    n_since_change: int
    p_value: float         # permutation test of the change-point
    confidence: str = ""   # HIGH | MEDIUM | LOW — how sure are we?
    duration: int = 0      # STABLE: how many trades it has held steady
    variance_ratio: float = 1.0     # recent outcome-variance ÷ baseline
    winrate_shift: float = 0.0      # recent win-rate − baseline win-rate
    risk_profile_changed: bool = False  # mean held but the RISK profile moved
    insight: str = ""      # plain-English, user-facing


_CONF_WORD = {"HIGH": "High confidence", "MEDIUM": "Medium confidence",
              "LOW": "Low confidence"}


def _confidence_tier(p_value: float, n_since_change: int) -> str:
    """How much to trust a drift call — a strong p-value over MANY recent trades
    is HIGH; a marginal one over few is LOW. Prevents reacting to a random
    fluctuation as if it were a regime change."""
    if p_value < 0.01 and n_since_change >= 30:
        return "HIGH"
    if p_value < 0.03 and n_since_change >= 15:
        return "MEDIUM"
    return "LOW"


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


def _variance_changepoint_p(x: np.ndarray, min_seg: int, n_perm: int,
                            seed: int | None) -> tuple[int, float, float]:
    """Detect a change in the VARIANCE of outcomes (risk profile), independent
    of the mean. Runs the same max-split machinery on the squared deviations
    from a rolling mean → a mean-shift in |deviation| IS a variance shift.
    Returns (change_point, p_value, variance_ratio recent÷baseline)."""
    if x.size < 2 * min_seg:
        return -1, 1.0, 1.0
    dev = (x - x.mean()) ** 2                      # squared deviations
    cp, t = _max_split_t(dev, min_seg)
    if cp < 0:
        return -1, 1.0, 1.0
    p = _permutation_pvalue(dev, abs(t), min_seg, n_perm, seed)
    vb = float(x[:cp].var(ddof=1)) if cp > 1 else 0.0
    va = float(x[cp:].var(ddof=1)) if (x.size - cp) > 1 else 0.0
    ratio = (va / vb) if vb > 0 else 1.0
    return cp, p, ratio


def _dip_recovery_stat(x: np.ndarray, min_seg: int) -> tuple[float, int, float, float, float]:
    """Statistic for a DIP-AND-RECOVER (V-shape): the edge fell from an early
    level into an interior trough, then climbed back. A single change-point split
    can't see this — neither half of a V separates cleanly — so we test it
    directly. Statistic = min(early_mean, recent_mean) − trough_mean, i.e. how
    far BOTH ends sit above the worst interior stretch. Returns
    (stat, trough_index, early_mean, trough_mean, recent_mean)."""
    n = x.size
    if n < 3 * min_seg:
        return 0.0, -1, 0.0, 0.0, 0.0
    early = float(x[:min_seg].mean())
    recent = float(x[-min_seg:].mean())
    # lowest-mean window of width min_seg strictly INTERIOR (a real trough, not
    # just the two ends) — vectorised rolling mean via prefix sums.
    s = np.concatenate([[0.0], np.cumsum(x)])
    lo, hi = min_seg, n - min_seg                    # window starts in [lo, hi)
    starts = np.arange(lo, hi)
    if starts.size == 0:
        return 0.0, -1, 0.0, 0.0, 0.0
    means = (s[starts + min_seg] - s[starts]) / min_seg
    j = int(np.argmin(means))
    trough = float(means[j])
    trough_idx = int(starts[j] + min_seg // 2)
    stat = min(early, recent) - trough
    return stat, trough_idx, early, trough, recent


def _dip_recovery_pvalue(x: np.ndarray, min_seg: int, n_perm: int,
                         seed: int | None) -> tuple[float, int, float, float, float]:
    """Permutation p-value for the dip-and-recover statistic. Shuffling destroys
    the time-ordering, so the fraction of shuffles whose V-statistic matches or
    beats the observed one is the honest p — and it accounts for having scanned
    every trough position. Returns (p, trough_idx, early, trough, recent)."""
    obs, tidx, early, trough, recent = _dip_recovery_stat(x, min_seg)
    if tidx < 0 or obs <= 0:
        return 1.0, tidx, early, trough, recent
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        st, *_ = _dip_recovery_stat(rng.permutation(x), min_seg)
        if st >= obs:
            exceed += 1
    return (exceed + 1) / (n_perm + 1), tidx, early, trough, recent


def assess_drift(r_stream, min_samples: int = _MIN_N, min_seg: int = _MIN_SEG,
                 alpha: float = _SIG_ALPHA, n_perm: int = 300,
                 seed: int | None = 0) -> DriftResult:
    """Locate the most likely change-point (max-|t| split), CONFIRM it by
    permutation p-value, then enrich the read:
      • RECOVERING — a decay that has since significantly rebounded.
      • confidence — HIGH/MEDIUM/LOW from the p-value × recent-trade count.
      • multi-metric — the outcome VARIANCE and WIN-RATE shift too, so a stable
        AVERAGE that masks a changed risk profile is still surfaced (the mean
        can hold flat while the tails get uglier).
    Absent a significant mean shift → STABLE, but with its duration and a
    risk-profile flag if the variance moved."""
    x = np.asarray(r_stream, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)

    if n < min_samples:
        mn = float(x.mean()) if n else 0.0
        return DriftResult("STABLE", n, round(mn, 3), round(mn, 3), 0.0, None, 0,
                           1.0, duration=n,
                           insight=f"Only {n} outcomes — need ~{min_samples} to judge drift.")

    cp, t = _max_split_t(x, min_seg)
    p = _permutation_pvalue(x, abs(t), min_seg, n_perm, seed) if cp >= 0 else 1.0
    baseline = float(x[:cp].mean()) if cp > 0 else float(x.mean())
    recent = float(x[cp:].mean()) if cp >= 0 else float(x.mean())
    d = recent - baseline
    mean_shifted = cp >= 0 and p < alpha and abs(d) >= 1e-9

    # variance & win-rate context (relative to the primary split, or halves)
    split = cp if cp >= 0 else n // 2
    var_b = float(x[:split].var(ddof=1)) if split > 1 else 0.0
    var_a = float(x[split:].var(ddof=1)) if (n - split) > 1 else 0.0
    var_ratio = round(var_a / var_b, 2) if var_b > 0 else 1.0
    wr_b = float((x[:split] > 0).mean()) if split > 0 else 0.0
    wr_a = float((x[split:] > 0).mean()) if (n - split) > 0 else 0.0
    wr_shift = round(wr_a - wr_b, 3)

    # ── DIP-AND-RECOVER (V-shape) — a single split can't see a V (neither half
    # separates cleanly), so test it directly and let it OVERRIDE the split
    # verdict: an edge that fell into a trough and has since climbed back near
    # its earlier level is RECOVERING, not a fresh strengthening and not a decay.
    rec_p, tidx, early_m, trough_m, recent_m = _dip_recovery_pvalue(
        x, min_seg, n_perm, seed)
    recovered = (rec_p < alpha
                 and (early_m - trough_m) >= 0.25     # a real fall from the top
                 and (recent_m - trough_m) >= 0.25    # a real climb back up
                 and recent_m >= early_m - 0.30)      # back near the earlier level
    if recovered:
        n_since = n - tidx
        conf = _confidence_tier(rec_p, n_since)
        insight = (f"{_CONF_WORD[conf]}: edge RECOVERED — dipped to {trough_m:+.2f}R "
                   f"then climbed back to {recent_m:+.2f}R over the last {n_since} "
                   f"trades. Trust can be restored.")
        if var_ratio >= 1.5 or var_ratio <= 0.67:
            insight += (f" Outcome volatility {'up' if var_ratio > 1 else 'down'} "
                        f"~{abs(var_ratio-1)*100:.0f}% too.")
        return DriftResult("RECOVERING", n, round(trough_m, 3), round(recent_m, 3),
                           round(recent_m - trough_m, 3), int(tidx), int(n_since),
                           round(rec_p, 4), confidence=conf, variance_ratio=var_ratio,
                           winrate_shift=wr_shift,
                           risk_profile_changed=bool(var_ratio >= 1.5 or var_ratio <= 0.67),
                           insight=insight)

    if not mean_shifted:
        # mean held — did the RISK PROFILE move underneath it?
        vcp, vp, vratio = _variance_changepoint_p(x, min_seg, n_perm, seed)
        mn = float(x.mean())
        if vp < 0.01 and (vratio >= 2.0 or (0 < vratio <= 0.5)):
            move = "rose" if vratio > 1 else "fell"
            return DriftResult(
                "STABLE", n, round(mn, 3), round(mn, 3), 0.0, None, 0, round(vp, 4),
                confidence=_confidence_tier(vp, n - vcp if vcp >= 0 else 0),
                duration=n, variance_ratio=round(vratio, 2),
                risk_profile_changed=True,
                insight=(f"Average held ({mn:+.2f}R) BUT outcome volatility "
                         f"{move} ~{abs(vratio-1)*100:.0f}% — risk profile shifted, "
                         f"not the edge. Same size ≠ same risk."))
        return DriftResult("STABLE", n, round(mn, 3), round(mn, 3), 0.0, None, 0,
                           round(p, 4), duration=n, variance_ratio=var_ratio,
                           insight=f"Edge stable across {n} trades — no significant shift.")

    # a confirmed one-directional mean shift — decay or strengthening (a V-shaped
    # recovery was already caught and returned above).
    status = "DECAYING" if d < 0 else "STRENGTHENING"
    n_since = n - cp
    conf = _confidence_tier(p, n_since)
    verb = "deterioration" if d < 0 else "improvement"
    insight = (f"{_CONF_WORD[conf]}: sustained {verb} — {baseline:+.2f}R → "
               f"{recent:+.2f}R ({d:+.2f}R) over {n_since} trades (p={p:.03f}).")
    # note a co-occurring risk-profile change
    if var_ratio >= 1.5 or var_ratio <= 0.67:
        insight += f" Outcome volatility {'up' if var_ratio > 1 else 'down'} ~{abs(var_ratio-1)*100:.0f}% too."
    return DriftResult(status, n, round(baseline, 3), round(recent, 3),
                       round(recent - baseline, 3), int(cp), int(n_since),
                       round(p, 4), confidence=conf, variance_ratio=var_ratio,
                       winrate_shift=wr_shift,
                       risk_profile_changed=bool(var_ratio >= 1.5 or var_ratio <= 0.67),
                       insight=insight)


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
    """Per-signal drift status from the live log. Returns signals in a CONFIRMED
    mean drift OR a stable-mean-but-shifted-RISK-profile state (steady STABLE
    ones omitted), strongest change first. Fail-open."""
    out: list[dict] = []
    for sig, rs in _signal_r_streams().items():
        if len(rs) < min_n:
            continue
        d = assess_drift(rs)
        if d.status != "STABLE" or d.risk_profile_changed:
            out.append({"signal": sig, "status": d.status,
                        "confidence": d.confidence, "delta_r": d.delta_r,
                        "baseline_r": d.baseline_r, "recent_r": d.recent_r,
                        "n": d.n, "n_since_change": d.n_since_change,
                        "variance_ratio": d.variance_ratio,
                        "risk_profile_changed": d.risk_profile_changed,
                        "insight": d.insight})
    # money-moving decays first, then recoveries/improvements, then risk-only
    _rank = {"DECAYING": 0, "RECOVERING": 1, "STRENGTHENING": 2, "STABLE": 3}
    return sorted(out, key=lambda r: (_rank.get(r["status"], 9), -abs(r["delta_r"])))


def drift_directives(max_items: int = 3) -> list[dict]:
    """Brain-ready directives for signals in confirmed drift. DECAYING warns
    (size down); RECOVERING/STRENGTHENING inform (trust can return / lean in);
    a stable-mean-but-shifted-risk STABLE warns (same size ≠ same risk). Only
    HIGH/MEDIUM-confidence calls become directives — LOW ones stay monitoring-
    only, so the Brain never reacts to a random fluctuation. Fail-open → []."""
    try:
        from scan.unified_scanner import SIGNAL_META
    except Exception:
        SIGNAL_META = {}
    dirs: list[dict] = []
    for r in drift_report()[:max_items + 2]:
        if r["confidence"] == "LOW":
            continue                                  # monitoring-only, no alert
        label = SIGNAL_META.get(r["signal"], (r["signal"],))[0]
        st = r["status"]
        if st == "DECAYING":
            # enrich the warning with WHAT changed (regime rotation / quality) so
            # it's actionable, not just "it decayed". Lazy import breaks the
            # drift↔attribution cycle; fail-open to no enrichment.
            why = ""
            try:
                from research.drift_attribution import attribution
                a = attribution(r["signal"])
                if a.get("drift") and a.get("drivers"):
                    why = " " + a["summary"]
            except Exception:
                why = ""
            dirs.append({"severity": "warn",
                         "text": f"📉 {label} — {r['insight']} Size down / demote it.{why}"})
        elif st == "RECOVERING":
            dirs.append({"severity": "info",
                         "text": f"📈 {label} — {r['insight']}"})
        elif st == "STRENGTHENING":
            dirs.append({"severity": "info",
                         "text": f"📈 {label} — {r['insight']} Edge improving."})
        elif r["risk_profile_changed"]:
            dirs.append({"severity": "warn",
                         "text": f"📊 {label} — {r['insight']}"})
        if len(dirs) >= max_items:
            break
    return dirs
