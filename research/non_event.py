"""
🕳️ Non-Event Learning — the 1,900 stocks you DIDN'T trade are the control group.

The scanner evaluates ~2,000 names a cycle and acts on a handful. The executed
trades answer only "what happened after I traded?" The REJECTED and NEAR-MISS
names answer the far more valuable question — "what would have happened if I had,
and was my 'no' correct?" That is the difference between a P&L and a controlled
experiment, and it is the rarest dataset in retail trading (deliberately-ignored
opportunities reveal selection bias professional shops pay dearly for).

This module makes those non-events first-class observations in the Feature
Platform, with two refinements that turn them into research gold:

  • STRUCTURED CAUSES. Every rejection carries a canonical `reason` code
    (EXTENSION, LOW_CONVICTION, POOR_BREADTH, RISK_LIMIT, CORRELATION, …). Years
    on you can ask "which rejection reasons consistently SAVE money, and which are
    too conservative?" — `rejection_analysis()` answers exactly that, harness-
    gated so a reason is only judged on adequate, significant evidence.

  • TWO NEAR-MISS TYPES, stored separately (they have very different research
    value):
      ALMOST — one threshold away (needed vol ≥2×, saw 1.92×). Carries the gap
               {feature, needed, observed}, which powers decision-boundary
               REPLAY: "if RSI cap moved 72→74, how many rejects flip?" answered
               from stored observations, with NO historical rescan.
      FADED  — qualified, then reversed before entry (a gate fired late).

Outcomes are settled from official bhavcopy forward returns (invariant #1: real
data only — a rejected name with no history is skipped, never simulated). All
emission + settlement is fail-open: non-event capture must never slow or crash a
scan.
"""
from __future__ import annotations

import time as _time
from datetime import datetime, timedelta

from research import feature_store as _FS

# ── canonical rejection causes — the single source of truth for "why not" ──
REASONS = (
    "EXTENSION",        # extended from 20-EMA / 50-DMA (chase risk)
    "LOW_CONVICTION",   # composite conviction below the bar
    "WEAK_CLOSE",       # weak Close Location Value (bull-trap risk)
    "BLOWOFF_RSI",      # RSI too hot (no room to run)
    "LAGGARD",          # too far below the 52-week high
    "POOR_BREADTH",     # market breadth NARROW
    "RISK_LIMIT",       # portfolio open-risk cap hit
    "CORRELATION",      # too correlated with existing positions
    "LIQUIDITY",        # turnover too thin to trade
    "MACRO",            # macro RISK_OFF holding back
    "ALREADY_OWNED",    # position already exists
    "DRIFT",            # signal in confirmed edge-decay
    "OTHER",
)

# ── near-miss subtypes (stored separately — different research value) ──
ALMOST = "ALMOST"   # Type A: one threshold away
FADED = "FADED"     # Type B: qualified then failed before entry
SUBTYPES = (ALMOST, FADED)

# forward-return horizon + the "counts as a move" band (mirrors the signal
# outcome tracker: +2% up = a real winner, −1% = a real loser).
_HORIZON_DAYS = 5
_WIN_PCT = 2.0
_LOSS_PCT = -1.0
_MIN_FOR_CLAIM = 30     # harness floor — no verdict on a reason below this


def _norm_reason(reason: str | None) -> str:
    r = (reason or "").strip().upper()
    return r if r in REASONS else "OTHER"


def _today() -> str:
    return _time.strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════════════
# Emission — the scanner calls these (fail-open, deterministic-id deduped)
# ══════════════════════════════════════════════════════════════════════════════

def capture_rejection(symbol: str, features: dict, reason: str,
                      ts: str | None = None, meta: dict | None = None) -> dict:
    """Freeze a REJECTION observation with its structured cause. The id is
    deterministic per (day, symbol, reason) so the same rejection logged every
    15-min cycle is stored ONCE (write-once immutability dedupes for free).
    Fail-open."""
    r = _norm_reason(reason)
    oid = f"{(ts or _today())[:10]}:{(symbol or '').upper()}:REJ:{r}"
    return _FS.snapshot(oid, symbol, "REJECTION", features, ts=ts,
                        reason=r, meta=meta)


def record_scan_batch(results, regime: str = "", breadth_pct=None,
                      almost_pivot_pct: float = 1.5) -> dict:
    """Map a scan's WATCH / rejected StockSignals into structured non-event
    observations — the control group the executed BUYs are drawn from. Confident,
    unambiguous mapping only (never guesses a cause it can't know):
      • chase_risk fired            → REJECTION / EXTENSION
      • WATCH, just below the pivot  → NEAR_MISS / ALMOST (Type A, gap recorded)
      • WATCH otherwise             → REJECTION / LOW_CONVICTION
    BUYs are skipped here (they're tracked as executed signals elsewhere). Returns
    a small counts summary. Fail-open — never raises into the scan loop."""
    counts = {"extension": 0, "almost": 0, "low_conviction": 0}
    try:
        for r in results:
            verdict = getattr(r, "verdict", "")
            if verdict == "BUY":
                continue
            feats = {"rsi": getattr(r, "rsi", None),
                     "quality_score": getattr(r, "score", None),
                     "regime": regime or None,
                     "breadth_pct_above_50dma": breadth_pct}
            sym = getattr(r, "symbol", "")
            if getattr(r, "chase_risk", False):
                capture_rejection(sym, feats, "EXTENSION")
                counts["extension"] += 1
                continue
            pivot_dist = float(getattr(r, "pivot_distance_pct", 0.0) or 0.0)
            if 0.0 < pivot_dist <= almost_pivot_pct:
                # just below the trigger — one nudge from qualifying (Type A)
                capture_near_miss(sym, feats, ALMOST,
                                  gap={"feature": "pivot_distance_pct",
                                       "needed": 0.0, "observed": pivot_dist})
                counts["almost"] += 1
            else:
                capture_rejection(sym, feats, "LOW_CONVICTION")
                counts["low_conviction"] += 1
    except Exception:
        pass
    return counts


def capture_near_miss(symbol: str, features: dict, subtype: str,
                      gap: dict | None = None, reason: str | None = None,
                      ts: str | None = None) -> dict:
    """Freeze a NEAR_MISS observation. `subtype` is ALMOST (one threshold away —
    pass `gap={feature, needed, observed}`) or FADED (qualified then reversed).
    Fail-open."""
    st = subtype if subtype in SUBTYPES else ALMOST
    oid = f"{(ts or _today())[:10]}:{(symbol or '').upper()}:NM:{st}"
    meta = {"gap": gap} if gap else None
    return _FS.snapshot(oid, symbol, "NEAR_MISS", features, ts=ts,
                        reason=_norm_reason(reason) if reason else None,
                        subtype=st, meta=meta)


# ══════════════════════════════════════════════════════════════════════════════
# Outcome settlement — real bhavcopy forward returns only
# ══════════════════════════════════════════════════════════════════════════════

def _forward_return(symbol: str, from_date: str, horizon: int = _HORIZON_DAYS):
    """% change from the close on/after `from_date` to `horizon` trading days
    later, from official bhavcopy. None if history is missing (never simulated)."""
    try:
        import pandas as pd
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv(symbol)
        if df is None or df.empty or "close" not in df.columns:
            return None
        after = df[df.index >= pd.Timestamp(str(from_date)[:10])]
        if len(after) < horizon + 1:
            return None
        entry = float(after["close"].iloc[0])
        exit_ = float(after["close"].iloc[horizon])
        if entry <= 0:
            return None
        return (exit_ - entry) / entry * 100.0
    except Exception:
        return None


def settle_outcomes(lookback_days: int = 45, limit: int = 4000) -> int:
    """Fill forward-return outcomes on REJECTION / NEAR_MISS observations old
    enough to have resolved (≥ horizon) but not older than `lookback_days`.
    Returns how many were settled. Fail-open → 0."""
    try:
        cutoff_new = (datetime.now() - timedelta(days=_HORIZON_DAYS + 1)).strftime("%Y-%m-%d")
        cutoff_old = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        c = _FS._conn()
        try:
            rows = c.execute(
                "SELECT observation_id, symbol, ts FROM observations "
                "WHERE kind IN ('REJECTION','NEAR_MISS') AND outcome IS NULL "
                "AND ts <= ? AND ts >= ? LIMIT ?",
                (cutoff_new, cutoff_old, limit)).fetchall()
        finally:
            c.close()
    except Exception:
        return 0
    settled = 0
    for r in rows:
        fr = _forward_return(r["symbol"], r["ts"])
        if fr is None:
            continue
        if _FS.set_outcome(r["observation_id"], fr).get("status") == "settled":
            settled += 1
    return settled


# ══════════════════════════════════════════════════════════════════════════════
# Counterfactual analysis — which "no"s earn, which are too conservative
# ══════════════════════════════════════════════════════════════════════════════

def _mean_vs_zero_p(arr) -> float | None:
    """Two-sided p-value that the mean differs from 0 (either sign). Zero-variance
    / tiny-sample safe. None when no test can be formed."""
    import numpy as np
    a = np.asarray(arr, float)
    a = a[~np.isnan(a)]
    if a.size < 2:
        return None
    sd = a.std(ddof=1)
    if sd < 1e-12:
        return 0.0 if abs(a.mean()) > 0 else 1.0
    try:
        from scipy.stats import ttest_1samp
        p = float(ttest_1samp(a, 0.0).pvalue)
        return p if p == p else None            # guard NaN
    except Exception:
        return None


# Modeling assumption for the OPTIONAL hypothetical-R view. A rejected trade had
# no real stop, so any R is MODELED, never observed — this constant is the stated
# assumption, surfaced alongside every modeled number so it can never masquerade
# as an observed one.
_ATR_STOP_MULT = float(__import__("os").getenv("QT_CF_ATR_STOP_MULT", "2.0") or 2.0)
_MODELED_ASSUMPTION = f"ATR-{_ATR_STOP_MULT:g}x stop (counterfactual — no real stop existed)"


def modeled_r(fwd_pct, atr_pct, atr_mult: float = _ATR_STOP_MULT):
    """MODELED (not observed) R-multiple for a counterfactual trade: forward
    return ÷ a hypothetical ATR-based stop distance. Explicitly an assumption —
    the observed world uses observed metrics (forward return %); the hypothetical
    world uses modeled ones (R). None when ATR is missing/zero (no model → no
    claim)."""
    try:
        atr = float(atr_pct)
        fwd = float(fwd_pct)
    except (TypeError, ValueError):
        return None
    stop_dist = atr_mult * atr
    if stop_dist <= 0:
        return None
    return fwd / stop_dist


def _load_settled(kind: str) -> list[dict]:
    try:
        c = _FS._conn()
        try:
            rows = c.execute(
                "SELECT symbol, reason, subtype, outcome, meta, features FROM "
                "observations WHERE kind=? AND outcome IS NOT NULL",
                (kind,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            c.close()
    except Exception:
        return []


def rejection_analysis() -> list[dict]:
    """Per-reason counterfactual: of the names a reason rejected, how many rose
    (missed winners) vs fell (correctly avoided), the average forward return, and
    a harness-gated verdict. avg_fwd < 0 ⇒ the reason EARNS its keep (it avoids
    losers); avg_fwd clearly > 0 over many names ⇒ TOO_CONSERVATIVE (it's costing
    winners). Below the evidence floor ⇒ INSUFFICIENT. Fail-open → []."""
    import json
    import numpy as np
    rows = _load_settled("REJECTION")
    by_reason: dict[str, list[float]] = {}
    modeled_by_reason: dict[str, list[float]] = {}
    for r in rows:
        reason = _norm_reason(r["reason"])
        by_reason.setdefault(reason, []).append(float(r["outcome"]))
        # OPTIONAL modeled R — only when ATR is present; kept in a SEPARATE stream
        try:
            atr = (json.loads(r.get("features") or "{}") or {}).get("atr_pct")
        except Exception:
            atr = None
        mr = modeled_r(r["outcome"], atr) if atr is not None else None
        if mr is not None:
            modeled_by_reason.setdefault(reason, []).append(mr)
    out = []
    for reason, fwds in by_reason.items():
        arr = np.array(fwds, float)
        n = arr.size
        rose = int((arr >= _WIN_PCT).sum())
        fell = int((arr <= _LOSS_PCT).sum())
        avg = float(arr.mean())
        # two-sided significance of the mean forward-return vs 0 (a rejection
        # reason EARNS if the names it passed fell SIGNIFICANTLY; it's too
        # conservative if they ROSE significantly). expectancy_stats' p is
        # one-sided-for-positive, so it can't judge the negative case — test here.
        # The VERDICT is driven ONLY by observed forward return, never the model.
        p = _mean_vs_zero_p(arr)
        if n < _MIN_FOR_CLAIM:
            verdict = "INSUFFICIENT"
        elif avg <= 0 and (p is None or p < 0.10):
            verdict = "EARNING"          # rejecting these avoided net-down names
        elif avg >= _WIN_PCT and (p is None or p < 0.10):
            verdict = "TOO_CONSERVATIVE"  # these rose after we passed
        else:
            verdict = "NEUTRAL"
        mrs = modeled_by_reason.get(reason, [])
        row = {"reason": reason, "n": n, "missed_winners": rose,
               "correctly_avoided": fell, "avg_fwd_pct": round(avg, 2),
               "p_value": round(p, 4) if p is not None else None,
               "verdict": verdict,
               # modeled view — clearly namespaced + assumption stated so it can
               # never be mistaken for the observed metric.
               "modeled_avg_r": round(float(np.mean(mrs)), 2) if mrs else None,
               "modeled_n": len(mrs),
               "modeled_assumption": _MODELED_ASSUMPTION if mrs else None}
        out.append(row)
    order = {"TOO_CONSERVATIVE": 0, "EARNING": 1, "NEUTRAL": 2, "INSUFFICIENT": 3}
    return sorted(out, key=lambda d: (order.get(d["verdict"], 9), -d["n"]))


def replay_threshold(feature: str, old: float, new: float,
                     direction: str = "ceiling") -> dict:
    """Decision-boundary REPLAY from stored ALMOST near-misses — no rescan.
    For a `ceiling` gate (reject when feature > threshold, e.g. RSI cap) that is
    RELAXED old→new (new > old), count the ALMOST rejects whose observed value
    falls in (old, new] — i.e. names that WOULD now qualify — and, where settled,
    what those names went on to do. `floor` mirrors it for lower-bound gates
    (e.g. volume ≥ threshold). Fail-open → an empty replay."""
    import numpy as np
    rows = [r for r in _load_settled("NEAR_MISS")]
    flipped, outcomes = [], []
    for r in rows:
        meta = r.get("meta")
        gap = None
        if meta:
            import json
            try:
                gap = (json.loads(meta) or {}).get("gap")
            except Exception:
                gap = None
        if not gap or gap.get("feature") != feature:
            continue
        obs = gap.get("observed")
        if obs is None:
            continue
        obs = float(obs)
        # would this observation cross the boundary under the new threshold?
        crosses = (direction == "ceiling" and old < obs <= new) or \
                  (direction == "floor" and new <= obs < old)
        if crosses:
            flipped.append(r["symbol"])
            if r.get("outcome") is not None:
                outcomes.append(float(r["outcome"]))
    arr = np.array(outcomes, float) if outcomes else np.array([])
    return {"feature": feature, "old": old, "new": new, "direction": direction,
            "would_qualify": len(flipped), "symbols": flipped[:25],
            "settled": int(arr.size),
            "avg_fwd_pct": round(float(arr.mean()), 2) if arr.size else None,
            "winners": int((arr >= _WIN_PCT).sum()) if arr.size else 0,
            "losers": int((arr <= _LOSS_PCT).sum()) if arr.size else 0}


def reason_trend(reason: str) -> str:
    """Is a reason's counterfactual getting BETTER or WORSE lately? Split its
    settled rejections chronologically (recent third vs the rest) and compare the
    avg forward return the reason let slip. ↑ = the names it passes are rising
    more (getting more conservative), ↓ = falling more (earning more), → = flat /
    too few. Fail-open → '→'."""
    import numpy as np
    r = _norm_reason(reason)
    try:
        c = _FS._conn()
        try:
            rows = c.execute(
                "SELECT outcome FROM observations WHERE kind='REJECTION' AND "
                "reason=? AND outcome IS NOT NULL ORDER BY ts ASC", (r,)).fetchall()
        finally:
            c.close()
    except Exception:
        return "→"
    vals = [float(x["outcome"]) for x in rows]
    if len(vals) < 12:
        return "→"
    k = max(4, len(vals) // 3)
    recent = float(np.mean(vals[-k:]))
    prior = float(np.mean(vals[:-k]))
    if recent > prior + 0.75:
        return "↑"
    if recent < prior - 0.75:
        return "↓"
    return "→"


def rejection_directives(max_items: int = 1) -> list[dict]:
    """Brain-ready: a rejection reason the evidence now says is TOO_CONSERVATIVE
    (costing more winners than it saves) — a gate to loosen. Demote-only,
    harness-gated. Fail-open → []."""
    dirs = []
    for a in rejection_analysis():
        if a["verdict"] == "TOO_CONSERVATIVE":
            dirs.append({"severity": "info",
                         "text": f"🕳️ Rejection reason {a['reason']} looks too "
                                 f"strict — {a['missed_winners']}/{a['n']} names it "
                                 f"passed rose (avg {a['avg_fwd_pct']:+.1f}%). "
                                 f"Consider loosening it."})
            if len(dirs) >= max_items:
                break
    return dirs
