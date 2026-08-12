"""
🧪 Data Integrity — catch the corrupt input BEFORE it becomes a confident lie.

The professor's disqualifier: a backtest (or a live signal) on unadjusted or
broken price data is not merely wrong, it is *confidently* wrong. NSE bhavcopy is
unadjusted for splits/bonuses, so a 1:1 bonus reads as a −50% crash — a phantom
gap that fabricates stop-hits and signals. This module is the guard:

  • phantom_gaps() — pure scan for single-session moves too large to be real
    price action (the fingerprint of an un-adjusted corporate action or a bad
    print). Unit-tested.
  • check_symbol() / integrity_report() — the fail-open I/O layer that runs the
    scan + a freshness check over the store and hands a verdict to the Governance
    Sentinel (a corporate-action mismatch is a HALT condition).

Detection lives here; the FIX lives in `data.corporate_actions` (back-adjust from
a real NSE CA table, applied on read). `verify_ca_adjustment()` closes the loop —
it re-scans through the adjusted read path and only PASSES when a table is loaded
and the phantom-gap rate has collapsed to ~0. Fail-open: an error yields
"unknown", never a false all-clear.
"""
from __future__ import annotations

import os as _os

import numpy as np

# A one-session move beyond this is not normal price action — it's the signature
# of an un-adjusted split/bonus or a bad print. 35% is safely above even circuit
# limits stacked with a gap, so a flag here is a real data problem, not a mover.
_GAP_PCT = float(_os.getenv("QT_INTEGRITY_GAP_PCT", "35") or 35)
_STALE_DAYS = int(_os.getenv("QT_INTEGRITY_STALE_DAYS", "7") or 7)


def phantom_gaps(closes, threshold_pct: float = _GAP_PCT) -> list[dict]:
    """Indices where |session-over-session % change| exceeds `threshold_pct` — the
    fingerprint of an un-adjusted corporate action or a data error. Pure. Returns
    [{index, pct}] (index is the position of the *later* bar)."""
    c = np.asarray(closes, dtype=float)
    c = c[~np.isnan(c)]
    out: list[dict] = []
    if c.size < 2:
        return out
    prev = c[:-1]
    chg = np.where(prev > 0, (c[1:] - prev) / prev * 100.0, 0.0)
    for i, pct in enumerate(chg):
        if abs(pct) >= threshold_pct:
            out.append({"index": int(i + 1), "pct": round(float(pct), 1)})
    return out


def check_symbol(symbol: str) -> dict:
    """Integrity of one symbol's stored history: phantom gaps + freshness.
    Returns {symbol, ok, gaps, stale_days, issues}. Fail-open → ok=None
    ('unknown', never a false all-clear)."""
    try:
        import pandas as pd
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv(symbol)
    except Exception:
        return {"symbol": symbol, "ok": None, "issues": ["read failed"]}
    if df is None or df.empty or "close" not in df.columns:
        return {"symbol": symbol, "ok": None, "issues": ["no data"]}
    gaps = phantom_gaps(df["close"].to_numpy(dtype=float))
    issues = []
    if gaps:
        issues.append(f"{len(gaps)} phantom gap(s) — possible un-adjusted "
                      f"corporate action")
    stale_days = None
    try:
        import pandas as pd
        last = df.index[-1]
        stale_days = int((pd.Timestamp.now().normalize() - last.normalize()).days)
        if stale_days > _STALE_DAYS:
            issues.append(f"stale: last bar {stale_days}d old")
    except Exception:
        pass
    return {"symbol": symbol, "ok": len(issues) == 0, "gaps": gaps,
            "stale_days": stale_days, "issues": issues}


def verify_ca_adjustment(sample: int = 200) -> dict:
    """Acceptance test for corporate-action adjustment quality.

    IMPORTANT (remediation R1/R6): the research-quality measure is the rate of
    *unresolved consecutive-session* discontinuities after adjustment — NOT the
    raw frequency of any large adjacent-bar move (which incorrectly counted
    suspension/sparse-trading spans as CA failures).

    ``passed`` requires an events table loaded AND
    ``unresolved_symbol_rate ≤ 0.002``. Fail-open → passed=False.
    """
    try:
        from data.corporate_actions import load_events
        n_events = len(load_events())
    except Exception:
        n_events = 0
    try:
        from research.intelligence.data.discontinuity_audit import audit_universe
        audit = audit_universe(sample=sample)
        checked = int(audit.get("symbols_checked") or 0)
        gap_rate = float(audit.get("unresolved_symbol_rate") or 1.0)
        flagged = [
            r["symbol"] for r in (audit.get("top_unresolved") or [])
        ]
        # unique preserve order
        seen = set()
        flagged_u = []
        for s in flagged:
            if s not in seen:
                seen.add(s)
                flagged_u.append(s)
        still = int(audit.get("unresolved_consecutive_events") or 0)
        passed = bool(n_events > 0 and checked > 0 and gap_rate <= 0.002)
        return {
            "checked": checked,
            "still_flagged": len(flagged_u),
            "gap_rate": gap_rate,
            "ca_events_loaded": n_events,
            "flagged": flagged_u[:15],
            "passed": passed,
            "metric": "unresolved_consecutive_session_symbol_rate",
            "legacy_note": audit.get("legacy_any_large_move_symbol_rate_note"),
            "by_class": audit.get("by_class"),
            "verified_ca_events": audit.get("verified_ca_events"),
            "sparse_or_suspension_events": audit.get("sparse_or_suspension_events"),
            "note": (
                "PASS — unresolved consecutive discontinuities collapsed"
                if passed else
                "FAIL — unresolved consecutive-session discontinuities remain "
                "(suspension/sparse spans are classified separately and do not "
                "alone fail this gate)"
                if n_events == 0 or gap_rate > 0.002 else
                "no data to check"
            ),
            "unresolved_consecutive_events": still,
        }
    except Exception as exc:
        # Fall back to legacy integrity_report if audit import/runtime fails
        rep = integrity_report(sample=sample)
        checked = rep.get("checked", 0)
        gap_rate = rep.get("gap_rate", 1.0)
        passed = bool(n_events > 0 and checked > 0 and gap_rate <= 0.002)
        return {"checked": checked, "still_flagged": rep.get("with_phantom_gaps", 0),
                "gap_rate": gap_rate, "ca_events_loaded": n_events,
                "flagged": rep.get("flagged", []), "passed": passed,
                "metric": "legacy_any_large_move_symbol_rate",
                "fallback_error": str(exc),
                "note": ("PASS — data continuous after adjustment" if passed else
                         "FAIL — supply/complete logs/ca_events.json until gap_rate≈0"
                         if n_events == 0 or gap_rate > 0.002 else "no data to check")}


def integrity_report(sample: int = 120) -> dict:
    """Store-wide data-health headline over a sample of symbols → the input the
    Governance Sentinel reads. Fail-open → {'checked': 0}.

    ``gap_rate`` here remains the *legacy* any-large-move rate for operational
    HALT signalling. Research certification must use ``verify_ca_adjustment`` /
    ``discontinuity_audit.audit_universe`` (unresolved consecutive rate).
    """
    try:
        from data.bhavcopy_store import store_symbols
        syms = store_symbols()[:sample]
    except Exception:
        return {"checked": 0, "ca_mismatch": False, "stale": False,
                "note": "store unavailable"}
    checked = 0
    with_gaps = 0
    stale = 0
    flagged: list[str] = []
    for s in syms:
        r = check_symbol(s)
        if r.get("ok") is None:
            continue
        checked += 1
        if r.get("gaps"):
            with_gaps += 1
            if len(flagged) < 15:
                flagged.append(s)
        if r.get("stale_days") is not None and r["stale_days"] > _STALE_DAYS:
            stale += 1
    gap_rate = (with_gaps / checked) if checked else 0.0
    return {
        "checked": checked,
        "with_phantom_gaps": with_gaps,
        "gap_rate": round(gap_rate, 3),
        "stale_symbols": stale,
        "flagged": flagged,
        # HALT-worthy operational signal (legacy). Research uses discontinuity_audit.
        "ca_mismatch": gap_rate > 0.02,
        "stale": checked > 0 and (stale / checked) > 0.5,
        "metric": "legacy_any_large_move_symbol_rate",
    }
