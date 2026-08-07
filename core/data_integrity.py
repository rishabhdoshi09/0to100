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
    """Acceptance test for Phase-1 corporate-action adjustment: read the store
    THROUGH the adjust-on-read path and confirm the phantom-gap rate has collapsed
    to ~0. This is what turns 'we applied a CA table' into 'the data is now
    trustworthy'. Returns {checked, still_flagged, gap_rate, ca_events_loaded,
    passed}. `passed` requires an events table to actually be loaded AND a
    post-adjustment gap_rate ≤ 0.2% — an empty table that leaves gaps must FAIL,
    never green-light un-adjusted data. Fail-open → passed=False."""
    try:
        from data.corporate_actions import load_events
        n_events = len(load_events())
    except Exception:
        n_events = 0
    rep = integrity_report(sample=sample)
    checked = rep.get("checked", 0)
    gap_rate = rep.get("gap_rate", 1.0)
    passed = bool(n_events > 0 and checked > 0 and gap_rate <= 0.002)
    return {"checked": checked, "still_flagged": rep.get("with_phantom_gaps", 0),
            "gap_rate": gap_rate, "ca_events_loaded": n_events,
            "flagged": rep.get("flagged", []), "passed": passed,
            "note": ("PASS — data continuous after adjustment" if passed else
                     "FAIL — supply/complete logs/ca_events.json until gap_rate≈0"
                     if n_events == 0 or gap_rate > 0.002 else "no data to check")}


def integrity_report(sample: int = 120) -> dict:
    """Store-wide data-health headline over a sample of symbols → the input the
    Governance Sentinel reads. Fail-open → {'checked': 0}."""
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
        # HALT-worthy: a broad phantom-gap rate means the store is likely
        # un-adjusted for corporate actions — every backtest number is suspect.
        "ca_mismatch": gap_rate > 0.02,
        "stale": checked > 0 and (stale / checked) > 0.5,
    }
