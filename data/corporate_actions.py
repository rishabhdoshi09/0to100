"""
🔧 Corporate-Action Adjustment — make historical prices continuous & trustworthy.

The professor's disqualifier, fixed: NSE bhavcopy is UNADJUSTED, so a 1:1 bonus
or a 1→5 split reads as a phantom −50% / −80% crash. `core.data_integrity`
DETECTS those gaps; this module REMOVES them, so a backtest measures price
action, not accounting.

Design (deliberately minimal, and honest about data):

  • An "event" is {symbol, ex_date, factor, type}. `factor` = the multiple by
    which the SHARE COUNT rose on the ex-date: a 1:1 bonus → 2.0, a 1→5 split
    → 5.0, a 3:1 bonus (3 new per 1 held) → 4.0. Prices BEFORE the ex-date are
    divided by the cumulative factor; volumes are multiplied — the standard
    back-adjustment that pins history to today's share base.
  • adjust_frame() is a PURE function over a datetime-indexed OHLCV frame.
  • load_events() reads a real CA table from `logs/ca_events.json`. If that file
    is ABSENT it returns {} and the whole system behaves exactly as before —
    there is NO synthesised or guessed adjustment (invariant #1: no fake data).
    The events themselves must come from NSE corporate-actions archives; this
    module cannot and will not invent them.

Adjustment is applied ON READ (data.bhavcopy_store.get_ohlcv), so the on-disk
store stays raw, there is no double-adjustment across rebuilds, and an updated
CA table takes effect with no re-download.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_CA_FILE = Path(__file__).resolve().parent.parent / "logs" / "ca_events.json"
_VALID_TYPES = {"split", "bonus", "consolidation", "dividend"}


def _events_path() -> Path:
    override = os.getenv("QT_CA_EVENTS_FILE")
    return Path(override) if override else _CA_FILE


def load_events(path=None) -> dict:
    """Read the corporate-action table → {SYMBOL: [event, ...]}. Returns {} when
    the file is absent or unreadable — NEVER a fabricated table (no-file ⇒ the
    system runs exactly as it did before, un-adjusted but flagged by the
    integrity guard). Each event: {ex_date, factor>0, type}. Malformed rows are
    dropped, not guessed."""
    import pandas as pd
    p = Path(path) if path else _events_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return {}
    out: dict[str, list[dict]] = {}
    for row in raw if isinstance(raw, list) else []:
        try:
            sym = str(row["symbol"]).strip().upper()
            factor = float(row["factor"])
            ex = pd.Timestamp(row["ex_date"])
            typ = str(row.get("type", "split")).lower()
            if not sym or factor <= 0 or factor == 1.0 or pd.isna(ex):
                continue
            if typ not in _VALID_TYPES:
                continue
            out.setdefault(sym, []).append(
                {"ex_date": ex, "factor": factor, "type": typ})
        except Exception:
            continue
    for sym in out:
        out[sym].sort(key=lambda e: e["ex_date"])
    return out


def adjust_frame(df, events):
    """Back-adjust one symbol's OHLCV frame for its corporate actions. PURE.

    `df` is datetime-indexed with any of open/high/low/close/volume (deliv_per, a
    percentage, is left untouched). `events` is that symbol's event list. Bars
    STRICTLY BEFORE an ex-date have prices divided by the event's factor and
    volume multiplied by it, applied cumulatively across all events. Returns a new
    frame; the input is not mutated. Empty/absent events → the frame is returned
    unchanged (a copy)."""
    if df is None or getattr(df, "empty", True) or not events:
        return df.copy() if df is not None else df
    import numpy as np
    import pandas as pd
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    # divisor[i] = product of factors of every event whose ex_date is AFTER bar i
    divisor = np.ones(len(out), dtype=float)
    for e in events:
        before = idx < e["ex_date"]
        divisor[before.to_numpy() if hasattr(before, "to_numpy") else before] *= e["factor"]
    price_cols = [c for c in ("open", "high", "low", "close") if c in out.columns]
    for c in price_cols:
        out[c] = out[c].to_numpy(dtype=float) / divisor
    if "volume" in out.columns:
        out["volume"] = out["volume"].to_numpy(dtype=float) * divisor
    return out


def is_continuous(df, threshold_pct: float = None) -> bool:
    """True when the (adjusted) close series has NO phantom gap — the acceptance
    test for a correct adjustment. Delegates to the same detector the integrity
    guard uses, so 'adjusted' means exactly 'the guard now passes'."""
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return True
    from core.data_integrity import phantom_gaps, _GAP_PCT
    thr = threshold_pct if threshold_pct is not None else _GAP_PCT
    return len(phantom_gaps(df["close"].to_numpy(dtype=float), thr)) == 0
