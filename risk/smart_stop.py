"""
Smart Stop Placement — places stops at technically significant levels,
NOT at obvious round numbers where stop hunts cluster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

# Round numbers that attract stop hunts
_ROUND_LEVELS = [100, 250, 500, 1000, 2500, 5000]
_ROUND_PROXIMITY_PCT = 0.3   # within 0.3% of a round number = danger zone
_ROUND_AVOID_SHIFT   = 0.5   # move stop 0.5% further from entry


def _is_near_round_number(price: float, tolerance_pct: float = _ROUND_PROXIMITY_PCT) -> bool:
    """True if price is within tolerance_pct% of a known round number."""
    for base in _ROUND_LEVELS:
        nearest = round(price / base) * base
        if nearest > 0 and abs(price - nearest) / price * 100 < tolerance_pct:
            return True
    return False


def _compute_atr14(df: pd.DataFrame) -> float:
    """Returns 14-period ATR. Returns 0.0 on failure."""
    try:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        if len(h) < 15:
            return 0.0
        tr = np.array([
            max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            for i in range(1, len(h))
        ])
        return float(tr[-14:].mean())
    except Exception:
        return 0.0


def suggest_smart_stop(
    symbol: str,
    df: pd.DataFrame,
    entry_price: float,
    archetype: str,
) -> dict:
    """
    Suggest a smart stop level for an entry.

    Returns:
        suggested_stop  : float  — final recommended stop price
        raw_stop        : float  — original ATR-based stop
        adjusted        : bool   — True if moved away from round number
        reason          : str    — explanation
        risk_pct        : float  — stop distance as % of entry
    """
    if df is None or len(df) < 15 or entry_price <= 0:
        raw = round(entry_price * 0.95, 2)
        return {
            "suggested_stop": raw,
            "raw_stop":        raw,
            "adjusted":        False,
            "reason":          "Insufficient data — using 5% default stop",
            "risk_pct":        5.0,
        }

    atr14 = _compute_atr14(df)
    if atr14 <= 0:
        atr14 = entry_price * 0.015  # 1.5% fallback

    # 1. ATR-based stop: 1.5× ATR below entry
    atr_stop = round(entry_price * (1 - 1.5 * atr14 / entry_price), 2)
    adjusted = False
    reason = "1.5× ATR below entry"

    # 2. If ATR stop is near a round number, shift it further from entry
    if _is_near_round_number(atr_stop):
        shift_amt  = entry_price * _ROUND_AVOID_SHIFT / 100
        atr_stop   = round(atr_stop - shift_amt, 2)
        adjusted   = True
        log.debug("smart_stop_shifted_from_round", symbol=symbol,
                  original=round(entry_price * (1 - 1.5 * atr14 / entry_price), 2),
                  shifted=atr_stop)

    # 3. Volume support stop: highest-volume bar close in last 20 days
    vol_stop: float | None = None
    try:
        if "volume" in df.columns and len(df) >= 20:
            recent = df.iloc[-20:].copy()
            peak_vol_idx = recent["volume"].idxmax()
            vol_support  = float(recent.loc[peak_vol_idx, "close"])
            # Only use it if it's below entry and above zero
            if 0 < vol_support < entry_price:
                vol_stop = round(vol_support, 2)
    except Exception:
        vol_stop = None

    # 4. Choose better stop: the one higher (closer to entry) but still valid
    final_stop = atr_stop
    used_vol_support = False
    if vol_stop is not None and vol_stop > atr_stop:
        final_stop       = vol_stop
        used_vol_support = True
        adjusted         = False  # vol support overrides the round-number shift status
        reason           = f"below volume support at ₹{vol_stop:,.0f}"

    # Build final reason
    raw_atr_stop = round(entry_price * (1 - 1.5 * atr14 / entry_price), 2)
    if used_vol_support:
        pass  # reason already set
    elif adjusted:
        near_round = round(raw_atr_stop / min(_ROUND_LEVELS, key=lambda b: abs(raw_atr_stop - round(raw_atr_stop / b) * b)))
        reason = f"moved from round number zone near ₹{raw_atr_stop:,.0f}"
    else:
        reason = "1.5× ATR below entry"

    risk_pct = round((entry_price - final_stop) / entry_price * 100, 2) if entry_price > 0 else 0.0

    return {
        "suggested_stop": final_stop,
        "raw_stop":        raw_atr_stop,
        "adjusted":        adjusted,
        "reason":          reason,
        "risk_pct":        risk_pct,
    }


def format_stop_note(result: dict) -> str:
    """
    Returns a plain-English note about the stop placement.

    Examples:
      "Stop ₹2,347 (moved from ₹2,350 — too obvious, stop hunt zone)"
      "Stop ₹2,340 (below volume support at ₹2,345)"
      "Stop ₹2,347 (1.5× ATR below entry)"
    """
    stop      = result.get("suggested_stop", 0)
    raw_stop  = result.get("raw_stop", 0)
    adjusted  = result.get("adjusted", False)
    reason    = result.get("reason", "")

    stop_fmt = f"₹{stop:,.0f}"

    if adjusted and raw_stop and abs(raw_stop - stop) > 0.5:
        raw_fmt = f"₹{raw_stop:,.0f}"
        return f"Stop {stop_fmt} (moved from {raw_fmt} — too obvious, stop hunt zone)"
    elif "volume support" in reason:
        return f"Stop {stop_fmt} ({reason})"
    else:
        return f"Stop {stop_fmt} (1.5× ATR below entry)"
