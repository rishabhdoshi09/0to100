"""Causal SMA trend flags. Fail-closed on short history."""
from __future__ import annotations

import numpy as np


def sma_at(close, j: int, window: int) -> float | None:
    if j < window - 1 or window < 2:
        return None
    w = np.asarray(close[j - window + 1: j + 1], dtype=float)
    if w.size < window or np.any(~np.isfinite(w)) or np.any(w <= 0):
        return None
    return float(w.mean())


def trend_flag(close, j: int, window: int = 200, slope: int = 21, require_slope: bool = True) -> bool | None:
    now = sma_at(close, j, window)
    px = float(close[j]) if j >= 0 else float("nan")
    if now is None or not np.isfinite(px) or px <= 0:
        return None
    if not require_slope:
        return bool(px > now)
    prev = sma_at(close, j - slope, window)
    if prev is None:
        return None
    return bool(px > now and now > prev)


def dist_above_sma(close, j: int, window: int = 200) -> float | None:
    now = sma_at(close, j, window)
    px = float(close[j]) if j >= 0 else float("nan")
    if now is None or now <= 0 or not np.isfinite(px) or px <= 0:
        return None
    return px / now - 1.0
