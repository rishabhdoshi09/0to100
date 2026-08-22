"""Causal proximity-to-high. Fail-closed on short / invalid windows."""
from __future__ import annotations

import numpy as np


def proximity_to_high(close, j: int, window: int = 252) -> float | None:
    """close[j] / max(close[j-window+1:j+1]). None if the window is unusable."""
    if j < window - 1 or window < 2:
        return None
    w = np.asarray(close[j - window + 1: j + 1], dtype=float)
    if w.size < window or np.any(~np.isfinite(w)) or np.any(w <= 0):
        return None
    peak = float(w.max())
    last = float(w[-1])
    if peak <= 0 or last <= 0:
        return None
    return last / peak
