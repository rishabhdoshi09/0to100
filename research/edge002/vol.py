"""Causal realized volatility. Fail-closed on short or non-finite history."""
from __future__ import annotations

import numpy as np


def realized_vol(close, j: int, lookback: int) -> float | None:
    """Annualized stdev of log returns using close[j-lookback : j] inclusive of j.

    Requires j >= lookback and lookback >= 2. Future bars after j are unused.
    """
    if j < lookback or lookback < 2:
        return None
    window = np.asarray(close[j - lookback: j + 1], dtype=float)
    if window.size < lookback + 1:
        return None
    if np.any(~np.isfinite(window)) or np.any(window <= 0):
        return None
    logp = np.log(window)
    r = np.diff(logp)
    if r.size < 2 or np.any(~np.isfinite(r)):
        return None
    sd = float(np.std(r, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return sd * float(np.sqrt(252.0))
