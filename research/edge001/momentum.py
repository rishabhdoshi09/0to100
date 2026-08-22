"""Causal momentum scores. Fail-closed on missing history."""
from __future__ import annotations


def skip_momentum(close, j: int, lookback: int, skip: int = 21) -> float | None:
    """close[j-skip] / close[j-lookback] - 1. Requires j >= lookback."""
    if j < lookback or lookback <= skip:
        return None
    start = close[j - lookback]
    end = close[j - skip]
    if start is None or end is None:
        return None
    try:
        p0, p1 = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if p0 <= 0 or p1 <= 0 or p0 != p0 or p1 != p1:
        return None
    return p1 / p0 - 1.0


def incl_momentum(close, j: int, lookback: int) -> float | None:
    if j < lookback:
        return None
    start = close[j - lookback]
    end = close[j]
    try:
        p0, p1 = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if p0 <= 0 or p1 <= 0 or p0 != p0 or p1 != p1:
        return None
    return p1 / p0 - 1.0
