"""
Setup Freshness — how long has this pattern been forming?
Sweet spot: VCP 3-8 weeks, Accumulation 4-12 weeks, Flag 1-3 weeks.
Outside window = stale or too early.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# ── Optimal freshness windows (trading days) ──────────────────────────────────
_FRESHNESS_WINDOWS: dict[str, tuple[int, int]] = {
    "VCP_BREAKOUT":          (21, 56),   # 3-8 weeks
    "HIGH_TIGHT_FLAG":       (5,  21),   # 1-3 weeks
    "ACCUMULATION_BREAKOUT": (28, 84),   # 4-12 weeks
    "MOMENTUM_EXPANSION":    (5,  30),   # 1-6 weeks
    "EARNINGS_CONTINUATION": (3,  15),   # days after earnings
    "TREND_CONTINUATION":    (10, 60),   # 2-12 weeks
    "MEAN_REVERSION":        (5,  20),   # 1-4 weeks
    "EARLY_LEADER":          (10, 40),   # 2-8 weeks
}

# Default window for unknown archetypes
_DEFAULT_WINDOW = (10, 50)


def _detect_pattern_start(df: pd.DataFrame, lookback: int = 90) -> int:
    """
    Estimate how many bars ago the consolidation pattern started.
    Heuristic: find the bar with the lowest low in the last `lookback` bars.
    Returns the number of bars since that low (i.e., pattern age in trading days).
    """
    try:
        low = df["low"].values if "low" in df.columns else df["Low"].values
        window = low[-lookback:] if len(low) >= lookback else low
        # Index of the lowest low within the window (relative to window start)
        rel_idx = int(np.argmin(window))
        # Age = bars since that low
        age = len(window) - 1 - rel_idx
        return max(1, age)
    except Exception:
        return 20  # neutral fallback


def _freshness_score(age: int, min_days: int, max_days: int) -> float:
    """
    Score peaks at 100 at the midpoint of [min_days, max_days].
    Falls to 0 at the edges (age == min_days or age == max_days).
    Outside the window it decays further below 0 (clamped to 0).
    """
    mid = (min_days + max_days) / 2.0
    half_width = (max_days - min_days) / 2.0
    if half_width == 0:
        return 100.0 if age == mid else 0.0
    distance_from_mid = abs(age - mid)
    raw = 1.0 - (distance_from_mid / half_width)
    return max(0.0, min(100.0, raw * 100.0))


def _freshness_status(age: int, min_days: int, max_days: int) -> str:
    """
    Classify freshness status based on age vs optimal window.
    """
    if age < max(1, min_days - 5):
        return "EARLY"
    if age < min_days:
        return "FRESH"
    if age <= max_days:
        mid = (min_days + max_days) / 2.0
        half = (max_days - min_days) / 2.0
        if abs(age - mid) <= half * 0.35:
            return "OPTIMAL"
        return "MATURE" if age > mid else "FRESH"
    if age <= max_days + max(5, int(max_days * 0.15)):
        return "MATURE"
    return "STALE"


def _make_note(age: int, status: str, min_days: int, max_days: int) -> str:
    weeks = age / 5.0
    if status == "OPTIMAL":
        return f"Forming for {weeks:.1f} weeks — optimal window"
    elif status == "FRESH":
        return f"Forming for {weeks:.1f} weeks — entering optimal range"
    elif status == "EARLY":
        remaining = (min_days - age)
        return f"Only {age} days old — optimal starts in ~{remaining} days"
    elif status == "MATURE":
        remaining = max_days - age
        return f"Forming for {weeks:.1f} weeks — {remaining} days left in window"
    else:  # STALE
        over = age - max_days
        return f"Forming for {weeks:.1f} weeks — {over} days past optimal window"


def compute_freshness(
    symbol: str,
    archetype: str,
    df: pd.DataFrame,
) -> dict:
    """
    Detect how long the setup has been forming and score its freshness.

    Returns:
        {
            "age_days": int,
            "status": "EARLY" | "FRESH" | "OPTIMAL" | "MATURE" | "STALE",
            "score": float,   # 0-100, peaks at optimal window midpoint
            "note": str,
        }
    """
    min_days, max_days = _FRESHNESS_WINDOWS.get(archetype.upper(), _DEFAULT_WINDOW)

    age = _detect_pattern_start(df, lookback=90)
    status = _freshness_status(age, min_days, max_days)
    score  = _freshness_score(age, min_days, max_days)
    note   = _make_note(age, status, min_days, max_days)

    return {
        "age_days": age,
        "status":   status,
        "score":    round(score, 1),
        "note":     note,
    }


def freshness_badge(result: dict) -> str:
    """
    Returns an HTML badge for the freshness status.
    OPTIMAL = green, FRESH/MATURE = amber, EARLY/STALE = red.
    """
    status = result.get("status", "FLAT")
    age    = result.get("age_days", 0)
    weeks  = age / 5.0

    if status == "OPTIMAL":
        color = "#00d4a0"
        label = f"OPTIMAL ({weeks:.1f}w)"
    elif status in ("FRESH", "MATURE"):
        color = "#f59e0b"
        label = f"{status} ({weeks:.1f}w)"
    else:  # EARLY or STALE
        color = "#ff4b4b"
        label = f"{status} ({weeks:.1f}w)"

    return (
        f"<span style='color:{color};font-size:.65rem;font-weight:700;"
        f"background:{color}15;padding:1px 6px;border-radius:4px;"
        f"border:1px solid {color}44'>{label}</span>"
    )
