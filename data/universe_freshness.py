"""Generic investability freshness — a last print is not a living listing.

A name that once traded must not stay investable forever after it stops printing.
Holidays are official missing sessions, not death. Multi-session silence is.
"""
from __future__ import annotations

from typing import Any

# Must have an official bar on the as-of session (EDGE-001 live_on_session).
REQUIRE_BAR_ON_SESSION = 0
# Allow a short holiday/halt gap (calendar days of official sessions, not wall days).
DEFAULT_MAX_STALE_SESSIONS = 5
# After this many official sessions without a print, treat as not currently tradable
# even if the listing master still shows the name (suspension / effective death).
HARD_STALE_SESSIONS = 40


def last_bar_date(frame) -> str | None:
    if frame is None or len(frame) == 0:
        return None
    try:
        import pandas as pd
        return str(pd.Timestamp(frame.index[-1]).date())
    except Exception:
        return None


def sessions_between(calendar: list[str], start: str, end: str) -> int:
    """Official sessions in (start, end] using a sorted ISO calendar."""
    n = 0
    for d in calendar:
        if start < d <= end:
            n += 1
    return n


def investability(
    *,
    symbol: str,
    as_of: str,
    listed: bool,
    delisted: bool,
    last_bar: str | None,
    calendar: list[str] | None = None,
    max_stale_sessions: int = REQUIRE_BAR_ON_SESSION,
    suspended: bool = False,
) -> dict[str, Any]:
    """Return tradable flag + reason. Does not invent listing/delist dates."""
    as_of = str(as_of)[:10]
    if delisted:
        return {
            "symbol": symbol, "as_of": as_of, "tradable": False,
            "reason": "delisted", "stale_sessions": None,
        }
    if suspended:
        return {
            "symbol": symbol, "as_of": as_of, "tradable": False,
            "reason": "suspended", "stale_sessions": None,
        }
    if not listed:
        return {
            "symbol": symbol, "as_of": as_of, "tradable": False,
            "reason": "not_listed", "stale_sessions": None,
        }
    if not last_bar:
        return {
            "symbol": symbol, "as_of": as_of, "tradable": False,
            "reason": "no_history", "stale_sessions": None,
        }
    if last_bar > as_of:
        return {
            "symbol": symbol, "as_of": as_of, "tradable": False,
            "reason": "future_bar_rejected", "stale_sessions": None,
        }
    if last_bar == as_of:
        return {
            "symbol": symbol, "as_of": as_of, "tradable": True,
            "reason": "bar_on_session", "stale_sessions": 0,
        }
    # last_bar < as_of
    if calendar:
        stale = sessions_between(calendar, last_bar, as_of)
    else:
        # Fallback: calendar-day gap is NOT session gap. Be conservative:
        # without a session calendar, only same-day bar is tradable when
        # max_stale_sessions == 0.
        stale = None
    if max_stale_sessions == REQUIRE_BAR_ON_SESSION:
        return {
            "symbol": symbol, "as_of": as_of, "tradable": False,
            "reason": "missing_session_bar", "stale_sessions": stale,
            "last_bar": last_bar,
        }
    if stale is None:
        return {
            "symbol": symbol, "as_of": as_of, "tradable": False,
            "reason": "stale_unknown_calendar", "last_bar": last_bar,
        }
    if stale > HARD_STALE_SESSIONS:
        return {
            "symbol": symbol, "as_of": as_of, "tradable": False,
            "reason": "hard_stale", "stale_sessions": stale, "last_bar": last_bar,
        }
    if stale > max_stale_sessions:
        return {
            "symbol": symbol, "as_of": as_of, "tradable": False,
            "reason": "stale", "stale_sessions": stale, "last_bar": last_bar,
        }
    return {
        "symbol": symbol, "as_of": as_of, "tradable": True,
        "reason": "fresh_enough", "stale_sessions": stale, "last_bar": last_bar,
    }
