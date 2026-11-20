"""Exact official-session outcome windows, shared by the walk-forward and every settler.

The one rulebook (core.evaluation_spec) gives a trade ``horizon`` official NSE
sessions. A window built from a symbol's next ``horizon`` *rows* silently
stretches over a suspension or a missing archive, so a target first touched in
the 11th official session could be booked as a 10-session WIN (GPT
cross-review 2026-11-20, A-7/B-4/B-5).

``official_session_index`` is the ordinal map those windows are checked
against:

* the realised official sessions (the bhavcopy archive dates; weekends and
  calendar-closed days dropped, as ``scan.signal_backtest.realised_session_index``);
* PLUS every session the verified exchange calendar lists inside the realised
  span for the years it covers, so a session whose archive is missing
  market-wide leaves a hole rather than compressing two days together;
* MINUS dates in years after the last covered year, which the calendar cannot
  vouch for yet: a window reaching into them is never verified.

Uncovered PAST years (before the calendar's first covered year) keep the
realised archive dates only. That residual is stated, not hidden: there a
market-wide missing archive is not detectable from the calendar.
"""
from __future__ import annotations

import bisect
import threading
from datetime import date, timedelta
from typing import Iterable

_CACHE_LOCK = threading.Lock()
_CACHE: tuple | None = None


def official_session_index(dates: Iterable) -> dict[str, int]:
    """{YYYY-MM-DD: ordinal} of verified official sessions (see the module doc)."""
    from scan.signal_backtest import realised_session_index

    base = realised_session_index(dates)
    if not base:
        return {}
    try:
        from research.intelligence.data.nse_calendar import covered_years, is_session, load_holidays

        holidays = load_holidays()
        years = covered_years(holidays)
    except Exception:
        years, holidays, is_session = set(), set(), None
    keep = set(base)
    if years and is_session is not None:
        lo = date.fromisoformat(min(keep))
        hi = date.fromisoformat(max(keep))
        last_covered = max(years)
        for year in sorted(years):
            cur = max(lo, date(year, 1, 1))
            end = min(hi, date(year, 12, 31))
            while cur <= end:
                if is_session(cur, holidays):
                    keep.add(cur.isoformat())
                cur += timedelta(days=1)
        keep = {d for d in keep if int(d[:4]) <= last_covered}
    return {d: i for i, d in enumerate(sorted(keep))}


def official_dates() -> list[str]:
    """Sorted verified official sessions from the bhavcopy archive, [] when unavailable.

    Cached on the archive listing's identity (length and end points), so the
    settlement loops do not rebuild it per row.
    """
    global _CACHE
    try:
        from data.bhavcopy_store import _dates_on_disk

        disk = [d.isoformat() if hasattr(d, "isoformat") else str(d)[:10] for d in _dates_on_disk()]
    except Exception:
        return []
    if not disk:
        return []
    key = (len(disk), disk[0], disk[-1])
    with _CACHE_LOCK:
        if _CACHE is not None and _CACHE[0] == key:
            return list(_CACHE[1])
    out = sorted(official_session_index(disk))
    with _CACHE_LOCK:
        _CACHE = (key, tuple(out))
    return out


def verified_prefix(bar_days: list[str], decided_on: str, sessions: list[str]) -> int:
    """How many leading bars are exactly the next official sessions after ``decided_on``.

    ``bar_days`` are a symbol's bar dates strictly after the decision, in order.
    The count stops at the first bar that is not the expected session (a
    missing session, an unofficial bar) and at the end of the verified list.
    """
    pos = bisect.bisect_right(sessions, str(decided_on)[:10])
    n = 0
    for got, want in zip(bar_days, sessions[pos:]):
        if got != want:
            break
        n += 1
    return n
