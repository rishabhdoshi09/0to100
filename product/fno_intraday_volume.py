"""Same-clock cumulative volume confirmation for live NSE stock F&O scans.

A partial session must NEVER be compared against 20 completed daily volumes.
Read only Kite's historical minute candles for the small set of stocks whose
price has already crossed a 20-day level. No broker mutation, synthetic volume,
look-ahead, extrapolation, or unavailable-data approval occurs here.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
import math
from statistics import median
from typing import Any, Mapping
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
OPEN = time(9, 15)
LAST_ENTRY = time(15, 30)
PRIOR_SESSIONS = 5
MAX_PRIOR_SESSIONS = 10
MIN_MINUTE_COVERAGE = 0.80


def _minute_stamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt.replace(tzinfo=IST) if dt.tzinfo is None else dt.astimezone(IST)


def same_clock_rvol(
    *,
    client: Any,
    instrument_token: int,
    as_of: date,
    observed_at: datetime,
    cumulative_volume: float,
    minimum_sessions: int = PRIOR_SESSIONS,
) -> dict[str, Any]:
    """Compare today's observed cumulative volume with genuine prior sessions.

    Each prior session uses ONLY observed Kite minute bars from 09:15 through
    the same minute as the current quote. Incomplete prior sessions are omitted.
    Historical daily volume is not substituted for unavailable minute evidence.
    """
    def unavailable(reason: str, **extra: Any) -> dict[str, Any]:
        return {"available": False, "reason": reason, "source": "KITE_PRIOR_MINUTE_CUMULATIVE", **extra}

    now = _minute_stamp(observed_at)
    if now is None or now.date() != as_of or not (time(9, 30) <= now.time() < LAST_ENTRY):
        return unavailable("OUTSIDE_CURRENT_ENTRY_SESSION")
    if instrument_token <= 0:
        return unavailable("NSE_EQUITY_INSTRUMENT_UNAVAILABLE")
    try:
        current = float(cumulative_volume)
    except (TypeError, ValueError, OverflowError):
        return unavailable("CURRENT_VOLUME_INVALID")
    if not math.isfinite(current) or current < 0:
        return unavailable("CURRENT_VOLUME_INVALID")
    if not callable(getattr(client, "historical", None)):
        return unavailable("KITE_MINUTE_HISTORY_UNAVAILABLE")

    cutoff = now.hour * 60 + now.minute
    expected = cutoff - (9 * 60 + 15) + 1
    start = (as_of - timedelta(days=28)).isoformat()
    end = (as_of - timedelta(days=1)).isoformat()
    try:
        bars = client.historical(instrument_token, start, end, interval="minute")
    except Exception:
        return unavailable("KITE_MINUTE_HISTORY_FETCH_FAILED")

    by_day: dict[date, dict[int, float]] = defaultdict(dict)
    for row in bars or ():
        if not isinstance(row, Mapping):
            continue
        dt = _minute_stamp(row.get("date") or row.get("timestamp"))
        if dt is None or not (as_of - timedelta(days=28) <= dt.date() < as_of):
            continue
        minute = dt.hour * 60 + dt.minute
        if not (9 * 60 + 15 <= minute <= cutoff):
            continue
        try:
            volume = float(row.get("volume"))
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(volume) or volume < 0:
            continue
        # Deduplication prevents replayed Kite bars from doubling a session.
        by_day[dt.date()][minute] = volume

    totals: list[float] = []
    for session in sorted(by_day, reverse=True):
        minute_volumes = by_day[session]
        if (len(minute_volumes) < math.ceil(MIN_MINUTE_COVERAGE * expected)
                or min(minute_volumes) > 9 * 60 + 16
                or max(minute_volumes) < cutoff - 2):
            continue
        total = sum(minute_volumes.values())
        if total > 0:
            totals.append(total)
        if len(totals) >= MAX_PRIOR_SESSIONS:
            break

    if len(totals) < max(1, int(minimum_sessions)):
        return unavailable(
            "SAME_CLOCK_HISTORY_INSUFFICIENT",
            prior_sessions=len(totals), required_sessions=max(1, int(minimum_sessions)),
        )

    # Median protects against a single exceptionally high or low prior day.
    baseline = median(totals)
    if baseline <= 0:
        return unavailable("SAME_CLOCK_BASELINE_INVALID")
    return {
        "available": True, "reason": "SAME_CLOCK_PRIOR_SESSIONS_CONFIRMED",
        "source": "KITE_PRIOR_MINUTE_CUMULATIVE",
        "rvol": round(current / baseline, 4),
        "current_cumulative_volume": round(current, 2),
        "prior_median_cumulative_volume": round(baseline, 2),
        "prior_sessions": len(totals),
        "as_of": as_of.isoformat(), "minute_ist": now.strftime("%H:%M"),
        "synthetic": False, "paper_only": True,
    }
