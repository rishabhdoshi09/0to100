"""
📅 NSE session freshness — production-accurate replacement for the `latest >= as_of` placeholder.

Answers ONE question for PAPER_AUTO: given the wall clock (Asia/Kolkata) and the snapshot's latest
available session, is the data fresh enough to open NEW entries? Weekends, NSE holidays, the
intraday/pre-close window and the daily bhavcopy publication delay are all handled via the trading
calendar — never by comparing calendar dates alone.

Holidays are read from an existing calendar file if present (`data/nse_holidays.json` or
`logs/nse_holidays.json`, a JSON list of YYYY-MM-DD); otherwise weekends-only, reported honestly.
This module trades nothing and adds no data source.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from core.runtime_paths import logs_dir

_CLOSE_HM = (15, 30)                 # NSE regular close (IST)
_PUBLISH_CUTOFF_HM = (18, 0)         # after this IST time the day's bhavcopy is expected available
#: By this IST time on the NEXT trading session, the previous session's bhavcopy
#: is MANDATORY. Publication lag is tolerated through the evening and overnight;
#: it is never tolerated into the following session.
_NEXT_SESSION_DEADLINE_HM = (9, 0)


def _now_ist() -> datetime:
    try:
        from core.market_clock import now_ist_naive
        return now_ist_naive()
    except Exception:
        return datetime.now()


def load_holidays() -> set:
    for p in (Path("data/nse_holidays.json"),
              logs_dir() / "nse_holidays.json"):
        try:
            if p.exists():
                return {str(x) for x in json.loads(p.read_text())}
        except Exception:
            continue
    return set()                     # weekends-only (honest: no holiday table on disk)


def is_session(d: date, holidays: set) -> bool:
    return d.weekday() < 5 and d.isoformat() not in holidays


def previous_session(d: date, holidays: set) -> date:
    x = d - timedelta(days=1)
    while not is_session(x, holidays):
        x -= timedelta(days=1)
    return x


def latest_required_session(now: datetime, holidays: set, cutoff=_PUBLISH_CUTOFF_HM) -> date:
    """The most recent COMPLETED-and-PUBLISHED NSE session as of `now`.
    Today counts only once it's a session AND past the publication cutoff; otherwise the prior
    session is the latest required one (so pre-close / pre-publish never demands today's file)."""
    today = now.date()
    if is_session(today, holidays) and (now.hour, now.minute) >= cutoff:
        return today
    return previous_session(today, holidays)


def next_session(d: date, holidays: set) -> date:
    x = d + timedelta(days=1)
    while not is_session(x, holidays):
        x += timedelta(days=1)
    return x


def completed_session(now: datetime, holidays: set, close=_CLOSE_HM) -> date:
    """The most recent session that has CLOSED.

    Distinct from published: the exchange is done with this session, but its
    bhavcopy may not exist yet. Never confuse the two.
    """
    today = now.date()
    if is_session(today, holidays) and (now.hour, now.minute) >= close:
        return today
    return previous_session(today, holidays)


def publication_window(
    now: datetime | None = None,
    holidays: set | None = None,
    *,
    close=_CLOSE_HM,
    deadline=_NEXT_SESSION_DEADLINE_HM,
) -> dict:
    """THE canonical publication-cycle authority.

    Every freshness verdict in the product derives from this one function, so
    two callers with the same clock and the same session state can never
    disagree. It answers three separate questions that used to be conflated:

    ``completed_session``
        The latest session the exchange has finished. Says nothing about files.
    ``minimum_required_official_session``
        The oldest session whose archive is mandatory RIGHT NOW. Between a
        session's close and the next session's pre-open deadline this is the
        session BEFORE the completed one -- that is the publication grace.
    ``publication_deadline``
        The instant ``completed_session``'s archive stops being optional.

    The grace is time-bounded by construction: it expires at a wall-clock
    instant on the next trading day, never after "one more session".
    """
    now = now or _now_ist()
    holidays = holidays if holidays is not None else load_holidays()
    completed = completed_session(now, holidays, close)
    deadline_at = datetime.combine(
        next_session(completed, holidays), time(*deadline)
    )
    in_grace = now < deadline_at
    minimum_required = previous_session(completed, holidays) if in_grace else completed
    return {
        "completed_session": completed,
        "minimum_required_official_session": minimum_required,
        "publication_deadline": deadline_at,
        "in_publication_grace": bool(in_grace),
    }


def sessions_gap(latest: date, required: date, holidays: set) -> int:
    """Number of trading sessions in (latest, required]; 0 when latest >= required."""
    if latest >= required:
        return 0
    n, x = 0, latest
    while x < required:
        x += timedelta(days=1)
        if is_session(x, holidays):
            n += 1
    return n


def snapshot_freshness(latest_iso: str, *, now: datetime | None = None, holidays: set | None = None,
                       cutoff=_PUBLISH_CUTOFF_HM, allowance_sessions: int = 0) -> dict:
    """Verdict for the snapshot's latest session, derived from publication_window.

    The publication grace now comes from the window and is bounded by a wall
    clock, so ``allowance_sessions`` defaults to 0. It used to default to 1,
    which tolerated a missing session indefinitely -- a file absent all of the
    next trading day still read as FRESH. Anything above 0 re-opens that hole
    and no production caller passes it.
    """
    now = now or _now_ist()
    holidays = holidays if holidays is not None else load_holidays()
    try:
        latest = date.fromisoformat(latest_iso)
    except Exception:
        return {"fresh": False, "reason": "latest session date unparseable", "required": None}

    if latest > now.date():
        return {"fresh": False, "reason": "future-dated bars (latest > today)",
                "required": None, "latest": latest_iso}
    window = publication_window(now, holidays)
    completed = window["completed_session"]
    minimum_required = window["minimum_required_official_session"]
    gap = sessions_gap(latest, completed, holidays)
    # The window decides. allowance_sessions survives for callers that want to
    # widen tolerance FURTHER; it can no longer contradict the window, which is
    # how snapshot_freshness and official_history_freshness used to disagree.
    tolerated = minimum_required
    for _ in range(max(0, int(allowance_sessions))):
        tolerated = previous_session(tolerated, holidays)
    fresh = latest >= tolerated
    pending = fresh and latest < completed
    return {"fresh": bool(fresh), "required": completed.isoformat(), "latest": latest_iso,
            "sessions_behind": gap, "holidays_loaded": len(holidays),
            "completed_session": completed.isoformat(),
            "minimum_required_official_session": minimum_required.isoformat(),
            "publication_deadline": window["publication_deadline"].isoformat(),
            "publication_pending": bool(pending),
            "reason": ("publication pending — the completed session's archive has not "
                       "landed yet and is not mandatory until the deadline" if pending else
                       "fresh" if fresh else
                       f"{gap} completed session(s) missing; the archive for "
                       f"{minimum_required.isoformat()} is mandatory now — blocking new entries")}


def has_duplicate_sessions(rows) -> bool:
    """True if any (symbol, date) appears more than once — a data defect the snapshot must reject."""
    seen = set()
    for r in rows:
        key = (r[0], r[1])
        if key in seen:
            return True
        seen.add(key)
    return False
