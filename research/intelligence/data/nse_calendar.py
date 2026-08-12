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
from datetime import date, datetime, timedelta
from pathlib import Path

_CLOSE_HM = (15, 30)                 # NSE regular close (IST)
_PUBLISH_CUTOFF_HM = (18, 0)         # after this IST time the day's bhavcopy is expected available


def _now_ist() -> datetime:
    try:
        from core.market_clock import now_ist_naive
        return now_ist_naive()
    except Exception:
        return datetime.now()


def load_holidays() -> set:
    for p in (Path("data/nse_holidays.json"),
              Path(__file__).resolve().parents[3] / "logs" / "nse_holidays.json"):
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
                       cutoff=_PUBLISH_CUTOFF_HM, allowance_sessions: int = 1) -> dict:
    """Verdict for the snapshot's latest session. `allowance_sessions` is the publication grace:
    right after the cutoff the just-completed file may not be imported yet, so a gap up to the
    allowance stays FRESH; beyond it is STALE (block new entries)."""
    now = now or _now_ist()
    holidays = holidays if holidays is not None else load_holidays()
    try:
        latest = date.fromisoformat(latest_iso)
    except Exception:
        return {"fresh": False, "reason": "latest session date unparseable", "required": None}

    if latest > now.date():
        return {"fresh": False, "reason": "future-dated bars (latest > today)",
                "required": None, "latest": latest_iso}
    required = latest_required_session(now, holidays, cutoff)
    gap = sessions_gap(latest, required, holidays)
    fresh = gap <= allowance_sessions
    return {"fresh": bool(fresh), "required": required.isoformat(), "latest": latest_iso,
            "sessions_behind": gap, "holidays_loaded": len(holidays),
            "reason": ("fresh" if fresh else
                       f"{gap} completed session(s) missing beyond the {allowance_sessions}-session "
                       "publication allowance — blocking new entries")}


def has_duplicate_sessions(rows) -> bool:
    """True if any (symbol, date) appears more than once — a data defect the snapshot must reject."""
    seen = set()
    for r in rows:
        key = (r[0], r[1])
        if key in seen:
            return True
        seen.add(key)
    return False
