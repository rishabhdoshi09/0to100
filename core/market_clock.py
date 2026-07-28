"""
Market clock — exchange ka time hi sach hai, machine ka nahi.

NSE gates (market hours, entry windows, daily-limit dates) pehle naive
`datetime.now()` / `date.today()` pe chalte the — jo MACHINE ke timezone
mein hota hai. Mac (IST) pe sahi; kisi UTC server/VPS pe **sab 5.5 ghante
shift**: 9:15 ka gate 2:45pm ban jata, daily limits subah 5:30 IST pe
reset hote. Ye woh bug hai jo ghar pe kabhi nahi dikhta aur production
server pe pehle din kaat leta hai.

Ab: har NSE-side gate is module ke IST-explicit clock se poochhta hai —
machine ka TZ kuch bhi ho, jawab wahi. (US side pehle se ET-explicit hai —
us_autopilot pytz America/New_York use karta hai; wahi discipline ab
IST side par.)
"""
from __future__ import annotations

from datetime import datetime, date

try:                                    # stdlib (3.9+); tzdata system se
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:                       # pragma: no cover — pytz fallback
    import pytz
    IST = pytz.timezone("Asia/Kolkata")


def now_ist() -> datetime:
    """Timezone-aware 'abhi' in IST — the only clock NSE gates trust."""
    return datetime.now(IST)


def today_ist() -> date:
    """IST calendar date — daily limits / one-shot-per-day keys yahi se."""
    return now_ist().date()


# ── Persisted-timestamp contract (canonical, see docs/architecture) ────────────
# STORAGE CONVENTION (legacy, documented): trade/journal timestamps are persisted
# as NAIVE IST wall-clock (`now_ist_naive().isoformat()`). Every "today" query must
# resolve the IST trading day via `ist_day_of()` / `is_ist_today()` — NEVER compare
# a naive machine `datetime.now()` against an IST date. A future migration to
# tz-aware UTC storage is deferred; until then this convention is single-sourced
# here so no writer/query can drift off it (root cause of audit item C-13).

def now_ist_naive() -> datetime:
    """The STORAGE clock: current IST wall-clock as a naive datetime. This is the
    canonical value to persist for trade/journal timestamps."""
    return now_ist().replace(tzinfo=None)


def ist_day_of(ts) -> str:
    """The IST trading-day (YYYY-MM-DD) of a stored timestamp. Accepts a naive-IST
    datetime/ISO string (the storage convention → its own date) OR a tz-aware value
    (converted into IST first). Robust to either so a query can never mis-bucket a
    trade across the UTC↔IST midnight boundary."""
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            return ts[:10]                     # already a date-prefixed string
    else:
        dt = ts
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.astimezone(IST)
    return dt.date().isoformat()


def is_ist_today(ts, today: str | None = None) -> bool:
    """True when `ts` falls on the IST trading day `today` (default: today IST).
    Pass an explicit `today` (from a single `today_ist()` read) so a batch of
    queries is internally consistent and deterministically testable."""
    return ist_day_of(ts) == (today or today_ist().isoformat())


def system_tz_is_ist() -> bool:
    """Diagnostics: kya machine ka local clock IST hai? Gates ab IST-explicit
    hain isliye galat TZ par bhi SAHI chalenge — par logs/cron timestamps
    shift dikhenge, isliye Diagnostics mein flag hota hai."""
    try:
        local = datetime.now().astimezone()
        return local.utcoffset() == now_ist().utcoffset()
    except Exception:
        return True                     # benefit of doubt — warn mat karo
