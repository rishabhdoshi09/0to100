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


def system_tz_is_ist() -> bool:
    """Diagnostics: kya machine ka local clock IST hai? Gates ab IST-explicit
    hain isliye galat TZ par bhi SAHI chalenge — par logs/cron timestamps
    shift dikhenge, isliye Diagnostics mein flag hota hai."""
    try:
        local = datetime.now().astimezone()
        return local.utcoffset() == now_ist().utcoffset()
    except Exception:
        return True                     # benefit of doubt — warn mat karo
