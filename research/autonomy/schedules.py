"""Pure IST/NSE schedule helpers for the durable autonomy supervisor.

There are explicit windows: midnight is not "premarket", scans do not repeat unchanged off-hours,
and new paper risk is structurally restricted to 09:30–15:15 IST on an NSE session day.
"""
from __future__ import annotations

from datetime import time as _time

AUTH_HEALTH = "auth_health"
INSTRUMENT_REFRESH = "instrument_refresh"
DATA_REFRESH = "data_refresh"
BHAVCOPY_UPDATE = "bhavcopy_update"
CORPORATE_ACTIONS = "corporate_actions"
UNIVERSE_HISTORY = "universe_history"
OPTIONS_EOD = "options_eod"
INDEX_WARMUP = "index_warmup"
MARKET_SCAN = "market_scan"
NEWS_REFRESH = "news_refresh"
STREET_PULSE = "street_pulse"
PAPER_CYCLE = "paper_cycle"
OUTCOME_RESOLUTION = "outcome_resolution"
LEARNING_CYCLE = "learning_cycle"
RESEARCH_CYCLE = "research_cycle"
LONG_TERM_SCAN = "long_term_scan"
LONG_TERM_REFRESH = "long_term_refresh"

ALL_JOB_TYPES = (
    AUTH_HEALTH, INSTRUMENT_REFRESH, DATA_REFRESH, BHAVCOPY_UPDATE, CORPORATE_ACTIONS,
    UNIVERSE_HISTORY, OPTIONS_EOD, INDEX_WARMUP, MARKET_SCAN, NEWS_REFRESH, STREET_PULSE,
    PAPER_CYCLE, OUTCOME_RESOLUTION, LEARNING_CYCLE, RESEARCH_CYCLE, LONG_TERM_SCAN,
    LONG_TERM_REFRESH,
)
CRITICAL_JOBS = {AUTH_HEALTH, DATA_REFRESH, PAPER_CYCLE, OUTCOME_RESOLUTION}

AUTH_WINDOW_START = _time(7, 30)
AUTH_WINDOW_END = _time(10, 0)
PREMARKET_SCAN_START = _time(9, 0)
PREMARKET_SCAN_END = _time(9, 10)
MARKET_OPEN = _time(9, 15)
ENTRY_WINDOW_START = _time(9, 30)
ENTRY_WINDOW_END = _time(15, 15)
MARKET_CLOSE = _time(15, 30)
EOD_WINDOW_START = _time(18, 5)
EOD_WINDOW_END = _time(20, 0)
SCAN_INTERVAL_MIN = 15


def _is_session_day(now_ist, holidays=None) -> bool:
    return now_ist.weekday() < 5 and not (holidays and now_ist.date() in holidays)


def in_auth_window(now_ist, holidays=None) -> bool:
    return _is_session_day(now_ist, holidays) and AUTH_WINDOW_START <= now_ist.time() <= AUTH_WINDOW_END


def in_premarket_window(now_ist, holidays=None) -> bool:
    return (_is_session_day(now_ist, holidays)
            and PREMARKET_SCAN_START <= now_ist.time() <= PREMARKET_SCAN_END)


def market_is_open(now_ist, holidays=None) -> bool:
    return _is_session_day(now_ist, holidays) and MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE


def in_scan_window(now_ist, holidays=None) -> bool:
    return (_is_session_day(now_ist, holidays)
            and ENTRY_WINDOW_START <= now_ist.time() <= ENTRY_WINDOW_END)


def in_eod_window(now_ist, holidays=None) -> bool:
    return (_is_session_day(now_ist, holidays)
            and EOD_WINDOW_START <= now_ist.time() <= EOD_WINDOW_END)


def entries_allowed_by_clock(now_ist, holidays=None) -> bool:
    return in_scan_window(now_ist, holidays)


def in_opening_noise(now_ist, holidays=None) -> bool:
    return (_is_session_day(now_ist, holidays)
            and MARKET_OPEN <= now_ist.time() < ENTRY_WINDOW_START)


def session_phase(now_ist, holidays=None) -> str:
    if not _is_session_day(now_ist, holidays):
        return "off_session"
    if in_premarket_window(now_ist, holidays):
        return "premarket"
    if in_opening_noise(now_ist, holidays):
        return "opening_noise"
    if in_scan_window(now_ist, holidays):
        return "intraday"
    if in_eod_window(now_ist, holidays):
        return "eod"
    return "off_session"


def kite_login_optional(now_ist, holidays=None) -> bool:
    """True when Zerodha session is not required for bhavcopy-first product paths."""
    if not _is_session_day(now_ist, holidays):
        return True
    if in_auth_window(now_ist, holidays) or market_is_open(now_ist, holidays):
        return False
    return session_phase(now_ist, holidays) == "off_session"


def next_session_start(now_ist, holidays=None):
    """Next NSE session day at AUTH_WINDOW_START (IST-aware when input is aware)."""
    from datetime import datetime, timedelta

    holidays = holidays or set()
    tz = getattr(now_ist, "tzinfo", None)
    probe = now_ist.date()
    for _ in range(14):
        if probe.weekday() < 5 and probe not in holidays:
            start = datetime.combine(probe, AUTH_WINDOW_START)
            if tz is not None:
                start = start.replace(tzinfo=tz)
            if start > now_ist:
                return start
        probe += timedelta(days=1)
    return None


def scan_slot(now_ist, holidays=None) -> str | None:
    """Deterministic scan slot, or None when no scan is scheduled."""
    if in_premarket_window(now_ist, holidays):
        return "premarket"
    if in_scan_window(now_ist, holidays):
        minute = now_ist.hour * 60 + now_ist.minute
        bucket = minute - (minute % SCAN_INTERVAL_MIN)
        return f"intraday-{bucket // 60:02d}{bucket % 60:02d}"
    if in_eod_window(now_ist, holidays):
        return "eod"
    return None


def scan_due(now_ist, last_scan_slot: str | None, holidays=None) -> bool:
    slot = scan_slot(now_ist, holidays)
    return bool(slot and slot != last_scan_slot)


def auth_probe_bucket(now_ist) -> str:
    """Five-minute probes in the login window; 30-minute buckets outside it."""
    minutes = now_ist.hour * 60 + now_ist.minute
    size = 5 if AUTH_WINDOW_START <= now_ist.time() <= AUTH_WINDOW_END else 30
    bucket = minutes - minutes % size
    return f"{bucket // 60:02d}{bucket % 60:02d}"


def paper_cycle_key(snapshot_id: str, session_and_slot: str) -> str:
    return f"paper_cycle:{snapshot_id}:{session_and_slot}"


def scan_key(snapshot_id: str, slot: str, session_date: str | None = None) -> str:
    suffix = f":{session_date}" if session_date else ""
    return f"market_scan:{snapshot_id}:{slot}{suffix}"


def data_refresh_key(session_date: str) -> str:
    return f"data_refresh:{session_date}"


def eod_data_refresh_key(session_date: str) -> str:
    return f"data_refresh:{session_date}:eod"


def eod_bhavcopy_key(session_date: str) -> str:
    return f"bhavcopy_update:{session_date}:eod"


def instrument_key(session_date: str) -> str:
    return f"instrument_refresh:{session_date}"


def bhavcopy_key(session_date: str) -> str:
    return f"bhavcopy_update:{session_date}"


def corporate_actions_key(session_date: str) -> str:
    return f"corporate_actions:{session_date}"


def universe_history_key(session_date: str) -> str:
    return f"universe_history:{session_date}"


def options_eod_key(session_date: str) -> str:
    return f"options_eod:{session_date}"


def eod_options_key(session_date: str) -> str:
    return f"options_eod:{session_date}:eod"


def index_warmup_key(session_date: str) -> str:
    return f"index_warmup:{session_date}"


def news_key(session_date: str, bucket: str) -> str:
    return f"news_refresh:{session_date}:{bucket}"


def street_pulse_key(session_date: str, slot: str) -> str:
    return f"street_pulse:{session_date}:{slot}"


def learning_key(session_date: str) -> str:
    return f"learning_cycle:{session_date}"


def outcome_key(session_date: str) -> str:
    return f"outcome_resolution:{session_date}"


def research_key(session_date: str) -> str:
    return f"research_cycle:{session_date}"


def long_term_weekly_due(now_ist, holidays=None) -> bool:
    """Friday EOD is the automatic long-term review window."""
    return in_eod_window(now_ist, holidays) and now_ist.weekday() == 4


def long_term_key(session_date: str, *, refresh: bool = False) -> str:
    kind = "refresh" if refresh else "scan"
    return f"long_term_{kind}:{session_date}"
