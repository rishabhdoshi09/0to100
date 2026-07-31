"""
⏰ Job cadence — market-calendar and opening-noise aware.

Pure helpers the supervisor uses to decide WHEN a job is due and WHAT its idempotency key is. All
times are IST. New paper ENTRIES are forbidden before 09:30 (opening noise) and after 15:15; the
whole-market scan may still run for observation, but the paper cycle must not open new risk in the
opening-noise window.
"""
from __future__ import annotations

from datetime import time as _time

# job types
AUTH_HEALTH = "auth_health"
INSTRUMENT_REFRESH = "instrument_refresh"
DATA_REFRESH = "data_refresh"
BHAVCOPY_UPDATE = "bhavcopy_update"
CORPORATE_ACTIONS = "corporate_actions"
UNIVERSE_HISTORY = "universe_history"
INDEX_WARMUP = "index_warmup"
MARKET_SCAN = "market_scan"
NEWS_REFRESH = "news_refresh"
PAPER_CYCLE = "paper_cycle"
OUTCOME_RESOLUTION = "outcome_resolution"
LEARNING_CYCLE = "learning_cycle"

CRITICAL_JOBS = {AUTH_HEALTH, DATA_REFRESH, PAPER_CYCLE, OUTCOME_RESOLUTION}

# NSE cash session and the retail entry window (opening noise excluded)
MARKET_OPEN = _time(9, 15)
ENTRY_WINDOW_START = _time(9, 30)      # no new entries before this — opening noise
ENTRY_WINDOW_END = _time(15, 15)
MARKET_CLOSE = _time(15, 30)
SCAN_INTERVAL_MIN = 15


def _is_session_day(now_ist, holidays=None) -> bool:
    if now_ist.weekday() >= 5:
        return False
    if holidays and now_ist.date() in holidays:
        return False
    return True


def market_is_open(now_ist, holidays=None) -> bool:
    return _is_session_day(now_ist, holidays) and MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE


def entries_allowed_by_clock(now_ist, holidays=None) -> bool:
    """New paper ENTRIES require a live session AND to be past the opening-noise window."""
    return (_is_session_day(now_ist, holidays)
            and ENTRY_WINDOW_START <= now_ist.time() <= ENTRY_WINDOW_END)


def in_opening_noise(now_ist, holidays=None) -> bool:
    return (_is_session_day(now_ist, holidays)
            and MARKET_OPEN <= now_ist.time() < ENTRY_WINDOW_START)


def scan_slot(now_ist) -> str:
    """A deterministic label for the current scan slot (used in the idempotency key)."""
    t = now_ist.time()
    if t < ENTRY_WINDOW_START:
        return "premarket"
    if t > MARKET_CLOSE:
        return "eod"
    minute = (now_ist.hour * 60 + now_ist.minute)
    bucket = minute - (minute % SCAN_INTERVAL_MIN)
    return f"intraday-{bucket // 60:02d}{bucket % 60:02d}"


def scan_due(now_ist, last_scan_slot: str | None, holidays=None) -> bool:
    if not _is_session_day(now_ist, holidays):
        return False
    return scan_slot(now_ist) != last_scan_slot     # one immutable scan per slot; no unchanged re-scan


# ── idempotency keys — the same logical operation runs at most once ──────────────
def paper_cycle_key(snapshot_id: str, session_date: str) -> str:
    return f"paper_cycle:{snapshot_id}:{session_date}"


def scan_key(snapshot_id: str, slot: str) -> str:
    return f"market_scan:{snapshot_id}:{slot}"


def data_refresh_key(session_date: str) -> str:
    return f"data_refresh:{session_date}"


def learning_key(session_date: str) -> str:
    return f"learning_cycle:{session_date}"


def outcome_key(session_date: str) -> str:
    return f"outcome_resolution:{session_date}"
