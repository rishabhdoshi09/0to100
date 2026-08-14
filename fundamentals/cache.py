"""
SQLite-backed cache for fundamental data.

Table schema:
  symbol     TEXT PRIMARY KEY
  data_json  TEXT
  fetched_at REAL  (Unix timestamp)

Freshness = same IST calendar day as fetch (once per trading day).
Rolling 24h is NOT used — yesterday's scrape is stale at IST midnight.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from logger import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = _ROOT / "data" / "fundamentals_cache.db"


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_cache (
            symbol     TEXT PRIMARY KEY,
            data_json  TEXT NOT NULL,
            fetched_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _ist_day_of_epoch(fetched_at: float) -> str:
    try:
        from core.market_clock import IST
        return datetime.fromtimestamp(float(fetched_at), tz=IST).date().isoformat()
    except Exception:
        return datetime.utcfromtimestamp(float(fetched_at)).date().isoformat()


def _today_ist() -> str:
    try:
        from core.market_clock import today_ist
        return today_ist().isoformat()
    except Exception:
        return datetime.utcnow().date().isoformat()


def is_fresh_fetch(fetched_at: float, *, today: str | None = None) -> bool:
    """True when the payload was fetched on the given IST calendar day."""
    return _ist_day_of_epoch(fetched_at) == (today or _today_ist())


class FundamentalsCache:
    """Thread-unsafe SQLite cache — use one instance per process."""

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return cached data only if fetched today (IST). Else None."""
        symbol = symbol.upper()
        with _connect() as conn:
            row = conn.execute(
                "SELECT data_json, fetched_at FROM fundamentals_cache WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        data_json, fetched_at = row
        if not is_fresh_fetch(fetched_at):
            log.debug(
                "fundamentals_cache_stale_day",
                symbol=symbol,
                fetched_day=_ist_day_of_epoch(fetched_at),
                today=_today_ist(),
            )
            return None
        log.info(
            "fundamentals_cache_hit",
            symbol=symbol,
            age_minutes=round((time.time() - fetched_at) / 60, 1),
        )
        data = json.loads(data_json)
        if isinstance(data, dict):
            data = {
                **data,
                "_qt_cache_status": "TODAY",
                "_qt_cache_day": _ist_day_of_epoch(fetched_at),
                "_qt_fetched_at": float(fetched_at),
            }
        return data

    def get_any(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return cached payload regardless of day (offline / scrape-fail fallback).

        Callers must treat this as STALE when ``get()`` returned None — never as
        current fundamentals.
        """
        symbol = symbol.upper()
        with _connect() as conn:
            row = conn.execute(
                "SELECT data_json, fetched_at FROM fundamentals_cache WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        fetched_at = float(row[1] or 0)
        if isinstance(data, dict):
            day = _ist_day_of_epoch(fetched_at) if fetched_at else ""
            data = {
                **data,
                "_qt_cache_status": "TODAY" if is_fresh_fetch(fetched_at) else "STALE",
                "_qt_cache_day": day,
                "_qt_fetched_at": fetched_at,
            }
        return data

    def has(self, symbol: str) -> bool:
        symbol = symbol.upper()
        with _connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM fundamentals_cache WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return row is not None

    def stats(self) -> dict[str, Any]:
        """Row counts for API status — fresh = fetched on today's IST date."""
        today = _today_ist()
        with _connect() as conn:
            rows = conn.execute(
                "SELECT fetched_at FROM fundamentals_cache"
            ).fetchall()
        total = len(rows)
        fresh = sum(1 for (fa,) in rows if is_fresh_fetch(float(fa or 0), today=today))
        return {
            "symbols_cached": int(total or 0),
            "symbols_fresh": int(fresh),
            "symbols_stale": max(0, int(total) - int(fresh)),
            "db_path": str(_DB_PATH),
            "freshness": "ist_calendar_day",
            "today_ist": today,
            "lazy_sync": True,
        }

    def set(self, symbol: str, data: Dict[str, Any]) -> None:
        """Store data with current timestamp."""
        symbol = symbol.upper()
        clean = {
            k: v for k, v in dict(data or {}).items()
            if not str(k).startswith("_qt_")
        }
        payload = json.dumps(clean, ensure_ascii=False)
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO fundamentals_cache (symbol, data_json, fetched_at)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE
                    SET data_json  = excluded.data_json,
                        fetched_at = excluded.fetched_at
                """,
                (symbol, payload, time.time()),
            )
            conn.commit()
        log.info("fundamentals_cache_written", symbol=symbol, bytes=len(payload))

    def clear_old(self) -> int:
        """Delete entries not fetched on today's IST day. Returns count deleted."""
        today = _today_ist()
        with _connect() as conn:
            rows = conn.execute(
                "SELECT symbol, fetched_at FROM fundamentals_cache"
            ).fetchall()
            stale_syms = [
                sym for sym, fa in rows
                if not is_fresh_fetch(float(fa or 0), today=today)
            ]
            count = 0
            for sym in stale_syms:
                conn.execute(
                    "DELETE FROM fundamentals_cache WHERE symbol = ?", (sym,)
                )
                count += 1
            conn.commit()
        if count:
            log.info("fundamentals_cache_cleared_old", count=count, today_ist=today)
        return count

    def invalidate(self, symbol: str) -> None:
        """Force-expire a single symbol."""
        symbol = symbol.upper()
        with _connect() as conn:
            conn.execute(
                "DELETE FROM fundamentals_cache WHERE symbol = ?", (symbol,)
            )
            conn.commit()
        log.debug("fundamentals_cache_invalidated", symbol=symbol)
