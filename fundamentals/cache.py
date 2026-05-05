"""
SQLite-backed cache for fundamental data.

Table schema:
  symbol     TEXT PRIMARY KEY
  data_json  TEXT
  fetched_at REAL  (Unix timestamp)

TTL = 86 400 seconds (1 trading day).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from logger import get_logger

log = get_logger(__name__)

_DB_PATH = Path("data/fundamentals_cache.db")
_TTL = 86_400  # 1 day in seconds


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


class FundamentalsCache:
    """Thread-unsafe SQLite cache — use one instance per process."""

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return cached data if fresh, else None."""
        symbol = symbol.upper()
        with _connect() as conn:
            row = conn.execute(
                "SELECT data_json, fetched_at FROM fundamentals_cache WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        data_json, fetched_at = row
        age = time.time() - fetched_at
        if age > _TTL:
            log.debug("fundamentals_cache_stale", symbol=symbol, age_hours=round(age / 3600, 1))
            return None
        log.info("fundamentals_cache_hit", symbol=symbol, age_minutes=round(age / 60, 1))
        return json.loads(data_json)

    def set(self, symbol: str, data: Dict[str, Any]) -> None:
        """Store data with current timestamp."""
        symbol = symbol.upper()
        payload = json.dumps(data, ensure_ascii=False)
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
        """Delete entries older than TTL. Returns count deleted."""
        cutoff = time.time() - _TTL
        with _connect() as conn:
            cursor = conn.execute(
                "DELETE FROM fundamentals_cache WHERE fetched_at < ?", (cutoff,)
            )
            conn.commit()
            count = cursor.rowcount
        if count:
            log.info("fundamentals_cache_cleared_old", count=count)
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
