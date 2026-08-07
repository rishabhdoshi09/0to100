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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from logger import get_logger

log = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = _ROOT / "data" / "fundamentals_cache.db"
_TTL = 86_400  # 1 day in seconds


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection that is always closed.

    ``with sqlite3.connect(...)`` alone does not close the FD on modern CPython.
    """
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=5.0)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fundamentals_cache (
                symbol     TEXT PRIMARY KEY,
                data_json  TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
            """
        )
        conn.commit()
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


class FundamentalsCache:
    """SQLite cache — short-lived connections, always closed after each call."""

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

    def get_any(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return cached payload regardless of TTL (for lazy fallback after failed scrape)."""
        symbol = symbol.upper()
        with _connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM fundamentals_cache WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def has(self, symbol: str) -> bool:
        symbol = symbol.upper()
        with _connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM fundamentals_cache WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return row is not None

    def stats(self) -> dict[str, Any]:
        """Row counts for API status — no bulk backfill required."""
        with _connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM fundamentals_cache").fetchone()[0]
            fresh_cutoff = time.time() - _TTL
            fresh = conn.execute(
                "SELECT COUNT(*) FROM fundamentals_cache WHERE fetched_at >= ?",
                (fresh_cutoff,),
            ).fetchone()[0]
        return {
            "symbols_cached": int(total or 0),
            "symbols_fresh": int(fresh or 0),
            "symbols_stale": max(0, int(total or 0) - int(fresh or 0)),
            "db_path": str(_DB_PATH),
            "ttl_hours": round(_TTL / 3600, 1),
            "lazy_sync": True,
        }

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
        log.info("fundamentals_cache_written", symbol=symbol, bytes=len(payload))

    def clear_old(self) -> int:
        """Delete entries older than TTL. Returns count deleted."""
        cutoff = time.time() - _TTL
        with _connect() as conn:
            cursor = conn.execute(
                "DELETE FROM fundamentals_cache WHERE fetched_at < ?", (cutoff,)
            )
            count = int(cursor.rowcount or 0)
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
        log.debug("fundamentals_cache_invalidated", symbol=symbol)
