"""Cheap TTL-aware key/value cache backed by SQLite.

Used by every external-API integration (Alpha Vantage, NewsAPI, Screener.in,
Kite REST history, etc.) to respect free-tier rate limits and keep the
laptop snappy.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


_LOCK = threading.RLock()


def _db_path() -> str:
    return os.environ.get("SQ_DB_PATH", "./data/sq_ai.db")


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    Path(_db_path()).parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        con = sqlite3.connect(_db_path(), check_same_thread=False, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS kv_cache ("
                "  key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                "  expires_at REAL NOT NULL)"
            )
            yield con
            con.commit()
        finally:
            con.close()


def cache_get(key: str) -> Any | None:
    with _conn() as con:
        row = con.execute(
            "SELECT value, expires_at FROM kv_cache WHERE key=?", (key,)
        ).fetchone()
    if not row:
        return None
    if row["expires_at"] != 0 and row["expires_at"] < time.time():
        cache_delete(key)
        return None
    try:
        return json.loads(row["value"])
    except Exception:
        return None


def cache_set(key: str, value: Any, ttl_seconds: int = 3600) -> None:
    """``ttl_seconds=0`` means *forever* (e.g., transcripts)."""
    expires = 0 if ttl_seconds == 0 else time.time() + ttl_seconds
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO kv_cache(key, value, expires_at) VALUES(?,?,?)",
            (key, json.dumps(value, default=str), expires),
        )


def cache_delete(key: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM kv_cache WHERE key=?", (key,))


def cached(prefix: str, ttl_seconds: int = 3600) -> Callable:
    """Decorator: cache by ``prefix:args:kwargs``."""
    def deco(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            key = f"{prefix}:{json.dumps([args, kwargs], default=str)}"
            hit = cache_get(key)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            if result is not None:
                cache_set(key, result, ttl_seconds)
            return result
        wrapper.__wrapped__ = fn
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco
