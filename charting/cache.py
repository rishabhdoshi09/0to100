"""SQLite cache for computed charting artefacts (volume profile, footprint)."""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from logger import get_logger

log = get_logger(__name__)

_DB_PATH = Path("data/charting_cache.db")
_TTL = 3_600  # 1 hour


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS charting_cache (
            cache_key  TEXT PRIMARY KEY,
            data_blob  BLOB NOT NULL,
            fetched_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _make_key(prefix: str, params: dict) -> str:
    raw = json.dumps(params, sort_keys=True)
    return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def get(prefix: str, params: dict) -> Optional[Any]:
    key = _make_key(prefix, params)
    with _connect() as conn:
        row = conn.execute(
            "SELECT data_blob, fetched_at FROM charting_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
    if row is None:
        return None
    data_blob, fetched_at = row
    if time.time() - fetched_at > _TTL:
        return None
    log.debug("charting_cache_hit", key=key)
    return pickle.loads(data_blob)


def put(prefix: str, params: dict, data: Any) -> None:
    key = _make_key(prefix, params)
    blob = pickle.dumps(data)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO charting_cache (cache_key, data_blob, fetched_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE
                SET data_blob = excluded.data_blob,
                    fetched_at = excluded.fetched_at
            """,
            (key, blob, time.time()),
        )
        conn.commit()
    log.debug("charting_cache_written", key=key, bytes=len(blob))


def clear_old() -> int:
    cutoff = time.time() - _TTL
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM charting_cache WHERE fetched_at < ?", (cutoff,)
        )
        conn.commit()
    return cursor.rowcount
