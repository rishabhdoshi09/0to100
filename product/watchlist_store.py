"""User watchlist persistence for the retail terminal API.

SQLite store at logs/watchlist.db — shared schema with Streamlit watchlist UI.
No broker actions; read/write list metadata only.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Optional

DEFAULT_DB = Path("logs/watchlist.db")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path = DEFAULT_DB) -> None:
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                added_date TEXT NOT NULL,
                buy_zone_low REAL,
                buy_zone_high REAL,
                target_price REAL,
                stop_price REAL,
                notes TEXT,
                added_price REAL
            )
            """
        )
        conn.commit()


def list_items(path: Path = DEFAULT_DB) -> list[dict[str, Any]]:
    init_db(path)
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM watchlist ORDER BY added_date DESC, symbol ASC"
        ).fetchall()
    return [dict(row) for row in rows]


def add_item(
    symbol: str,
    *,
    buy_low: Optional[float] = None,
    buy_high: Optional[float] = None,
    target: Optional[float] = None,
    stop: Optional[float] = None,
    notes: str = "",
    added_price: Optional[float] = None,
    path: Path = DEFAULT_DB,
) -> dict[str, Any]:
    init_db(path)
    sym = str(symbol or "").strip().upper()
    if not sym:
        raise ValueError("symbol is required")
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO watchlist
                (symbol, added_date, buy_zone_low, buy_zone_high,
                 target_price, stop_price, notes, added_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sym,
                date.today().isoformat(),
                buy_low,
                buy_high,
                target,
                stop,
                str(notes or "").strip(),
                added_price,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM watchlist WHERE id = last_insert_rowid()"
        ).fetchone()
    return dict(row) if row else {"symbol": sym}


def remove_item(row_id: int, path: Path = DEFAULT_DB) -> bool:
    init_db(path)
    with _connect(path) as conn:
        cur = conn.execute("DELETE FROM watchlist WHERE id = ?", (int(row_id),))
        conn.commit()
    return int(cur.rowcount or 0) > 0
