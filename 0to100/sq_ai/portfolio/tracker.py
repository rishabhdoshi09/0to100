"""SQLite persistence layer for portfolio, trades, signals and equity curve."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    entry_date  TEXT NOT NULL,
    entry_price REAL NOT NULL,
    qty INTEGER NOT NULL,
    exit_date  TEXT,
    exit_price REAL,
    pnl REAL,
    stop_initial REAL,
    target REAL,
    status TEXT DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,           -- BUY / SELL / HOLD
    confidence REAL,
    regime INTEGER,
    claude_reasoning TEXT,
    extra_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, timestamp);

CREATE TABLE IF NOT EXISTS daily_equity (
    date TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS screener_results (
    date TEXT NOT NULL,
    rank INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    score REAL,
    reasoning TEXT,
    PRIMARY KEY(date, rank)
);
"""


class PortfolioTracker:
    """Thin SQLite repo. Thread-safe via a single lock + check_same_thread=False."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.environ.get("SQ_DB_PATH", "./data/sq_ai.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ------------------------------------------------------------------ utils
    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            con = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
            con.row_factory = sqlite3.Row
            try:
                yield con
                con.commit()
            finally:
                con.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --------------------------------------------------------------- prices
    def upsert_price(self, symbol: str, date: str, o: float, h: float,
                     low_: float, c: float, v: float) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO prices VALUES(?,?,?,?,?,?,?)",
                (symbol, date, o, h, low_, c, v),
            )

    def latest_close(self, symbol: str) -> float | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT close FROM prices WHERE symbol=? ORDER BY date DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        return row["close"] if row else None

    # --------------------------------------------------------------- trades
    def open_trade(self, symbol: str, entry_price: float, qty: int,
                   stop: float, target: float, ts: str | None = None) -> int:
        ts = ts or self._now()
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO trades(symbol, entry_date, entry_price, qty, "
                "stop_initial, target, status) VALUES(?,?,?,?,?,?,'open')",
                (symbol, ts, entry_price, qty, stop, target),
            )
            return cur.lastrowid

    def close_trade(self, trade_id: int, exit_price: float,
                    ts: str | None = None) -> float:
        ts = ts or self._now()
        with self._conn() as con:
            row = con.execute(
                "SELECT entry_price, qty FROM trades WHERE id=?", (trade_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"trade {trade_id} not found")
            pnl = (exit_price - row["entry_price"]) * row["qty"]
            con.execute(
                "UPDATE trades SET exit_date=?, exit_price=?, pnl=?, status='closed' "
                "WHERE id=?",
                (ts, exit_price, pnl, trade_id),
            )
        return pnl

    def open_positions(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY entry_date DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def closed_trades(self, limit: int = 100) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM trades WHERE status='closed' "
                "ORDER BY exit_date DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------------- signals
    def log_signal(self, symbol: str, action: str, confidence: float,
                   regime: int, reasoning: str, extra: dict | None = None) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO signals(timestamp, symbol, action, confidence, "
                "regime, claude_reasoning, extra_json) VALUES(?,?,?,?,?,?,?)",
                (self._now(), symbol, action, confidence, regime, reasoning,
                 json.dumps(extra or {})),
            )

    def latest_signals(self, limit: int = 20) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --------------------------------------------------------------- equity
    def record_equity(self, equity: float, cash: float,
                      date: str | None = None) -> None:
        date = date or datetime.now(timezone.utc).date().isoformat()
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO daily_equity(date, equity, cash) VALUES(?,?,?)",
                (date, equity, cash),
            )

    def equity_curve(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT date, equity, cash FROM daily_equity ORDER BY date ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- screener
    def save_screener(self, date: str, ranked: list[dict[str, Any]]) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM screener_results WHERE date=?", (date,))
            con.executemany(
                "INSERT INTO screener_results(date, rank, symbol, score, reasoning) "
                "VALUES(?,?,?,?,?)",
                [(date, i + 1, r["symbol"], r.get("score", 0.0),
                  r.get("reasoning", "")) for i, r in enumerate(ranked)],
            )

    def latest_screener(self) -> list[dict]:
        with self._conn() as con:
            row = con.execute(
                "SELECT MAX(date) as d FROM screener_results"
            ).fetchone()
            if not row or not row["d"]:
                return []
            rows = con.execute(
                "SELECT * FROM screener_results WHERE date=? ORDER BY rank",
                (row["d"],),
            ).fetchall()
        return [dict(r) for r in rows]
