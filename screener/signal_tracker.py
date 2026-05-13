"""
Signal Tracker — SQLite-backed feedback loop for scanner signals.

Records every BUY/WATCH signal from the momentum scanner and checks
what the price did 3, 5, and 10 trading days later via yfinance.
"""
from __future__ import annotations

import sqlite3
import os
from datetime import date, timedelta
from typing import Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "signal_tracker.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS signal_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol           TEXT,
    signal           TEXT,
    price_at_signal  REAL,
    logged_at        TEXT,
    score            REAL,
    rsi              REAL,
    volume_ratio     REAL,
    price_3d         REAL,
    price_5d         REAL,
    price_10d        REAL,
    return_3d        REAL,
    return_5d        REAL,
    return_10d       REAL,
    outcome          TEXT
)
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_price(symbol: str, as_of: date) -> float | None:
    """Return the closing price of an NSE symbol on/after `as_of` using yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        # Pull a 5-day window starting from as_of to handle weekends / holidays
        end = as_of + timedelta(days=7)
        hist = ticker.history(start=as_of.isoformat(), end=end.isoformat(), interval="1d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[0])
    except Exception:
        return None


class SignalTracker:
    """SQLite-backed tracker for scanner BUY/WATCH signals."""

    def __init__(self) -> None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = _connect()
        conn.execute(_CREATE_TABLE)
        conn.commit()
        conn.close()

    # ── Write ──────────────────────────────────────────────────────────────

    def log_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        score: float,
        rsi: float,
        volume_ratio: float,
    ) -> None:
        """Insert a new signal row with outcome='PENDING'. Skips if already logged today."""
        today = date.today().isoformat()
        with _connect() as conn:
            existing = conn.execute(
                "SELECT id FROM signal_log WHERE symbol=? AND logged_at=?",
                (symbol, today),
            ).fetchone()
            if existing:
                return  # already logged today for this symbol
            conn.execute(
                """INSERT INTO signal_log
                   (symbol, signal, price_at_signal, logged_at, score, rsi, volume_ratio, outcome)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING')""",
                (symbol, signal, price, today, score, rsi, volume_ratio),
            )
            conn.commit()

    # ── Update outcomes ────────────────────────────────────────────────────

    def update_outcomes(self) -> int:
        """
        For all PENDING rows where enough calendar days have elapsed, fetch
        prices at the 3d / 5d / 10d marks and compute returns + outcome.

        Returns the number of rows updated.
        """
        today = date.today()
        updated = 0

        with _connect() as conn:
            pending = conn.execute(
                "SELECT id, symbol, price_at_signal, logged_at FROM signal_log WHERE outcome='PENDING'"
            ).fetchall()

            for row in pending:
                row_id     = row["id"]
                symbol     = row["symbol"]
                base_price = row["price_at_signal"]
                logged_at  = date.fromisoformat(row["logged_at"])

                days_elapsed = (today - logged_at).days
                if days_elapsed < 3:
                    continue  # too early to check anything

                updates: dict[str, Any] = {}

                # Fetch prices at each horizon if enough time has passed
                for horizon, col_p, col_r in [
                    (3,  "price_3d",  "return_3d"),
                    (5,  "price_5d",  "return_5d"),
                    (10, "price_10d", "return_10d"),
                ]:
                    if days_elapsed < horizon:
                        continue
                    # Only fetch if not already stored
                    existing_price = conn.execute(
                        f"SELECT {col_p} FROM signal_log WHERE id=?", (row_id,)
                    ).fetchone()[0]
                    if existing_price is not None:
                        updates[col_p] = existing_price
                        if base_price and base_price > 0:
                            updates[col_r] = round((existing_price - base_price) / base_price * 100, 2)
                        continue
                    target_date = logged_at + timedelta(days=horizon + 2)  # +2 buffer for weekends
                    price = _fetch_price(symbol, logged_at + timedelta(days=horizon))
                    if price:
                        updates[col_p] = price
                        if base_price and base_price > 0:
                            updates[col_r] = round((price - base_price) / base_price * 100, 2)

                if not updates:
                    continue

                # Determine outcome once we have a 5d return
                return_5d = updates.get("return_5d")
                if return_5d is None:
                    # Try reading from DB in case it was already stored
                    stored = conn.execute(
                        "SELECT return_5d FROM signal_log WHERE id=?", (row_id,)
                    ).fetchone()
                    if stored and stored[0] is not None:
                        return_5d = stored[0]

                if return_5d is not None:
                    if return_5d > 2.0:
                        outcome = "WIN"
                    elif return_5d < -2.0:
                        outcome = "LOSS"
                    else:
                        outcome = "NEUTRAL"
                    updates["outcome"] = outcome
                else:
                    updates["outcome"] = "PENDING"

                # Build SET clause
                set_clause = ", ".join(f"{k}=?" for k in updates)
                values = list(updates.values()) + [row_id]
                conn.execute(
                    f"UPDATE signal_log SET {set_clause} WHERE id=?",
                    values,
                )
                updated += 1

            conn.commit()

        return updated

    # ── Read ───────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate stats across all logged signals."""
        with _connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM signal_log").fetchone()[0]
            if total == 0:
                return {
                    "total_signals":    0,
                    "win_rate_pct":     0.0,
                    "avg_return_5d":    0.0,
                    "best_trade":       None,
                    "worst_trade":      None,
                    "signals_by_type":  {"BUY": 0, "WATCH": 0},
                }

            decided = conn.execute(
                "SELECT COUNT(*) FROM signal_log WHERE outcome IN ('WIN','LOSS','NEUTRAL')"
            ).fetchone()[0]
            wins = conn.execute(
                "SELECT COUNT(*) FROM signal_log WHERE outcome='WIN'"
            ).fetchone()[0]
            win_rate = round(wins / decided * 100, 1) if decided > 0 else 0.0

            avg_5d_row = conn.execute(
                "SELECT AVG(return_5d) FROM signal_log WHERE return_5d IS NOT NULL"
            ).fetchone()[0]
            avg_5d = round(avg_5d_row, 2) if avg_5d_row is not None else 0.0

            best_row = conn.execute(
                "SELECT symbol, return_5d FROM signal_log WHERE return_5d IS NOT NULL ORDER BY return_5d DESC LIMIT 1"
            ).fetchone()
            worst_row = conn.execute(
                "SELECT symbol, return_5d FROM signal_log WHERE return_5d IS NOT NULL ORDER BY return_5d ASC LIMIT 1"
            ).fetchone()

            by_type: dict[str, int] = {}
            for sig_type in ("BUY", "WATCH"):
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM signal_log WHERE signal=?", (sig_type,)
                ).fetchone()[0]
                by_type[sig_type] = cnt

        return {
            "total_signals":   total,
            "win_rate_pct":    win_rate,
            "avg_return_5d":   avg_5d,
            "best_trade":      {"symbol": best_row["symbol"],  "return_5d": round(best_row["return_5d"],  2)} if best_row  else None,
            "worst_trade":     {"symbol": worst_row["symbol"], "return_5d": round(worst_row["return_5d"], 2)} if worst_row else None,
            "signals_by_type": by_type,
        }

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent signals as a list of dicts."""
        with _connect() as conn:
            rows = conn.execute(
                """SELECT id, symbol, signal, price_at_signal, logged_at, score, rsi,
                          volume_ratio, price_3d, price_5d, price_10d,
                          return_3d, return_5d, return_10d, outcome
                   FROM signal_log
                   ORDER BY logged_at DESC, id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_symbol_history(self, symbol: str) -> list[dict[str, Any]]:
        """Return all signals for a specific symbol, newest first."""
        symbol = symbol.upper().strip()
        with _connect() as conn:
            rows = conn.execute(
                """SELECT id, symbol, signal, price_at_signal, logged_at, score, rsi,
                          volume_ratio, price_3d, price_5d, price_10d,
                          return_3d, return_5d, return_10d, outcome
                   FROM signal_log
                   WHERE symbol=?
                   ORDER BY logged_at DESC, id DESC""",
                (symbol,),
            ).fetchall()
        return [dict(r) for r in rows]
