"""
Breakout Memory — tracks which stocks have a history of false breakouts.
Quietly downweights confidence for repeat offenders.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Optional

# ── DB path ───────────────────────────────────────────────────────────────────
_DB_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_DB_PATH = os.path.join(_DB_DIR, "breakout_memory.db")


def _ensure_db() -> None:
    """Create the logs directory and DB table if they don't exist."""
    os.makedirs(_DB_DIR, exist_ok=True)
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS breakout_history (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol             TEXT,
                date               TEXT,
                pivot              REAL,
                was_real           INTEGER,   -- 1=real breakout held, 0=false
                follow_through_pct REAL
            )
            """
        )
        conn.commit()


@contextmanager
def _get_conn():
    """Yield a SQLite connection; auto-close on exit."""
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    try:
        yield conn
    finally:
        conn.close()


# ── Public API ────────────────────────────────────────────────────────────────

def record_breakout(
    symbol: str,
    pivot: float,
    was_real: bool,
    follow_through_pct: float = 0.0,
    breakout_date: Optional[str] = None,
) -> None:
    """
    Insert a breakout record into the history table.

    Args:
        symbol:             NSE symbol (e.g. 'INFY')
        pivot:              Pivot price at breakout
        was_real:           True if breakout held, False if it failed
        follow_through_pct: % move from pivot after the breakout (negative = failed)
        breakout_date:      ISO date string; defaults to today
    """
    _ensure_db()
    dt = breakout_date or date.today().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO breakout_history (symbol, date, pivot, was_real, follow_through_pct) "
            "VALUES (?, ?, ?, ?, ?)",
            (symbol.upper(), dt, float(pivot), int(was_real), float(follow_through_pct)),
        )
        conn.commit()


def get_false_breakout_rate(
    symbol: str,
    min_breakouts: int = 3,
) -> dict:
    """
    Returns false-breakout statistics for a symbol.

    Returns:
        {
            "total":       int,
            "false_count": int,
            "false_rate":  float,   # 0.0-1.0
            "warning":     str | None,
        }
    """
    _empty = {"total": 0, "false_count": 0, "false_rate": 0.0, "warning": None}
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT was_real FROM breakout_history WHERE symbol = ?",
                (symbol.upper(),),
            ).fetchall()
    except Exception:
        return _empty

    total = len(rows)
    if total < min_breakouts:
        return _empty

    false_count = sum(1 for (r,) in rows if r == 0)
    false_rate  = false_count / total

    warning: Optional[str] = None
    if false_rate > 0.5:
        warning = (
            f"{symbol.upper()}: {false_count} of {total} breakouts failed "
            f"({false_rate * 100:.0f}%)"
        )

    return {
        "total":       total,
        "false_count": false_count,
        "false_rate":  round(false_rate, 3),
        "warning":     warning,
    }


def confidence_penalty(symbol: str) -> float:
    """
    Returns a score multiplier (0.5–1.0) based on breakout history.

    - false_rate > 0.8  → 0.5  (severe penalty)
    - false_rate > 0.6  → 0.7  (moderate penalty)
    - otherwise         → 1.0
    """
    try:
        stats = get_false_breakout_rate(symbol)
        rate  = stats["false_rate"]
        if stats["total"] == 0:
            return 1.0
        if rate > 0.8:
            return 0.5
        if rate > 0.6:
            return 0.7
        return 1.0
    except Exception:
        return 1.0


def detect_and_record_breakouts(
    setups: list,
    days_later: int = 5,
) -> None:
    """
    Called from the engine pipeline after `days_later` trading days.
    For each setup that is old enough, checks whether the breakout held
    by comparing the current price to the pivot. Records the result.

    A breakout is considered real if the current price is >= pivot * 0.99
    (i.e. the stock has not fallen more than 1% below the pivot).

    Args:
        setups:     List of setup dicts with keys: symbol, price, pivot, date (ISO str)
        days_later: Number of trading days to wait before judging
    """
    try:
        cutoff = date.today() - timedelta(days=days_later)

        for setup in setups:
            try:
                sym   = setup.get("symbol", "")
                pivot = float(setup.get("pivot", 0))
                price = float(setup.get("price", 0))
                setup_date_str = setup.get("date", "")

                if not sym or pivot <= 0 or price <= 0:
                    continue

                # Only process setups old enough to be judged
                if setup_date_str:
                    try:
                        setup_date = date.fromisoformat(setup_date_str)
                        if setup_date > cutoff:
                            continue
                    except ValueError:
                        pass

                # Check if already recorded for this symbol + date
                with _get_conn() as conn:
                    already = conn.execute(
                        "SELECT id FROM breakout_history WHERE symbol=? AND date=? AND pivot=?",
                        (sym.upper(), setup_date_str, pivot),
                    ).fetchone()
                    if already:
                        continue

                # Determine outcome: current price vs pivot
                follow_through_pct = (price - pivot) / pivot * 100.0
                was_real = price >= pivot * 0.99  # held above pivot

                record_breakout(
                    symbol=sym,
                    pivot=pivot,
                    was_real=was_real,
                    follow_through_pct=follow_through_pct,
                    breakout_date=setup_date_str or date.today().isoformat(),
                )
            except Exception:
                continue  # never crash the pipeline

    except Exception:
        pass  # graceful fallback — breakout memory is non-critical
