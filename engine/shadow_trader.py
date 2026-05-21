"""
Shadow Trader — parallel paper simulation alongside live trading.

Every live trade decision is mirrored in paper. After 20+ shadow trades,
compare paper vs live PnL to detect:
  - Slippage/routing issues (paper beats live by >15%)
  - Unusually good fills (live beats paper by >15%)
  - Divergence >20% either direction → auto-pause and alert

SQLite-backed. Thread-safe.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

_DB = Path("logs/shadow_trades.db")
_LOCK = threading.Lock()
_DIVERGENCE_THRESHOLD = 0.20   # 20% PnL divergence triggers alert
_MIN_SAMPLE = 20               # minimum trades before divergence analysis


def _init_db() -> None:
    _DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(_DB)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol       TEXT NOT NULL,
                archetype    TEXT,
                entry_price  REAL,
                stop_level   REAL,
                target_price REAL,
                live_entry   REAL,
                paper_entry  REAL,
                live_exit    REAL,
                paper_exit   REAL,
                live_pnl     REAL,
                paper_pnl    REAL,
                entered_at   TEXT,
                exited_at    TEXT,
                status       TEXT DEFAULT 'open'
            )
        """)
        con.commit()


_init_db()


def record_entry(
    symbol: str,
    archetype: str,
    signal_price: float,    # the price the LLM/system decided on
    live_fill_price: float, # actual Kite fill price
    stop_level: float,
    target_price: float,
) -> int:
    """Record a new shadow trade. Returns trade_id."""
    with _LOCK:
        with sqlite3.connect(str(_DB)) as con:
            cur = con.execute(
                "INSERT INTO shadow_trades "
                "(symbol, archetype, entry_price, stop_level, target_price, "
                " live_entry, paper_entry, entered_at, status) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (symbol, archetype, signal_price, stop_level, target_price,
                 live_fill_price, signal_price,   # paper gets signal price (no slippage)
                 datetime.now().isoformat(), "open"),
            )
            trade_id = cur.lastrowid
            con.commit()
    log.info("shadow_entry_recorded", symbol=symbol, trade_id=trade_id,
             live_fill=live_fill_price, paper_fill=signal_price,
             slippage_pct=round((live_fill_price - signal_price) / signal_price * 100, 3))
    return trade_id


def record_exit(
    trade_id: int,
    live_exit_price: float,
    paper_exit_price: float,
) -> Optional[dict]:
    """
    Record exit prices and compute PnL divergence.
    Returns divergence report dict if threshold breached, else None.
    """
    with _LOCK:
        with sqlite3.connect(str(_DB)) as con:
            row = con.execute(
                "SELECT live_entry, paper_entry, symbol FROM shadow_trades WHERE id=?",
                (trade_id,)
            ).fetchone()
            if not row:
                return None
            live_entry, paper_entry, symbol = row

            live_pnl  = (live_exit_price  - live_entry)  / live_entry  * 100
            paper_pnl = (paper_exit_price - paper_entry) / paper_entry * 100

            con.execute(
                "UPDATE shadow_trades SET live_exit=?, paper_exit=?, live_pnl=?, "
                "paper_pnl=?, exited_at=?, status='closed' WHERE id=?",
                (live_exit_price, paper_exit_price, live_pnl, paper_pnl,
                 datetime.now().isoformat(), trade_id),
            )
            con.commit()

    divergence = paper_pnl - live_pnl
    log.info("shadow_exit_recorded", symbol=symbol, trade_id=trade_id,
             live_pnl=round(live_pnl, 2), paper_pnl=round(paper_pnl, 2),
             divergence=round(divergence, 2))

    if abs(divergence) > _DIVERGENCE_THRESHOLD * 100:
        return _build_divergence_alert(symbol, live_pnl, paper_pnl, divergence)
    return None


def _build_divergence_alert(symbol, live_pnl, paper_pnl, divergence) -> dict:
    if divergence > 0:
        direction = "Paper beating live"
        cause = "Possible slippage or poor fill quality on live execution."
        action = "Check Kite order logs for slippage. Consider limit orders."
    else:
        direction = "Live beating paper"
        cause = "Unusually good fills — verify no data error."
        action = "Confirm fills in Kite console. Could be price improvement."
    return {
        "alert": True,
        "symbol": symbol,
        "direction": direction,
        "live_pnl": round(live_pnl, 2),
        "paper_pnl": round(paper_pnl, 2),
        "divergence_pct": round(divergence, 2),
        "cause": cause,
        "action": action,
    }


def get_divergence_report() -> dict:
    """
    Aggregate divergence analysis across all closed shadow trades.
    Returns empty dict if < MIN_SAMPLE trades closed.
    """
    with sqlite3.connect(str(_DB)) as con:
        rows = con.execute(
            "SELECT live_pnl, paper_pnl, symbol, exited_at "
            "FROM shadow_trades WHERE status='closed' "
            "ORDER BY exited_at DESC LIMIT 100"
        ).fetchall()

    if len(rows) < _MIN_SAMPLE:
        return {"status": "insufficient_data", "closed_trades": len(rows), "required": _MIN_SAMPLE}

    total_live  = sum(r[0] for r in rows)
    total_paper = sum(r[1] for r in rows)
    avg_divergence = (total_paper - total_live) / len(rows)
    pct_divergence = (total_paper - total_live) / max(abs(total_live), 0.01) * 100

    status = "healthy"
    warning = None
    if abs(pct_divergence) > 20:
        status = "DIVERGENCE_ALERT"
        warning = (
            f"{'Paper' if pct_divergence > 0 else 'Live'} is outperforming by "
            f"{abs(pct_divergence):.1f}% over {len(rows)} trades. Investigate execution quality."
        )

    return {
        "status": status,
        "closed_trades": len(rows),
        "total_live_pnl_pct": round(total_live, 2),
        "total_paper_pnl_pct": round(total_paper, 2),
        "avg_divergence_pct": round(avg_divergence, 2),
        "overall_divergence_pct": round(pct_divergence, 2),
        "warning": warning,
    }


def get_open_shadow_trades() -> list[dict]:
    with sqlite3.connect(str(_DB)) as con:
        rows = con.execute(
            "SELECT id, symbol, archetype, live_entry, paper_entry, entered_at "
            "FROM shadow_trades WHERE status='open'"
        ).fetchall()
    return [{"id": r[0], "symbol": r[1], "archetype": r[2],
             "live_entry": r[3], "paper_entry": r[4], "entered_at": r[5]}
            for r in rows]
