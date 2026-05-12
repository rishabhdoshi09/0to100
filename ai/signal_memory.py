"""
Signal Memory — persistent trade outcome tracking with feedback loop.

Every confirmed trade is stored with its signal fingerprint (indicators,
agent scores, confidence). When a new signal fires, similar past signals
are retrieved and their outcomes adjust the current confidence.

This creates a closed loop: backtest + live outcomes → better live signals.

SQLite table: signal_memory
  id, symbol, action, confidence, agent_scores (JSON), indicators (JSON),
  outcome (WIN/LOSS/UNKNOWN), pnl_pct, timestamp, notes
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

_DB_PATH = Path("data/signal_memory.db")


@dataclass
class SignalRecord:
    symbol: str
    action: str                        # BUY | SELL | HOLD
    confidence: float
    agent_scores: dict = field(default_factory=dict)   # {technical: 78, ...}
    indicators: dict = field(default_factory=dict)     # {rsi_14: 62, ...}
    outcome: str = "UNKNOWN"           # WIN | LOSS | UNKNOWN
    pnl_pct: float = 0.0
    timestamp: str = ""
    notes: str = ""
    id: Optional[int] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class MemoryInsight:
    similar_count: int
    win_rate: float                  # among similar past signals
    avg_pnl_pct: float
    confidence_adjustment: float     # +/- to apply to current confidence
    summary: str


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_memory (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                action      TEXT NOT NULL,
                confidence  REAL NOT NULL,
                agent_scores TEXT DEFAULT '{}',
                indicators  TEXT DEFAULT '{}',
                outcome     TEXT DEFAULT 'UNKNOWN',
                pnl_pct     REAL DEFAULT 0.0,
                timestamp   TEXT NOT NULL,
                notes       TEXT DEFAULT ''
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_sm_symbol ON signal_memory(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_sm_outcome ON signal_memory(outcome)")
        conn.commit()


_init_db()


# ── Write ─────────────────────────────────────────────────────────────────────

def record_signal(record: SignalRecord) -> int:
    """Store a new signal. Returns the row id."""
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO signal_memory
               (symbol, action, confidence, agent_scores, indicators,
                outcome, pnl_pct, timestamp, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.symbol, record.action, record.confidence,
                json.dumps(record.agent_scores), json.dumps(record.indicators),
                record.outcome, record.pnl_pct, record.timestamp, record.notes,
            ),
        )
        conn.commit()
        log.info("signal_recorded", symbol=record.symbol, action=record.action,
                 confidence=record.confidence, id=cur.lastrowid)
        return cur.lastrowid


def update_outcome(record_id: int, outcome: str, pnl_pct: float) -> None:
    """Update WIN/LOSS + P&L after a trade closes."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE signal_memory SET outcome=?, pnl_pct=? WHERE id=?",
            (outcome, pnl_pct, record_id),
        )
        conn.commit()
    log.info("signal_outcome_updated", id=record_id, outcome=outcome, pnl_pct=pnl_pct)


# ── Read + Feedback ───────────────────────────────────────────────────────────

def query_similar(
    symbol: str,
    action: str,
    indicators: dict,
    lookback: int = 50,
) -> MemoryInsight:
    """
    Find past signals for this symbol+action and compute a confidence adjustment.

    Similarity is based on RSI range bucket (oversold/neutral/overbought) and
    momentum direction — a lightweight heuristic that doesn't require embeddings.

    Returns a MemoryInsight with the suggested confidence delta.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT outcome, pnl_pct, indicators, confidence
               FROM signal_memory
               WHERE symbol=? AND action=? AND outcome != 'UNKNOWN'
               ORDER BY id DESC LIMIT ?""",
            (symbol, action, lookback),
        ).fetchall()

    if not rows:
        return MemoryInsight(
            similar_count=0,
            win_rate=0.5,
            avg_pnl_pct=0.0,
            confidence_adjustment=0.0,
            summary="No historical signals found — no adjustment applied.",
        )

    current_rsi = indicators.get("rsi_14", 50)
    current_mom = indicators.get("momentum_5d_pct", 0)

    # Bucket current conditions
    rsi_bucket = _rsi_bucket(current_rsi)
    mom_bucket = _mom_bucket(current_mom)

    similar, wins, total_pnl = 0, 0, 0.0

    for row in rows:
        past_ind = json.loads(row["indicators"] or "{}")
        past_rsi = past_ind.get("rsi_14", 50)
        past_mom = past_ind.get("momentum_5d_pct", 0)

        if _rsi_bucket(past_rsi) == rsi_bucket and _mom_bucket(past_mom) == mom_bucket:
            similar += 1
            if row["outcome"] == "WIN":
                wins += 1
            total_pnl += row["pnl_pct"]

    if similar == 0:
        # Fall back to all results for this symbol+action
        similar = len(rows)
        wins = sum(1 for r in rows if r["outcome"] == "WIN")
        total_pnl = sum(r["pnl_pct"] for r in rows)

    win_rate = wins / similar if similar > 0 else 0.5
    avg_pnl = total_pnl / similar if similar > 0 else 0.0

    # Confidence adjustment: positive if win_rate > 60%, negative if < 40%
    if win_rate >= 0.60:
        adj = min(0.08, (win_rate - 0.5) * 0.4)   # max +8%
    elif win_rate <= 0.40:
        adj = max(-0.12, (win_rate - 0.5) * 0.4)  # max -12%
    else:
        adj = 0.0

    summary = (
        f"{similar} similar past signals → win rate {win_rate:.0%}, "
        f"avg P&L {avg_pnl:+.1f}%. "
        f"Confidence adjustment: {adj:+.0%}."
    )

    log.info("memory_insight", symbol=symbol, action=action,
             similar=similar, win_rate=win_rate, adj=adj)

    return MemoryInsight(
        similar_count=similar,
        win_rate=win_rate,
        avg_pnl_pct=avg_pnl,
        confidence_adjustment=adj,
        summary=summary,
    )


def get_performance_stats(symbol: str | None = None) -> dict:
    """Return aggregate win/loss stats. Pass symbol=None for global stats."""
    with _get_conn() as conn:
        base = "WHERE outcome != 'UNKNOWN'"
        params: tuple = ()
        if symbol:
            base += " AND symbol=?"
            params = (symbol,)
        row = conn.execute(
            f"""SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as wins,
                AVG(pnl_pct) as avg_pnl,
                MAX(pnl_pct) as best,
                MIN(pnl_pct) as worst
                FROM signal_memory {base}""",
            params,
        ).fetchone()

    total = row["total"] or 0
    wins  = row["wins"] or 0
    return {
        "total": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": wins / total if total > 0 else 0.0,
        "avg_pnl_pct": row["avg_pnl"] or 0.0,
        "best_pnl_pct": row["best"] or 0.0,
        "worst_pnl_pct": row["worst"] or 0.0,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rsi_bucket(rsi: float) -> str:
    if rsi < 35:
        return "oversold"
    if rsi > 65:
        return "overbought"
    return "neutral"


def _mom_bucket(mom_pct: float) -> str:
    if mom_pct > 2:
        return "strong_up"
    if mom_pct > 0:
        return "mild_up"
    if mom_pct < -2:
        return "strong_down"
    return "mild_down"
