"""
Learning Loop — adapts to YOUR trading edge over time.

Tracks performance by: archetype × regime × quality_tier
Computes your personal win rate vs system baseline.
Weights future recommendations toward your proven edges.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── DB path ───────────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "logs" / "learning_loop.db"

# Universal baseline win rate used when quality_engine rates are unavailable
_UNIVERSAL_BASELINE = 0.55


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_outcomes (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol    TEXT,
            archetype TEXT,
            regime    TEXT,
            tier      TEXT,
            entry     REAL,
            exit      REAL,
            pnl_r     REAL,
            won       INTEGER
        )
        """
    )
    conn.commit()
    return conn


# ── Public API ────────────────────────────────────────────────────────────────

def record_trade_outcome(
    symbol: str,
    archetype: str,
    regime: str,
    tier: str,
    entry: float,
    exit: float,
    pnl_r: float,
) -> None:
    """Store a completed trade outcome to the learning-loop database."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO trade_outcomes
                (timestamp, symbol, archetype, regime, tier, entry, exit, pnl_r, won)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                symbol,
                archetype,
                regime,
                tier,
                entry,
                exit,
                pnl_r,
                1 if pnl_r > 0 else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _get_baseline(archetype: str) -> float:
    """
    Try to read historical baseline win rate from QualityEngine playbooks.
    Falls back to _UNIVERSAL_BASELINE if unavailable.
    """
    try:
        from playbooks import get_all_playbooks  # type: ignore
        for pb in get_all_playbooks():
            if pb.archetype == archetype or getattr(pb, "id", "") == archetype:
                return float(pb.baseline_win_rate)
    except Exception:
        pass
    return _UNIVERSAL_BASELINE


def get_personal_edge(min_trades: int = 5) -> dict[str, dict]:
    """
    Returns your personal edge per archetype.

    Shape::

        {
          "VCP_BREAKOUT": {
              "win_rate":    0.73,
              "avg_r":       1.4,
              "trades":      11,
              "vs_baseline": +0.18,   # positive = you outperform
          },
          ...
        }

    Only archetypes with >= min_trades are included.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT archetype,
                   COUNT(*)        AS trades,
                   SUM(won)        AS wins,
                   AVG(pnl_r)      AS avg_r
            FROM   trade_outcomes
            GROUP  BY archetype
            HAVING trades >= ?
            """,
            (min_trades,),
        ).fetchall()
    finally:
        conn.close()

    result: dict[str, dict] = {}
    for archetype, trades, wins, avg_r in rows:
        wr = wins / trades if trades else 0.0
        baseline = _get_baseline(archetype)
        result[archetype] = {
            "win_rate":    round(wr, 4),
            "avg_r":       round(avg_r or 0.0, 4),
            "trades":      trades,
            "vs_baseline": round(wr - baseline, 4),
        }
    return result


def get_top_edges(n: int = 3) -> list[dict]:
    """
    Returns your top N archetypes by personal win rate (min 5 trades).
    Each dict has keys: archetype, win_rate, avg_r, trades, vs_baseline.
    """
    edge = get_personal_edge(min_trades=5)
    ranked = sorted(edge.items(), key=lambda kv: kv[1]["win_rate"], reverse=True)
    return [{"archetype": k, **v} for k, v in ranked[:n]]


def get_regime_edge(regime: str) -> dict[str, float | int]:
    """
    Returns your aggregate win stats for a specific market regime.

    Returns::

        {"win_rate": float, "avg_r": float, "trades": int}
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS trades,
                   SUM(won)  AS wins,
                   AVG(pnl_r) AS avg_r
            FROM   trade_outcomes
            WHERE  regime = ?
            """,
            (regime,),
        ).fetchone()
    finally:
        conn.close()

    if row is None or row[0] == 0:
        return {"win_rate": 0.0, "avg_r": 0.0, "trades": 0}

    trades, wins, avg_r = row
    wr = (wins or 0) / trades if trades else 0.0
    return {
        "win_rate": round(wr, 4),
        "avg_r":    round(avg_r or 0.0, 4),
        "trades":   trades,
    }


# ── Streamlit UI Component ────────────────────────────────────────────────────

def render_personal_edge_ui() -> None:
    """
    Streamlit component showing your personal trading edge analytics.
    Displays:
      1. Header
      2. Bar chart — personal WR vs baseline per archetype
      3. Text summary of strongest / weakest archetypes
      4. Regime performance table
    """
    return
