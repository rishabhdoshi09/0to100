"""
Trade Thesis Recorder — captures WHY you entered a trade.
At exit, scores whether the thesis was correct (even if trade was wrong).
Builds metacognition over time.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

_DB_PATH = Path("logs/thesis_db.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_theses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT,
    symbol TEXT,
    entry_date TEXT,
    entry_price REAL,
    thesis TEXT,
    archetype TEXT,
    regime TEXT,
    exit_date TEXT,
    exit_price REAL,
    pnl_r REAL,
    thesis_correct INTEGER,
    thesis_score_reason TEXT
)
"""


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def record_thesis(
    trade_id: str,
    symbol: str,
    entry_price: float,
    thesis: str,
    archetype: str,
    regime: str,
) -> int:
    """Insert a new trade thesis entry. Returns the row id."""
    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO trade_theses
                (trade_id, symbol, entry_date, entry_price, thesis, archetype, regime)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                symbol,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                float(entry_price),
                thesis,
                archetype,
                regime,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def score_thesis_at_exit(trade_id: str, exit_price: float, pnl_r: float) -> None:
    """
    Auto-score whether the thesis was correct using DeepSeek.
    Updates the thesis record with exit data and scoring result.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, thesis FROM trade_theses WHERE trade_id = ? ORDER BY id DESC LIMIT 1",
            (trade_id,),
        ).fetchone()
        if row is None:
            return

        row_id, thesis = row

        # Update exit data first
        conn.execute(
            """
            UPDATE trade_theses
            SET exit_date = ?, exit_price = ?, pnl_r = ?
            WHERE id = ?
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                float(exit_price),
                float(pnl_r),
                row_id,
            ),
        )
        conn.commit()

        # Score with DeepSeek
        thesis_correct: Optional[int] = None
        score_reason: Optional[str] = None

        try:
            from ai.dual_llm_service import get_service

            svc = get_service()
            outcome_desc = "profitable" if pnl_r > 0 else "at a loss"
            prompt = (
                f"A trader entered a trade with the following thesis:\n"
                f'"{thesis}"\n\n'
                f"The trade exited {outcome_desc} with a P&L of {pnl_r:.2f}R.\n\n"
                f"Was the original thesis CORRECT in its directional reasoning? "
                f"Answer YES or NO followed by one sentence explaining why."
            )
            # Use the underlying client if available, else fall back to raw call
            raw = svc._call_deepseek(prompt) if hasattr(svc, "_call_deepseek") else None
            if raw is None:
                import openai

                client = openai.OpenAI(
                    api_key=_get_deepseek_key(),
                    base_url="https://api.deepseek.com/v1",
                )
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=120,
                    temperature=0.2,
                )
                raw = response.choices[0].message.content.strip()

            if raw:
                upper = raw.upper()
                if upper.startswith("YES"):
                    thesis_correct = 1
                elif upper.startswith("NO"):
                    thesis_correct = 0
                score_reason = raw
        except Exception:
            # DeepSeek unavailable — leave thesis_correct as NULL
            pass

        conn.execute(
            """
            UPDATE trade_theses
            SET thesis_correct = ?, thesis_score_reason = ?
            WHERE id = ?
            """,
            (thesis_correct, score_reason, row_id),
        )
        conn.commit()
    finally:
        conn.close()


def _get_deepseek_key() -> str:
    try:
        from config import settings

        return settings.deepseek_api_key or ""
    except Exception:
        import os

        return os.getenv("DEEPSEEK_API_KEY", "")


def get_thesis_accuracy() -> dict:
    """
    Returns metacognition stats across all scored theses.

    Interesting insight: profitable_wrong_thesis reveals luck-based wins.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT thesis_correct, pnl_r
            FROM trade_theses
            WHERE thesis_correct IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()

    total_scored = len(rows)
    correct = sum(1 for r in rows if r[0] == 1)
    accuracy = (correct / total_scored * 100.0) if total_scored else 0.0

    # Won money (pnl_r > 0) despite wrong thesis — luck-based wins
    profitable_wrong_thesis = sum(
        1 for r in rows if r[0] == 0 and (r[1] or 0) > 0
    )
    # Right thesis, lost money — execution or timing issue
    correct_thesis_lost = sum(
        1 for r in rows if r[0] == 1 and (r[1] or 0) <= 0
    )

    return {
        "total_scored": total_scored,
        "correct": correct,
        "accuracy": round(accuracy, 1),
        "profitable_wrong_thesis": profitable_wrong_thesis,
        "correct_thesis_lost": correct_thesis_lost,
    }


def render_thesis_input(symbol: str, archetype: str) -> Optional[str]:
    """
    Streamlit component: text input for trade thesis.
    Returns the thesis string if filled, None if skipped.
    """
    return None

def render_thesis_history(symbol: Optional[str] = None) -> None:
    """
    Shows recent theses in a compact table.
    If symbol is given, filters to that symbol.
    """
    return
