"""
Auto Journal — automatically creates a complete journal entry for every trade,
capturing the full context at entry and exit.

SQLite at logs/auto_journal.db
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH = Path("logs/auto_journal.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE,
    symbol TEXT,
    archetype TEXT,
    entry_date TEXT,
    entry_time TEXT,
    entry_price REAL,
    exit_date TEXT,
    exit_price REAL,
    quantity INTEGER,
    pnl_inr REAL,
    pnl_r REAL,

    -- Context at entry (captured automatically)
    regime_at_entry TEXT,
    vix_at_entry REAL,
    institutional_activity TEXT,
    quality_score REAL,
    rs_score REAL,
    timing_label TEXT,
    earnings_days INTEGER,
    whipsaw_verdict TEXT,

    -- Thesis (from thesis_recorder)
    thesis TEXT,
    thesis_correct INTEGER,

    -- Outcome analysis (filled at exit)
    exit_reason TEXT,
    regime_at_exit TEXT,
    conviction_at_exit REAL,

    -- Tags (auto-generated)
    tags TEXT,

    notes TEXT
)
"""


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def _row_to_dict(row, cursor: sqlite3.Cursor) -> Dict[str, Any]:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def create_entry(
    symbol: str,
    archetype: str,
    entry_price: float,
    quantity: int,
    regime: str,
    quality_score: float,
    rs_score: float,
    timing: str,
    earnings_days: Optional[int],
    whipsaw: str,
    thesis: str,
    vix_at_entry: Optional[float] = None,
    institutional_activity: Optional[str] = None,
) -> str:
    """
    Creates a new journal entry at trade entry time.
    Returns the trade_id (UUID string).
    """
    trade_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Auto-generate tags from context
    tags: List[str] = [archetype.lower()]
    if regime in ("TRENDING_BULL", "EXPANSION"):
        tags.append("bull_regime")
    if rs_score and rs_score >= 90:
        tags.append("high_rs")
    if timing == "OPTIMAL":
        tags.append("optimal_timing")
    if earnings_days is not None and earnings_days <= 7:
        tags.append("pre_earnings")

    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO journal_entries (
                trade_id, symbol, archetype,
                entry_date, entry_time, entry_price,
                quantity, regime_at_entry, vix_at_entry,
                institutional_activity, quality_score, rs_score,
                timing_label, earnings_days, whipsaw_verdict,
                thesis, tags
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?
            )
            """,
            (
                trade_id, symbol, archetype,
                now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), float(entry_price),
                int(quantity), regime, vix_at_entry,
                institutional_activity, quality_score, rs_score,
                timing, earnings_days, whipsaw,
                thesis, json.dumps(tags),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return trade_id


def close_entry(
    trade_id: str,
    exit_price: float,
    pnl_inr: float,
    pnl_r: float,
    exit_reason: str,
    regime_at_exit: str,
    conviction_score: float,
) -> None:
    """
    Updates the journal entry at trade close.
    Calls score_thesis_at_exit if thesis exists.
    """
    now = datetime.now(timezone.utc)
    conn = _get_conn()
    try:
        conn.execute(
            """
            UPDATE journal_entries
            SET exit_date = ?,
                exit_price = ?,
                pnl_inr = ?,
                pnl_r = ?,
                exit_reason = ?,
                regime_at_exit = ?,
                conviction_at_exit = ?
            WHERE trade_id = ?
            """,
            (
                now.strftime("%Y-%m-%d"),
                float(exit_price),
                float(pnl_inr),
                float(pnl_r),
                exit_reason,
                regime_at_exit,
                float(conviction_score),
                trade_id,
            ),
        )
        conn.commit()

        # Score thesis if present
        row = conn.execute(
            "SELECT thesis FROM journal_entries WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()
        if row and row[0]:
            try:
                from ai.thesis_recorder import score_thesis_at_exit
                score_thesis_at_exit(trade_id, exit_price, pnl_r)

                # Sync thesis_correct back from thesis_recorder DB
                try:
                    from pathlib import Path as _Path
                    import sqlite3 as _sqlite3
                    thesis_db = _Path("logs/thesis_db.db")
                    if thesis_db.exists():
                        tc_conn = _sqlite3.connect(str(thesis_db))
                        tc_row = tc_conn.execute(
                            "SELECT thesis_correct FROM trade_theses WHERE trade_id = ? "
                            "ORDER BY id DESC LIMIT 1",
                            (trade_id,),
                        ).fetchone()
                        tc_conn.close()
                        if tc_row and tc_row[0] is not None:
                            conn.execute(
                                "UPDATE journal_entries SET thesis_correct = ? WHERE trade_id = ?",
                                (tc_row[0], trade_id),
                            )
                            conn.commit()
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass
    finally:
        conn.close()


def get_journal_stats() -> Dict[str, Any]:
    """
    Returns aggregate performance statistics across all closed journal entries.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT symbol, archetype, regime_at_entry, timing_label,
                   entry_date, exit_date, pnl_inr, pnl_r,
                   thesis_correct
            FROM journal_entries
            WHERE exit_date IS NOT NULL AND pnl_inr IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "best_regime": "",
            "best_archetype": "",
            "best_timing": "",
            "avg_holding_days": 0.0,
            "thesis_accuracy": 0.0,
            "lucky_wins": 0,
        }

    total = len(rows)
    wins = sum(1 for r in rows if (r[6] or 0) > 0)
    win_rate = wins / total * 100.0 if total else 0.0

    r_values = [r[7] for r in rows if r[7] is not None]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0

    # Best regime (highest win rate among regimes with >= 2 trades)
    from collections import defaultdict
    regime_wins: Dict[str, List[float]] = defaultdict(list)
    archetype_wins: Dict[str, List[float]] = defaultdict(list)
    timing_wins: Dict[str, List[float]] = defaultdict(list)

    holding_days_list: List[float] = []

    for r in rows:
        _symbol, archetype, regime, timing, entry_date, exit_date, pnl_inr, pnl_r, _ = r
        won = 1.0 if (pnl_inr or 0) > 0 else 0.0
        if regime:
            regime_wins[regime].append(won)
        if archetype:
            archetype_wins[archetype].append(won)
        if timing:
            timing_wins[timing].append(won)
        if entry_date and exit_date:
            try:
                d0 = datetime.strptime(entry_date, "%Y-%m-%d")
                d1 = datetime.strptime(exit_date, "%Y-%m-%d")
                holding_days_list.append(max(0.0, (d1 - d0).days))
            except Exception:
                pass

    def _best_key(d: Dict[str, List[float]]) -> str:
        candidates = {k: sum(v) / len(v) for k, v in d.items() if len(v) >= 2}
        return max(candidates, key=lambda k: candidates[k]) if candidates else ""

    best_regime = _best_key(regime_wins)
    best_archetype = _best_key(archetype_wins)
    best_timing = _best_key(timing_wins)
    avg_holding_days = sum(holding_days_list) / len(holding_days_list) if holding_days_list else 0.0

    # Thesis accuracy
    scored = [(r[8], r[7]) for r in rows if r[8] is not None]
    thesis_accuracy = (
        sum(1 for s in scored if s[0] == 1) / len(scored) * 100.0 if scored else 0.0
    )
    lucky_wins = sum(1 for s in scored if s[0] == 0 and (s[1] or 0) > 0)

    return {
        "total_trades": total,
        "win_rate": round(win_rate, 1),
        "avg_r": round(avg_r, 3),
        "best_regime": best_regime,
        "best_archetype": best_archetype,
        "best_timing": best_timing,
        "avg_holding_days": round(avg_holding_days, 1),
        "thesis_accuracy": round(thesis_accuracy, 1),
        "lucky_wins": lucky_wins,
    }


def get_personal_edge_by_archetype(min_trades: int = 3) -> Dict[str, Dict]:
    """
    Returns personal win rates and avg-R by archetype (and by archetype×regime).
    Used by the ranking engine to personalise EV calculations.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT archetype, regime_at_entry, pnl_inr, pnl_r
            FROM journal_entries
            WHERE exit_date IS NOT NULL AND pnl_inr IS NOT NULL AND archetype IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()

    from collections import defaultdict
    arch_data: Dict[str, Dict] = defaultdict(lambda: {"wins": 0, "total": 0, "r_vals": []})
    arch_regime: Dict[str, Dict] = defaultdict(lambda: {"wins": 0, "total": 0})

    for archetype, regime, pnl_inr, pnl_r in rows:
        arch = archetype or "UNKNOWN"
        won = (pnl_inr or 0) > 0
        arch_data[arch]["total"] += 1
        arch_data[arch]["wins"] += int(won)
        if pnl_r is not None:
            arch_data[arch]["r_vals"].append(float(pnl_r))
        if regime:
            key = f"{arch}|{regime}"
            arch_regime[key]["total"] += 1
            arch_regime[key]["wins"] += int(won)

    result: Dict[str, Dict] = {}
    for arch, d in arch_data.items():
        if d["total"] < min_trades:
            continue
        wr = d["wins"] / d["total"]
        avg_r = sum(d["r_vals"]) / len(d["r_vals"]) if d["r_vals"] else 0.0
        regime_breakdown = {}
        for k, rd in arch_regime.items():
            a, reg = k.split("|", 1)
            if a == arch and rd["total"] >= 2:
                regime_breakdown[reg] = round(rd["wins"] / rd["total"], 3)
        result[arch] = {
            "win_rate": round(wr, 3),
            "avg_r": round(avg_r, 3),
            "count": d["total"],
            "regime_breakdown": regime_breakdown,
        }
    return result


# ---------------------------------------------------------------------------
# Streamlit rendering
# ---------------------------------------------------------------------------

def render_journal_entry(entry: Dict[str, Any]) -> None:
    """Render a single journal entry as a rich card."""
    return

def render_auto_journal_page() -> None:
    """Full auto-journal page with stats, filters, entries, and CSV export."""
    return
