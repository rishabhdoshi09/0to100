"""
Signal Outcome Tracker — the self-improving loop.

Every time the scanner marks a stock READY_TO_TRADE, we log it.
The canonical outcome resolver settles it on official bhavcopy: first-touch
stop-before-target when geometry exists, otherwise the horizon-th session
close. Never a live quote. Never “five calendar days then today’s price.”
"""
from __future__ import annotations

import sqlite3
import os
import threading
from datetime import datetime, timedelta  # noqa: F401 — kept for callers that import the name

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "signal_outcomes.db")
_DB_PATH = os.path.normpath(_DB_PATH)


def _now() -> datetime:
    """Naive IST wall-clock time — same storage format as plain
    datetime.now() (no offset suffix, backward-compatible with existing
    rows), but correct regardless of the host's own timezone. A UTC-hosted
    VPS (the documented 24/7 setup — docs/ALWAYS_ON.md) running naive
    datetime.now() here would date-bucket/window every signal 5.5 hours
    off, same class of bug market_clock.py exists to prevent elsewhere.
    (Only used for date-bucketing/windowing — the _error_cache cooldown
    below is a pure elapsed-time delta and doesn't need this.)"""
    from core.market_clock import now_ist
    return now_ist().replace(tzinfo=None)

# Error cache to skip recently-failed symbols
_error_cache: dict[str, datetime] = {}
_error_lock = threading.Lock()
_ERROR_COOLDOWN_MIN = 30  # don't retry a failed symbol for 30 minutes


# ── Schema ─────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS signal_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    logged_at TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    archetype TEXT,
    regime TEXT,
    entry_price REAL,
    pivot_price REAL,
    stop_price REAL,
    target_price REAL,
    quality_score REAL,
    accum_score REAL,
    outcome_checked_at TEXT,
    outcome_price REAL,
    outcome_pct REAL,
    worked INTEGER
);
CREATE INDEX IF NOT EXISTS idx_signal_log_symbol ON signal_log(symbol);
CREATE INDEX IF NOT EXISTS idx_signal_log_logged_at ON signal_log(logged_at);
CREATE INDEX IF NOT EXISTS idx_signal_log_worked ON signal_log(worked);
"""


def _get_conn() -> sqlite3.Connection:
    """Return a connection to the SQLite DB, creating the schema if needed."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    for stmt in _DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    return conn


# ── Public API ────────────────────────────────────────────────────────────────

def log_signal(
    symbol: str,
    signal_type: str,
    entry_price: float,
    pivot_price: float,
    stop_price: float,
    target_price: float,
    quality_score: float,
    accum_score: float,
    archetype: str,
    regime: str,
) -> None:
    """
    Log a signal. Dedupes — won't log same symbol+date twice.
    """
    try:
        today = _now().strftime("%Y-%m-%d")
        now_iso = _now().isoformat(timespec="seconds")
        conn = _get_conn()
        try:
            # Dedupe: if the same symbol was logged today with the same signal_type, skip
            existing = conn.execute(
                "SELECT id FROM signal_log WHERE symbol=? AND logged_at LIKE ? AND signal_type=?",
                (symbol.upper(), f"{today}%", signal_type),
            ).fetchone()
            if existing:
                return
            conn.execute(
                """
                INSERT INTO signal_log
                    (symbol, logged_at, signal_type, archetype, regime,
                     entry_price, pivot_price, stop_price, target_price,
                     quality_score, accum_score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    symbol.upper(), now_iso, signal_type, archetype, regime,
                    float(entry_price), float(pivot_price), float(stop_price),
                    float(target_price), float(quality_score), float(accum_score),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass  # never crash on tracker errors


# ── Canonical resolution (same clock as decision_journal / Cases) ─────────────
# Geometry present → first-touch stop-before-target on official bars.
# No geometry → close-to-close at the horizon-th trading session.
# Never a live quote. A delayed runner must not mark day-8 as day-5.
_OPEN = object()                         # sentinel: trade still live, leave NULL
_HORIZON_SESSIONS = int(os.getenv("QT_OUTCOME_HORIZON", "15") or 15)


def _resolve_via_path(row) -> object | None:
    """First-touch outcome from official bars via the canonical resolver.

    Returns (id, price, pct, worked), the _OPEN sentinel (still live or
    missing bars), or None (no usable geometry — caller may use the
    session-close path). Stop is checked before target.
    """
    try:
        entry = float(row["entry_price"] or 0)
        stop = float(row["stop_price"] or 0)
        target = float(row["target_price"] or 0)
    except Exception:
        return None
    if entry <= 0 or stop <= 0 or target <= entry or stop >= entry:
        return None
    try:
        from core.outcome_resolver import first_touch_path
        got = first_touch_path(
            row["symbol"],
            str(row["logged_at"] or "")[:10],
            entry, stop, target,
        )
    except Exception:
        return _OPEN
    if got is None:
        return _OPEN
    price, pct, worked = got
    return (row["id"], price, pct, worked)


def update_outcomes(lookback_days: int = 30) -> None:
    """Settle OPEN signals through the canonical outcome resolver.

    Lookback is only a search window. A row is due when official bars
    actually contain the horizon-th session — never because five
    calendar days have passed and a live quote exists.
    """
    try:
        too_old = (_now() - timedelta(days=lookback_days)).isoformat(timespec="seconds")

        conn = _get_conn()
        try:
            rows = conn.execute(
                """
                SELECT id, symbol, entry_price, stop_price, target_price, logged_at
                FROM signal_log
                WHERE worked IS NULL
                  AND logged_at >= ?
                """,
                (too_old,),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return

        def _entry_was_triggered(symbol: str, entry: float, logged_at: str) -> bool:
            """Fill-awareness: a breakout entry sits ABOVE the market."""
            try:
                import pandas as pd
                from data.bhavcopy_store import get_ohlcv
                df = get_ohlcv(symbol)
                if df is None or df.empty or "high" not in df.columns:
                    return True
                since = df[df.index >= pd.Timestamp(str(logged_at)[:10])]
                if since.empty:
                    return True
                return float(since["high"].max()) >= entry * 0.999
            except Exception:
                return True

        results: list[tuple[int, float, float, int | None]] = []
        for row in rows:
            symbol = row["symbol"]
            entry = row["entry_price"]
            try:
                from data.dead_symbols import is_dead
                if is_dead(symbol):
                    results.append((row["id"], 0.0, 0.0, -1))
                    continue
            except Exception:
                pass
            pv = _resolve_via_path(row)
            if pv is _OPEN:
                continue
            if pv is not None:
                results.append(pv)
                continue
            if entry and not _entry_was_triggered(symbol, float(entry),
                                                  row["logged_at"]):
                results.append((row["id"], 0.0, 0.0, -1))
                continue
            if not entry or float(entry) <= 0:
                continue
            try:
                from core.outcome_resolver import session_close_return
                resolved = session_close_return(
                    symbol, str(row["logged_at"] or "")[:10],
                )
            except Exception:
                continue
            if resolved is None:
                continue
            price, _ = resolved
            pct = (float(price) - float(entry)) / float(entry) * 100.0
            worked = 1 if pct >= 2.0 else (0 if pct <= -1.0 else None)
            results.append((row["id"], float(price), pct, worked))

        if not results:
            return

        now_iso = _now().isoformat(timespec="seconds")
        conn = _get_conn()
        try:
            for (row_id, price, pct, worked) in results:
                _price = None if worked == -1 else price
                _pct = None if worked == -1 else pct
                conn.execute(
                    """
                    UPDATE signal_log
                    SET outcome_checked_at=?, outcome_price=?, outcome_pct=?, worked=?
                    WHERE id=?
                    """,
                    (now_iso, _price, _pct, worked, row_id),
                )
            conn.commit()
        finally:
            conn.close()

    except Exception:
        pass  # never crash


def reresolve_history(max_rows: int = 8000) -> int:
    """One-time back-data correction. Rows closed by the OLD crude ±band method
    (win = +2% now / loss = −1% now) are re-judged by TRUE target-vs-stop
    first-touch from official bhavcopy, so the whole learning stack (live-edge,
    EV, drift, beliefs, calibration, equity curve) reflects how the trades
    actually resolved — not a noisy proxy. Only rows the path can resolve
    definitively are rewritten; anything the path can't judge, or that is still
    live within its horizon, keeps its existing value. Returns rows corrected.
    Fail-open → 0."""
    try:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT id, symbol, entry_price, stop_price, target_price,
                          logged_at, worked, outcome_pct
                   FROM signal_log
                   WHERE worked IS NOT NULL AND entry_price > 0
                     AND stop_price > 0 AND target_price > 0
                   ORDER BY id DESC LIMIT ?""", (max_rows,)).fetchall()
            updates = []
            now_iso = _now().isoformat(timespec="seconds")
            for row in rows:
                pv = _resolve_via_path(dict(row))
                if pv is _OPEN or pv is None:
                    continue                       # can't/shouldn't re-judge
                _id, price, pct, worked = pv
                new_price = None if worked == -1 else price
                new_pct = None if worked == -1 else pct
                # write only on a real change (avoid pointless churn)
                changed = worked != row["worked"]
                if not changed and new_pct is not None and row["outcome_pct"] is not None:
                    changed = abs(float(new_pct) - float(row["outcome_pct"])) > 0.05
                if not changed and (new_pct is None) != (row["outcome_pct"] is None):
                    changed = True
                if changed:
                    updates.append((now_iso, new_price, new_pct, worked, _id))
            for u in updates:
                conn.execute(
                    "UPDATE signal_log SET outcome_checked_at=?, outcome_price=?, "
                    "outcome_pct=?, worked=? WHERE id=?", u)
            conn.commit()
            return len(updates)
        finally:
            conn.close()
    except Exception:
        return 0


def get_accuracy_report() -> dict:
    """
    Returns a dict with overall accuracy, accuracy by archetype, regime, and weekly.
    """
    try:
        conn = _get_conn()
        try:
            # Total counts
            row = conn.execute(
                "SELECT COUNT(*) as total, SUM(CASE WHEN worked IS NULL THEN 1 ELSE 0 END) as open, "
                "SUM(CASE WHEN worked=1 THEN 1 ELSE 0 END) as wins, "
                "SUM(CASE WHEN worked=0 THEN 1 ELSE 0 END) as losses "
                "FROM signal_log"
            ).fetchone()

            total = row["total"] or 0
            open_signals = row["open"] or 0
            wins = row["wins"] or 0
            losses = row["losses"] or 0
            closed = wins + losses
            overall_accuracy = (wins / closed * 100.0) if closed > 0 else 0.0

            # Win/loss avg pct
            avg_row = conn.execute(
                "SELECT AVG(CASE WHEN worked=1 THEN outcome_pct END) as avg_win, "
                "AVG(CASE WHEN worked=0 THEN outcome_pct END) as avg_loss "
                "FROM signal_log WHERE worked IS NOT NULL"
            ).fetchone()
            avg_win_pct = float(avg_row["avg_win"] or 0.0)
            avg_loss_pct = float(avg_row["avg_loss"] or 0.0)

            # Expectancy: win_rate * avg_win - loss_rate * avg_loss
            win_rate = wins / closed if closed > 0 else 0.0
            loss_rate = losses / closed if closed > 0 else 0.0
            system_edge = win_rate * avg_win_pct + loss_rate * avg_loss_pct

            # By archetype
            arch_rows = conn.execute(
                "SELECT archetype, "
                "COUNT(*) as cnt, "
                "SUM(CASE WHEN worked=1 THEN 1 ELSE 0 END) as w, "
                "SUM(CASE WHEN worked=0 THEN 1 ELSE 0 END) as l "
                "FROM signal_log WHERE worked IS NOT NULL AND archetype IS NOT NULL "
                "GROUP BY archetype"
            ).fetchall()

            accuracy_by_archetype: dict[str, dict] = {}
            best_archetype = ""
            best_arch_acc = -1.0
            for r in arch_rows:
                c = (r["w"] or 0) + (r["l"] or 0)
                if c == 0:
                    continue
                acc = (r["w"] or 0) / c * 100.0
                accuracy_by_archetype[r["archetype"]] = {"accuracy": round(acc, 1), "count": c}
                if acc > best_arch_acc:
                    best_arch_acc = acc
                    best_archetype = r["archetype"]

            # By regime
            regime_rows = conn.execute(
                "SELECT regime, "
                "COUNT(*) as cnt, "
                "SUM(CASE WHEN worked=1 THEN 1 ELSE 0 END) as w, "
                "SUM(CASE WHEN worked=0 THEN 1 ELSE 0 END) as l "
                "FROM signal_log WHERE worked IS NOT NULL AND regime IS NOT NULL "
                "GROUP BY regime"
            ).fetchall()

            accuracy_by_regime: dict[str, dict] = {}
            best_regime = ""
            best_regime_acc = -1.0
            for r in regime_rows:
                c = (r["w"] or 0) + (r["l"] or 0)
                if c == 0:
                    continue
                acc = (r["w"] or 0) / c * 100.0
                accuracy_by_regime[r["regime"]] = {"accuracy": round(acc, 1), "count": c}
                if acc > best_regime_acc:
                    best_regime_acc = acc
                    best_regime = r["regime"]

            # Weekly accuracy — last 8 weeks
            weekly_rows = conn.execute(
                "SELECT strftime('%Y-W%W', logged_at) as week, "
                "COUNT(*) as cnt, "
                "SUM(CASE WHEN worked=1 THEN 1 ELSE 0 END) as w, "
                "SUM(CASE WHEN worked=0 THEN 1 ELSE 0 END) as l "
                "FROM signal_log WHERE worked IS NOT NULL "
                "GROUP BY week ORDER BY week DESC LIMIT 8"
            ).fetchall()

            accuracy_by_week = []
            for r in weekly_rows:
                c = (r["w"] or 0) + (r["l"] or 0)
                if c == 0:
                    continue
                acc = (r["w"] or 0) / c * 100.0
                accuracy_by_week.append({"week": r["week"], "accuracy": round(acc, 1), "count": c})
            accuracy_by_week.reverse()  # oldest first

        finally:
            conn.close()

        return {
            "overall_accuracy": round(overall_accuracy, 1),
            "total_signals": total,
            "open_signals": open_signals,
            "wins": wins,
            "losses": losses,
            "accuracy_by_archetype": accuracy_by_archetype,
            "accuracy_by_regime": accuracy_by_regime,
            "accuracy_by_week": accuracy_by_week,
            "avg_win_pct": round(avg_win_pct, 2),
            "avg_loss_pct": round(avg_loss_pct, 2),
            "best_archetype": best_archetype,
            "best_regime": best_regime,
            "system_edge": round(system_edge, 2),
        }

    except Exception:
        return {
            "overall_accuracy": 0.0,
            "total_signals": 0,
            "open_signals": 0,
            "wins": 0,
            "losses": 0,
            "accuracy_by_archetype": {},
            "accuracy_by_regime": {},
            "accuracy_by_week": [],
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "best_archetype": "",
            "best_regime": "",
            "system_edge": 0.0,
        }


def get_recent_signals(limit: int = 20) -> list[dict]:
    """Returns most recent signals with their outcome status."""
    try:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM signal_log ORDER BY logged_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception:
        return []


def get_weekly_accuracy() -> float:
    """Quick single number: accuracy of signals from last 7 days that have closed."""
    try:
        cutoff = (_now() - timedelta(days=7)).isoformat(timespec="seconds")
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT SUM(CASE WHEN worked=1 THEN 1 ELSE 0 END) as wins, "
                "SUM(CASE WHEN worked=0 THEN 1 ELSE 0 END) as losses "
                "FROM signal_log WHERE logged_at >= ? AND worked IS NOT NULL",
                (cutoff,),
            ).fetchone()
        finally:
            conn.close()
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        closed = wins + losses
        return round(wins / closed * 100.0, 1) if closed > 0 else 0.0
    except Exception:
        return 0.0
