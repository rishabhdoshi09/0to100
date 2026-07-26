"""
Signal Outcome Tracker — the self-improving loop.

Every time the scanner marks a stock READY_TO_TRADE, we log it.
5 days later we check the actual price outcome and mark it as WIN/LOSS/OPEN.
Over time this builds a dataset of what actually works vs what the system thinks works.
The accuracy report feeds back into JARVIS context and the quality engine thresholds.
"""
from __future__ import annotations

import sqlite3
import os
import threading
from datetime import datetime, timedelta
from typing import Optional

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


# ── True target-vs-stop first-touch resolution (matches the backtest) ─────────
# The point-in-time quote method below marks a signal win/loss by ±band at check
# time — a noisy proxy that over-counts losses (a −1% wiggle looks like a loss
# even when the real stop is −6%) and caps wins near +2%. When we have a valid
# target/stop and official bhavcopy bars, we instead judge the outcome the way
# the trade was actually set up and the way the backtest measures it: did the
# target hit before the stop? Everything downstream (live-edge, EV, drift,
# beliefs, calibration) inherits this accuracy.
_OPEN = object()                         # sentinel: trade still live, leave NULL
_HORIZON_SESSIONS = int(os.getenv("QT_OUTCOME_HORIZON", "15") or 15)


def _resolve_via_path(row) -> object | None:
    """First-touch outcome from bhavcopy OHLC. Returns (id, price, pct, worked),
    the _OPEN sentinel (still live within horizon), or None (no target/path data
    → caller falls back to the point-in-time quote method). Conservative within a
    bar: the stop is checked before the target (never overstates a win)."""
    try:
        entry = float(row["entry_price"] or 0)
        stop = float(row["stop_price"] or 0)
        target = float(row["target_price"] or 0)
    except Exception:
        return None
    if entry <= 0 or stop <= 0 or target <= entry or stop >= entry:
        return None                      # need real geometry to path-resolve
    try:
        import pandas as pd
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv(row["symbol"])
    except Exception:
        return None
    if df is None or df.empty or not {"high", "low", "close"} <= set(df.columns):
        return None
    since = df[df.index >= pd.Timestamp(str(row["logged_at"])[:10])]
    if since.empty:
        return None
    highs = since["high"].to_numpy(dtype=float)
    lows = since["low"].to_numpy(dtype=float)
    closes = since["close"].to_numpy(dtype=float)
    n = len(highs)
    horizon = min(n, _HORIZON_SESSIONS)
    filled = False
    for i in range(horizon):
        if not filled:                   # breakout entries sit above the market
            if highs[i] >= entry:
                filled = True
            else:
                continue
        if lows[i] <= stop:              # stop first (pessimistic)
            return (row["id"], stop, (stop - entry) / entry * 100.0, 0)
        if highs[i] >= target:
            return (row["id"], target, (target - entry) / entry * 100.0, 1)
    if not filled:
        # never triggered — NO_FILL only once the horizon has fully elapsed
        return (row["id"], 0.0, 0.0, -1) if n >= _HORIZON_SESSIONS else _OPEN
    if n >= _HORIZON_SESSIONS:           # filled, no touch → mark to market
        last = float(closes[horizon - 1])
        return (row["id"], last, (last - entry) / entry * 100.0,
                1 if last >= entry else 0)
    return _OPEN                         # still live, leave it open


def update_outcomes(lookback_days: int = 30) -> None:
    """
    Check outcomes for signals logged > 5 days ago that are still OPEN (worked=NULL).
    Prices come from the unified quote chain in ONE bulk call (Kite→NSE→Google);
    the slow per-symbol path survives only as fallback for misses.
    outcome_pct = (current_price - entry_price) / entry_price * 100
    worked = 1 if outcome_pct >= 2.0 else (0 if outcome_pct <= -1.0 else None)
    """
    from concurrent.futures import ThreadPoolExecutor

    try:
        cutoff = (_now() - timedelta(days=5)).isoformat(timespec="seconds")
        too_old = (_now() - timedelta(days=lookback_days)).isoformat(timespec="seconds")

        conn = _get_conn()
        try:
            rows = conn.execute(
                """
                SELECT id, symbol, entry_price, stop_price, target_price, logged_at
                FROM signal_log
                WHERE worked IS NULL
                  AND logged_at <= ?
                  AND logged_at >= ?
                """,
                (cutoff, too_old),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return

        # One bulk quote call covers most symbols cheaply
        _bulk_prices: dict[str, float] = {}
        try:
            from data.live_quotes import get_live_quotes
            _bulk = get_live_quotes(sorted({r["symbol"] for r in rows}))
            _bulk_prices = {s: float(q["price"]) for s, q in _bulk.items()
                            if q.get("price")}
        except Exception:
            pass

        def _entry_was_triggered(symbol: str, entry: float, logged_at: str) -> bool:
            """
            Fill-awareness: a breakout entry sits ABOVE the market. If the
            high since the signal never reached it, the trade never existed —
            judging it by today's price would poison the accuracy stats.
            Unknown history → assume triggered (fail open, never lose data).
            """
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

        # Filter out recently errored symbols
        def _should_skip(symbol: str) -> bool:
            with _error_lock:
                ts = _error_cache.get(symbol)
            if ts and (datetime.now() - ts).total_seconds() < _ERROR_COOLDOWN_MIN * 60:
                return True
            return False

        def _fetch_price(row) -> Optional[tuple[int, float, float, int | None]]:
            """Returns (id, outcome_price, outcome_pct, worked) or None on error."""
            symbol = row["symbol"]
            entry = row["entry_price"]
            # Delisted symbol can NEVER resolve — close it as VOID (worked=-1,
            # excluded from accuracy math) instead of retrying every cycle
            try:
                from data.dead_symbols import is_dead
                if is_dead(symbol):
                    return (row["id"], 0.0, 0.0, -1)
            except Exception:
                pass
            # PREFERRED: true target-vs-stop first-touch from official bhavcopy
            # (accurate, matches the backtest). Falls through to the point-in-time
            # quote method only when there's no target / no path data.
            pv = _resolve_via_path(row)
            if pv is _OPEN:
                return None              # still live — leave the row open (NULL)
            if pv is not None:
                return pv                # definitive first-touch resolution
            # No-fill check first: order never triggered = not a trade.
            # worked=-1 marks NO_FILL (excluded from all accuracy math).
            if entry and not _entry_was_triggered(symbol, float(entry),
                                                  row["logged_at"]):
                return (row["id"], 0.0, 0.0, -1)
            # Bulk-quote fast path — no per-symbol HTTP
            price = _bulk_prices.get(symbol, 0.0)
            if price > 0 and entry > 0:
                pct = (price - entry) / entry * 100.0
                worked = 1 if pct >= 2.0 else (0 if pct <= -1.0 else None)
                return (row["id"], price, pct, worked)
            if _should_skip(symbol):
                return None
            try:
                from agents.tools import get_technical_indicators
                data = get_technical_indicators(symbol)
                if "error" in data:
                    with _error_lock:
                        _error_cache[symbol] = datetime.now()
                    return None
                price = float(data.get("ltp") or data.get("price") or 0)
                if price <= 0 or entry <= 0:
                    return None
                pct = (price - entry) / entry * 100.0
                if pct >= 2.0:
                    worked = 1
                elif pct <= -1.0:
                    worked = 0
                else:
                    worked = None
                return (row["id"], price, pct, worked)
            except Exception:
                with _error_lock:
                    _error_cache[symbol] = datetime.now()
                return None

        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(_fetch_price, row) for row in rows]
            results = []
            for f in futures:
                try:
                    res = f.result(timeout=30)
                    if res is not None:
                        results.append(res)
                except Exception:
                    pass

        if not results:
            return

        now_iso = _now().isoformat(timespec="seconds")
        conn = _get_conn()
        try:
            for (row_id, price, pct, worked) in results:
                # NO_FILL rows carry NULL price/pct — they must never look
                # like flat trades to the Report Card or accuracy math.
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
