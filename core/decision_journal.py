"""
Decision Journal — every decision becomes a prediction market.

Trades were already logged; DECISIONS were not. This records every candidate
the autopilot evaluated — TAKEN or REJECTED, with the reason and whatever
prediction the system held at that moment (EV%, P(win), confidence) — and
then resolves the outcome 5 sessions later exactly like the signal tracker.

Six months from now this answers the questions that actually build trust:

  • Were rejected trades better than accepted ones? (Which gates EARN money
    and which gates COST money — per rejection reason.)
  • When the system said "70% win", did ~70% actually win? (Calibration —
    can the probabilities be trusted?)

Read-only measurement infrastructure: nothing here changes a gate or places
a trade. One row per symbol × day × decision (a candidate re-scanned every
15 min doesn't flood the journal).
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta

from logger import get_logger

log = get_logger(__name__)


def _now() -> datetime:
    """Naive IST wall-clock time — same storage format as plain
    datetime.now() (no offset suffix, backward-compatible with existing
    rows), but correct regardless of the host's own timezone. A UTC-hosted
    VPS (the documented 24/7 setup — docs/ALWAYS_ON.md) running naive
    datetime.now() here would date-bucket/window every decision 5.5 hours
    off, same class of bug market_clock.py exists to prevent elsewhere."""
    from core.market_clock import now_ist
    return now_ist().replace(tzinfo=None)

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "logs", "decisions.db")

_DDL = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    decision TEXT NOT NULL,          -- TAKEN | REJECTED
    reason TEXT,                     -- rejection reason category ('' for TAKEN)
    source TEXT,
    entry_ref REAL,                  -- reference price at decision time
    stop_ref REAL,
    score REAL,
    ev_pct REAL,                     -- prediction at decision time (nullable)
    p_win REAL,
    confidence TEXT,
    outcome_checked_at TEXT,
    outcome_price REAL,
    outcome_pct REAL
);
CREATE INDEX IF NOT EXISTS idx_dec_symbol_date ON decisions(symbol, decided_at);
CREATE INDEX IF NOT EXISTS idx_dec_outcome ON decisions(outcome_pct);
"""

# Same convention as signal_outcome_tracker: ≥+2% in 5 sessions = worked.
WIN_PCT = 2.0


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    c = sqlite3.connect(_DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    for stmt in _DDL.strip().split(";"):
        if stmt.strip():
            c.execute(stmt)
    c.commit()
    return c


def log_decision(symbol: str, decision: str, reason: str = "",
                 source: str = "", entry_ref: float = 0.0,
                 stop_ref: float = 0.0, score: float = 0.0,
                 ev_pct: float | None = None, p_win: float | None = None,
                 confidence: str | None = None) -> None:
    """One row per symbol × day × decision type (dedup — scans repeat)."""
    try:
        if entry_ref <= 0:
            return                              # no reference price → no claim
        today = _now().strftime("%Y-%m-%d")
        c = _conn()
        try:
            dup = c.execute(
                "SELECT id FROM decisions WHERE symbol=? AND decision=? "
                "AND decided_at LIKE ?",
                (symbol.upper(), decision, f"{today}%")).fetchone()
            if dup:
                return
            c.execute(
                "INSERT INTO decisions (symbol, decided_at, decision, reason, "
                "source, entry_ref, stop_ref, score, ev_pct, p_win, confidence) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (symbol.upper(),
                 _now().isoformat(timespec="seconds"),
                 decision, reason[:120], source, float(entry_ref),
                 float(stop_ref), float(score), ev_pct, p_win, confidence))
            c.commit()
        finally:
            c.close()
    except Exception as exc:
        log.debug("decision_log_failed", error=str(exc))


def update_outcomes(check_after_days: int = 5, lookback_days: int = 40) -> int:
    """Resolve outcomes for decisions ≥5 sessions old (same convention as the
    signal tracker). Returns rows updated. Quote misses stay pending — a
    missing price is never written as an outcome (no fake data)."""
    try:
        cutoff = (_now() - timedelta(days=check_after_days)) \
            .isoformat(timespec="seconds")
        too_old = (_now() - timedelta(days=lookback_days)) \
            .isoformat(timespec="seconds")
        c = _conn()
        try:
            rows = c.execute(
                "SELECT id, symbol, entry_ref FROM decisions "
                "WHERE outcome_pct IS NULL AND decided_at <= ? "
                "AND decided_at >= ?", (cutoff, too_old)).fetchall()
            if not rows:
                return 0
            from data.live_quotes import get_live_quotes
            quotes = get_live_quotes(sorted({r["symbol"] for r in rows}))
            n = 0
            now_iso = _now().isoformat(timespec="seconds")
            for r in rows:
                q = quotes.get(r["symbol"]) or {}
                px = float(q.get("price") or 0)
                if px <= 0 or not r["entry_ref"]:
                    continue
                pct = (px - r["entry_ref"]) / r["entry_ref"] * 100
                c.execute("UPDATE decisions SET outcome_checked_at=?, "
                          "outcome_price=?, outcome_pct=? WHERE id=?",
                          (now_iso, px, round(pct, 2), r["id"]))
                n += 1
            c.commit()
            if n:
                log.info("decision_outcomes_updated", n=n)
            return n
        finally:
            c.close()
    except Exception as exc:
        log.debug("decision_outcomes_failed", error=str(exc))
        return 0


def _resolved() -> list[dict]:
    try:
        c = _conn()
        try:
            return [dict(r) for r in c.execute(
                "SELECT * FROM decisions WHERE outcome_pct IS NOT NULL")]
        finally:
            c.close()
    except Exception:
        return []


def decision_report(min_n: int = 10) -> dict:
    """Accepted vs rejected, and per-rejection-reason outcomes — which gates
    EARN and which gates COST. Empty verdict below min_n resolved rows."""
    rows = _resolved()

    def _agg(sub: list[dict]) -> dict:
        n = len(sub)
        if not n:
            return {"n": 0, "avg_outcome_pct": 0.0, "win_rate": 0.0}
        wins = sum(1 for r in sub if r["outcome_pct"] >= WIN_PCT)
        return {"n": n,
                "avg_outcome_pct": round(sum(r["outcome_pct"] for r in sub) / n, 2),
                "win_rate": round(wins / n * 100, 1)}

    taken = _agg([r for r in rows if r["decision"] == "TAKEN"])
    rejected = _agg([r for r in rows if r["decision"] == "REJECTED"])
    by_reason: dict[str, dict] = {}
    for r in rows:
        if r["decision"] == "REJECTED" and r.get("reason"):
            by_reason.setdefault(r["reason"], []).append(r)
    by_reason = {k: _agg(v) for k, v in by_reason.items()}

    verdict = ""
    if taken["n"] >= min_n and rejected["n"] >= min_n:
        gap = taken["avg_outcome_pct"] - rejected["avg_outcome_pct"]
        if gap > 0.5:
            verdict = (f"✅ Gates kaam kar rahe hain: taken avg "
                       f"{taken['avg_outcome_pct']:+.1f}% vs rejected "
                       f"{rejected['avg_outcome_pct']:+.1f}% ({gap:+.1f}pp).")
        elif gap < -0.5:
            verdict = (f"⚠️ Rejected trades taken se BEHTAR nikle "
                       f"({rejected['avg_outcome_pct']:+.1f}% vs "
                       f"{taken['avg_outcome_pct']:+.1f}%) — koi gate paisa kha "
                       f"raha hai, breakdown dekho.")
        else:
            verdict = "➖ Taken vs rejected mein abhi koi meaningful gap nahi."
    return {"taken": taken, "rejected": rejected, "by_reason": by_reason,
            "verdict": verdict}


def calibration_report(min_n: int = 20) -> dict:
    """When the system said P(win)=X, did ~X% actually win? Buckets of
    predicted p_win vs realized win-rate — the honesty check on our own
    probabilities. {buckets: [...], verdict}."""
    rows = [r for r in _resolved() if r.get("p_win") is not None]
    spans = [(50, 60), (60, 70), (70, 101)]
    buckets = []
    worst_gap = 0.0
    for lo, hi in spans:
        sub = [r for r in rows if lo <= float(r["p_win"]) < hi]
        n = len(sub)
        if n < min_n:
            buckets.append({"range": f"{lo}-{hi - 1 if hi < 101 else 100}%",
                            "n": n, "predicted": None, "actual": None})
            continue
        predicted = sum(float(r["p_win"]) for r in sub) / n
        actual = sum(1 for r in sub if r["outcome_pct"] >= WIN_PCT) / n * 100
        worst_gap = max(worst_gap, abs(predicted - actual))
        buckets.append({"range": f"{lo}-{hi - 1 if hi < 101 else 100}%",
                        "n": n, "predicted": round(predicted, 1),
                        "actual": round(actual, 1)})
    scored = [b for b in buckets if b["predicted"] is not None]
    if not scored:
        verdict = ("Calibration ke liye data jama ho raha hai — buckets "
                   "min 20 resolved predictions maangte hain.")
    elif worst_gap <= 8:
        verdict = (f"✅ Probabilities bharosemand — worst bucket gap "
                   f"{worst_gap:.0f}pp. HIGH ka matlab sach mein HIGH hai.")
    else:
        verdict = (f"⚠️ Calibration off — worst bucket gap {worst_gap:.0f}pp. "
                   f"Jab tak sudhre, confidence tiers ko discount karo.")
    return {"buckets": buckets, "verdict": verdict}
