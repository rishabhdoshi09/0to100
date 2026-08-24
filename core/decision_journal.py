"""
Decision Journal — every YES, NO and WAIT becomes an experiment.

Trades were already logged; DECISIONS were not. This records every candidate
the desk evaluated — TAKEN, REJECTED, or WAIT — with the reason and whatever
prediction the system held, then resolves the outcome on the horizon-th
official trading session (canonical `outcome_resolver`, never a live quote).

Six months from now this answers the questions that actually build trust:

  • Were rejected trades better than accepted ones? (Which gates EARN money
    and which gates COST money — per rejection reason.)
  • When the system said "70% win", did ~70% actually win? (Calibration.)

Read-only measurement. Places no orders.
One row per symbol × day × decision × reason (a 15-min rescan doesn't flood
the journal; two different gates on the same name the same day both survive).

Aggregation splits two units:
  opportunity = symbol × day × stance  — Shadow Book / calibration / WAIT
  reason-row  = one gate on that opportunity — gate attribution only

WAIT is timing patience, not a second NO: later cheaper entry, ran away, or failed.
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
    decision TEXT NOT NULL,          -- TAKEN | REJECTED | WAIT
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
    outcome_pct REAL,
    wait_result TEXT                 -- OFFERED | RAN_AWAY | FAILED (WAIT only)
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
    cols = {r[1] for r in c.execute("PRAGMA table_info(decisions)")}
    if "wait_result" not in cols:
        c.execute("ALTER TABLE decisions ADD COLUMN wait_result TEXT")
    c.commit()
    return c


def _norm_reason(reason: str, decision: str) -> str:
    text = str(reason or "").strip()
    kind = str(decision or "").upper()
    if kind in {"TAKEN", "WAIT"}:
        return text[:120]
    if not text:
        return "OTHER"
    key = text.upper().replace(" ", "_")
    known = {
        "EXTENSION", "LOW_CONVICTION", "WEAK_CLOSE", "BLOWOFF_RSI",
        "LAGGARD", "POOR_BREADTH", "RISK_LIMIT", "CORRELATION",
        "LIQUIDITY", "MACRO", "ALREADY_OWNED", "DRIFT", "OTHER",
    }
    if key in known:
        return key
    t = text.lower()
    if "extend" in t or "chase" in t:
        return "EXTENSION"
    if "weak close" in t or "weak_close" in t or "bull-trap" in t or "bull trap" in t:
        return "WEAK_CLOSE"
    if "rsi" in t or "blow" in t or "overheat" in t:
        return "BLOWOFF_RSI"
    if "breadth" in t or "narrow" in t:
        return "POOR_BREADTH"
    if "corr" in t:
        return "CORRELATION"
    if "liquid" in t:
        return "LIQUIDITY"
    if "laggard" in t:
        return "LAGGARD"
    if "decay" in t or "drift" in t:
        return "DRIFT"
    if "macro" in t or "risk_off" in t or "risk-off" in t:
        return "MACRO"
    if "convic" in t:
        return "LOW_CONVICTION"
    return text[:120]


def log_decision(symbol: str, decision: str, reason: str = "",
                 source: str = "", entry_ref: float = 0.0,
                 stop_ref: float = 0.0, score: float = 0.0,
                 ev_pct: float | None = None, p_win: float | None = None,
                 confidence: str | None = None) -> None:
    """One row per symbol × day × decision × reason.

    Scans repeat every few minutes — those collapse. Two different rejection
    reasons on the same name the same day do NOT: gate attribution needs both.
    """
    try:
        if entry_ref <= 0:
            return                              # no reference price → no claim
        today = _now().strftime("%Y-%m-%d")
        kind = str(decision or "").strip().upper() or "REJECTED"
        why = _norm_reason(reason, kind)
        c = _conn()
        try:
            dup = c.execute(
                "SELECT id FROM decisions WHERE symbol=? AND decision=? "
                "AND IFNULL(reason,'')=? AND decided_at LIKE ?",
                (symbol.upper(), kind, why, f"{today}%")).fetchone()
            if dup:
                return
            c.execute(
                "INSERT INTO decisions (symbol, decided_at, decision, reason, "
                "source, entry_ref, stop_ref, score, ev_pct, p_win, confidence) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (symbol.upper(),
                 _now().isoformat(timespec="seconds"),
                 kind, why, source, float(entry_ref),
                 float(stop_ref), float(score), ev_pct, p_win, confidence))
            c.commit()
        finally:
            c.close()
    except Exception as exc:
        log.debug("decision_log_failed", error=str(exc))


def _opportunity_key(row: dict | sqlite3.Row) -> tuple[str, str, str]:
    """One economic decision: symbol × day × stance. Reasons are attributes."""
    return (
        str(row["symbol"] if not isinstance(row, dict) else row.get("symbol") or "").upper(),
        str(row["decided_at"] if not isinstance(row, dict) else row.get("decided_at") or "")[:10],
        str(row["decision"] if not isinstance(row, dict) else row.get("decision") or "").upper(),
    )


def fold_opportunities(rows: list[dict]) -> list[dict]:
    """Collapse reason-rows into one opportunity. Outcome taken from the first
    settled row — same path, so later reasons must not double-count P&L."""
    folded: dict[tuple[str, str, str], dict] = {}
    for raw in rows:
        row = dict(raw)
        key = _opportunity_key(row)
        why = str(row.get("reason") or "").strip()
        if key not in folded:
            item = dict(row)
            item["reasons"] = [why] if why else []
            folded[key] = item
            continue
        if why and why not in folded[key]["reasons"]:
            folded[key]["reasons"].append(why)
    return list(folded.values())


def update_outcomes(check_after_days: int = 5, lookback_days: int = 40) -> int:
    """Resolve outcomes after the horizon-th official trading session.

    `check_after_days` is kept for call-site compatibility and is NOT the
    resolution clock — a delayed runner must not mark day-8 as day-5.
    WAIT uses wait_patience (later entry / ran away / failed), not a second
    close-to-close NO. Missing bars stay pending. Never writes a live quote.
    """
    try:
        too_old = (_now() - timedelta(days=lookback_days)) \
            .isoformat(timespec="seconds")
        c = _conn()
        try:
            rows = c.execute(
                "SELECT id, symbol, decided_at, decision, entry_ref, stop_ref "
                "FROM decisions WHERE outcome_pct IS NULL AND decided_at >= ?",
                (too_old,),
            ).fetchall()
            if not rows:
                return 0
            from core.outcome_resolver import session_close_return, wait_patience
            n = 0
            now_iso = _now().isoformat(timespec="seconds")
            for r in rows:
                entry = float(r["entry_ref"] or 0)
                if entry <= 0:
                    continue
                kind = str(r["decision"] or "").upper()
                day = str(r["decided_at"])[:10]
                wait_result = None
                if kind == "WAIT":
                    resolved = wait_patience(r["symbol"], day, entry)
                    if resolved is None:
                        continue
                    _px, pct, wait_result = resolved
                else:
                    resolved = session_close_return(r["symbol"], day)
                    if resolved is None:
                        continue
                    _px, pct = resolved
                    pct = (_px - entry) / entry * 100.0
                c.execute(
                    "UPDATE decisions SET outcome_checked_at=?, "
                    "outcome_price=?, outcome_pct=?, wait_result=? WHERE id=?",
                    (now_iso, float(_px), round(pct, 2), wait_result, r["id"]),
                )
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
    """Accepted vs rejected opportunities, plus per-gate reason rows.

    Shadow Book n is opportunities (BLUSPRING rejected for two gates still
    counts once). `by_reason` still counts each gate once, because that is
    how you evaluate a guard. WAIT is timing patience, not a second NO.
    """
    rows = _resolved()
    ops = fold_opportunities(rows)

    def _agg(sub: list[dict]) -> dict:
        n = len(sub)
        if not n:
            return {"n": 0, "avg_outcome_pct": 0.0, "win_rate": 0.0}
        wins = sum(1 for r in sub if r["outcome_pct"] >= WIN_PCT)
        return {"n": n,
                "avg_outcome_pct": round(sum(r["outcome_pct"] for r in sub) / n, 2),
                "win_rate": round(wins / n * 100, 1)}

    taken = _agg([r for r in ops if r["decision"] == "TAKEN"])
    rejected = _agg([r for r in ops if r["decision"] == "REJECTED"])
    waited_ops = [r for r in ops if r["decision"] == "WAIT"]
    waited = _agg(waited_ops)
    wait_timing = {"OFFERED": 0, "RAN_AWAY": 0, "FAILED": 0}
    for r in waited_ops:
        key = str(r.get("wait_result") or "")
        if key in wait_timing:
            wait_timing[key] += 1
    waited["timing"] = wait_timing
    waited["unit"] = "opportunity"
    taken["unit"] = "opportunity"
    rejected["unit"] = "opportunity"
    by_reason: dict[str, dict] = {}
    for r in rows:
        if r["decision"] == "REJECTED" and r.get("reason"):
            by_reason.setdefault(r["reason"], []).append(r)
    by_reason = {k: _agg(v) for k, v in by_reason.items()}
    for stats in by_reason.values():
        stats["unit"] = "gate"

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
    return {"taken": taken, "rejected": rejected, "wait": waited,
            "by_reason": by_reason, "verdict": verdict}


def calibration_report(min_n: int = 20) -> dict:
    """When the system said P(win)=X, did ~X% actually win? Buckets of
    predicted p_win vs realized win-rate — the honesty check on our own
    probabilities. {buckets: [...], verdict}."""
    rows = fold_opportunities([r for r in _resolved() if r.get("p_win") is not None])
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
