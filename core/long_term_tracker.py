"""
💎 Long-Term Pick Tracker — remember every long-term call, and REVISE it.

A long-term pick isn't fire-and-forget. This logs each pick when it's made, then
re-checks the open ones on a schedule and REVISES the call when the thesis breaks
— the stock loses its 200-day uptrend, its long-term momentum turns negative, or
its score collapses. A revision is surfaced (and pushed to Telegram) as an EXIT
so a holder isn't left in a name that stopped being a compounder.

Pure lifecycle logic (`_revision`) is unit-tested; the SQLite layer uses a
monkeypatchable path and fails safe (a broken tracker degrades to "no changes",
never to a false exit signal).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent / "logs" / "long_term.db"

ACTIVE = "ACTIVE"
EXITED = "EXITED"

_DDL = """
CREATE TABLE IF NOT EXISTS lt_picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    added_at TEXT NOT NULL,
    entry_price REAL,
    score REAL,
    thesis TEXT,
    factors TEXT,                  -- json
    status TEXT NOT NULL,          -- ACTIVE | EXITED
    last_reviewed_at TEXT,
    review_note TEXT,
    exit_price REAL,
    return_pct REAL
);
CREATE INDEX IF NOT EXISTS idx_lt_status ON lt_picks(status);
CREATE INDEX IF NOT EXISTS idx_lt_symbol ON lt_picks(symbol);
"""


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    c = sqlite3.connect(_DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    for stmt in _DDL.strip().split(";"):
        if stmt.strip():
            c.execute(stmt)
    c.commit()
    return c


# ══════════════════════════════════════════════════════════════════════════════
# Pure revision logic — when does a long-term call get pulled?
# ══════════════════════════════════════════════════════════════════════════════

def _revision(current: dict) -> tuple[bool, str]:
    """Given a fresh long_term_score for a held pick, decide whether to REVISE
    (exit). Exit when the structural thesis has broken. Returns (exit?, reason).
    Pure."""
    if not current or current.get("verdict") == "SKIP":
        # SKIP already means a gate broke (below 200-DMA / illiquid) or score died
        if not current.get("above_200dma", True):
            return True, "lost its 200-day uptrend (fell below 200-DMA)"
        return True, "no longer meets the long-term bar"
    if not current.get("dma200_rising", True) and current.get("mom_12m_pct", 0) < 0:
        return True, "200-DMA flat/falling and 12-month momentum turned negative"
    if current.get("score", 100) < 40:
        return True, f"quality score fell to {current.get('score')}"
    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════

def record_picks(picks: list[dict]) -> list[dict]:
    """Log new long-term picks. A symbol already ACTIVE is not re-added (no
    duplicate calls). Returns the picks that were newly recorded. Fail-open."""
    added: list[dict] = []
    try:
        c = _conn()
        try:
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            active = {r["symbol"] for r in c.execute(
                "SELECT symbol FROM lt_picks WHERE status=?", (ACTIVE,)).fetchall()}
            for p in picks or []:
                sym = (p.get("symbol") or "").upper()
                if not sym or sym in active:
                    continue
                c.execute(
                    "INSERT INTO lt_picks (symbol, added_at, entry_price, score, "
                    "thesis, factors, status, last_reviewed_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (sym, now, float(p.get("price") or 0), float(p.get("score") or 0),
                     p.get("thesis", ""), json.dumps(p.get("factors", [])),
                     ACTIVE, now))
                active.add(sym)
                added.append({"symbol": sym, "price": p.get("price"),
                              "score": p.get("score"), "thesis": p.get("thesis", "")})
            c.commit()
        finally:
            c.close()
    except Exception:
        pass
    return added


def review_picks() -> list[dict]:
    """Re-check every ACTIVE pick against fresh data and REVISE (exit) the ones
    whose thesis has broken. Returns the revisions (each {symbol, reason,
    return_pct, entry, price}). Fail-open → []."""
    revisions: list[dict] = []
    try:
        from data.bhavcopy_store import get_ohlcv
        from scan.long_term import long_term_score
    except Exception:
        return []
    try:
        c = _conn()
        try:
            rows = [dict(r) for r in c.execute(
                "SELECT * FROM lt_picks WHERE status=?", (ACTIVE,)).fetchall()]
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            for row in rows:
                df = None
                try:
                    df = get_ohlcv(row["symbol"])
                except Exception:
                    df = None
                if df is None or len(df) < 60:
                    continue                       # can't judge → hold
                cur = long_term_score(df)
                price = float(cur.get("price") or 0)
                do_exit, reason = _revision(cur)
                if do_exit and price > 0:
                    entry = float(row["entry_price"] or 0)
                    ret = ((price - entry) / entry * 100) if entry > 0 else 0.0
                    c.execute(
                        "UPDATE lt_picks SET status=?, last_reviewed_at=?, "
                        "review_note=?, exit_price=?, return_pct=? WHERE id=?",
                        (EXITED, now, reason, price, round(ret, 1), row["id"]))
                    revisions.append({"symbol": row["symbol"], "reason": reason,
                                      "return_pct": round(ret, 1), "entry": entry,
                                      "price": price})
                else:
                    c.execute("UPDATE lt_picks SET last_reviewed_at=?, score=? "
                              "WHERE id=?", (now, float(cur.get("score") or 0),
                                             row["id"]))
            c.commit()
        finally:
            c.close()
    except Exception:
        pass
    return revisions


def active_picks() -> list[dict]:
    """Current open long-term picks, strongest first. Fail-open → []."""
    try:
        c = _conn()
        try:
            rows = c.execute("SELECT * FROM lt_picks WHERE status=? "
                             "ORDER BY score DESC", (ACTIVE,)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["factors"] = json.loads(d["factors"] or "[]")
                out.append(d)
            return out
        finally:
            c.close()
    except Exception:
        return []


def exited_picks(limit: int = 30) -> list[dict]:
    """Recently revised (exited) picks — the honest record of calls pulled."""
    try:
        c = _conn()
        try:
            rows = c.execute("SELECT * FROM lt_picks WHERE status=? "
                             "ORDER BY last_reviewed_at DESC LIMIT ?",
                             (EXITED, limit)).fetchall()
            return [dict(r) for r in rows]
        finally:
            c.close()
    except Exception:
        return []
