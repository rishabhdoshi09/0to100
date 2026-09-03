"""Persistent candidate lifecycle. Survives scan reruns and process restart."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "logs" / "product" / "candidates.db"

DISCOVERED = "DISCOVERED"
SCREENED = "SCREENED"
WATCH = "WATCH"
RESEARCHING = "RESEARCHING"
QUALIFIED = "QUALIFIED"
READY = "READY"
ENTERED = "ENTERED"
WAIT = "WAIT"
REJECTED = "REJECTED"
INVALIDATED = "INVALIDATED"
EXPIRED = "EXPIRED"
CLOSED = "CLOSED"
WAIT_EVIDENCE = "WAIT_EVIDENCE"

STATES = (
    DISCOVERED, SCREENED, WATCH, RESEARCHING, QUALIFIED, READY, ENTERED,
    WAIT, WAIT_EVIDENCE, REJECTED, INVALIDATED, EXPIRED, CLOSED,
)

_RANK = {
    DISCOVERED: 0, SCREENED: 1, WATCH: 1, RESEARCHING: 2, WAIT_EVIDENCE: 2,
    QUALIFIED: 3, WAIT: 3, REJECTED: 3, READY: 4, ENTERED: 5,
    INVALIDATED: 6, EXPIRED: 6, CLOSED: 6,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_path(path: Path | None = None) -> Path:
    target = path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _connect(path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path(path)))
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            session_date TEXT,
            state TEXT NOT NULL,
            reason TEXT,
            first_seen_at TEXT,
            updated_at TEXT,
            scan_run_id TEXT,
            recommendation_id TEXT,
            decision_id TEXT,
            paper_intent_id TEXT,
            outcome_id TEXT,
            payload_json TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL,
            from_state TEXT,
            to_state TEXT NOT NULL,
            reason TEXT,
            at TEXT NOT NULL,
            trigger TEXT
        )
        """
    )
    return con


def candidate_id(symbol: str, session_date: str) -> str:
    return f"{str(session_date or '')[:10]}:{str(symbol or '').upper()}"


def recommendation_id(scan_run_id: str, symbol: str, tier: str = "") -> str:
    return f"{scan_run_id}:{str(symbol or '').upper()}:{tier}"


def upsert(
    *,
    symbol: str,
    session_date: str,
    state: str,
    reason: str = "",
    scan_run_id: str = "",
    recommendation_id_value: str = "",
    decision_id_value: str = "",
    paper_intent_id: str = "",
    outcome_id: str = "",
    payload: Mapping[str, Any] | None = None,
    trigger: str = "",
    path: Path | None = None,
    demote: bool = False,
) -> dict[str, Any]:
    cid = candidate_id(symbol, session_date)
    now = _now()
    con = _connect(path)
    prev = con.execute("SELECT * FROM candidates WHERE candidate_id=?", (cid,)).fetchone()
    if prev is None:
        con.execute(
            """INSERT INTO candidates (
                candidate_id, symbol, session_date, state, reason, first_seen_at, updated_at,
                scan_run_id, recommendation_id, decision_id, paper_intent_id, outcome_id, payload_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                cid, str(symbol).upper(), str(session_date)[:10], state, reason, now, now,
                scan_run_id, recommendation_id_value, decision_id_value, paper_intent_id,
                outcome_id, json.dumps(dict(payload or {}), default=str),
            ),
        )
        con.execute(
            "INSERT INTO transitions (candidate_id, from_state, to_state, reason, at, trigger) VALUES (?,?,?,?,?,?)",
            (cid, "", state, reason, now, trigger),
        )
    else:
        merged = dict(prev)
        next_state = state or merged["state"]
        if not demote and _RANK.get(next_state, 0) < _RANK.get(merged["state"], 0):
            next_state = merged["state"]
        if next_state != merged["state"] or reason:
            con.execute(
                "INSERT INTO transitions (candidate_id, from_state, to_state, reason, at, trigger) VALUES (?,?,?,?,?,?)",
                (cid, merged["state"], next_state, reason or merged["reason"], now, trigger),
            )
        body = dict(payload or {})
        if merged.get("payload_json"):
            try:
                old = json.loads(merged["payload_json"])
                if isinstance(old, dict):
                    old.update(body)
                    body = old
            except Exception:
                pass
        con.execute(
            """UPDATE candidates SET state=?, reason=?, updated_at=?,
               scan_run_id=COALESCE(NULLIF(?, ''), scan_run_id),
               recommendation_id=COALESCE(NULLIF(?, ''), recommendation_id),
               decision_id=COALESCE(NULLIF(?, ''), decision_id),
               paper_intent_id=COALESCE(NULLIF(?, ''), paper_intent_id),
               outcome_id=COALESCE(NULLIF(?, ''), outcome_id),
               payload_json=?
               WHERE candidate_id=?""",
            (
                next_state, reason or merged["reason"], now,
                scan_run_id, recommendation_id_value, decision_id_value,
                paper_intent_id, outcome_id, json.dumps(body, default=str), cid,
            ),
        )
    con.commit()
    row = dict(con.execute("SELECT * FROM candidates WHERE candidate_id=?", (cid,)).fetchone())
    con.close()
    return row


def get(cid: str, path: Path | None = None) -> dict[str, Any] | None:
    con = _connect(path)
    row = con.execute("SELECT * FROM candidates WHERE candidate_id=?", (cid,)).fetchone()
    con.close()
    return dict(row) if row else None


def list_candidates(*, session_date: str = "", states: tuple[str, ...] = (), limit: int = 200, path: Path | None = None) -> list[dict[str, Any]]:
    con = _connect(path)
    q = "SELECT * FROM candidates"
    args: list[Any] = []
    clauses = []
    if session_date:
        clauses.append("session_date=?")
        args.append(str(session_date)[:10])
    if states:
        clauses.append("state IN (%s)" % ",".join("?" * len(states)))
        args.extend(states)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY updated_at DESC LIMIT ?"
    args.append(int(limit))
    rows = [dict(r) for r in con.execute(q, args)]
    con.close()
    return rows


def transitions_for(cid: str, path: Path | None = None) -> list[dict[str, Any]]:
    con = _connect(path)
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM transitions WHERE candidate_id=? ORDER BY id", (cid,)
    )]
    con.close()
    return rows


def state_counts(session_date: str = "", path: Path | None = None) -> dict[str, int]:
    con = _connect(path)
    if session_date:
        rows = con.execute(
            "SELECT state, count(*) c FROM candidates WHERE session_date=? GROUP BY 1",
            (str(session_date)[:10],),
        )
    else:
        rows = con.execute("SELECT state, count(*) c FROM candidates GROUP BY 1")
    out = {str(r["state"]): int(r["c"]) for r in rows}
    con.close()
    return out
