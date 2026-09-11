"""Opportunity memory — the same name across sessions is one evolving opportunity."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from core.runtime_paths import logs_dir, logs_path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = logs_dir() / "product" / "opportunity_memory.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def opportunity_id(symbol: str) -> str:
    return str(symbol or "").upper()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    from product.sqlite_runtime import connect

    con = connect(path or DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS opportunities (
            opportunity_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            first_seen_at TEXT,
            first_setup TEXT,
            peak_tier TEXT,
            peak_rank INTEGER,
            last_session TEXT,
            last_scan_run_id TEXT,
            last_state TEXT,
            last_decision TEXT,
            last_entry_state TEXT,
            last_execution_state TEXT,
            last_reason TEXT,
            wait_trigger_json TEXT,
            research_started_at TEXT,
            research_completed_at TEXT,
            wake_event TEXT,
            payload_json TEXT,
            updated_at TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS opportunity_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            opportunity_id TEXT NOT NULL,
            at TEXT NOT NULL,
            event TEXT,
            old_state TEXT,
            new_state TEXT,
            reason TEXT,
            session_date TEXT,
            old_decision TEXT,
            new_decision TEXT,
            old_entry_state TEXT,
            new_entry_state TEXT,
            old_execution_state TEXT,
            new_execution_state TEXT
        )
        """
    )
    # Existing desks may already have the pre-decision-transition event schema.
    # Additive migrations keep the ledger readable without replacing the DB.
    event_cols = {str(row[1]) for row in con.execute("PRAGMA table_info(opportunity_events)")}
    for name in (
        "old_decision",
        "new_decision",
        "old_entry_state",
        "new_entry_state",
        "old_execution_state",
        "new_execution_state",
    ):
        if name not in event_cols:
            con.execute(f"ALTER TABLE opportunity_events ADD COLUMN {name} TEXT")
    return con


def remember(
    *,
    symbol: str,
    session_date: str,
    scan_run_id: str = "",
    state: str = "",
    decision: str = "",
    entry_state: str = "",
    execution_state: str = "",
    reason: str = "",
    setup: str = "",
    tier: str = "",
    rank: int | None = None,
    wait_trigger: Mapping[str, Any] | None = None,
    research_started: bool = False,
    research_completed: bool = False,
    wake_event: str = "",
    payload: Mapping[str, Any] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    oid = opportunity_id(symbol)
    now = _now()
    con = _connect(path)
    prev = con.execute("SELECT * FROM opportunities WHERE opportunity_id=?", (oid,)).fetchone()
    if prev is None:
        con.execute(
            """INSERT INTO opportunities (
                opportunity_id, symbol, first_seen_at, first_setup, peak_tier, peak_rank,
                last_session, last_scan_run_id, last_state, last_decision, last_entry_state,
                last_execution_state, last_reason, wait_trigger_json, research_started_at,
                research_completed_at, wake_event, payload_json, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                oid, oid, now, setup, tier, rank, str(session_date)[:10], scan_run_id,
                state, decision, entry_state, execution_state, reason,
                json.dumps(dict(wait_trigger or {}), default=str),
                now if research_started else "",
                now if research_completed else "",
                wake_event, json.dumps(dict(payload or {}), default=str), now,
            ),
        )
        con.execute(
            """INSERT INTO opportunity_events (
                opportunity_id, at, event, old_state, new_state, reason, session_date,
                old_decision, new_decision, old_entry_state, new_entry_state,
                old_execution_state, new_execution_state
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                oid, now, "DISCOVERED", "", state, reason, str(session_date)[:10],
                "", decision, "", entry_state, "", execution_state,
            ),
        )
    else:
        merged = dict(prev)
        peak_tier = merged.get("peak_tier") or tier
        if tier == "high_conviction" or (tier == "good_setup" and peak_tier not in {"high_conviction"}):
            peak_tier = tier
        peak_rank = merged.get("peak_rank")
        if rank is not None and (peak_rank is None or int(rank) < int(peak_rank)):
            peak_rank = rank
        started = merged.get("research_started_at") or (now if research_started else "")
        completed = merged.get("research_completed_at") or (now if research_completed else "")
        old_state = str(merged.get("last_state") or "")
        old_decision = str(merged.get("last_decision") or "")
        old_entry_state = str(merged.get("last_entry_state") or "")
        old_execution_state = str(merged.get("last_execution_state") or "")
        next_state = str(state or old_state)
        next_decision = str(decision or old_decision)
        next_entry_state = str(entry_state or old_entry_state)
        next_execution_state = str(execution_state or old_execution_state)
        if state and next_state != old_state:
            con.execute(
                """INSERT INTO opportunity_events (
                    opportunity_id, at, event, old_state, new_state, reason, session_date,
                    old_decision, new_decision, old_entry_state, new_entry_state,
                    old_execution_state, new_execution_state
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    oid, now, wake_event or "REEVALUATED", old_state, next_state, reason,
                    str(session_date)[:10], old_decision, next_decision,
                    old_entry_state, next_entry_state, old_execution_state, next_execution_state,
                ),
            )
        if decision and next_decision != old_decision:
            con.execute(
                """INSERT INTO opportunity_events (
                    opportunity_id, at, event, old_state, new_state, reason, session_date,
                    old_decision, new_decision, old_entry_state, new_entry_state,
                    old_execution_state, new_execution_state
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    oid, now, "DECISION_CHANGED", old_state, next_state, reason,
                    str(session_date)[:10], old_decision, next_decision,
                    old_entry_state, next_entry_state, old_execution_state, next_execution_state,
                ),
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
            """UPDATE opportunities SET
               last_session=?, last_scan_run_id=?, last_state=?, last_decision=?,
               last_entry_state=?, last_execution_state=?, last_reason=?,
               wait_trigger_json=?, peak_tier=?, peak_rank=?,
               research_started_at=?, research_completed_at=?,
               wake_event=?, payload_json=?, updated_at=?,
               first_setup=COALESCE(NULLIF(first_setup,''), ?)
               WHERE opportunity_id=?""",
            (
                str(session_date)[:10], scan_run_id or merged.get("last_scan_run_id"),
                next_state, next_decision, next_entry_state, next_execution_state,
                reason or merged.get("last_reason"),
                json.dumps(dict(wait_trigger or json.loads(merged.get("wait_trigger_json") or "{}")), default=str),
                peak_tier, peak_rank, started, completed, wake_event or merged.get("wake_event"),
                json.dumps(body, default=str), now, setup, oid,
            ),
        )
    con.commit()
    row = dict(con.execute("SELECT * FROM opportunities WHERE opportunity_id=?", (oid,)).fetchone())
    con.close()
    return row


def get(symbol: str, path: Path | None = None) -> dict[str, Any] | None:
    con = _connect(path)
    row = con.execute("SELECT * FROM opportunities WHERE opportunity_id=?", (opportunity_id(symbol),)).fetchone()
    con.close()
    return dict(row) if row else None


def events_for(symbol: str, path: Path | None = None) -> list[dict[str, Any]]:
    con = _connect(path)
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM opportunity_events WHERE opportunity_id=? ORDER BY id",
        (opportunity_id(symbol),),
    )]
    con.close()
    return rows


def list_open(*, states: tuple[str, ...] = (), limit: int = 80, path: Path | None = None) -> list[dict[str, Any]]:
    con = _connect(path)
    q = "SELECT * FROM opportunities"
    args: list[Any] = []
    if states:
        q += " WHERE last_state IN (%s)" % ",".join("?" * len(states))
        args.extend(states)
    q += " ORDER BY updated_at DESC LIMIT ?"
    args.append(int(limit))
    rows = [dict(r) for r in con.execute(q, args)]
    con.close()
    return rows


def next_session_set(session: str = "", path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    rows = list_open(path=path)
    buckets = {
        "READY": [],
        "WAIT_ENTRY": [],
        "WATCH_HIGH_PRIORITY": [],
        "RESEARCH_PENDING": [],
        "EVENT_PENDING": [],
    }
    for row in rows:
        state = str(row.get("last_state") or "")
        item = {
            "symbol": row.get("symbol"),
            "state": state,
            "decision": row.get("last_decision"),
            "why": row.get("last_reason"),
            "next": "",
            "first_seen_at": row.get("first_seen_at"),
            "peak_tier": row.get("peak_tier"),
            "wait_trigger": {},
        }
        try:
            item["wait_trigger"] = json.loads(row.get("wait_trigger_json") or "{}")
        except Exception:
            item["wait_trigger"] = {}
        trigger = item["wait_trigger"] if isinstance(item["wait_trigger"], dict) else {}
        item["next"] = str(trigger.get("reconsider_when") or "")
        if state == "READY":
            item["next"] = item["next"] or "await broker login or next entry window"
            buckets["READY"].append(item)
        elif state == "WAIT":
            buckets["WAIT_ENTRY"].append(item)
        elif state == "WAIT_EVIDENCE":
            item["next"] = item["next"] or "acquire missing evidence"
            buckets["RESEARCH_PENDING"].append(item)
        elif state == "WATCH":
            buckets["WATCH_HIGH_PRIORITY"].append(item)
        elif trigger.get("kind"):
            buckets["EVENT_PENDING"].append(item)
    return buckets


def wake_candidates(scan: Mapping[str, Any], *, path: Path | None = None) -> list[dict[str, Any]]:
    """Return remembered WAIT names whose price trigger is now crossed."""
    records = {str(r.get("symbol") or "").upper(): r for r in (scan.get("records") or []) if isinstance(r, dict)}
    woken = []
    for row in list_open(states=("WAIT", "WAIT_EVIDENCE", "WATCH"), path=path):
        symbol = str(row.get("symbol") or "").upper()
        rec = records.get(symbol)
        if not rec:
            continue
        try:
            trigger = json.loads(row.get("wait_trigger_json") or "{}")
        except Exception:
            continue
        kind = str(trigger.get("kind") or "")
        level = trigger.get("price")
        price = rec.get("price") or rec.get("entry")
        try:
            price_f = float(price) if price is not None else None
            level_f = float(level) if level is not None else None
        except (TypeError, ValueError):
            price_f = level_f = None
        hit = False
        if kind == "PRICE_LTE" and price_f is not None and level_f is not None and price_f <= level_f:
            hit = True
        if kind == "PRICE_GTE" and price_f is not None and level_f is not None and price_f >= level_f:
            hit = True
        if kind == "EVIDENCE_ACQUIRED":
            continue
        if hit:
            woken.append({
                "symbol": symbol,
                "wake_event": kind,
                "old_state": row.get("last_state"),
                "price": price_f,
                "trigger": trigger,
            })
    return woken
