"""Write-once FEATURE-002 shadow ledger. Outcomes attach later; features never rewrite."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from research.feature002.constants import (
    DB_PATH,
    FEATURE_SET_VERSION,
    FORWARD_START_DATE,
    LEDGER_DIR,
    eligible_primary,
    event_id,
)


_DDL = """
CREATE TABLE IF NOT EXISTS candidate_sets (
    candidate_set_id TEXT PRIMARY KEY,
    scan_cycle_id TEXT NOT NULL,
    session_date TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    n_candidates INTEGER NOT NULL,
    family_composition TEXT,
    source TEXT NOT NULL,
    feature_set_version TEXT NOT NULL,
    protocol_hash TEXT
);
CREATE TABLE IF NOT EXISTS observations (
    event_id TEXT PRIMARY KEY,
    candidate_set_id TEXT NOT NULL,
    scan_cycle_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL,
    session_date TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    source TEXT NOT NULL,
    families TEXT NOT NULL,
    primary_family TEXT,
    feature_set_version TEXT NOT NULL,
    production_rank_version TEXT NOT NULL,
    shadow_rank_version TEXT NOT NULL,
    protocol_hash TEXT,
    production_score REAL,
    production_rank INTEGER,
    production_verdict TEXT,
    production_signals TEXT,
    production_decision TEXT,
    would_trade INTEGER,
    ready_status TEXT,
    entry REAL,
    stop REAL,
    target REAL,
    chase_risk INTEGER,
    n_structure_passed INTEGER,
    structure_pass INTEGER,
    rs_percentile REAL,
    rs_score REAL,
    rs_rank INTEGER,
    trend_rank INTEGER,
    combined_shadow_rank INTEGER,
    r3_score REAL,
    regime_label TEXT,
    sector TEXT,
    sector_map_version TEXT,
    feature_snapshot TEXT NOT NULL,
    data_quality TEXT,
    eligible_primary INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    event_id TEXT PRIMARY KEY,
    resolved_at TEXT,
    next_open REAL,
    ret_1d REAL,
    ret_5d REAL,
    ret_10d REAL,
    ret_20d REAL,
    mae REAL,
    mfe REAL,
    hit_1r INTEGER,
    hit_2r INTEGER,
    production_traded INTEGER,
    production_outcome TEXT,
    unresolved_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_session ON observations(session_date, source);
CREATE INDEX IF NOT EXISTS idx_obs_set ON observations(candidate_set_id);
"""


def _conn(path: Path | None = None) -> sqlite3.Connection:
    db = path or DB_PATH
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(db), timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    for stmt in _DDL.strip().split(";"):
        if stmt.strip():
            c.execute(stmt)
    c.commit()
    return c


def insert_candidate_set(row: dict[str, Any], *, path: Path | None = None) -> str:
    c = _conn(path)
    try:
        c.execute(
            """INSERT OR IGNORE INTO candidate_sets (
                candidate_set_id, scan_cycle_id, session_date, recorded_at,
                n_candidates, family_composition, source, feature_set_version, protocol_hash
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                row["candidate_set_id"], row["scan_cycle_id"], row["session_date"],
                row["recorded_at"], int(row["n_candidates"]),
                json.dumps(row.get("family_composition") or {}, sort_keys=True),
                row["source"], row["feature_set_version"], row.get("protocol_hash"),
            ),
        )
        c.commit()
        return row["candidate_set_id"]
    finally:
        c.close()


def insert_observation(row: dict[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """First write wins. Later writes do not change the feature snapshot."""
    eid = row["event_id"]
    c = _conn(path)
    try:
        existing = c.execute("SELECT event_id FROM observations WHERE event_id=?", (eid,)).fetchone()
        if existing:
            return {"status": "exists", "event_id": eid, "wrote": False}
        if str(row["session_date"]) < FORWARD_START_DATE and row.get("source") == "live_scan":
            return {"status": "pre_freeze_refused", "event_id": eid, "wrote": False}
        c.execute(
            """INSERT INTO observations (
                event_id, candidate_set_id, scan_cycle_id, symbol, exchange,
                session_date, recorded_at, source, families, primary_family,
                feature_set_version, production_rank_version, shadow_rank_version,
                protocol_hash, production_score, production_rank, production_verdict,
                production_signals, production_decision, would_trade, ready_status,
                entry, stop, target, chase_risk, n_structure_passed, structure_pass,
                rs_percentile, rs_score, rs_rank, trend_rank, combined_shadow_rank,
                r3_score, regime_label, sector, sector_map_version, feature_snapshot,
                data_quality, eligible_primary
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                eid, row["candidate_set_id"], row["scan_cycle_id"], row["symbol"],
                row.get("exchange") or "NSE", row["session_date"], row["recorded_at"],
                row["source"], json.dumps(row.get("families") or []),
                row.get("primary_family"), row["feature_set_version"],
                row["production_rank_version"], row["shadow_rank_version"],
                row.get("protocol_hash"), row.get("production_score"),
                row.get("production_rank"), row.get("production_verdict"),
                json.dumps(row.get("production_signals") or []),
                row.get("production_decision"), int(bool(row.get("would_trade"))),
                row.get("ready_status"), row.get("entry"), row.get("stop"),
                row.get("target"), int(bool(row.get("chase_risk"))),
                row.get("n_structure_passed"),
                None if row.get("structure_pass") is None else int(bool(row.get("structure_pass"))),
                row.get("rs_percentile"), row.get("rs_score"), row.get("rs_rank"),
                row.get("trend_rank"), row.get("combined_shadow_rank"),
                row.get("r3_score"), row.get("regime_label"), row.get("sector"),
                row.get("sector_map_version"), json.dumps(row.get("feature_snapshot") or {}),
                row.get("data_quality"),
                int(eligible_primary(row["session_date"], row["source"],
                                     row["recorded_at"], row["feature_set_version"])),
            ),
        )
        c.commit()
        return {"status": "inserted", "event_id": eid, "wrote": True}
    finally:
        c.close()


def attach_outcome(event_id_value: str, outcome: dict[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """Write/replace outcome only. Never touches observations.feature_snapshot."""
    c = _conn(path)
    try:
        obs = c.execute("SELECT event_id FROM observations WHERE event_id=?", (event_id_value,)).fetchone()
        if not obs:
            return {"status": "missing_observation", "wrote": False}
        c.execute(
            """INSERT INTO outcomes (
                event_id, resolved_at, next_open, ret_1d, ret_5d, ret_10d, ret_20d,
                mae, mfe, hit_1r, hit_2r, production_traded, production_outcome, unresolved_reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(event_id) DO UPDATE SET
                resolved_at=excluded.resolved_at,
                next_open=excluded.next_open,
                ret_1d=excluded.ret_1d,
                ret_5d=excluded.ret_5d,
                ret_10d=excluded.ret_10d,
                ret_20d=excluded.ret_20d,
                mae=excluded.mae,
                mfe=excluded.mfe,
                hit_1r=excluded.hit_1r,
                hit_2r=excluded.hit_2r,
                production_traded=excluded.production_traded,
                production_outcome=excluded.production_outcome,
                unresolved_reason=excluded.unresolved_reason
            """,
            (
                event_id_value, outcome.get("resolved_at"), outcome.get("next_open"),
                outcome.get("ret_1d"), outcome.get("ret_5d"), outcome.get("ret_10d"),
                outcome.get("ret_20d"), outcome.get("mae"), outcome.get("mfe"),
                outcome.get("hit_1r"), outcome.get("hit_2r"),
                outcome.get("production_traded"), outcome.get("production_outcome"),
                outcome.get("unresolved_reason"),
            ),
        )
        c.commit()
        return {"status": "outcome_written", "wrote": True, "event_id": event_id_value}
    finally:
        c.close()


def get_observation(event_id_value: str, *, path: Path | None = None) -> dict[str, Any] | None:
    c = _conn(path)
    try:
        row = c.execute("SELECT * FROM observations WHERE event_id=?", (event_id_value,)).fetchone()
        return dict(row) if row else None
    finally:
        c.close()


def feature_snapshot(event_id_value: str, *, path: Path | None = None) -> dict[str, Any] | None:
    row = get_observation(event_id_value, path=path)
    if not row:
        return None
    return json.loads(row["feature_snapshot"])


def list_primary_observations(*, path: Path | None = None) -> list[dict[str, Any]]:
    c = _conn(path)
    try:
        rows = c.execute(
            """SELECT o.*, oc.ret_1d, oc.ret_5d, oc.ret_10d, oc.ret_20d, oc.mae, oc.mfe,
                      oc.hit_1r, oc.hit_2r, oc.unresolved_reason, oc.resolved_at
               FROM observations o
               LEFT JOIN outcomes oc ON oc.event_id = o.event_id
               WHERE o.eligible_primary=1 AND o.feature_set_version=?""",
            (FEATURE_SET_VERSION,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        c.close()


def counts(*, path: Path | None = None) -> dict[str, int]:
    c = _conn(path)
    try:
        n_obs = c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        n_pri = c.execute("SELECT COUNT(*) FROM observations WHERE eligible_primary=1").fetchone()[0]
        n_res = c.execute(
            """SELECT COUNT(*) FROM observations o
               JOIN outcomes oc ON oc.event_id=o.event_id
               WHERE o.eligible_primary=1 AND oc.ret_5d IS NOT NULL"""
        ).fetchone()[0]
        n_sets = c.execute("SELECT COUNT(*) FROM candidate_sets").fetchone()[0]
        return {
            "observations": int(n_obs),
            "primary": int(n_pri),
            "resolved_primary_5d": int(n_res),
            "candidate_sets": int(n_sets),
        }
    finally:
        c.close()


def export_jsonl(*, path: Path | None = None, dest: Path | None = None) -> Path:
    dest = dest or (LEDGER_DIR / "observations.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    c = _conn(path)
    try:
        rows = c.execute("SELECT * FROM observations").fetchall()
        with dest.open("w") as f:
            for r in rows:
                f.write(json.dumps(dict(r), default=str) + "\n")
    finally:
        c.close()
    return dest
