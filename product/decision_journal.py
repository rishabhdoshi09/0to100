"""Canonical judgment records. References snapshots; does not copy research blobs."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "logs" / "product" / "decisions.db"
JSONL_PATH = ROOT / "logs" / "product" / "decisions.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    from product.sqlite_runtime import connect

    target = path or DB_PATH
    con = connect(target)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            decision_id TEXT PRIMARY KEY,
            candidate_id TEXT,
            opportunity_id TEXT,
            scan_run_id TEXT,
            recommendation_id TEXT,
            symbol TEXT NOT NULL,
            decision_time TEXT,
            market_as_of TEXT,
            evidence_cutoff TEXT,
            candidate_state TEXT,
            decision TEXT,
            entry_state TEXT,
            execution_state TEXT,
            reason_code TEXT,
            reason TEXT,
            tier TEXT,
            framework_id TEXT,
            evidence_coverage_pct REAL,
            payload_json TEXT
        )
        """
    )
    return con


def persist(record: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    row = dict(record)
    did = str(row.get("decision_id") or "")
    if not did:
        return row
    con = _connect(path)
    con.execute(
        """INSERT OR REPLACE INTO decisions (
            decision_id, candidate_id, opportunity_id, scan_run_id, recommendation_id,
            symbol, decision_time, market_as_of, evidence_cutoff, candidate_state,
            decision, entry_state, execution_state, reason_code, reason, tier,
            framework_id, evidence_coverage_pct, payload_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            did, row.get("candidate_id"), row.get("opportunity_id"), row.get("scan_run_id"),
            row.get("recommendation_id"), row.get("symbol"), row.get("decision_time") or _now(),
            row.get("market_as_of"), row.get("evidence_cutoff"), row.get("candidate_state"),
            row.get("decision"), row.get("entry_state"), row.get("execution_state"),
            row.get("reason_code"), row.get("reason"), row.get("tier"),
            row.get("framework_id"), row.get("evidence_coverage_pct"),
            json.dumps({
                "vetoes": row.get("vetoes") or [],
                "methods_buy": row.get("methods_buy") or [],
                "methods_wait": row.get("methods_wait") or [],
                "methods_avoid": row.get("methods_avoid") or [],
                "disagreement": row.get("disagreement"),
                "wait_trigger": row.get("wait_trigger") or {},
                "positives": row.get("positives") or [],
                "missing_critical": row.get("missing_critical") or [],
                "families": row.get("families") or {},
                "method_votes": row.get("method_votes") or [],
                "evidence_family_votes": row.get("evidence_family_votes") or {},
                "dependency_notes": row.get("dependency_notes") or [],
                "effective_confirmation_count": row.get("effective_confirmation_count"),
                "family_gate_ok": row.get("family_gate_ok"),
                "risk_audit": row.get("risk_audit") or {},
                "portfolio": row.get("portfolio") or {},
                "shadow_status": row.get("shadow_status"),
                "references": row.get("references") or {},
                "entry": row.get("entry"),
                "stop": row.get("stop"),
                "target": row.get("target"),
            }, default=str),
        ),
    )
    con.commit()
    con.close()
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def get(decision_id: str, path: Path | None = None) -> dict[str, Any] | None:
    con = _connect(path)
    row = con.execute("SELECT * FROM decisions WHERE decision_id=?", (decision_id,)).fetchone()
    con.close()
    return dict(row) if row else None


def list_for_session(session: str, *, limit: int = 200, path: Path | None = None) -> list[dict[str, Any]]:
    con = _connect(path)
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM decisions WHERE market_as_of=? ORDER BY decision_time DESC LIMIT ?",
        (str(session)[:10], int(limit)),
    )]
    con.close()
    return rows


def counts(session: str = "", path: Path | None = None) -> dict[str, int]:
    con = _connect(path)
    if session:
        rows = con.execute(
            "SELECT decision, count(*) c FROM decisions WHERE market_as_of=? GROUP BY 1",
            (str(session)[:10],),
        )
    else:
        rows = con.execute("SELECT decision, count(*) c FROM decisions GROUP BY 1")
    out = {str(r["decision"]): int(r["c"]) for r in rows}
    con.close()
    return out
