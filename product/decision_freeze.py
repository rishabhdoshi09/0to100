"""Immutable freeze of a meaningful forward/replay decision.

Original decisions are never rewritten after outcomes arrive.
Settlement attaches beside the freeze; it does not mutate it.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from product.pit_versions import current_versions

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "logs" / "product" / "decision_freeze.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    from product.sqlite_runtime import connect

    con = connect(path or DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS freezes (
            freeze_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            symbol TEXT,
            as_of TEXT,
            decision TEXT,
            frozen_at TEXT,
            payload_json TEXT NOT NULL
        )
        """
    )
    return con


def evidence_fingerprint(rec: Mapping[str, Any]) -> str:
    blob = json.dumps({
        "symbol": rec.get("symbol"),
        "as_of": rec.get("as_of") or rec.get("market_as_of") or rec.get("pit_as_of"),
        "decision": rec.get("decision"),
        "families": rec.get("evidence_family_votes") or rec.get("families"),
        "method_votes": rec.get("method_votes"),
        "vetoes": rec.get("vetoes"),
        "entry": rec.get("entry"),
        "stop": rec.get("stop"),
        "target": rec.get("target"),
        "reason_code": rec.get("reason_code"),
        "versions": rec.get("versions") or current_versions().as_dict(),
    }, sort_keys=True, default=str).encode()
    return sha256(blob).hexdigest()[:24]


def freeze(rec: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """INSERT OR IGNORE. Seeing the outcome later cannot rewrite this row."""
    versions = dict(rec.get("versions") or current_versions().as_dict())
    payload = {
        "symbol": rec.get("symbol"),
        "as_of": str(rec.get("as_of") or rec.get("market_as_of") or rec.get("pit_as_of") or "")[:10],
        "decision": rec.get("decision"),
        "candidate_state": rec.get("candidate_state"),
        "entry_state": rec.get("entry_state"),
        "execution_state": rec.get("execution_state"),
        "reason_code": rec.get("reason_code"),
        "entry": rec.get("entry"),
        "stop": rec.get("stop"),
        "target": rec.get("target"),
        "regime": rec.get("regime") or (rec.get("references") or {}).get("regime"),
        "portfolio_context": rec.get("portfolio") or (rec.get("references") or {}).get("portfolio"),
        "family_votes": rec.get("evidence_family_votes") or rec.get("families"),
        "method_votes": rec.get("method_votes"),
        "vetoes": rec.get("vetoes"),
        "market_cutoff": rec.get("evidence_cutoff") or rec.get("as_of"),
        "policy_version": versions.get("decision_engine_version"),
        "committee_version": versions.get("committee_version"),
        "framework_version": versions.get("framework_version"),
        "risk_version": versions.get("risk_policy_version"),
        "versions": versions,
        "immutable": True,
        "rewritten_after_outcome": False,
    }
    fp = evidence_fingerprint({**rec, "versions": versions})
    freeze_id = str(rec.get("decision_id") or f"{payload['symbol']}:{payload['as_of']}:{fp}")
    payload["fingerprint"] = fp
    payload["freeze_id"] = freeze_id
    payload["frozen_at"] = _now()
    con = _connect(path)
    existing = con.execute("SELECT payload_json FROM freezes WHERE freeze_id=?", (freeze_id,)).fetchone()
    if existing:
        con.close()
        return json.loads(existing["payload_json"])
    con.execute(
        "INSERT INTO freezes (freeze_id, fingerprint, symbol, as_of, decision, frozen_at, payload_json) VALUES (?,?,?,?,?,?,?)",
        (freeze_id, fp, payload["symbol"], payload["as_of"], payload["decision"], payload["frozen_at"], json.dumps(payload, default=str)),
    )
    con.commit()
    con.close()
    return payload


def get_freeze(freeze_id: str, *, path: Path | None = None) -> dict[str, Any] | None:
    con = _connect(path)
    row = con.execute("SELECT payload_json FROM freezes WHERE freeze_id=?", (freeze_id,)).fetchone()
    con.close()
    return json.loads(row["payload_json"]) if row else None
