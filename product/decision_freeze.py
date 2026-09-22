"""Immutable decision-time fingerprints for forward and historical judgments.

A decision freeze is the audit boundary between what QuantTerm knew at T and what
happened later. Outcomes may be attached elsewhere, but the decision-time identity
must never be rewritten or silently reused for materially different evidence.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from product.pit_versions import current_versions
from core.runtime_paths import logs_dir

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = logs_dir() / "product" / "decision_freeze.db"
FINGERPRINT_SCHEMA_VERSION = 2


class DecisionIdentityCollision(RuntimeError):
    """The same decision_id was presented with different decision-time evidence."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def freeze_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_DECISION_FREEZE")
    if override:
        return Path(override)
    return DB_PATH


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    from product.sqlite_runtime import connect

    con = connect(freeze_path(path))
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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def decision_identity_material(
    rec: Mapping[str, Any],
    *,
    versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical material that must be identical for one immutable decision ID."""
    row = dict(rec or {})
    version_map = dict(versions or row.get("versions") or current_versions().as_dict())
    provenance = _mapping(row.get("provenance"))
    references = _mapping(row.get("references"))
    policy = _mapping(row.get("policy"))
    thesis = _mapping(row.get("thesis"))
    portfolio = row.get("portfolio") or references.get("portfolio") or {}
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "symbol": str(row.get("symbol") or "").upper(),
        "as_of": str(
            row.get("as_of")
            or row.get("market_as_of")
            or row.get("pit_as_of")
            or row.get("decision_as_of")
            or ""
        )[:10],
        "decision": row.get("decision") or row.get("selection_result"),
        "reason_code": row.get("reason_code"),
        "candidate_state": row.get("candidate_state"),
        "entry_state": row.get("entry_state"),
        "execution_state": row.get("execution_state"),
        "entry": row.get("entry"),
        "stop": row.get("stop"),
        "target": row.get("target"),
        "setup": row.get("setup_label") or row.get("setup") or row.get("primary_thesis"),
        "sector": row.get("sector") or references.get("sector"),
        "regime": row.get("regime") or references.get("regime"),
        "families": row.get("evidence_family_votes") or row.get("families"),
        "method_votes": row.get("method_votes"),
        "method_panel": row.get("methods") or row.get("method_panel"),
        "empirical": row.get("empirical"),
        "dd_status": row.get("dd_status") or row.get("dd_verdict"),
        "entry_quality": row.get("entry_quality"),
        "chase_risk": row.get("chase_risk"),
        "extension_pct": row.get("extension_pct"),
        "family_confirms": row.get("family_confirms"),
        "missing_evidence": row.get("missing_evidence"),
        "vetoes": row.get("vetoes"),
        "risk_audit": row.get("risk_audit"),
        "portfolio_context": portfolio,
        "selection_score": row.get("selection_score"),
        "policy_effect": row.get("policy_effect") or policy.get("final_effect"),
        "market_cutoff": row.get("evidence_cutoff") or row.get("as_of"),
        "thesis_hash": (
            row.get("thesis_hash")
            or provenance.get("thesis_hash")
            or thesis.get("thesis_hash")
            or ""
        ),
        "calibration_snapshot_id": (
            row.get("calibration_snapshot_id")
            or provenance.get("calibration_snapshot_id")
            or ""
        ),
        "data_snapshot_id": (
            row.get("data_snapshot_id")
            or row.get("source_snapshot_id")
            or provenance.get("data_snapshot_id")
            or ""
        ),
        "source_scan_id": (
            row.get("source_scan_id")
            or row.get("scan_run_id")
            or row.get("scan_scanned_at")
            or provenance.get("source_scan_id")
            or ""
        ),
        "evidence_class": (
            row.get("evidence_class")
            or provenance.get("evidence_class")
            or ""
        ),
        "versions": version_map,
    }


def evidence_fingerprint(rec: Mapping[str, Any]) -> str:
    material = decision_identity_material(rec)
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(blob).hexdigest()[:24]


def freeze(rec: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """Insert one immutable decision or fail on decision-ID identity collision."""
    row = dict(rec or {})
    versions = dict(row.get("versions") or current_versions().as_dict())
    material = decision_identity_material(row, versions=versions)
    fp = sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:24]
    freeze_id = str(
        row.get("decision_id")
        or f"{material['symbol']}:{material['as_of']}:{fp}"
    )
    payload = {
        **material,
        "fingerprint": fp,
        "freeze_id": freeze_id,
        "frozen_at": _now(),
        "immutable": True,
        "rewritten_after_outcome": False,
    }

    con = _connect(path)
    existing = con.execute(
        "SELECT fingerprint, payload_json FROM freezes WHERE freeze_id=?",
        (freeze_id,),
    ).fetchone()
    if existing:
        saved = json.loads(existing["payload_json"])
        saved_fp = str(existing["fingerprint"] or saved.get("fingerprint") or "")
        con.close()
        if saved_fp != fp:
            raise DecisionIdentityCollision(
                f"decision freeze collision for {freeze_id}: {saved_fp} != {fp}"
            )
        return saved

    con.execute(
        "INSERT INTO freezes (freeze_id, fingerprint, symbol, as_of, decision, frozen_at, payload_json) "
        "VALUES (?,?,?,?,?,?,?)",
        (
            freeze_id,
            fp,
            material["symbol"],
            material["as_of"],
            material["decision"],
            payload["frozen_at"],
            json.dumps(payload, default=str),
        ),
    )
    con.commit()
    con.close()
    return payload


def get_freeze(freeze_id: str, *, path: Path | None = None) -> dict[str, Any] | None:
    con = _connect(path)
    row = con.execute(
        "SELECT payload_json FROM freezes WHERE freeze_id=?",
        (freeze_id,),
    ).fetchone()
    con.close()
    return json.loads(row["payload_json"]) if row else None
