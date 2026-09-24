"""Immutable audit journal for information-driven historical replay acquisition.

Records why a historical session was selected and, later, what evidence it
actually produced.  This journal is research provenance only: records are
explicitly HISTORICAL_REPLAY and cannot confer forward or live-money authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1
EVIDENCE_ORIGIN = "HISTORICAL_REPLAY"
DEFAULT_PATH = logs_path("research", "acquisition_journal.jsonl")

_REQUIRED_IDENTITY = (
    "request_id",
    "acquisition_fingerprint",
    "session_date",
    "strategy_id",
    "thesis_hash",
    "universe_snapshot_id",
    "data_version",
    "feature_version",
    "model_version",
    "signal_registry_version",
    "decision_fingerprint",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    material = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _validated_acquisition(acquisition: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(acquisition or {})
    if str(row.get("evidence_origin") or "").upper() != EVIDENCE_ORIGIN:
        raise ValueError("acquisition journal accepts HISTORICAL_REPLAY evidence only")
    missing = [key for key in _REQUIRED_IDENTITY if not str(row.get(key) or "").strip()]
    if missing:
        raise ValueError("missing immutable acquisition identity: " + ",".join(missing))
    return row


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def record_selection(
    acquisition: Mapping[str, Any], *, path: str | Path | None = None
) -> dict[str, Any]:
    """Persist one selected acquisition exactly once by immutable fingerprint."""
    row = _validated_acquisition(acquisition)
    target = Path(path) if path is not None else DEFAULT_PATH
    fingerprint = str(row["acquisition_fingerprint"])
    for existing in _read(target):
        if existing.get("event") == "SELECTED" and existing.get("acquisition_fingerprint") == fingerprint:
            return existing

    payload = {
        "schema_version": SCHEMA_VERSION,
        "event": "SELECTED",
        "recorded_at": _now(),
        "evidence_origin": EVIDENCE_ORIGIN,
        **{key: row[key] for key in _REQUIRED_IDENTITY},
        "score": float(row.get("score") or 0.0),
        "rationale": list(row.get("rationale") or []),
    }
    payload["record_fingerprint"] = "acqsel_" + _stable_hash({
        key: payload[key] for key in (
            "event", "evidence_origin", "request_id", "acquisition_fingerprint",
            "session_date", "strategy_id", "thesis_hash", "universe_snapshot_id",
            "data_version", "feature_version", "model_version",
            "signal_registry_version", "decision_fingerprint", "score", "rationale",
        )
    })[:20]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return payload


def record_realized_gain(
    acquisition: Mapping[str, Any],
    *,
    eligible_samples: int,
    metrics_realized: list[str] | tuple[str, ...] = (),
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Append the observed evidence yield without rewriting selection history."""
    row = _validated_acquisition(acquisition)
    target = Path(path) if path is not None else DEFAULT_PATH
    fingerprint = str(row["acquisition_fingerprint"])
    metrics = sorted({str(x) for x in metrics_realized if str(x)})
    stable = {
        "event": "REALIZED",
        "evidence_origin": EVIDENCE_ORIGIN,
        "request_id": str(row["request_id"]),
        "acquisition_fingerprint": fingerprint,
        "session_date": str(row["session_date"])[:10],
        "eligible_samples": max(0, int(eligible_samples or 0)),
        "metrics_realized": metrics,
    }
    realized_id = "acqreal_" + _stable_hash(stable)[:20]
    for existing in _read(target):
        if existing.get("record_fingerprint") == realized_id:
            return existing
    payload = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": _now(),
        **stable,
        "record_fingerprint": realized_id,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return payload


def records_for_request(request_id: str, *, path: str | Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else DEFAULT_PATH
    wanted = str(request_id or "")
    return [row for row in _read(target) if str(row.get("request_id") or "") == wanted]
