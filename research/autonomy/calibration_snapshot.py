"""Immutable, provenance-keyed calibration snapshots.

Calibration is evidence, not execution authority.  Every snapshot is bound to
its data/feature/thesis/model/signal-registry identity and evidence lane so
historical replay can never be silently mixed with forward paper evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1
DEFAULT_PATH = logs_path("research/calibration_snapshots.jsonl")
ALLOWED_ORIGINS = {"HISTORICAL_REPLAY", "FORWARD_PAPER"}
IDENTITY_KEYS = (
    "data_version",
    "feature_version",
    "thesis_hash",
    "model_version",
    "signal_registry_version",
)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finite(value: Any) -> float:
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError("calibration metric must be finite")
    return out


@dataclass(frozen=True)
class CalibrationSnapshot:
    snapshot_id: str
    evidence_origin: str
    data_version: str
    feature_version: str
    thesis_hash: str
    model_version: str
    signal_registry_version: str
    sample_count: int
    brier_score: float
    expected_calibration_error: float
    bins: tuple[Mapping[str, Any], ...]
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["schema_version"] = SCHEMA_VERSION
        row["bins"] = [dict(x) for x in self.bins]
        return row


def build_snapshot(payload: Mapping[str, Any]) -> CalibrationSnapshot:
    origin = str(payload.get("evidence_origin") or "").upper()
    if origin not in ALLOWED_ORIGINS:
        raise ValueError("evidence_origin must be HISTORICAL_REPLAY or FORWARD_PAPER")
    identity = {key: str(payload.get(key) or "").strip() for key in IDENTITY_KEYS}
    missing = [key for key, value in identity.items() if not value]
    if missing:
        raise ValueError("missing immutable calibration identity: " + ", ".join(missing))
    n = int(payload.get("sample_count") or 0)
    if n <= 0:
        raise ValueError("sample_count must be positive")
    brier = _finite(payload.get("brier_score"))
    ece = _finite(payload.get("expected_calibration_error"))
    if brier < 0 or brier > 1 or ece < 0 or ece > 1:
        raise ValueError("calibration scores must be within [0, 1]")
    bins = tuple(dict(x) for x in (payload.get("bins") or ()))
    key = {"evidence_origin": origin, **identity, "sample_count": n,
           "brier_score": brier, "expected_calibration_error": ece, "bins": bins}
    snapshot_id = "cal_" + _stable_hash(key)[:24]
    created_at = str(payload.get("created_at") or datetime.now(timezone.utc).isoformat())
    return CalibrationSnapshot(snapshot_id, origin, *identity.values(), n, brier, ece, bins, created_at)


def append_snapshot(payload: Mapping[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    """Append one immutable snapshot; identical retries are idempotent.

    A snapshot id collision with different content is rejected fail-closed.
    Historical and forward rows remain separate because evidence_origin is part
    of the immutable snapshot identity.
    """
    target = Path(path) if path is not None else DEFAULT_PATH
    snapshot = build_snapshot(payload).as_dict()
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("snapshot_id") or "")
            if sid:
                existing[sid] = row
    prior = existing.get(snapshot["snapshot_id"])
    if prior is not None:
        # created_at is observation metadata, not identity; preserve first write.
        comparable = dict(snapshot)
        comparable["created_at"] = prior.get("created_at")
        if prior != comparable:
            raise ValueError("immutable calibration snapshot collision")
        return prior
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return snapshot


def load_snapshots(*, path: str | Path | None = None,
                   evidence_origin: str | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else DEFAULT_PATH
    origin = str(evidence_origin or "").upper()
    if origin and origin not in ALLOWED_ORIGINS:
        raise ValueError("invalid evidence_origin")
    if not target.exists():
        return []
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if not origin or str(row.get("evidence_origin") or "").upper() == origin]
