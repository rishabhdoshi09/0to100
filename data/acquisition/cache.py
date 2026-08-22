"""Immutable raw-source cache for ingest. Checksum + resume."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2] / "logs" / "acquisition"
RAW = ROOT / "raw"
MANIFESTS = ROOT / "manifests"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_path(*parts: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_raw(rel: str, data: bytes, *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    path = raw_path(rel)
    if path.exists() and path.stat().st_size == len(data):
        digest = sha256_bytes(path.read_bytes())
        if digest == sha256_bytes(data):
            return {"path": str(path), "sha256": digest, "bytes": len(data), "cached": True}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    rec = {
        "path": str(path),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "cached": False,
        "written_at": _now(),
    }
    if meta:
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        sidecar.write_text(json.dumps({**meta, **rec}, indent=2, default=str), encoding="utf-8")
    return rec


def write_manifest(name: str, payload: dict[str, Any]) -> Path:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.setdefault("written_at", _now())
    p = MANIFESTS / f"{name}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)
    return p
