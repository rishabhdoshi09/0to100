"""Cheap persisted operations status for the API GET path.

The worker writes this on the heartbeat. /api/operations must not open the
operations SQLite writer (30s busy timeout) on every request.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "market_ops" / "operations_status.json"
CURRENT = "CURRENT"
STALE = "STALE"
UNAVAILABLE = "UNAVAILABLE"
SNAPSHOT_CURRENT_S = 20.0
SNAPSHOT_STALE_S = 120.0


def snapshot_path() -> Path:
    raw = str(os.environ.get("QT_OPERATIONS_SNAPSHOT") or "").strip()
    return Path(raw) if raw else DEFAULT_PATH


FAT_BLOB_KEYS = {"payload", "articles", "downloads", "texts", "raw", "rows"}
_KEEP_NESTED = {"summary", "history", "telegram", "long_term_overlay"}


def slim_result_value(key: str, value: Any) -> Any:
    """Keep status/summary fields; drop embedded scan/filing blobs from GET payloads."""
    if key in FAT_BLOB_KEYS and isinstance(value, (dict, list)):
        if isinstance(value, list):
            return len(value)
        return {
            "n_keys": len(value),
            "keys": sorted(str(item) for item in list(value.keys())[:16]),
        }
    if key == "records" and isinstance(value, list):
        return len(value)
    if isinstance(value, list) and len(value) > 24:
        return {"n": len(value)}
    if isinstance(value, dict) and key not in _KEEP_NESTED:
        encoded = json.dumps(value, default=str)
        if len(encoded) > 4000:
            return {
                "n_keys": len(value),
                "keys": sorted(str(item) for item in list(value.keys())[:16]),
            }
    return value


def slim_operation_record(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    out = dict(row)
    payload = out.get("payload")
    if isinstance(payload, dict):
        out["payload"] = {str(key): slim_result_value(str(key), item) for key, item in payload.items()}
    result = out.get("result")
    if isinstance(result, dict):
        out["result"] = {str(key): slim_result_value(str(key), item) for key, item in result.items()}
    return out


def slim_operations_status(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    for key in ("active", "recent"):
        rows = out.get(key)
        if isinstance(rows, list):
            out[key] = [slim_operation_record(item) if isinstance(item, dict) else item for item in rows]
    latest = out.get("latest")
    if isinstance(latest, dict):
        out["latest"] = {
            str(key): slim_operation_record(item) if isinstance(item, dict) else item
            for key, item in latest.items()
        }
    return out


def persist_operations_snapshot(payload: dict[str, Any], *, path: Path | None = None) -> Path:
    dest = path or snapshot_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{time.time_ns()}.tmp")
    body = slim_operations_status(dict(payload))
    body.setdefault("generated_at", time.time())
    body.setdefault("freshness", CURRENT)
    try:
        tmp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def load_operations_snapshot(
    *,
    path: Path | None = None,
    current_s: float = SNAPSHOT_CURRENT_S,
    stale_s: float = SNAPSHOT_STALE_S,
) -> tuple[dict[str, Any] | None, str]:
    dest = path or snapshot_path()
    try:
        payload = json.loads(dest.read_text(encoding="utf-8"))
    except Exception:
        return None, UNAVAILABLE
    if not isinstance(payload, dict):
        return None, UNAVAILABLE
    try:
        generated = float(payload.get("generated_at") or dest.stat().st_mtime)
    except (TypeError, ValueError, OSError):
        generated = 0.0
    age = time.time() - generated if generated > 0 else 10**9
    if age <= float(current_s):
        freshness = CURRENT
    elif age <= float(stale_s):
        freshness = STALE
    else:
        freshness = STALE
    payload = slim_operations_status(dict(payload))
    payload["freshness"] = freshness
    payload["snapshot_age_s"] = round(age, 3)
    payload["available"] = True
    return payload, freshness


def unavailable_operations_payload(*, error: str = "") -> dict[str, Any]:
    return {
        "available": False,
        "freshness": UNAVAILABLE,
        "generated_at": time.time(),
        "running": False,
        "worker_pid": None,
        "heartbeat": "",
        "active_lanes": {},
        "counts": {},
        "active": [],
        "recent": [],
        "latest": {},
        "error": error or "Operations status store is unavailable",
    }
