"""Persisted startup trade-discovery projection.

The whole-market scan is the expensive trigger. Once it finishes, QuantTerm
projects the canonical decision board exactly once and stores it here. The
Decision Simulation gate then remains a cheap read path instead of rebuilding
recommendations/rankings inside an HTTP request.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1


def _path() -> Path:
    return logs_path("product/startup_trade_discovery.json")


def save(
    board: Mapping[str, Any],
    *,
    scan_scanned_at: str,
    long_term_scanned_at: str,
    thesis_hash: str,
) -> Path:
    target = _path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scan_scanned_at": str(scan_scanned_at or ""),
        "long_term_scanned_at": str(long_term_scanned_at or ""),
        "thesis_hash": str(thesis_hash or ""),
        "board": dict(board),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load(
    *,
    scan_scanned_at: str,
    long_term_scanned_at: str,
    thesis_hash: str,
) -> dict[str, Any] | None:
    target = _path()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("schema_version") or 0) != SCHEMA_VERSION:
        return None
    if str(payload.get("scan_scanned_at") or "") != str(scan_scanned_at or ""):
        return None
    if str(payload.get("long_term_scanned_at") or "") != str(long_term_scanned_at or ""):
        return None
    if str(payload.get("thesis_hash") or "") != str(thesis_hash or ""):
        return None
    board = payload.get("board")
    return dict(board) if isinstance(board, dict) else None
