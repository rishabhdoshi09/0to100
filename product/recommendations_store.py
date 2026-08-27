"""Persisted Recommendations desk from the last whole-market scan.

GET reads this file. Scan Now writes it. Page-open does not rescan.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

DEFAULT_RECO_PATH = Path("logs/product/latest_recommendations.json")


def save_recommendations(payload: Mapping[str, Any], path: str | Path = DEFAULT_RECO_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_recommendations(path: str | Path = DEFAULT_RECO_PATH) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0) or 0) < 4:
            return None
        if not isinstance(payload.get("categories"), list):
            return None
        return payload
    except Exception:
        return None


def reco_matches_scan(
    saved: Mapping[str, Any] | None,
    *,
    scan_scanned_at: str,
    long_term_scanned_at: str,
) -> bool:
    if not saved:
        return False
    return (
        str(saved.get("scan_scanned_at") or "") == str(scan_scanned_at or "")
        and str(saved.get("long_term_scanned_at") or "") == str(long_term_scanned_at or "")
    )
