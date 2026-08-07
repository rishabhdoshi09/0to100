"""Atomic persistence for the latest current long-term shortlist.

The payload is a present-day decision aid, not point-in-time research evidence.
Historical research continues to use the audited snapshot/evidence stores.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

DEFAULT_LONG_TERM_PATH = Path("logs/product/latest_long_term_scan.json")


def save_long_term_scan(payload: Mapping[str, Any], path: str | Path = DEFAULT_LONG_TERM_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_long_term_scan(path: str | Path = DEFAULT_LONG_TERM_PATH) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1 or not isinstance(payload.get("records"), list):
            return None
        return payload
    except Exception:
        return None
