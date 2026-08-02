"""Validated user import path for fundamentals and price files."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_platform.contracts import utc_now_iso


def inspect_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": "file not found", "path": str(path)}
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            keys = list(payload.keys()) if isinstance(payload, dict) else []
            return {"ok": True, "type": "json", "keys": keys[:20], "path": str(path)}
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": str(exc), "path": str(path)}
    if suffix == ".csv":
        try:
            with path.open(encoding="utf-8", newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, [])
            return {"ok": True, "type": "csv", "columns": header[:30], "path": str(path)}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "path": str(path)}
    return {"ok": False, "error": "unsupported extension", "path": str(path)}


def import_fundamentals_json(path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Import per-symbol fundamentals JSON into FundamentalsCache when higher priority allows."""
    inspection = inspect_file(path)
    if not inspection.get("ok"):
        return inspection
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    if not isinstance(payload, Mapping):
        return {"ok": False, "error": "expected object mapping symbol -> fundamentals"}
    from fundamentals.cache import FundamentalsCache
    cache = FundamentalsCache()
    imported = 0
    rejected: list[str] = []
    for symbol, row in payload.items():
        sym = str(symbol).strip().upper()
        if not sym or not isinstance(row, Mapping):
            rejected.append(str(symbol))
            continue
        if not overwrite and cache.get(sym):
            rejected.append(f"{sym}:existing")
            continue
        cache.set(sym, dict(row))
        imported += 1
    return {
        "ok": True,
        "imported": imported,
        "rejected": rejected[:20],
        "source": str(path),
        "retrieved_at": utc_now_iso(),
        "provenance": "user_import",
    }
