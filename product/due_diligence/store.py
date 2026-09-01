"""Durable last-good Due Diligence report store.

Due Diligence is expensive enough that a transient API/process failure must not
make the user's research disappear. Fresh GETs still rebuild from the canonical
files-on-disk engine; this module only preserves the last successfully built
report so the API can fail soft instead of returning an empty desk.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
STORE_ROOT = ROOT / "logs" / "product" / "due_diligence"
STORE_SCHEMA_VERSION = 1
_SYMBOL = re.compile(r"^[A-Z0-9&._-]{1,32}$")


def _clean_symbol(symbol: str) -> str:
    clean = str(symbol or "").strip().upper()
    if not _SYMBOL.fullmatch(clean):
        raise ValueError("Invalid symbol")
    return clean


def report_path(symbol: str, *, root: Path | None = None) -> Path:
    base = root or STORE_ROOT
    return base / f"{_clean_symbol(symbol)}.json"


def save_report(
    report: Mapping[str, Any],
    *,
    path: Path | None = None,
    saved_at: str | None = None,
) -> Path:
    symbol = _clean_symbol(str(report.get("symbol") or ""))
    target = path or report_path(symbol)
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = saved_at or datetime.now(timezone.utc).isoformat()
    payload = {
        "store_schema_version": STORE_SCHEMA_VERSION,
        "symbol": symbol,
        "saved_at": stamp,
        "report": dict(report),
    }
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def load_report(symbol: str, *, path: Path | None = None) -> dict[str, Any] | None:
    clean = _clean_symbol(symbol)
    target = path or report_path(clean)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("symbol") or "").upper() != clean:
        return None
    report = payload.get("report")
    if not isinstance(report, dict):
        return None
    if str(report.get("symbol") or "").upper() != clean:
        return None
    out = dict(report)
    out["snapshot_saved_at"] = payload.get("saved_at")
    return out


def fresh_delivery(report: Mapping[str, Any], *, saved_at: str | None = None) -> dict[str, Any]:
    out = dict(report)
    out["delivery_state"] = "FRESH"
    out["snapshot_saved_at"] = saved_at or datetime.now(timezone.utc).isoformat()
    out["backend_error"] = ""
    return out


def stale_delivery(report: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    out = dict(report)
    out["delivery_state"] = "STALE_LAST_GOOD"
    out["backend_error"] = str(error or "Due diligence rebuild failed")[:500]
    return out
