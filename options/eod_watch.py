"""Persist underlyings the user opened on F&O / Options so nightly EOD capture includes them.

DEFAULT_UNDERLYINGS stays NIFTY / BANKNIFTY / FINNIFTY. Opening a stock enqueues it
for the next options-eod job — it does not invent a chain on the weekend.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from options.eod_snapshot import DEFAULT_UNDERLYINGS

ROOT = Path(__file__).resolve().parents[1]
WATCH_PATH = ROOT / "logs" / "options" / "eod_watch.json"
MAX_WATCH = 40


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path = WATCH_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {"symbols": [], "updated_at": ""}


def _write(payload: dict[str, Any], path: Path = WATCH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clean(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def watched_symbols(path: Path = WATCH_PATH) -> list[str]:
    raw = _read(path).get("symbols") or []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        sym = _clean(item)
        if not sym or len(sym) > 32 or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def add_watch(symbol: str, *, path: Path = WATCH_PATH) -> dict[str, Any]:
    """Enqueue a name. FIFO cap. Idempotent. Never captures a chain here."""
    sym = _clean(symbol)
    if not sym or len(sym) > 32:
        return {"accepted": False, "symbol": sym, "watched": watched_symbols(path), "message": "invalid symbol"}
    current = watched_symbols(path)
    if sym in current:
        return {
            "accepted": True,
            "symbol": sym,
            "already": True,
            "watched": current,
            "capture_list": capture_list(path=path),
            "message": f"{sym} already on the next options EOD list.",
        }
    current.append(sym)
    if len(current) > MAX_WATCH:
        current = current[-MAX_WATCH:]
    _write({"symbols": current, "updated_at": _now_iso(), "latest": sym}, path)
    return {
        "accepted": True,
        "symbol": sym,
        "already": False,
        "watched": current,
        "capture_list": capture_list(path=path),
        "message": f"{sym} queued for the next options EOD capture. No live chain is invented now.",
    }


def capture_list(*, path: Path = WATCH_PATH) -> list[str]:
    """Index defaults first, then user-opened names — unique, stable order."""
    out: list[str] = []
    seen: set[str] = set()
    for item in list(DEFAULT_UNDERLYINGS) + watched_symbols(path):
        sym = _clean(item)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out
