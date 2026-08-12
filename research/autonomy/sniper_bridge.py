"""Breakout sniper bridge for the autonomy supervisor.

Keeps the Kite WebSocket sniper armed from the latest product scan payload —
the same path `run_quantterm.sh` already starts via `main.py autonomy`.
Does not enable legacy Streamlit daemons.
"""
from __future__ import annotations

from typing import Any, Mapping


def records_from_payload(payload: Mapping[str, Any] | None) -> list[dict]:
    if not payload:
        return []
    rows = payload.get("records")
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    return []


def ensure_breakout_sniper(payload: Mapping[str, Any] | None = None) -> dict:
    """Start (idempotent) and refresh the sniper watch map.

    Returns a small status dict for job metadata / logs. Never raises to callers.
    """
    try:
        from product.scan_store import load_scan
        from scan.breakout_sniper import refresh_watch, start_sniper
    except Exception as exc:
        return {"ok": False, "error": f"import_failed:{exc}", "watching": 0}

    try:
        from research.autonomy import schedules as SCH
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        if not SCH.market_is_open(now):
            return {"ok": False, "error": "market_closed", "watching": 0}
    except Exception:
        # If clock helper unavailable, still try — start_sniper needs Kite.
        pass

    data = payload if payload is not None else load_scan()
    records = records_from_payload(data)
    try:
        started = bool(start_sniper())
    except Exception as exc:
        return {"ok": False, "error": f"start_failed:{exc}", "watching": 0, "started": False}

    if not started:
        return {
            "ok": False,
            "error": "kite_unavailable",
            "watching": 0,
            "started": False,
            "hint": "Run python main.py login so the sniper can subscribe to ticks",
        }

    try:
        n = int(refresh_watch(records))
    except Exception as exc:
        return {"ok": False, "error": f"refresh_failed:{exc}", "watching": 0, "started": True}

    return {"ok": True, "started": True, "watching": n, "records": len(records)}
