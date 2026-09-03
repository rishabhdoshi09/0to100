"""Opportunity path analysis over opportunity memory events."""
from __future__ import annotations

import json
from typing import Any, Mapping

from product.opportunity_memory import events_for, get as mem_get, list_open


def analyze_path(symbol: str, *, path=None) -> dict[str, Any]:
    row = mem_get(symbol, path=path) or {}
    events = events_for(symbol, path=path)
    states = [str(e.get("new_state") or "") for e in events if e.get("new_state")]
    first_decision = str(row.get("last_decision") or "")
    wait_reasons = [e.get("reason") for e in events if str(e.get("old_state") or "") in {"WAIT", "WAIT_EVIDENCE"}]
    wakes = [e for e in events if str(e.get("event") or "") not in {"DISCOVERED", ""} and e.get("old_state")]
    return {
        "symbol": str(symbol).upper(),
        "opportunity_id": row.get("opportunity_id") or str(symbol).upper(),
        "first_seen": row.get("first_seen_at"),
        "first_setup": row.get("first_setup"),
        "first_decision": first_decision,
        "states": states,
        "wait_triggers": _trigger(row),
        "wake_events": wakes,
        "last_state": row.get("last_state"),
        "last_decision": row.get("last_decision"),
        "last_entry_state": row.get("last_entry_state"),
        "last_execution_state": row.get("last_execution_state"),
        "ready": row.get("last_state") == "READY",
        "entered": str(row.get("last_execution_state") or "") in {"PAPER_ENTERED", "ELIGIBLE"},
        "expired": row.get("last_state") in {"EXPIRED", "INVALIDATED", "CLOSED"},
    }


def _trigger(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(row.get("wait_trigger_json") or "{}")
    except Exception:
        return {}


def conversion_metrics(*, path=None) -> dict[str, Any]:
    rows = list_open(path=path, limit=500)
    n = len(rows)
    by_state: dict[str, int] = {}
    watch_to_ready = wait_to_ready = ready_entered = ready_expired = 0
    for row in rows:
        state = str(row.get("last_state") or "")
        by_state[state] = by_state.get(state, 0) + 1
        events = events_for(str(row.get("symbol") or ""), path=path)
        seen = [str(e.get("new_state") or "") for e in events]
        if "WATCH" in seen and "READY" in seen:
            watch_to_ready += 1
        if any(s in {"WAIT", "WAIT_EVIDENCE"} for s in seen) and "READY" in seen:
            wait_to_ready += 1
        if "READY" in seen and str(row.get("last_execution_state") or "") == "PAPER_ENTERED":
            ready_entered += 1
        if "READY" in seen and state in {"EXPIRED", "INVALIDATED"}:
            ready_expired += 1
    return {
        "n_opportunities": n,
        "by_state": by_state,
        "watch_to_ready": watch_to_ready,
        "wait_to_ready": wait_to_ready,
        "ready_to_entered": ready_entered,
        "ready_to_expired": ready_expired,
        "sample_size": n,
        "note": "Conversion counts need sample size. Do not rank from three names.",
    }
