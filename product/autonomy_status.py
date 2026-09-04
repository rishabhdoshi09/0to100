"""
Read-only projection of the autonomy supervisor for the retail UI.

Reads the supervisor's status snapshot + job ledger + dialogue log from disk. It NEVER starts the
supervisor and owns no truth — it only translates what the supervisor already recorded into plain
language for the control room.

Broker readiness is projected independently from supervisor health. Missing Zerodha auth disables
broker-dependent live-data/execution work; it must not make research/scanning autonomy look broken.
"""
from __future__ import annotations

from pathlib import Path

from research.autonomy import default_root
from research.autonomy import health as H


def _paths(root=None):
    root = Path(root) if root else default_root()
    return root / "status.json", root / "jobs.db", root / "dialogue.jsonl"


_STATE_PLAIN = {
    "STARTING": "QuantTerm is starting up.",
    "AUTH_REQUIRED": "Zerodha login is unavailable; non-broker autonomy can continue.",
    "DATA_REFRESHING": "Updating market history.",
    "DATA_BLOCKED": "Market data is not ready — new paper trades are paused.",
    "DATA_READY": "Market data is ready.",
    "OBSERVING": "QuantTerm is observing the market normally.",
    "PAPER_ACTIVE": "Automatic paper trading is active.",
    "RESEARCHING": "Testing new strategy ideas.",
    "DEGRADED": "Running with reduced capability — see details.",
    "HALTED": "Stopped. No new activity.",
    "UNKNOWN": "The autonomous supervisor is not running.",
}


def _broker_status() -> dict:
    """Canonical execution/live-data capability, fail-closed and independent."""
    try:
        from product.readiness import broker_status

        return dict(broker_status() or {})
    except Exception as exc:
        return {
            "state": "UNKNOWN",
            "ready": False,
            "live_data_ready": False,
            "execution_ready": False,
            "auth_ready": False,
            "login_required": False,
            "auth_status": "PROBE_FAILED",
            "reason_code": type(exc).__name__.upper(),
            "detail": str(exc)[:200],
            "snapshot_id": "",
        }


def read_autonomy_status(root=None) -> dict:
    """Plain-language, read-only status for the control room. Safe when the supervisor never ran."""
    status_path, jobs_db, dialogue_path = _paths(root)
    raw = H.read_status(state_path=status_path,
                        jobs_db=jobs_db if Path(jobs_db).exists() else None,
                        dialogue_path=dialogue_path if Path(dialogue_path).exists() else None)
    state = raw.get("state", "UNKNOWN")
    caps = H.capabilities(raw.get("active_failures", []) or [])  # recompute from recorded failures
    return {
        "running": raw.get("supervisor_running", False),
        "state": state,
        "plain_state": _STATE_PLAIN.get(state, state),
        "explanation": raw.get("explanation", ""),
        "reason_code": str(raw.get("reason_code") or ""),
        "heartbeat_ist": raw.get("heartbeat_ist", ""),
        "scheduler_owner_pid": raw.get("scheduler_owner_pid"),
        "active_job": dict(raw.get("active_job", {}) or {}),
        "snapshot_id": raw.get("snapshot_id", ""),
        "new_paper_entries": caps["new_paper_entries"],
        "existing_exits": caps["existing_exits"],
        "research": caps["research"],
        "capability_notes": caps["notes"],
        "active_failures": list(raw.get("active_failures") or []),
        "jobs": raw.get("jobs", {}),
        "recent_transitions": raw.get("recent_transitions", []),
        "recent_dialogue": raw.get("recent_dialogue", []),
        "owner_state": raw.get("owner_state", {}),
        "scheduler_of_record": raw.get("scheduler_of_record", ""),
        "last_cycle": raw.get("last_cycle", {}),
        "broker": _broker_status(),
    }
