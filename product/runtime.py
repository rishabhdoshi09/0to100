"""Idempotent startup wiring for the retail entrypoint."""
from __future__ import annotations


def ensure_runtime_started() -> dict:
    """Start the existing PAPER_AUTO worker when its persisted switch is on.

    No data download and no order operation occurs here. The existing brain is
    the sole owner of runtime state; repeated calls are safe.
    """
    state = {"paper_auto_enabled": False, "worker_running": False, "error": ""}
    try:
        from research.auto_research.scheduler import get_brain
        brain = get_brain()
        state["paper_auto_enabled"] = bool(brain.is_paper_auto_enabled())
        if state["paper_auto_enabled"]:
            brain.start()
        state["worker_running"] = bool(brain.state.running)
        state["error"] = str(brain.state.last_error or "")
    except Exception as exc:
        state["error"] = str(exc)
    return state
