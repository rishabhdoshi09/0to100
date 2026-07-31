"""Deprecated compatibility projection; Streamlit no longer starts runtime workers."""
from __future__ import annotations


def ensure_runtime_started() -> dict:
    """Return read-only supervisor status.

    Kept for old imports, but deliberately starts no brain, news thread or scheduler.  The dedicated
    ``python main.py autonomy`` process is the sole scheduler and mutation owner.
    """
    from product.autonomy_status import read_autonomy_status
    status = read_autonomy_status()
    return {"supervisor_running": status.get("running", False),
            "state": status.get("state", "UNKNOWN"), "error": status.get("explanation", "")}
