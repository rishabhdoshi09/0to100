"""Read-only projection of the canonical live-execution interlock.

This module never authorizes or executes a live order.  It exists so every
operator-facing safety surface reports the same broker-boundary truth.  Missing
or failed verification is UNKNOWN/UNVERIFIED, never a positive lock claim.
"""
from __future__ import annotations

from typing import Any


CANONICAL_SOURCE = "product.live_execution_interlock"


def live_safety_projection() -> dict[str, Any]:
    """Return canonical live-execution safety, failing closed on missing proof."""
    try:
        from product.live_execution_interlock import get_live_execution_state

        state = get_live_execution_state()
        payload = state.as_dict() if hasattr(state, "as_dict") else {}
        verified = getattr(state, "verified", False) is True
        if not verified:
            return {
                "live_locked": None,
                "live_lock_verified": False,
                "live_execution_authorized": None,
                "live_lock_status": "UNVERIFIED",
                "live_lock_reason": str(
                    payload.get("reason")
                    or getattr(state, "reason", "")
                    or "Canonical live-execution interlock reported an unverified state."
                ),
                "live_lock_source": str(
                    payload.get("source")
                    or getattr(state, "source", "")
                    or CANONICAL_SOURCE
                ),
            }

        authorized = getattr(state, "authorized", False) is True
        locked = bool(getattr(state, "locked", False) is True and not authorized)
        status = str(
            payload.get("status")
            or getattr(state, "status", "")
            or ("LOCKED" if locked else "AUTHORIZED" if authorized else "UNLOCKED")
        )
        return {
            "live_locked": locked,
            "live_lock_verified": True,
            "live_execution_authorized": authorized,
            "live_lock_status": status,
            "live_lock_reason": str(payload.get("reason") or getattr(state, "reason", "") or ""),
            "live_lock_source": str(
                payload.get("source")
                or getattr(state, "source", "")
                or CANONICAL_SOURCE
            ),
        }
    except Exception as exc:
        return {
            "live_locked": None,
            "live_lock_verified": False,
            "live_execution_authorized": None,
            "live_lock_status": "UNVERIFIED",
            "live_lock_reason": f"Canonical live-execution interlock could not be verified: {exc}",
            "live_lock_source": CANONICAL_SOURCE,
        }


def live_safety_is_verified_locked(payload: dict[str, Any] | None = None) -> bool:
    """True only when the canonical boundary was verified and is locked."""
    state = payload if isinstance(payload, dict) else live_safety_projection()
    return state.get("live_lock_verified") is True and state.get("live_locked") is True
