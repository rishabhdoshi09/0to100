"""Canonical broker-boundary live-execution interlock.

This module is the single source of truth for whether live broker mutations are
allowed.  Environment variables may request features elsewhere in the product,
but they are deliberately not authorization here.

The current production contract is PAPER/SHADOW ONLY.  Until a durable,
auditable graduation manifest is implemented and verified, the canonical state
remains locked and every broker mutation fails closed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class LiveExecutionBlocked(RuntimeError):
    """Raised before a live broker mutation can leave QuantTerm."""


@dataclass(frozen=True)
class LiveExecutionState:
    schema_version: int
    locked: bool
    authorized: bool
    verified: bool
    status: str
    reason: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_LOCK_REASON = (
    "Live broker execution is locked by the canonical QuantTerm interlock. "
    "Paper/shadow operation is permitted; live capital requires a future durable "
    "graduation authorization that is verified at this same broker boundary."
)


def get_live_execution_state() -> LiveExecutionState:
    """Return the authoritative live-execution state.

    Important: QT_LIVE_ENABLED, QT_ENABLE_UNSAFE_LEGACY_LIVE, UI arming, broker
    credentials, and similar flags are intentionally ignored.  They are not an
    authorization primitive.
    """
    return LiveExecutionState(
        schema_version=1,
        locked=True,
        authorized=False,
        verified=True,
        status="LOCKED",
        reason=_LOCK_REASON,
        source="product.live_execution_interlock",
    )


def assert_live_execution_allowed(operation: str = "broker_mutation") -> None:
    """Fail closed unless the canonical state is verified and authorized."""
    try:
        state = get_live_execution_state()
    except Exception as exc:  # pragma: no cover - defensive fail-closed boundary
        raise LiveExecutionBlocked(
            f"Live execution state could not be verified for {operation}: {exc}"
        ) from exc

    if not state.verified or state.locked or not state.authorized:
        raise LiveExecutionBlocked(f"{operation} blocked: {state.reason}")
