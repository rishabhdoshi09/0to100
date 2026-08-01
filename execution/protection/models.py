"""Broker-neutral protection plan states and snapshots."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

REQUIRED = "REQUIRED"
SUBMISSION_PENDING = "SUBMISSION_PENDING"
ACTIVE = "ACTIVE"
VERIFIED = "VERIFIED"
ADJUSTMENT_REQUIRED = "ADJUSTMENT_REQUIRED"
CANCEL_PENDING = "CANCEL_PENDING"
CANCELLED = "CANCELLED"
FAILED = "FAILED"
ORPHANED = "ORPHANED"
RECOVERY_REQUIRED = "RECOVERY_REQUIRED"

ALL_STATUSES = frozenset({
    REQUIRED,
    SUBMISSION_PENDING,
    ACTIVE,
    VERIFIED,
    ADJUSTMENT_REQUIRED,
    CANCEL_PENDING,
    CANCELLED,
    FAILED,
    ORPHANED,
    RECOVERY_REQUIRED,
})

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    REQUIRED: frozenset({SUBMISSION_PENDING, FAILED, CANCELLED, RECOVERY_REQUIRED}),
    SUBMISSION_PENDING: frozenset({
        ACTIVE,
        VERIFIED,
        ADJUSTMENT_REQUIRED,
        FAILED,
        RECOVERY_REQUIRED,
    }),
    ACTIVE: frozenset({VERIFIED, ADJUSTMENT_REQUIRED, CANCEL_PENDING, RECOVERY_REQUIRED}),
    VERIFIED: frozenset({ADJUSTMENT_REQUIRED, CANCEL_PENDING, RECOVERY_REQUIRED}),
    ADJUSTMENT_REQUIRED: frozenset({SUBMISSION_PENDING, CANCEL_PENDING, FAILED, RECOVERY_REQUIRED}),
    CANCEL_PENDING: frozenset({CANCELLED, ACTIVE, VERIFIED, RECOVERY_REQUIRED}),
    FAILED: frozenset({SUBMISSION_PENDING, CANCELLED, RECOVERY_REQUIRED}),
    RECOVERY_REQUIRED: frozenset({SUBMISSION_PENDING, ACTIVE, VERIFIED, CANCEL_PENDING, ORPHANED}),
    ORPHANED: frozenset({CANCEL_PENDING, CANCELLED, RECOVERY_REQUIRED}),
    CANCELLED: frozenset(),
}


class ProtectionError(RuntimeError):
    pass


class ProtectionNotFound(ProtectionError):
    pass


class InvalidProtectionPlan(ProtectionError):
    pass


class IllegalProtectionTransition(ProtectionError):
    pass


@dataclass(frozen=True)
class ProtectionPlanSnapshot:
    plan_id: str
    order_id: str
    symbol: str
    required_quantity: int
    protected_quantity: int
    stop_price: float
    target_price: float
    status: str
    request_token: str
    broker_protection_id: str
    stop_reference: str
    target_reference: str
    version: int
    created_at: str
    updated_at: str
    last_verified_at: str = ""
    last_error_code: str = ""
    last_error_message: str = ""

    @property
    def fully_protected(self) -> bool:
        return (
            self.status == VERIFIED
            and self.required_quantity > 0
            and self.protected_quantity == self.required_quantity
            and bool(self.stop_reference)
        )

    @property
    def missing_quantity(self) -> int:
        return max(0, self.required_quantity - self.protected_quantity)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerProtectionSnapshot:
    broker_protection_id: str
    order_id: str
    symbol: str
    active: bool
    quantity: int
    stop_price: float
    target_price: float
    stop_reference: str = ""
    target_reference: str = ""
    updated_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtectionTransitionSnapshot:
    transition_id: str
    plan_id: str
    sequence: int
    from_status: str
    to_status: str
    event_type: str
    event_at: str
    actor: str
    reason: str
    external_event_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
