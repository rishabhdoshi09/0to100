"""Typed states and snapshots for QuantTerm's broker-neutral OMS."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PROPOSED = "PROPOSED"
RISK_APPROVED = "RISK_APPROVED"
SUBMISSION_PENDING = "SUBMISSION_PENDING"
BROKER_ACKNOWLEDGED = "BROKER_ACKNOWLEDGED"
PARTIALLY_FILLED = "PARTIALLY_FILLED"
FILLED = "FILLED"
PROTECTION_PENDING = "PROTECTION_PENDING"
PROTECTED = "PROTECTED"
EXIT_PENDING = "EXIT_PENDING"
CLOSED = "CLOSED"

REJECTED = "REJECTED"
CANCELLED = "CANCELLED"
EXPIRED = "EXPIRED"
UNKNOWN = "UNKNOWN"
QUARANTINED = "QUARANTINED"
RECOVERY_REQUIRED = "RECOVERY_REQUIRED"

ALL_STATUSES = frozenset({
    PROPOSED,
    RISK_APPROVED,
    SUBMISSION_PENDING,
    BROKER_ACKNOWLEDGED,
    PARTIALLY_FILLED,
    FILLED,
    PROTECTION_PENDING,
    PROTECTED,
    EXIT_PENDING,
    CLOSED,
    REJECTED,
    CANCELLED,
    EXPIRED,
    UNKNOWN,
    QUARANTINED,
    RECOVERY_REQUIRED,
})

TERMINAL_STATUSES = frozenset({CLOSED, REJECTED, CANCELLED, EXPIRED})
UNCERTAIN_STATUSES = frozenset({SUBMISSION_PENDING, UNKNOWN, QUARANTINED, RECOVERY_REQUIRED})
PENDING_EXPOSURE_STATUSES = frozenset({
    RISK_APPROVED,
    SUBMISSION_PENDING,
    BROKER_ACKNOWLEDGED,
    PARTIALLY_FILLED,
    UNKNOWN,
    QUARANTINED,
    RECOVERY_REQUIRED,
})

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    PROPOSED: frozenset({RISK_APPROVED, REJECTED, CANCELLED, EXPIRED, QUARANTINED}),
    RISK_APPROVED: frozenset({SUBMISSION_PENDING, CANCELLED, EXPIRED, QUARANTINED}),
    SUBMISSION_PENDING: frozenset({
        BROKER_ACKNOWLEDGED,
        PARTIALLY_FILLED,
        FILLED,
        REJECTED,
        CANCELLED,
        UNKNOWN,
        QUARANTINED,
        RECOVERY_REQUIRED,
    }),
    BROKER_ACKNOWLEDGED: frozenset({
        PARTIALLY_FILLED,
        FILLED,
        REJECTED,
        CANCELLED,
        EXPIRED,
        UNKNOWN,
        QUARANTINED,
        RECOVERY_REQUIRED,
    }),
    PARTIALLY_FILLED: frozenset({
        FILLED,
        CANCELLED,
        EXPIRED,
        UNKNOWN,
        QUARANTINED,
        RECOVERY_REQUIRED,
    }),
    FILLED: frozenset({PROTECTION_PENDING, EXIT_PENDING, CLOSED, QUARANTINED}),
    PROTECTION_PENDING: frozenset({PROTECTED, EXIT_PENDING, QUARANTINED, RECOVERY_REQUIRED}),
    PROTECTED: frozenset({EXIT_PENDING, CLOSED, QUARANTINED}),
    EXIT_PENDING: frozenset({CLOSED, UNKNOWN, QUARANTINED, RECOVERY_REQUIRED}),
    UNKNOWN: frozenset({
        BROKER_ACKNOWLEDGED,
        PARTIALLY_FILLED,
        FILLED,
        REJECTED,
        CANCELLED,
        QUARANTINED,
        RECOVERY_REQUIRED,
    }),
    RECOVERY_REQUIRED: frozenset({
        BROKER_ACKNOWLEDGED,
        PARTIALLY_FILLED,
        FILLED,
        REJECTED,
        CANCELLED,
        UNKNOWN,
        QUARANTINED,
    }),
    QUARANTINED: frozenset({UNKNOWN, RECOVERY_REQUIRED, CANCELLED, CLOSED}),
    CLOSED: frozenset(),
    REJECTED: frozenset(),
    CANCELLED: frozenset(),
    EXPIRED: frozenset(),
}


class OmsError(RuntimeError):
    """Base class for deterministic OMS errors."""


class OrderNotFound(OmsError):
    pass


class InvalidIntent(OmsError):
    pass


class IdempotencyConflict(OmsError):
    pass


class IllegalTransition(OmsError):
    pass


class FillConflict(OmsError):
    pass


@dataclass(frozen=True)
class OrderSnapshot:
    order_id: str
    idempotency_key: str
    trade_intent_id: str
    intent_hash: str
    target_portfolio_id: str
    target_position_id: str
    strategy_id: str
    strategy_version: int
    symbol: str
    side: str
    requested_quantity: int
    approved_quantity: int
    filled_quantity: int
    average_fill_price: float
    intended_entry: float
    stop_price: float
    target_price: float
    intended_risk_pct: float
    max_capital: float
    status: str
    broker_order_id: str
    submission_token: str
    risk_decision_id: str
    protection_required: bool
    version: int
    created_at: str
    updated_at: str
    last_error_code: str = ""
    last_error_message: str = ""

    @property
    def remaining_quantity(self) -> int:
        return max(0, self.approved_quantity - self.filled_quantity)

    @property
    def risk_per_share(self) -> float:
        return max(0.0, self.intended_entry - self.stop_price)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionSnapshot:
    transition_id: str
    order_id: str
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


@dataclass(frozen=True)
class FillSnapshot:
    fill_id: str
    order_id: str
    external_fill_id: str
    quantity: int
    price: float
    filled_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
