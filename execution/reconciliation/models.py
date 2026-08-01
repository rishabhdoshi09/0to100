"""Broker-neutral reconciliation snapshots and reports.

Adapters may translate any broker into these immutable records. The reconciliation engine
contains no network access and never treats a missing endpoint response as an empty book.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

HEALTHY = "HEALTHY"
REPAIRABLE = "REPAIRABLE"
QUARANTINED = "QUARANTINED"
INCOMPLETE = "INCOMPLETE"

INFO = "INFO"
WARNING = "WARNING"
CRITICAL = "CRITICAL"

NO_ACTION = "NO_ACTION"
AUTO_REPAIR = "AUTO_REPAIR"
QUARANTINE = "QUARANTINE"
MANUAL_REVIEW = "MANUAL_REVIEW"
FREEZE_ENTRIES = "FREEZE_ENTRIES"


@dataclass(frozen=True)
class BrokerOrderSnapshot:
    broker_order_id: str
    status: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int = 0
    average_price: float = 0.0
    client_order_ref: str = ""
    submission_token: str = ""
    status_message: str = ""
    updated_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerTradeSnapshot:
    trade_id: str
    broker_order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    executed_at: str
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    symbol: str
    quantity: int
    average_price: float = 0.0
    product: str = ""
    protected_quantity: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InternalPositionSnapshot:
    symbol: str
    quantity: int
    average_price: float = 0.0
    protected_quantity: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrokerAccountSnapshot:
    snapshot_id: str
    observed_at: str
    source: str
    orders: tuple[BrokerOrderSnapshot, ...] = ()
    trades: tuple[BrokerTradeSnapshot, ...] = ()
    positions: tuple[BrokerPositionSnapshot, ...] = ()
    cash: float = 0.0
    available_margin: float = 0.0
    orders_complete: bool = False
    trades_complete: bool = False
    positions_complete: bool = False
    account_complete: bool = False
    errors: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return (
            bool(self.snapshot_id)
            and self.orders_complete
            and self.trades_complete
            and self.positions_complete
            and self.account_complete
            and not self.errors
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairAction:
    action_id: str
    action_type: str
    order_id: str = ""
    broker_order_id: str = ""
    external_event_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationIssue:
    issue_id: str
    code: str
    severity: str
    action: str
    message: str
    order_id: str = ""
    broker_order_id: str = ""
    symbol: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationReport:
    report_id: str
    broker_snapshot_id: str
    observed_at: str
    status: str
    entry_freeze_required: bool
    issues: tuple[ReconciliationIssue, ...] = ()
    repairs: tuple[RepairAction, ...] = ()
    matched_orders: int = 0
    internal_orders: int = 0
    broker_orders: int = 0
    internal_positions: int = 0
    broker_positions: int = 0
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def healthy(self) -> bool:
        return self.status == HEALTHY and not self.entry_freeze_required

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
