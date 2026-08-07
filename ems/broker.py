"""
🔌 BrokerAdapter contract + normalized broker types (Phase 5).

A broker adapter TRANSLATES broker-neutral commands. It must never select strategies, increase
quantity/risk, silently substitute an unsafe order type, or hide unsupported capabilities. It
normalizes statuses, fills, ids and timestamps. Real adapters (e.g. Kite) implement this later,
isolated and credential-gated; nothing here imports a real broker.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BrokerOrder:
    broker_order_id: str
    idempotency_key: str
    symbol: str
    side: str
    qty: int
    status: str                 # NORMALIZED: PENDING/OPEN/PARTIAL/COMPLETE/REJECTED/CANCELLED/UNKNOWN
    filled_qty: int = 0
    avg_price: float = 0.0
    reject_reason: str = ""


@dataclass
class BrokerFill:
    broker_order_id: str
    fill_id: str
    qty: int
    price: float
    fees: float = 0.0


@dataclass
class BrokerPosition:
    symbol: str
    qty: int
    avg_price: float


@dataclass
class BrokerProtection:
    protection_id: str
    symbol: str
    qty: int
    stop_price: float


class BrokerError(Exception):
    pass


class BrokerTimeout(BrokerError):
    """Ambiguous: the request may or may not have reached the exchange — NEVER assume failure."""


class BrokerAdapter:
    """Abstract broker-neutral interface. Concrete adapters normalize a real broker to this."""
    adapter_id = "abstract"

    def health(self) -> dict: raise NotImplementedError
    def capabilities(self) -> dict: raise NotImplementedError
    def funds(self) -> float: raise NotImplementedError
    def place_order(self, plan, idempotency_key: str) -> BrokerOrder: raise NotImplementedError
    def get_order(self, broker_order_id: str) -> BrokerOrder: raise NotImplementedError
    def find_by_idempotency(self, idempotency_key: str) -> BrokerOrder | None: raise NotImplementedError
    def cancel_order(self, broker_order_id: str) -> BrokerOrder: raise NotImplementedError
    def list_orders(self) -> list: raise NotImplementedError
    def list_positions(self) -> list: raise NotImplementedError
    def place_protection(self, plan) -> BrokerProtection: raise NotImplementedError
    def list_protections(self) -> list: raise NotImplementedError
