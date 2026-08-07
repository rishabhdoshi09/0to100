"""Deterministic internal position projection from durable OMS and protection state."""
from __future__ import annotations

from dataclasses import dataclass

from execution.oms import models as OM
from execution.protection import models as PM
from execution.reconciliation.models import InternalPositionSnapshot


@dataclass(frozen=True)
class InternalStateProjection:
    positions: tuple[InternalPositionSnapshot, ...]
    unresolved_order_ids: tuple[str, ...]
    filled_order_ids: tuple[str, ...]

    @property
    def reconciled_candidate(self) -> bool:
        return not self.unresolved_order_ids


def project_internal_positions(oms_store, protection_store=None) -> InternalStateProjection:
    """Fold durable order fills into expected net positions.

    Any recorded fill remains part of the expected position until the OMS order reaches CLOSED,
    including a partially filled order that is later cancelled. Unknown, quarantined and recovery
    states remain visible as unresolved rather than being discarded.
    """
    quantity_by_symbol: dict[str, int] = {}
    value_by_symbol: dict[str, float] = {}
    absolute_fill_by_symbol: dict[str, int] = {}
    protected_by_symbol: dict[str, int] = {}
    unresolved: list[str] = []
    filled_orders: list[str] = []

    for order in oms_store.list_orders():
        filled = max(0, int(order.filled_quantity))
        remaining = max(0, int(order.remaining_quantity))
        if order.status in {OM.UNKNOWN, OM.QUARANTINED, OM.RECOVERY_REQUIRED} and (filled or remaining):
            unresolved.append(order.order_id)
        if filled <= 0 or order.status == OM.CLOSED:
            continue
        filled_orders.append(order.order_id)
        symbol = order.symbol.upper()
        sign = 1 if order.side.upper() == "BUY" else -1
        signed_quantity = sign * filled
        quantity_by_symbol[symbol] = quantity_by_symbol.get(symbol, 0) + signed_quantity
        value_by_symbol[symbol] = (
            value_by_symbol.get(symbol, 0.0)
            + signed_quantity * float(order.average_fill_price or order.intended_entry)
        )
        absolute_fill_by_symbol[symbol] = absolute_fill_by_symbol.get(symbol, 0) + filled

        if protection_store is not None and sign > 0:
            plan = protection_store.get_by_order(order.order_id)
            if plan is not None and plan.status == PM.VERIFIED and plan.fully_protected:
                protected_by_symbol[symbol] = (
                    protected_by_symbol.get(symbol, 0)
                    + min(filled, int(plan.protected_quantity))
                )

    positions: list[InternalPositionSnapshot] = []
    for symbol in sorted(quantity_by_symbol):
        quantity = quantity_by_symbol[symbol]
        if quantity == 0:
            continue
        average_price = abs(value_by_symbol[symbol] / quantity) if quantity else 0.0
        protected = min(abs(quantity), protected_by_symbol.get(symbol, 0)) if quantity > 0 else 0
        positions.append(
            InternalPositionSnapshot(
                symbol=symbol,
                quantity=quantity,
                average_price=average_price,
                protected_quantity=protected,
            )
        )

    return InternalStateProjection(
        positions=tuple(positions),
        unresolved_order_ids=tuple(sorted(set(unresolved))),
        filled_order_ids=tuple(sorted(set(filled_orders))),
    )
