"""
🧪 Deterministic broker simulator (Phase 6) — testing infrastructure ONLY.

Certifies the EMS against realistic broker behaviour: acknowledgement, partial fills, rejects,
timeouts (ambiguous — the order MAY have reached the exchange), duplicate/out-of-order callbacks,
idempotency (a repeated place with the same key returns the SAME order), protection placement,
and external/manual positions. Its results are never market evidence.

A `script` dict configures per-symbol behaviour so tests are fully deterministic.
"""
from __future__ import annotations

from ems.broker import (BrokerAdapter, BrokerOrder, BrokerFill, BrokerPosition,
                        BrokerProtection, BrokerTimeout, BrokerError)


class SimBroker(BrokerAdapter):
    adapter_id = "sim"

    def __init__(self, *, funds: float = 1_000_000.0, script: dict | None = None,
                 healthy: bool = True):
        self._funds = funds
        self.script = script or {}                 # symbol -> behaviour dict
        self.healthy = healthy
        self._orders: dict[str, BrokerOrder] = {}
        self._by_key: dict[str, str] = {}          # idempotency_key -> broker_order_id
        self._protections: dict[str, BrokerProtection] = {}
        self._manual_positions: list = []
        self._seq = 0

    # ── introspection ─────────────────────────────────────────────────────────────
    def health(self) -> dict:
        return {"ok": self.healthy, "latency_ms": 5}
    def capabilities(self) -> dict:
        return {"order_types": ["LIMIT", "MARKET"], "protection": ["STOP"], "atomic_oco": False}
    def funds(self) -> float:
        return self._funds

    # ── order placement (idempotent) ─────────────────────────────────────────────
    def place_order(self, plan, idempotency_key: str) -> BrokerOrder:
        if not self.healthy:
            raise BrokerError("broker unhealthy")
        beh = self.script.get(plan.symbol, {})
        # idempotency: same key ⇒ the SAME broker order, never a duplicate
        if idempotency_key in self._by_key:
            return self._orders[self._by_key[idempotency_key]]
        if beh.get("timeout"):
            # the exchange DID accept it (ambiguous to the caller) — record it, then raise
            oid = self._create(plan, idempotency_key, beh)
            raise BrokerTimeout(f"timeout after accepting {oid}")
        if beh.get("reject"):
            oid = f"SIMO{self._next()}"
            o = BrokerOrder(oid, idempotency_key, plan.symbol, plan.side, plan.qty,
                            status="REJECTED", reject_reason=beh.get("reject_reason", "REJECTED"))
            self._orders[oid] = o; self._by_key[idempotency_key] = oid
            return o
        oid = self._create(plan, idempotency_key, beh)
        return self._orders[oid]

    def _create(self, plan, key, beh) -> str:
        oid = f"SIMO{self._next()}"
        fill = int(beh.get("fill_qty", plan.qty))
        price = float(beh.get("fill_price", plan.limit_price or 100.0))
        if fill >= plan.qty:
            status, filled = "COMPLETE", plan.qty
        elif fill > 0:
            status, filled = "PARTIAL", fill
        else:
            status, filled = "OPEN", 0
        o = BrokerOrder(oid, key, plan.symbol, plan.side, plan.qty, status=status,
                        filled_qty=filled, avg_price=price if filled else 0.0)
        self._orders[oid] = o; self._by_key[key] = oid
        return oid

    def get_order(self, broker_order_id: str) -> BrokerOrder:
        if broker_order_id not in self._orders:
            return BrokerOrder(broker_order_id, "", "", "", 0, status="UNKNOWN")
        return self._orders[broker_order_id]

    def find_by_idempotency(self, idempotency_key: str):
        oid = self._by_key.get(idempotency_key)
        return self._orders[oid] if oid else None

    def fills(self, broker_order_id: str) -> list:
        o = self._orders.get(broker_order_id)
        if not o or o.filled_qty <= 0:
            return []
        return [BrokerFill(broker_order_id, f"{broker_order_id}-F1", o.filled_qty, o.avg_price,
                           fees=round(o.filled_qty * o.avg_price * 0.001, 2))]

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        o = self._orders.get(broker_order_id)
        if o and o.status in ("OPEN", "PARTIAL"):
            o.status = "CANCELLED"
        return o or BrokerOrder(broker_order_id, "", "", "", 0, status="UNKNOWN")

    def list_orders(self) -> list:
        return list(self._orders.values())

    def list_positions(self) -> list:
        pos = []
        seen = {}
        for o in self._orders.values():
            if o.filled_qty > 0 and o.status in ("COMPLETE", "PARTIAL"):
                seen[o.symbol] = seen.get(o.symbol, 0) + o.filled_qty
        for sym, q in seen.items():
            pos.append(BrokerPosition(sym, q, 0.0))
        return pos + list(self._manual_positions)

    # ── protection ────────────────────────────────────────────────────────────────
    def place_protection(self, plan) -> BrokerProtection:
        if not self.healthy:
            raise BrokerError("broker unhealthy")
        beh = self.script.get(plan.symbol, {})
        if beh.get("protection_reject"):
            raise BrokerError("protection rejected")
        pid = f"SIMP{self._next()}"
        p = BrokerProtection(pid, plan.symbol, plan.qty, plan.stop_price)
        self._protections[plan.symbol] = p
        return p

    def list_protections(self) -> list:
        return list(self._protections.values())

    # ── test helpers ──────────────────────────────────────────────────────────────
    def add_manual_position(self, symbol: str, qty: int, avg_price: float = 100.0):
        self._manual_positions.append(BrokerPosition(symbol, qty, avg_price))

    def _next(self) -> int:
        self._seq += 1
        return self._seq
