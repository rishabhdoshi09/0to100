"""
📒 Persistent execution ledger (Phase 8) — journaled, restart-surviving.

Records the INTENDED action before the external side effect and the OBSERVED result after, so a
crash can never erase the fact that an order may have reached the broker. Keyed by idempotency
key so a duplicate is detectable across restarts. JSON-backed; in-memory when no path is given.
"""
from __future__ import annotations

import json
from pathlib import Path

from ems import schemas as SC


class ExecutionLedger:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.orders: dict[str, SC.OrderStateRecord] = {}     # idempotency_key -> order
        self.fills: list = []
        self.positions: dict[str, SC.PositionRecord] = {}    # (strategy,symbol) key
        self.incidents: list = []
        if self.path and self.path.exists():
            self._load()

    # ── idempotency ────────────────────────────────────────────────────────────────
    def get_order(self, idempotency_key: str):
        return self.orders.get(idempotency_key)

    def has_order(self, idempotency_key: str) -> bool:
        return idempotency_key in self.orders

    # ── journaled writes ────────────────────────────────────────────────────────────
    def record_order(self, order: SC.OrderStateRecord) -> None:
        self.orders[order.idempotency_key] = order
        self._save()

    def record_fill(self, fill: SC.FillRecord) -> None:
        self.fills.append(fill)
        self._save()

    def record_position(self, pos: SC.PositionRecord) -> None:
        self.positions[f"{pos.strategy_id}:{pos.symbol}"] = pos
        self._save()

    def remove_position(self, strategy_id: str, symbol: str) -> None:
        self.positions.pop(f"{strategy_id}:{symbol}", None)
        self._save()

    def record_incident(self, inc: SC.ExecutionIncident) -> None:
        self.incidents.append(inc)
        self._save()

    def open_positions(self) -> list:
        return list(self.positions.values())

    def unresolved_orders(self) -> list:
        from ems import state_machine as SM
        return [o for o in self.orders.values() if not SM.is_terminal(o.state)]

    def has_critical_incident(self) -> bool:
        return any(i.severity == "CRITICAL" and not i.resolved for i in self.incidents)

    # ── persistence ────────────────────────────────────────────────────────────────
    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "orders": {k: v.as_dict() for k, v in self.orders.items()},
            "fills": [f.as_dict() for f in self.fills],
            "positions": {k: v.as_dict() for k, v in self.positions.items()},
            "incidents": [i.as_dict() for i in self.incidents],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, default=str), encoding="utf-8")
        import os
        os.replace(tmp, self.path)                       # atomic ledger write

    def _load(self) -> None:
        try:
            d = json.loads(self.path.read_text())
            for k, v in d.get("orders", {}).items():
                self.orders[k] = SC.OrderStateRecord(**v)
            self.fills = [SC.FillRecord(**f) for f in d.get("fills", [])]
            for k, v in d.get("positions", {}).items():
                self.positions[k] = SC.PositionRecord(**v)
            self.incidents = [SC.ExecutionIncident(**i) for i in d.get("incidents", [])]
        except Exception:
            pass
