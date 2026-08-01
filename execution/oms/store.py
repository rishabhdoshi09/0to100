"""Durable, broker-neutral Order Management System.

The store persists an immutable TradeIntent before any external submission, validates every
lifecycle transition, deduplicates broker events and fills, and reconstructs exact state after
restart. It deliberately contains no Kite import and no network call.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from execution.oms import models as M

_SCHEMA = """
CREATE TABLE IF NOT EXISTS oms_orders (
    order_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    trade_intent_id TEXT NOT NULL UNIQUE,
    intent_hash TEXT NOT NULL,
    intent_json TEXT NOT NULL,
    target_portfolio_id TEXT NOT NULL,
    target_position_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    strategy_version INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    requested_quantity INTEGER NOT NULL CHECK(requested_quantity > 0),
    approved_quantity INTEGER NOT NULL DEFAULT 0 CHECK(approved_quantity >= 0),
    filled_quantity INTEGER NOT NULL DEFAULT 0 CHECK(filled_quantity >= 0),
    average_fill_price REAL NOT NULL DEFAULT 0,
    intended_entry REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    intended_risk_pct REAL NOT NULL,
    max_capital REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    broker_order_id TEXT UNIQUE,
    submission_token TEXT UNIQUE,
    risk_decision_id TEXT,
    protection_required INTEGER NOT NULL DEFAULT 1,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_error_code TEXT NOT NULL DEFAULT '',
    last_error_message TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS oms_transitions (
    transition_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES oms_orders(order_id),
    sequence INTEGER NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_at TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    external_event_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(order_id, sequence),
    UNIQUE(order_id, external_event_id)
);
CREATE TABLE IF NOT EXISTS oms_fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES oms_orders(order_id),
    external_fill_id TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    price REAL NOT NULL CHECK(price > 0),
    filled_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(order_id, external_fill_id)
);
CREATE INDEX IF NOT EXISTS idx_oms_orders_status ON oms_orders(status);
CREATE INDEX IF NOT EXISTS idx_oms_orders_symbol ON oms_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_oms_transitions_order ON oms_transitions(order_id, sequence);
CREATE INDEX IF NOT EXISTS idx_oms_fills_order ON oms_fills(order_id, filled_at);
"""


class OmsStore:
    """Single-writer durable OMS with explicit state transitions."""

    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.commit()

    def ingest_intent(self, intent) -> M.OrderSnapshot:
        """Persist a Target-Portfolio-linked intent idempotently in PROPOSED state."""
        payload = intent.as_dict()
        required_quantity = int(payload.get("required_quantity") or 0)
        target_portfolio_id = str(payload.get("target_portfolio_id") or "")
        target_position_id = str(payload.get("target_position_id") or "")
        if required_quantity <= 0:
            raise M.InvalidIntent("required_quantity must be positive")
        if not target_portfolio_id or not target_position_id:
            raise M.InvalidIntent("TradeIntent must reference a TargetPortfolio and TargetPosition")
        if float(payload.get("intended_entry") or 0) <= 0:
            raise M.InvalidIntent("intended_entry must be positive")
        if float(payload.get("stop_price") or 0) <= 0:
            raise M.InvalidIntent("stop_price must be positive")

        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        intent_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        idempotency_key = str(intent.record_id)
        order_id = f"oms-{hashlib.sha256(idempotency_key.encode()).hexdigest()[:20]}"
        now = self._now()
        side = "BUY" if str(payload.get("direction") or "LONG").upper() == "LONG" else "SELL"

        with self._transaction() as connection:
            existing = self._get_by_idempotency_conn(connection, idempotency_key)
            if existing is not None:
                if existing.intent_hash != intent_hash:
                    raise M.IdempotencyConflict(
                        f"idempotency key {idempotency_key} already owns different intent content"
                    )
                return existing
            connection.execute(
                """INSERT INTO oms_orders (
                    order_id,idempotency_key,trade_intent_id,intent_hash,intent_json,
                    target_portfolio_id,target_position_id,strategy_id,strategy_version,
                    symbol,side,requested_quantity,intended_entry,stop_price,target_price,
                    intended_risk_pct,max_capital,status,protection_required,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    order_id,
                    idempotency_key,
                    str(intent.record_id),
                    intent_hash,
                    canonical,
                    target_portfolio_id,
                    target_position_id,
                    str(payload.get("strategy_id") or ""),
                    int(payload.get("strategy_version") or 0),
                    str(payload.get("symbol") or "").upper(),
                    side,
                    required_quantity,
                    float(payload.get("intended_entry") or 0),
                    float(payload.get("stop_price") or 0),
                    float(payload.get("target_price") or 0),
                    float(payload.get("intended_risk_pct") or 0),
                    float(payload.get("max_capital") or 0),
                    M.PROPOSED,
                    1 if float(payload.get("stop_price") or 0) > 0 else 0,
                    now,
                    now,
                ),
            )
            self._append_transition_conn(
                connection,
                order_id,
                from_status="",
                to_status=M.PROPOSED,
                event_type="INTENT_ACCEPTED",
                actor="target_portfolio",
                reason="durable intent persisted before external submission",
                metadata={
                    "trade_intent_id": str(intent.record_id),
                    "target_portfolio_id": target_portfolio_id,
                    "target_position_id": target_position_id,
                },
            )
            return self._get_conn(connection, order_id)

    def approve_risk(
        self,
        order_id: str,
        *,
        risk_decision_id: str,
        approved_quantity: int | None = None,
        actor: str = "risk_governor",
        reason: str = "",
        external_event_id: str = "",
    ) -> M.OrderSnapshot:
        with self._transaction() as connection:
            order = self._get_conn(connection, order_id)
            quantity = order.requested_quantity if approved_quantity is None else int(approved_quantity)
            if quantity <= 0 or quantity > order.requested_quantity:
                raise M.InvalidIntent("approved quantity must be within requested quantity")
            if not risk_decision_id:
                raise M.InvalidIntent("risk_decision_id is required")
            return self._transition_conn(
                connection,
                order,
                M.RISK_APPROVED,
                event_type="RISK_APPROVED",
                actor=actor,
                reason=reason,
                external_event_id=external_event_id,
                updates={
                    "approved_quantity": quantity,
                    "risk_decision_id": risk_decision_id,
                    "last_error_code": "",
                    "last_error_message": "",
                },
                metadata={"approved_quantity": quantity, "risk_decision_id": risk_decision_id},
            )

    def prepare_submission(
        self,
        order_id: str,
        *,
        submission_token: str,
        actor: str = "ems",
    ) -> M.OrderSnapshot:
        """Persist SUBMISSION_PENDING and its unique token before calling a broker."""
        if not submission_token:
            raise M.InvalidIntent("submission_token is required")
        with self._transaction() as connection:
            order = self._get_conn(connection, order_id)
            if order.status == M.SUBMISSION_PENDING and order.submission_token == submission_token:
                return order
            return self._transition_conn(
                connection,
                order,
                M.SUBMISSION_PENDING,
                event_type="SUBMISSION_PREPARED",
                actor=actor,
                updates={"submission_token": submission_token},
                metadata={"submission_token": submission_token},
            )

    def acknowledge(
        self,
        order_id: str,
        *,
        broker_order_id: str,
        external_event_id: str = "",
        actor: str = "broker_adapter",
    ) -> M.OrderSnapshot:
        if not broker_order_id:
            raise M.InvalidIntent("broker_order_id is required")
        with self._transaction() as connection:
            duplicate = self._external_event_order_conn(connection, order_id, external_event_id)
            if duplicate is not None:
                return duplicate
            order = self._get_conn(connection, order_id)
            if order.broker_order_id and order.broker_order_id != broker_order_id:
                return self._quarantine_conn(
                    connection,
                    order,
                    code="BROKER_ORDER_ID_CONFLICT",
                    message=f"existing={order.broker_order_id}; received={broker_order_id}",
                    actor=actor,
                    external_event_id=external_event_id,
                )
            if order.status == M.BROKER_ACKNOWLEDGED and order.broker_order_id == broker_order_id:
                return order
            return self._transition_conn(
                connection,
                order,
                M.BROKER_ACKNOWLEDGED,
                event_type="BROKER_ACKNOWLEDGED",
                actor=actor,
                external_event_id=external_event_id,
                updates={"broker_order_id": broker_order_id},
                metadata={"broker_order_id": broker_order_id},
            )

    def mark_submission_uncertain(
        self,
        order_id: str,
        *,
        reason: str,
        error_code: str = "BROKER_RESPONSE_UNCERTAIN",
        actor: str = "broker_adapter",
    ) -> M.OrderSnapshot:
        with self._transaction() as connection:
            order = self._get_conn(connection, order_id)
            return self._transition_conn(
                connection,
                order,
                M.RECOVERY_REQUIRED,
                event_type="SUBMISSION_UNCERTAIN",
                actor=actor,
                reason=reason,
                updates={"last_error_code": error_code, "last_error_message": reason},
            )

    def record_fill(
        self,
        order_id: str,
        *,
        external_fill_id: str,
        quantity: int,
        price: float,
        filled_at: str | None = None,
        broker_order_id: str = "",
        metadata: dict[str, Any] | None = None,
        actor: str = "broker_adapter",
    ) -> M.OrderSnapshot:
        """Record a broker fill idempotently and derive partial/full/quarantine state."""
        if not external_fill_id:
            raise M.InvalidIntent("external_fill_id is required")
        quantity = int(quantity)
        price = float(price)
        if quantity <= 0 or price <= 0:
            raise M.InvalidIntent("fill quantity and price must be positive")
        filled_at = filled_at or self._now()
        metadata = dict(metadata or {})

        with self._transaction() as connection:
            order = self._get_conn(connection, order_id)
            if order.status not in {
                M.SUBMISSION_PENDING,
                M.BROKER_ACKNOWLEDGED,
                M.PARTIALLY_FILLED,
                M.UNKNOWN,
                M.RECOVERY_REQUIRED,
            }:
                raise M.IllegalTransition(f"cannot record fill while order is {order.status}")
            existing = connection.execute(
                "SELECT * FROM oms_fills WHERE order_id=? AND external_fill_id=?",
                (order_id, external_fill_id),
            ).fetchone()
            if existing is not None:
                if int(existing["quantity"]) != quantity or float(existing["price"]) != price:
                    raise M.FillConflict("external fill id was reused with different fill content")
                return self._get_conn(connection, order_id)

            fill_id = f"fill-{hashlib.sha256(f'{order_id}:{external_fill_id}'.encode()).hexdigest()[:20]}"
            connection.execute(
                """INSERT INTO oms_fills
                   (fill_id,order_id,external_fill_id,quantity,price,filled_at,metadata_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    fill_id,
                    order_id,
                    external_fill_id,
                    quantity,
                    price,
                    filled_at,
                    json.dumps(metadata, sort_keys=True, default=str),
                ),
            )
            totals = connection.execute(
                "SELECT COALESCE(SUM(quantity),0) qty, COALESCE(SUM(quantity*price),0) value "
                "FROM oms_fills WHERE order_id=?",
                (order_id,),
            ).fetchone()
            total_quantity = int(totals["qty"])
            average_price = float(totals["value"]) / total_quantity
            approved = order.approved_quantity
            if approved <= 0:
                return self._quarantine_conn(
                    connection,
                    order,
                    code="FILL_WITHOUT_APPROVED_QUANTITY",
                    message=f"received {total_quantity} filled shares without approved quantity",
                    actor=actor,
                    updates={
                        "filled_quantity": total_quantity,
                        "average_fill_price": average_price,
                        "broker_order_id": broker_order_id or order.broker_order_id,
                    },
                )
            if total_quantity > approved:
                return self._quarantine_conn(
                    connection,
                    order,
                    code="BROKER_OVERFILL",
                    message=f"filled={total_quantity}; approved={approved}",
                    actor=actor,
                    updates={
                        "filled_quantity": total_quantity,
                        "average_fill_price": average_price,
                        "broker_order_id": broker_order_id or order.broker_order_id,
                    },
                )

            target_status = M.FILLED if total_quantity == approved else M.PARTIALLY_FILLED
            updates = {
                "filled_quantity": total_quantity,
                "average_fill_price": average_price,
                "broker_order_id": broker_order_id or order.broker_order_id,
                "last_error_code": "",
                "last_error_message": "",
            }
            fill_meta = {
                "fill_id": fill_id,
                "external_fill_id": external_fill_id,
                "fill_quantity": quantity,
                "fill_price": price,
                "cumulative_quantity": total_quantity,
                "approved_quantity": approved,
                "average_fill_price": average_price,
            }
            if order.status == target_status:
                self._update_order_conn(connection, order_id, updates)
                self._append_transition_conn(
                    connection,
                    order_id,
                    from_status=target_status,
                    to_status=target_status,
                    event_type="FILL_RECORDED",
                    actor=actor,
                    metadata=fill_meta,
                )
                return self._get_conn(connection, order_id)
            return self._transition_conn(
                connection,
                order,
                target_status,
                event_type="FILL_RECORDED",
                actor=actor,
                updates=updates,
                metadata=fill_meta,
            )

    def mark_protection_pending(self, order_id: str, *, actor: str = "protection_manager"):
        return self.transition(
            order_id,
            M.PROTECTION_PENDING,
            event_type="PROTECTION_REQUESTED",
            actor=actor,
        )

    def mark_protected(
        self,
        order_id: str,
        *,
        protection_reference: str,
        external_event_id: str = "",
        actor: str = "protection_manager",
    ):
        if not protection_reference:
            raise M.InvalidIntent("protection_reference is required")
        return self.transition(
            order_id,
            M.PROTECTED,
            event_type="PROTECTION_VERIFIED",
            actor=actor,
            external_event_id=external_event_id,
            metadata={"protection_reference": protection_reference},
        )

    def mark_exit_pending(self, order_id: str, *, reason: str, actor: str = "ems"):
        return self.transition(
            order_id,
            M.EXIT_PENDING,
            event_type="EXIT_REQUESTED",
            actor=actor,
            reason=reason,
        )

    def mark_closed(self, order_id: str, *, reason: str, actor: str = "reconciliation"):
        return self.transition(
            order_id,
            M.CLOSED,
            event_type="POSITION_CLOSED",
            actor=actor,
            reason=reason,
        )

    def reject(self, order_id: str, *, reason: str, external_event_id: str = ""):
        return self.transition(
            order_id,
            M.REJECTED,
            event_type="ORDER_REJECTED",
            actor="broker_adapter",
            reason=reason,
            external_event_id=external_event_id,
            updates={"last_error_code": "BROKER_REJECTED", "last_error_message": reason},
        )

    def cancel(self, order_id: str, *, reason: str, external_event_id: str = ""):
        return self.transition(
            order_id,
            M.CANCELLED,
            event_type="ORDER_CANCELLED",
            actor="ems",
            reason=reason,
            external_event_id=external_event_id,
        )

    def expire(self, order_id: str, *, reason: str):
        return self.transition(
            order_id,
            M.EXPIRED,
            event_type="ORDER_EXPIRED",
            actor="oms",
            reason=reason,
        )

    def mark_unknown(self, order_id: str, *, reason: str, actor: str = "reconciliation"):
        return self.transition(
            order_id,
            M.UNKNOWN,
            event_type="ORDER_STATE_UNKNOWN",
            actor=actor,
            reason=reason,
            updates={"last_error_code": "STATE_UNKNOWN", "last_error_message": reason},
        )

    def quarantine(
        self,
        order_id: str,
        *,
        code: str,
        message: str,
        actor: str = "reconciliation",
    ):
        with self._transaction() as connection:
            order = self._get_conn(connection, order_id)
            return self._quarantine_conn(
                connection,
                order,
                code=code,
                message=message,
                actor=actor,
            )

    def transition(
        self,
        order_id: str,
        to_status: str,
        *,
        event_type: str,
        actor: str,
        reason: str = "",
        external_event_id: str = "",
        updates: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> M.OrderSnapshot:
        with self._transaction() as connection:
            duplicate = self._external_event_order_conn(connection, order_id, external_event_id)
            if duplicate is not None:
                return duplicate
            order = self._get_conn(connection, order_id)
            return self._transition_conn(
                connection,
                order,
                to_status,
                event_type=event_type,
                actor=actor,
                reason=reason,
                external_event_id=external_event_id,
                updates=updates,
                metadata=metadata,
            )

    def get(self, order_id: str) -> M.OrderSnapshot:
        with self._connect() as connection:
            return self._get_conn(connection, order_id)

    def get_by_intent(self, trade_intent_id: str) -> M.OrderSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM oms_orders WHERE trade_intent_id=?",
                (trade_intent_id,),
            ).fetchone()
            return self._order_from_row(row) if row is not None else None

    def list_orders(self, *, statuses: Iterable[str] | None = None) -> list[M.OrderSnapshot]:
        with self._connect() as connection:
            if statuses:
                values = tuple(statuses)
                placeholders = ",".join("?" for _ in values)
                rows = connection.execute(
                    f"SELECT * FROM oms_orders WHERE status IN ({placeholders}) ORDER BY created_at,order_id",
                    values,
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM oms_orders ORDER BY created_at,order_id"
                ).fetchall()
            return [self._order_from_row(row) for row in rows]

    def history(self, order_id: str) -> list[M.TransitionSnapshot]:
        with self._connect() as connection:
            self._get_conn(connection, order_id)
            rows = connection.execute(
                "SELECT * FROM oms_transitions WHERE order_id=? ORDER BY sequence",
                (order_id,),
            ).fetchall()
            return [self._transition_from_row(row) for row in rows]

    def fills(self, order_id: str) -> list[M.FillSnapshot]:
        with self._connect() as connection:
            self._get_conn(connection, order_id)
            rows = connection.execute(
                "SELECT * FROM oms_fills WHERE order_id=? ORDER BY filled_at,fill_id",
                (order_id,),
            ).fetchall()
            return [self._fill_from_row(row) for row in rows]

    def pending_exposure(self) -> dict[str, Any]:
        """Worst-case remaining entry exposure for Target Portfolio construction."""
        orders = self.list_orders(statuses=M.PENDING_EXPOSURE_STATUSES)
        quantities: dict[str, int] = {}
        risks: dict[str, float] = {}
        capital: dict[str, float] = {}
        uncertain: list[str] = []
        for order in orders:
            remaining = order.remaining_quantity
            if remaining <= 0:
                continue
            quantities[order.symbol] = quantities.get(order.symbol, 0) + remaining
            risks[order.symbol] = risks.get(order.symbol, 0.0) + remaining * order.risk_per_share
            capital[order.symbol] = capital.get(order.symbol, 0.0) + remaining * order.intended_entry
            if order.status in M.UNCERTAIN_STATUSES:
                uncertain.append(order.order_id)
        return {
            "pending_quantities": quantities,
            "pending_risk_amounts": risks,
            "pending_capital_amounts": capital,
            "uncertain_order_ids": uncertain,
            "orders": [order.as_dict() for order in orders],
        }

    def summary(self) -> dict[str, Any]:
        orders = self.list_orders()
        counts: dict[str, int] = {}
        for order in orders:
            counts[order.status] = counts.get(order.status, 0) + 1
        return {
            "orders": len(orders),
            "by_status": counts,
            "pending_exposure": self.pending_exposure(),
            "recovery_required": [
                order.order_id for order in orders
                if order.status in {M.UNKNOWN, M.QUARANTINED, M.RECOVERY_REQUIRED}
            ],
        }

    def _transition_conn(
        self,
        connection: sqlite3.Connection,
        order: M.OrderSnapshot,
        to_status: str,
        *,
        event_type: str,
        actor: str,
        reason: str = "",
        external_event_id: str = "",
        updates: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> M.OrderSnapshot:
        if to_status not in M.ALL_STATUSES:
            raise M.IllegalTransition(f"unknown OMS status {to_status}")
        if order.status == to_status:
            return order
        allowed = M.ALLOWED_TRANSITIONS.get(order.status, frozenset())
        if to_status not in allowed and not (
            to_status == M.QUARANTINED and order.status not in M.TERMINAL_STATUSES
        ):
            raise M.IllegalTransition(f"illegal OMS transition {order.status} -> {to_status}")
        merged = dict(updates or {})
        merged["status"] = to_status
        self._update_order_conn(connection, order.order_id, merged)
        self._append_transition_conn(
            connection,
            order.order_id,
            from_status=order.status,
            to_status=to_status,
            event_type=event_type,
            actor=actor,
            reason=reason,
            external_event_id=external_event_id,
            metadata=metadata,
        )
        return self._get_conn(connection, order.order_id)

    def _quarantine_conn(
        self,
        connection: sqlite3.Connection,
        order: M.OrderSnapshot,
        *,
        code: str,
        message: str,
        actor: str,
        external_event_id: str = "",
        updates: dict[str, Any] | None = None,
    ) -> M.OrderSnapshot:
        merged = dict(updates or {})
        merged.update(last_error_code=code, last_error_message=message)
        if order.status == M.QUARANTINED:
            self._update_order_conn(connection, order.order_id, merged)
            self._append_transition_conn(
                connection,
                order.order_id,
                from_status=M.QUARANTINED,
                to_status=M.QUARANTINED,
                event_type="QUARANTINE_UPDATED",
                actor=actor,
                reason=message,
                external_event_id=external_event_id,
                metadata={"code": code},
            )
            return self._get_conn(connection, order.order_id)
        return self._transition_conn(
            connection,
            order,
            M.QUARANTINED,
            event_type="ORDER_QUARANTINED",
            actor=actor,
            reason=message,
            external_event_id=external_event_id,
            updates=merged,
            metadata={"code": code},
        )

    def _update_order_conn(
        self,
        connection: sqlite3.Connection,
        order_id: str,
        updates: dict[str, Any],
    ) -> None:
        allowed = {
            "approved_quantity",
            "filled_quantity",
            "average_fill_price",
            "status",
            "broker_order_id",
            "submission_token",
            "risk_decision_id",
            "last_error_code",
            "last_error_message",
        }
        values = {key: value for key, value in updates.items() if key in allowed}
        if not values:
            return
        values["updated_at"] = self._now()
        assignments = ",".join(f"{key}=?" for key in values)
        parameters = list(values.values()) + [order_id]
        connection.execute(
            f"UPDATE oms_orders SET {assignments}, version=version+1 WHERE order_id=?",
            parameters,
        )

    def _append_transition_conn(
        self,
        connection: sqlite3.Connection,
        order_id: str,
        *,
        from_status: str,
        to_status: str,
        event_type: str,
        actor: str,
        reason: str = "",
        external_event_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if external_event_id:
            existing = connection.execute(
                "SELECT transition_id FROM oms_transitions WHERE order_id=? AND external_event_id=?",
                (order_id, external_event_id),
            ).fetchone()
            if existing is not None:
                return
        sequence = int(connection.execute(
            "SELECT COALESCE(MAX(sequence),0)+1 n FROM oms_transitions WHERE order_id=?",
            (order_id,),
        ).fetchone()["n"])
        fingerprint = f"{order_id}:{sequence}:{from_status}:{to_status}:{event_type}"
        transition_id = f"trn-{hashlib.sha256(fingerprint.encode()).hexdigest()[:20]}"
        connection.execute(
            """INSERT INTO oms_transitions
               (transition_id,order_id,sequence,from_status,to_status,event_type,event_at,
                actor,reason,external_event_id,metadata_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                transition_id,
                order_id,
                sequence,
                from_status,
                to_status,
                event_type,
                self._now(),
                actor,
                reason,
                external_event_id or None,
                json.dumps(dict(metadata or {}), sort_keys=True, default=str),
            ),
        )

    def _external_event_order_conn(
        self,
        connection: sqlite3.Connection,
        order_id: str,
        external_event_id: str,
    ) -> M.OrderSnapshot | None:
        if not external_event_id:
            return None
        row = connection.execute(
            "SELECT 1 FROM oms_transitions WHERE order_id=? AND external_event_id=?",
            (order_id, external_event_id),
        ).fetchone()
        return self._get_conn(connection, order_id) if row is not None else None

    def _get_conn(self, connection: sqlite3.Connection, order_id: str) -> M.OrderSnapshot:
        row = connection.execute(
            "SELECT * FROM oms_orders WHERE order_id=?",
            (order_id,),
        ).fetchone()
        if row is None:
            raise M.OrderNotFound(order_id)
        return self._order_from_row(row)

    def _get_by_idempotency_conn(
        self,
        connection: sqlite3.Connection,
        idempotency_key: str,
    ) -> M.OrderSnapshot | None:
        row = connection.execute(
            "SELECT * FROM oms_orders WHERE idempotency_key=?",
            (idempotency_key,),
        ).fetchone()
        return self._order_from_row(row) if row is not None else None

    @staticmethod
    def _order_from_row(row: sqlite3.Row) -> M.OrderSnapshot:
        return M.OrderSnapshot(
            order_id=str(row["order_id"]),
            idempotency_key=str(row["idempotency_key"]),
            trade_intent_id=str(row["trade_intent_id"]),
            intent_hash=str(row["intent_hash"]),
            target_portfolio_id=str(row["target_portfolio_id"]),
            target_position_id=str(row["target_position_id"]),
            strategy_id=str(row["strategy_id"]),
            strategy_version=int(row["strategy_version"]),
            symbol=str(row["symbol"]),
            side=str(row["side"]),
            requested_quantity=int(row["requested_quantity"]),
            approved_quantity=int(row["approved_quantity"]),
            filled_quantity=int(row["filled_quantity"]),
            average_fill_price=float(row["average_fill_price"]),
            intended_entry=float(row["intended_entry"]),
            stop_price=float(row["stop_price"]),
            target_price=float(row["target_price"]),
            intended_risk_pct=float(row["intended_risk_pct"]),
            max_capital=float(row["max_capital"]),
            status=str(row["status"]),
            broker_order_id=str(row["broker_order_id"] or ""),
            submission_token=str(row["submission_token"] or ""),
            risk_decision_id=str(row["risk_decision_id"] or ""),
            protection_required=bool(row["protection_required"]),
            version=int(row["version"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_error_code=str(row["last_error_code"] or ""),
            last_error_message=str(row["last_error_message"] or ""),
        )

    @staticmethod
    def _transition_from_row(row: sqlite3.Row) -> M.TransitionSnapshot:
        return M.TransitionSnapshot(
            transition_id=str(row["transition_id"]),
            order_id=str(row["order_id"]),
            sequence=int(row["sequence"]),
            from_status=str(row["from_status"]),
            to_status=str(row["to_status"]),
            event_type=str(row["event_type"]),
            event_at=str(row["event_at"]),
            actor=str(row["actor"]),
            reason=str(row["reason"] or ""),
            external_event_id=str(row["external_event_id"] or ""),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    @staticmethod
    def _fill_from_row(row: sqlite3.Row) -> M.FillSnapshot:
        return M.FillSnapshot(
            fill_id=str(row["fill_id"]),
            order_id=str(row["order_id"]),
            external_fill_id=str(row["external_fill_id"]),
            quantity=int(row["quantity"]),
            price=float(row["price"]),
            filled_at=str(row["filled_at"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def _now(self) -> str:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @contextmanager
    def _transaction(self):
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
