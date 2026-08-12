"""Durable broker-neutral Protection Manager state.

Protection plans track the exact filled quantity that requires exchange-side protection. The
store does not call a broker; it persists requests, acknowledgements, verification, quantity
adjustments, cancellation and recovery state across restarts.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from execution.protection import models as P

_SCHEMA = """
CREATE TABLE IF NOT EXISTS protection_plans (
    plan_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    required_quantity INTEGER NOT NULL CHECK(required_quantity >= 0),
    protected_quantity INTEGER NOT NULL DEFAULT 0 CHECK(protected_quantity >= 0),
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    status TEXT NOT NULL,
    request_token TEXT UNIQUE,
    broker_protection_id TEXT UNIQUE,
    stop_reference TEXT,
    target_reference TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_verified_at TEXT,
    last_error_code TEXT NOT NULL DEFAULT '',
    last_error_message TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS protection_transitions (
    transition_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES protection_plans(plan_id),
    sequence INTEGER NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_at TEXT NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    external_event_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(plan_id, sequence),
    UNIQUE(plan_id, external_event_id)
);
CREATE INDEX IF NOT EXISTS idx_protection_status ON protection_plans(status);
CREATE INDEX IF NOT EXISTS idx_protection_symbol ON protection_plans(symbol);
CREATE INDEX IF NOT EXISTS idx_protection_transitions
    ON protection_transitions(plan_id, sequence);
"""


class ProtectionStore:
    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.commit()

    def ensure_for_order(self, order, *, actor: str = "protection_manager") -> P.ProtectionPlanSnapshot:
        """Create or resize the plan to the OMS order's cumulative filled quantity."""
        filled = int(getattr(order, "filled_quantity", 0) or 0)
        stop = float(getattr(order, "stop_price", 0) or 0)
        target = float(getattr(order, "target_price", 0) or 0)
        symbol = str(getattr(order, "symbol", "") or "").upper()
        order_id = str(getattr(order, "order_id", "") or "")
        if not order_id or not symbol or filled <= 0:
            raise P.InvalidProtectionPlan("a protection plan requires an identified filled order")
        if not (math.isfinite(stop) and math.isfinite(target) and stop > 0 and target > stop):
            raise P.InvalidProtectionPlan("valid stop and target prices are required")
        plan_id = f"protect-{hashlib.sha256(order_id.encode()).hexdigest()[:20]}"
        now = self._now()

        with self._transaction() as connection:
            existing = self._get_by_order_conn(connection, order_id)
            if existing is None:
                connection.execute(
                    """INSERT INTO protection_plans
                       (plan_id,order_id,symbol,required_quantity,protected_quantity,
                        stop_price,target_price,status,version,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        plan_id,
                        order_id,
                        symbol,
                        filled,
                        0,
                        stop,
                        target,
                        P.REQUIRED,
                        1,
                        now,
                        now,
                    ),
                )
                self._append_transition_conn(
                    connection,
                    plan_id,
                    from_status="",
                    to_status=P.REQUIRED,
                    event_type="PROTECTION_REQUIRED",
                    actor=actor,
                    metadata={"required_quantity": filled},
                )
                return self._get_conn(connection, plan_id)

            identity_mismatch = (
                existing.symbol != symbol
                or abs(existing.stop_price - stop) > 1e-9
                or abs(existing.target_price - target) > 1e-9
            )
            if identity_mismatch:
                return self._transition_conn(
                    connection,
                    existing,
                    P.RECOVERY_REQUIRED,
                    event_type="PROTECTION_IDENTITY_CONFLICT",
                    actor=actor,
                    reason="OMS order protection terms changed after plan creation",
                    updates={
                        "last_error_code": "PROTECTION_IDENTITY_CONFLICT",
                        "last_error_message": "symbol, stop or target changed",
                    },
                    metadata={
                        "existing": {
                            "symbol": existing.symbol,
                            "stop_price": existing.stop_price,
                            "target_price": existing.target_price,
                        },
                        "received": {"symbol": symbol, "stop_price": stop, "target_price": target},
                    },
                )
            if filled < existing.required_quantity:
                return self._transition_conn(
                    connection,
                    existing,
                    P.RECOVERY_REQUIRED,
                    event_type="FILLED_QUANTITY_REGRESSED",
                    actor=actor,
                    reason="OMS cumulative filled quantity moved backwards",
                    updates={
                        "last_error_code": "FILLED_QUANTITY_REGRESSED",
                        "last_error_message": (
                            f"existing={existing.required_quantity}; received={filled}"
                        ),
                    },
                )
            if filled == existing.required_quantity:
                return existing

            updates = {"required_quantity": filled}
            if existing.status in {P.ACTIVE, P.VERIFIED}:
                target_status = P.ADJUSTMENT_REQUIRED
                event_type = "PROTECTION_QUANTITY_INCREASED"
            elif existing.status == P.SUBMISSION_PENDING:
                target_status = P.RECOVERY_REQUIRED
                event_type = "PENDING_REQUEST_QUANTITY_STALE"
                updates.update(
                    last_error_code="PENDING_REQUEST_QUANTITY_STALE",
                    last_error_message="filled quantity increased while protection request was pending",
                )
            elif existing.status == P.CANCELLED:
                target_status = P.RECOVERY_REQUIRED
                event_type = "CANCELLED_PLAN_HAS_NEW_FILL"
                updates.update(
                    last_error_code="CANCELLED_PLAN_HAS_NEW_FILL",
                    last_error_message="new fill arrived after protection cancellation",
                )
            else:
                self._update_plan_conn(connection, plan_id, updates)
                self._append_transition_conn(
                    connection,
                    plan_id,
                    from_status=existing.status,
                    to_status=existing.status,
                    event_type="REQUIRED_QUANTITY_UPDATED",
                    actor=actor,
                    metadata={
                        "previous_required_quantity": existing.required_quantity,
                        "required_quantity": filled,
                    },
                )
                return self._get_conn(connection, plan_id)
            return self._transition_conn(
                connection,
                existing,
                target_status,
                event_type=event_type,
                actor=actor,
                updates=updates,
                metadata={
                    "previous_required_quantity": existing.required_quantity,
                    "required_quantity": filled,
                },
            )

    def prepare_submission(
        self,
        plan_id: str,
        *,
        request_token: str,
        actor: str = "protection_adapter",
    ) -> P.ProtectionPlanSnapshot:
        if not request_token:
            raise P.InvalidProtectionPlan("request_token is required")
        with self._transaction() as connection:
            plan = self._get_conn(connection, plan_id)
            if plan.status == P.SUBMISSION_PENDING and plan.request_token == request_token:
                return plan
            return self._transition_conn(
                connection,
                plan,
                P.SUBMISSION_PENDING,
                event_type="PROTECTION_SUBMISSION_PREPARED",
                actor=actor,
                updates={"request_token": request_token},
                metadata={"request_token": request_token, "required_quantity": plan.required_quantity},
            )

    def acknowledge(
        self,
        plan_id: str,
        *,
        broker_protection_id: str,
        protected_quantity: int,
        stop_reference: str,
        target_reference: str = "",
        external_event_id: str = "",
        actor: str = "protection_adapter",
    ) -> P.ProtectionPlanSnapshot:
        if not broker_protection_id or not stop_reference:
            raise P.InvalidProtectionPlan("broker protection and stop references are required")
        protected_quantity = int(protected_quantity)
        if protected_quantity <= 0:
            raise P.InvalidProtectionPlan("protected quantity must be positive")
        with self._transaction() as connection:
            duplicate = self._external_event_plan_conn(connection, plan_id, external_event_id)
            if duplicate is not None:
                return duplicate
            plan = self._get_conn(connection, plan_id)
            updates = {
                "broker_protection_id": broker_protection_id,
                "protected_quantity": protected_quantity,
                "stop_reference": stop_reference,
                "target_reference": target_reference,
            }
            if protected_quantity > plan.required_quantity:
                return self._transition_conn(
                    connection,
                    plan,
                    P.RECOVERY_REQUIRED,
                    event_type="PROTECTION_OVER_COVER",
                    actor=actor,
                    reason="broker protection quantity exceeds filled quantity",
                    external_event_id=external_event_id,
                    updates={
                        **updates,
                        "last_error_code": "PROTECTION_OVER_COVER",
                        "last_error_message": (
                            f"protected={protected_quantity}; required={plan.required_quantity}"
                        ),
                    },
                )
            status = P.ACTIVE if protected_quantity == plan.required_quantity else P.ADJUSTMENT_REQUIRED
            return self._transition_conn(
                connection,
                plan,
                status,
                event_type="PROTECTION_ACKNOWLEDGED",
                actor=actor,
                external_event_id=external_event_id,
                updates=updates,
                metadata={
                    "broker_protection_id": broker_protection_id,
                    "protected_quantity": protected_quantity,
                },
            )

    def verify(
        self,
        plan_id: str,
        broker: P.BrokerProtectionSnapshot,
        *,
        price_tolerance: float = 0.01,
        external_event_id: str = "",
        actor: str = "protection_reconciliation",
    ) -> P.ProtectionPlanSnapshot:
        with self._transaction() as connection:
            duplicate = self._external_event_plan_conn(connection, plan_id, external_event_id)
            if duplicate is not None:
                return duplicate
            plan = self._get_conn(connection, plan_id)
            updates = {
                "broker_protection_id": broker.broker_protection_id,
                "protected_quantity": max(0, int(broker.quantity)),
                "stop_reference": broker.stop_reference,
                "target_reference": broker.target_reference,
            }
            conflicts: list[str] = []
            if broker.order_id and broker.order_id != plan.order_id:
                conflicts.append("ORDER_ID_MISMATCH")
            if broker.symbol.upper() != plan.symbol:
                conflicts.append("SYMBOL_MISMATCH")
            if not broker.active:
                conflicts.append("PROTECTION_INACTIVE")
            if abs(float(broker.stop_price) - plan.stop_price) > price_tolerance:
                conflicts.append("STOP_PRICE_MISMATCH")
            if abs(float(broker.target_price) - plan.target_price) > price_tolerance:
                conflicts.append("TARGET_PRICE_MISMATCH")
            if not broker.stop_reference:
                conflicts.append("STOP_REFERENCE_MISSING")
            if broker.quantity > plan.required_quantity:
                conflicts.append("PROTECTION_OVER_COVER")
            if conflicts:
                return self._transition_conn(
                    connection,
                    plan,
                    P.RECOVERY_REQUIRED,
                    event_type="PROTECTION_VERIFICATION_FAILED",
                    actor=actor,
                    reason="; ".join(conflicts),
                    external_event_id=external_event_id,
                    updates={
                        **updates,
                        "last_error_code": conflicts[0],
                        "last_error_message": "; ".join(conflicts),
                    },
                    metadata={"broker": broker.as_dict(), "conflicts": conflicts},
                )
            if broker.quantity < plan.required_quantity:
                return self._transition_conn(
                    connection,
                    plan,
                    P.ADJUSTMENT_REQUIRED,
                    event_type="PROTECTION_UNDER_COVERED",
                    actor=actor,
                    reason="broker protection quantity is below cumulative filled quantity",
                    external_event_id=external_event_id,
                    updates=updates,
                    metadata={
                        "protected_quantity": broker.quantity,
                        "required_quantity": plan.required_quantity,
                    },
                )
            updates.update(
                last_verified_at=self._now(),
                last_error_code="",
                last_error_message="",
            )
            if plan.status == P.VERIFIED:
                self._update_plan_conn(connection, plan_id, updates)
                self._append_transition_conn(
                    connection,
                    plan_id,
                    from_status=P.VERIFIED,
                    to_status=P.VERIFIED,
                    event_type="PROTECTION_REVERIFIED",
                    actor=actor,
                    external_event_id=external_event_id,
                    metadata={"broker_protection_id": broker.broker_protection_id},
                )
                return self._get_conn(connection, plan_id)
            return self._transition_conn(
                connection,
                plan,
                P.VERIFIED,
                event_type="PROTECTION_VERIFIED",
                actor=actor,
                external_event_id=external_event_id,
                updates=updates,
                metadata={"broker_protection_id": broker.broker_protection_id},
            )

    def request_cancel(
        self,
        plan_id: str,
        *,
        reason: str,
        actor: str = "protection_manager",
    ) -> P.ProtectionPlanSnapshot:
        with self._transaction() as connection:
            plan = self._get_conn(connection, plan_id)
            return self._transition_conn(
                connection,
                plan,
                P.CANCEL_PENDING,
                event_type="PROTECTION_CANCEL_REQUESTED",
                actor=actor,
                reason=reason,
            )

    def mark_cancelled(
        self,
        plan_id: str,
        *,
        external_event_id: str = "",
        actor: str = "protection_adapter",
    ) -> P.ProtectionPlanSnapshot:
        with self._transaction() as connection:
            duplicate = self._external_event_plan_conn(connection, plan_id, external_event_id)
            if duplicate is not None:
                return duplicate
            plan = self._get_conn(connection, plan_id)
            return self._transition_conn(
                connection,
                plan,
                P.CANCELLED,
                event_type="PROTECTION_CANCELLED",
                actor=actor,
                external_event_id=external_event_id,
                updates={"protected_quantity": 0},
            )

    def mark_failed(
        self,
        plan_id: str,
        *,
        code: str,
        message: str,
        actor: str = "protection_adapter",
    ) -> P.ProtectionPlanSnapshot:
        with self._transaction() as connection:
            plan = self._get_conn(connection, plan_id)
            return self._transition_conn(
                connection,
                plan,
                P.FAILED,
                event_type="PROTECTION_FAILED",
                actor=actor,
                reason=message,
                updates={"last_error_code": code, "last_error_message": message},
            )

    def require_recovery(
        self,
        plan_id: str,
        *,
        code: str,
        message: str,
        actor: str = "protection_reconciliation",
    ) -> P.ProtectionPlanSnapshot:
        with self._transaction() as connection:
            plan = self._get_conn(connection, plan_id)
            if plan.status == P.RECOVERY_REQUIRED:
                self._update_plan_conn(connection, plan_id, {
                    "last_error_code": code,
                    "last_error_message": message,
                })
                return self._get_conn(connection, plan_id)
            return self._transition_conn(
                connection,
                plan,
                P.RECOVERY_REQUIRED,
                event_type="PROTECTION_RECOVERY_REQUIRED",
                actor=actor,
                reason=message,
                updates={"last_error_code": code, "last_error_message": message},
            )

    def record_orphan(
        self,
        broker: P.BrokerProtectionSnapshot,
        *,
        actor: str = "protection_reconciliation",
    ) -> P.ProtectionPlanSnapshot:
        if not broker.broker_protection_id:
            raise P.InvalidProtectionPlan("orphan protection requires a broker reference")
        order_id = f"orphan:{broker.broker_protection_id}"
        plan_id = f"protect-orphan-{hashlib.sha256(broker.broker_protection_id.encode()).hexdigest()[:16]}"
        with self._transaction() as connection:
            existing = self._get_by_order_conn(connection, order_id)
            if existing is not None:
                return existing
            now = self._now()
            connection.execute(
                """INSERT INTO protection_plans
                   (plan_id,order_id,symbol,required_quantity,protected_quantity,
                    stop_price,target_price,status,broker_protection_id,stop_reference,
                    target_reference,version,created_at,updated_at,last_error_code,last_error_message)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    plan_id,
                    order_id,
                    broker.symbol.upper(),
                    max(0, int(broker.quantity)),
                    max(0, int(broker.quantity)),
                    float(broker.stop_price),
                    float(broker.target_price),
                    P.ORPHANED,
                    broker.broker_protection_id,
                    broker.stop_reference,
                    broker.target_reference,
                    1,
                    now,
                    now,
                    "ORPHAN_PROTECTION",
                    "broker protection has no internal owner",
                ),
            )
            self._append_transition_conn(
                connection,
                plan_id,
                from_status="",
                to_status=P.ORPHANED,
                event_type="ORPHAN_PROTECTION_DETECTED",
                actor=actor,
                metadata={"broker": broker.as_dict()},
            )
            return self._get_conn(connection, plan_id)

    def get(self, plan_id: str) -> P.ProtectionPlanSnapshot:
        with self._connect() as connection:
            return self._get_conn(connection, plan_id)

    def get_by_order(self, order_id: str) -> P.ProtectionPlanSnapshot | None:
        with self._connect() as connection:
            return self._get_by_order_conn(connection, order_id)

    def list_plans(self, *, statuses: Iterable[str] | None = None):
        with self._connect() as connection:
            if statuses:
                values = tuple(statuses)
                placeholders = ",".join("?" for _ in values)
                rows = connection.execute(
                    f"SELECT * FROM protection_plans WHERE status IN ({placeholders}) "
                    "ORDER BY created_at,plan_id",
                    values,
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM protection_plans ORDER BY created_at,plan_id"
                ).fetchall()
            return [self._plan_from_row(row) for row in rows]

    def history(self, plan_id: str):
        with self._connect() as connection:
            self._get_conn(connection, plan_id)
            rows = connection.execute(
                "SELECT * FROM protection_transitions WHERE plan_id=? ORDER BY sequence",
                (plan_id,),
            ).fetchall()
            return [self._transition_from_row(row) for row in rows]

    def summary(self):
        plans = self.list_plans()
        counts: dict[str, int] = {}
        for plan in plans:
            counts[plan.status] = counts.get(plan.status, 0) + 1
        unsafe = [
            plan.plan_id for plan in plans
            if plan.required_quantity > 0 and not plan.fully_protected and plan.status != P.CANCELLED
        ]
        return {
            "plans": len(plans),
            "by_status": counts,
            "fully_protected": sum(1 for plan in plans if plan.fully_protected),
            "unsafe_plan_ids": unsafe,
            "entry_freeze_required": bool(unsafe),
        }

    def _transition_conn(
        self,
        connection,
        plan,
        to_status,
        *,
        event_type,
        actor,
        reason="",
        external_event_id="",
        updates=None,
        metadata=None,
    ):
        if to_status not in P.ALL_STATUSES:
            raise P.IllegalProtectionTransition(f"unknown protection status {to_status}")
        if plan.status == to_status:
            if updates:
                self._update_plan_conn(connection, plan.plan_id, updates)
            return self._get_conn(connection, plan.plan_id)
        if to_status not in P.ALLOWED_TRANSITIONS.get(plan.status, frozenset()):
            raise P.IllegalProtectionTransition(
                f"illegal protection transition {plan.status} -> {to_status}"
            )
        merged = dict(updates or {})
        merged["status"] = to_status
        self._update_plan_conn(connection, plan.plan_id, merged)
        self._append_transition_conn(
            connection,
            plan.plan_id,
            from_status=plan.status,
            to_status=to_status,
            event_type=event_type,
            actor=actor,
            reason=reason,
            external_event_id=external_event_id,
            metadata=metadata,
        )
        return self._get_conn(connection, plan.plan_id)

    def _update_plan_conn(self, connection, plan_id, updates):
        allowed = {
            "required_quantity",
            "protected_quantity",
            "status",
            "request_token",
            "broker_protection_id",
            "stop_reference",
            "target_reference",
            "last_verified_at",
            "last_error_code",
            "last_error_message",
        }
        values = {key: value for key, value in dict(updates or {}).items() if key in allowed}
        if not values:
            return
        for nullable in ("request_token", "broker_protection_id", "stop_reference", "target_reference", "last_verified_at"):
            if nullable in values and values[nullable] == "":
                values[nullable] = None
        values["updated_at"] = self._now()
        assignments = ",".join(f"{key}=?" for key in values)
        connection.execute(
            f"UPDATE protection_plans SET {assignments},version=version+1 WHERE plan_id=?",
            [*values.values(), plan_id],
        )

    def _append_transition_conn(
        self,
        connection,
        plan_id,
        *,
        from_status,
        to_status,
        event_type,
        actor,
        reason="",
        external_event_id="",
        metadata=None,
    ):
        if external_event_id:
            existing = connection.execute(
                "SELECT 1 FROM protection_transitions WHERE plan_id=? AND external_event_id=?",
                (plan_id, external_event_id),
            ).fetchone()
            if existing is not None:
                return
        sequence = int(connection.execute(
            "SELECT COALESCE(MAX(sequence),0)+1 n FROM protection_transitions WHERE plan_id=?",
            (plan_id,),
        ).fetchone()["n"])
        fingerprint = f"{plan_id}:{sequence}:{from_status}:{to_status}:{event_type}"
        transition_id = f"ptrn-{hashlib.sha256(fingerprint.encode()).hexdigest()[:20]}"
        connection.execute(
            """INSERT INTO protection_transitions
               (transition_id,plan_id,sequence,from_status,to_status,event_type,event_at,
                actor,reason,external_event_id,metadata_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                transition_id,
                plan_id,
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

    def _external_event_plan_conn(self, connection, plan_id, external_event_id):
        if not external_event_id:
            return None
        row = connection.execute(
            "SELECT 1 FROM protection_transitions WHERE plan_id=? AND external_event_id=?",
            (plan_id, external_event_id),
        ).fetchone()
        return self._get_conn(connection, plan_id) if row is not None else None

    def _get_conn(self, connection, plan_id):
        row = connection.execute(
            "SELECT * FROM protection_plans WHERE plan_id=?",
            (plan_id,),
        ).fetchone()
        if row is None:
            raise P.ProtectionNotFound(plan_id)
        return self._plan_from_row(row)

    def _get_by_order_conn(self, connection, order_id):
        row = connection.execute(
            "SELECT * FROM protection_plans WHERE order_id=?",
            (order_id,),
        ).fetchone()
        return self._plan_from_row(row) if row is not None else None

    @staticmethod
    def _plan_from_row(row):
        return P.ProtectionPlanSnapshot(
            plan_id=str(row["plan_id"]),
            order_id=str(row["order_id"]),
            symbol=str(row["symbol"]),
            required_quantity=int(row["required_quantity"]),
            protected_quantity=int(row["protected_quantity"]),
            stop_price=float(row["stop_price"]),
            target_price=float(row["target_price"]),
            status=str(row["status"]),
            request_token=str(row["request_token"] or ""),
            broker_protection_id=str(row["broker_protection_id"] or ""),
            stop_reference=str(row["stop_reference"] or ""),
            target_reference=str(row["target_reference"] or ""),
            version=int(row["version"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_verified_at=str(row["last_verified_at"] or ""),
            last_error_code=str(row["last_error_code"] or ""),
            last_error_message=str(row["last_error_message"] or ""),
        )

    @staticmethod
    def _transition_from_row(row):
        return P.ProtectionTransitionSnapshot(
            transition_id=str(row["transition_id"]),
            plan_id=str(row["plan_id"]),
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

    def _connect(self):
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

    def _now(self):
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
