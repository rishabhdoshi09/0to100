"""Append-only durable store for independent Risk Governor decisions."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from risk.governor import GovernorDecision

_SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_decisions (
    decision_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    state_snapshot_id TEXT NOT NULL,
    action TEXT NOT NULL,
    approved_quantity INTEGER NOT NULL,
    requested_quantity INTEGER NOT NULL,
    decision_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_risk_decisions_order ON risk_decisions(order_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_risk_decisions_action ON risk_decisions(action, recorded_at);
"""


class RiskDecisionConflict(RuntimeError):
    pass


class RiskDecisionStore:
    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
            connection.commit()
        finally:
            connection.close()

    def record(self, decision: GovernorDecision) -> GovernorDecision:
        payload = json.dumps(decision.as_dict(), sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT decision_json FROM risk_decisions WHERE decision_id=?",
                    (decision.decision_id,),
                ).fetchone()
                if row is not None:
                    if str(row["decision_json"]) != payload:
                        raise RiskDecisionConflict(
                            f"decision id {decision.decision_id} already owns different content"
                        )
                    connection.commit()
                    return decision
                connection.execute(
                    """INSERT INTO risk_decisions
                       (decision_id,order_id,state_snapshot_id,action,approved_quantity,
                        requested_quantity,decision_json,recorded_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        decision.decision_id,
                        decision.order_id,
                        decision.state_snapshot_id,
                        decision.action,
                        decision.approved_quantity,
                        decision.requested_quantity,
                        payload,
                        self._now(),
                    ),
                )
                connection.commit()
                return decision
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def for_order(self, order_id: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT decision_json, recorded_at FROM risk_decisions "
                "WHERE order_id=? ORDER BY recorded_at, decision_id",
                (order_id,),
            ).fetchall()
            return [
                {**json.loads(row["decision_json"]), "recorded_at": str(row["recorded_at"])}
                for row in rows
            ]
        finally:
            connection.close()

    def latest(self, order_id: str) -> dict[str, Any] | None:
        decisions = self.for_order(order_id)
        return decisions[-1] if decisions else None

    def summary(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            total = int(connection.execute("SELECT COUNT(*) n FROM risk_decisions").fetchone()["n"])
            rows = connection.execute(
                "SELECT action, COUNT(*) n FROM risk_decisions GROUP BY action"
            ).fetchall()
            return {
                "decisions": total,
                "by_action": {str(row["action"]): int(row["n"]) for row in rows},
            }
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _now(self) -> str:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
