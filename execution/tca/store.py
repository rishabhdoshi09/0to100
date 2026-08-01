"""Append-only durable store for transaction-cost assessments."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from execution.tca.models import EntryExecutionAssessment

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tca_assessments (
    assessment_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    trade_intent_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    complete INTEGER NOT NULL,
    implementation_shortfall REAL NOT NULL,
    implementation_shortfall_bps REAL NOT NULL,
    assessment_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tca_order ON tca_assessments(order_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_tca_strategy ON tca_assessments(strategy_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_tca_symbol ON tca_assessments(symbol, recorded_at);
"""


class TcaAssessmentConflict(RuntimeError):
    pass


class TcaStore:
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

    def record(self, assessment: EntryExecutionAssessment) -> EntryExecutionAssessment:
        payload = json.dumps(
            assessment.as_dict(), sort_keys=True, separators=(",", ":"), default=str
        )
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT assessment_json FROM tca_assessments WHERE assessment_id=?",
                    (assessment.assessment_id,),
                ).fetchone()
                if row is not None:
                    if str(row["assessment_json"]) != payload:
                        raise TcaAssessmentConflict(
                            f"assessment id {assessment.assessment_id} owns different content"
                        )
                    connection.commit()
                    return assessment
                connection.execute(
                    """INSERT INTO tca_assessments
                       (assessment_id,order_id,trade_intent_id,strategy_id,symbol,quantity,
                        complete,implementation_shortfall,implementation_shortfall_bps,
                        assessment_json,recorded_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        assessment.assessment_id,
                        assessment.order_id,
                        assessment.trade_intent_id,
                        assessment.strategy_id,
                        assessment.symbol,
                        assessment.quantity,
                        1 if assessment.complete else 0,
                        assessment.implementation_shortfall,
                        assessment.implementation_shortfall_bps,
                        payload,
                        self._now(),
                    ),
                )
                connection.commit()
                return assessment
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def latest_for_order(self, order_id: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT assessment_json,recorded_at FROM tca_assessments "
                "WHERE order_id=? ORDER BY recorded_at DESC,assessment_id DESC LIMIT 1",
                (order_id,),
            ).fetchone()
            if row is None:
                return None
            return {**json.loads(row["assessment_json"]), "recorded_at": str(row["recorded_at"])}
        finally:
            connection.close()

    def list_assessments(self, *, strategy_id: str = "", symbol: str = ""):
        clauses: list[str] = []
        params: list[str] = []
        if strategy_id:
            clauses.append("strategy_id=?")
            params.append(strategy_id)
        if symbol:
            clauses.append("symbol=?")
            params.append(symbol.upper())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        connection = self._connect()
        try:
            rows = connection.execute(
                f"SELECT assessment_json,recorded_at FROM tca_assessments{where} "
                "ORDER BY recorded_at,assessment_id",
                params,
            ).fetchall()
            return [
                {**json.loads(row["assessment_json"]), "recorded_at": str(row["recorded_at"])}
                for row in rows
            ]
        finally:
            connection.close()

    def summary(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT COUNT(*) n,
                          COALESCE(SUM(implementation_shortfall),0) cost,
                          COALESCE(AVG(implementation_shortfall_bps),0) avg_bps,
                          COALESCE(SUM(CASE WHEN complete=1 THEN 1 ELSE 0 END),0) complete_n
                   FROM tca_assessments"""
            ).fetchone()
            return {
                "assessments": int(row["n"]),
                "complete_assessments": int(row["complete_n"]),
                "total_implementation_shortfall": float(row["cost"]),
                "average_implementation_shortfall_bps": float(row["avg_bps"]),
            }
        finally:
            connection.close()

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _now(self):
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
