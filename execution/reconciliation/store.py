"""Append-only durable store for broker reconciliation reports."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from execution.reconciliation.models import ReconciliationReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reconciliation_reports (
    report_id TEXT PRIMARY KEY,
    broker_snapshot_id TEXT NOT NULL,
    status TEXT NOT NULL,
    entry_freeze_required INTEGER NOT NULL,
    report_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reconciliation_snapshot
    ON reconciliation_reports(broker_snapshot_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_reconciliation_status
    ON reconciliation_reports(status, recorded_at);
"""


class ReconciliationReportConflict(RuntimeError):
    pass


class ReconciliationReportStore:
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

    def record(self, report: ReconciliationReport) -> ReconciliationReport:
        payload = json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT report_json FROM reconciliation_reports WHERE report_id=?",
                    (report.report_id,),
                ).fetchone()
                if row is not None:
                    if str(row["report_json"]) != payload:
                        raise ReconciliationReportConflict(
                            f"report id {report.report_id} already owns different content"
                        )
                    connection.commit()
                    return report
                connection.execute(
                    """INSERT INTO reconciliation_reports
                       (report_id,broker_snapshot_id,status,entry_freeze_required,
                        report_json,recorded_at)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        report.report_id,
                        report.broker_snapshot_id,
                        report.status,
                        1 if report.entry_freeze_required else 0,
                        payload,
                        self._now(),
                    ),
                )
                connection.commit()
                return report
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def latest(self) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT report_json, recorded_at FROM reconciliation_reports "
                "ORDER BY recorded_at DESC, report_id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return {**json.loads(row["report_json"]), "recorded_at": str(row["recorded_at"])}
        finally:
            connection.close()

    def for_snapshot(self, broker_snapshot_id: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT report_json, recorded_at FROM reconciliation_reports "
                "WHERE broker_snapshot_id=? ORDER BY recorded_at, report_id",
                (broker_snapshot_id,),
            ).fetchall()
            return [
                {**json.loads(row["report_json"]), "recorded_at": str(row["recorded_at"])}
                for row in rows
            ]
        finally:
            connection.close()

    def summary(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            total = int(connection.execute(
                "SELECT COUNT(*) n FROM reconciliation_reports"
            ).fetchone()["n"])
            rows = connection.execute(
                "SELECT status, COUNT(*) n FROM reconciliation_reports GROUP BY status"
            ).fetchall()
            latest = self.latest()
            return {
                "reports": total,
                "by_status": {str(row["status"]): int(row["n"]) for row in rows},
                "latest_status": str(latest.get("status", "")) if latest else "",
                "entry_freeze_required": bool(latest.get("entry_freeze_required")) if latest else True,
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
