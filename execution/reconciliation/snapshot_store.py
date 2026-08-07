"""Append-only durable store for broker reconciliation snapshots.

Complete and incomplete captures are both preserved. An incomplete snapshot remains evidence of
an unavailable lane; it is never promoted to the latest complete state or rewritten as an empty
account.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from execution.reconciliation.zerodha_snapshot import ZerodhaSnapshotBundle

_SCHEMA = """
CREATE TABLE IF NOT EXISTS broker_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    observed_at TEXT NOT NULL,
    source TEXT NOT NULL,
    account_complete INTEGER NOT NULL,
    protections_complete INTEGER NOT NULL,
    bundle_complete INTEGER NOT NULL,
    bundle_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_broker_snapshots_observed
    ON broker_snapshots(observed_at, snapshot_id);
CREATE INDEX IF NOT EXISTS idx_broker_snapshots_complete
    ON broker_snapshots(bundle_complete, observed_at);
"""


class BrokerSnapshotConflict(RuntimeError):
    pass


class BrokerSnapshotStore:
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

    def record(self, bundle: ZerodhaSnapshotBundle) -> ZerodhaSnapshotBundle:
        payload = json.dumps(
            bundle.as_dict(), sort_keys=True, separators=(",", ":"), default=str
        )
        snapshot_id = bundle.account.snapshot_id
        if not snapshot_id:
            raise ValueError("broker snapshot id is required")
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT bundle_json FROM broker_snapshots WHERE snapshot_id=?",
                    (snapshot_id,),
                ).fetchone()
                if row is not None:
                    if str(row["bundle_json"]) != payload:
                        raise BrokerSnapshotConflict(
                            f"snapshot id {snapshot_id} already owns different content"
                        )
                    connection.commit()
                    return bundle
                connection.execute(
                    """INSERT INTO broker_snapshots
                       (snapshot_id,observed_at,source,account_complete,protections_complete,
                        bundle_complete,bundle_json,recorded_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        snapshot_id,
                        bundle.account.observed_at,
                        bundle.account.source,
                        1 if bundle.account.complete else 0,
                        1 if bundle.protections_complete else 0,
                        1 if bundle.complete else 0,
                        payload,
                        self._now(),
                    ),
                )
                connection.commit()
                return bundle
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def latest(self) -> dict[str, Any] | None:
        return self._latest_where("")

    def latest_account_complete(self) -> dict[str, Any] | None:
        return self._latest_where("WHERE account_complete=1")

    def latest_complete(self) -> dict[str, Any] | None:
        return self._latest_where("WHERE bundle_complete=1")

    def get(self, snapshot_id: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT bundle_json,recorded_at FROM broker_snapshots WHERE snapshot_id=?",
                (snapshot_id,),
            ).fetchone()
            return self._decode(row) if row is not None else None
        finally:
            connection.close()

    def list_snapshots(self, *, limit: int = 100) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT bundle_json,recorded_at FROM broker_snapshots "
                "ORDER BY observed_at DESC,snapshot_id DESC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
            return [self._decode(row) for row in rows]
        finally:
            connection.close()

    def summary(self) -> dict[str, Any]:
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT COUNT(*) n,
                          COALESCE(SUM(account_complete),0) account_n,
                          COALESCE(SUM(protections_complete),0) protection_n,
                          COALESCE(SUM(bundle_complete),0) complete_n
                   FROM broker_snapshots"""
            ).fetchone()
            latest = self.latest()
            latest_complete = self.latest_complete()
            return {
                "snapshots": int(row["n"]),
                "account_complete_snapshots": int(row["account_n"]),
                "protection_complete_snapshots": int(row["protection_n"]),
                "complete_snapshots": int(row["complete_n"]),
                "latest_snapshot_id": (
                    str(latest["account"]["snapshot_id"]) if latest else ""
                ),
                "latest_complete_snapshot_id": (
                    str(latest_complete["account"]["snapshot_id"])
                    if latest_complete else ""
                ),
            }
        finally:
            connection.close()

    def _latest_where(self, clause: str) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                f"SELECT bundle_json,recorded_at FROM broker_snapshots {clause} "
                "ORDER BY observed_at DESC,snapshot_id DESC LIMIT 1"
            ).fetchone()
            return self._decode(row) if row is not None else None
        finally:
            connection.close()

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return {
            **json.loads(row["bundle_json"]),
            "recorded_at": str(row["recorded_at"]),
        }

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


def capture_and_store_zerodha_snapshot(
    store: BrokerSnapshotStore,
    client=None,
    *,
    observed_at: datetime | str | None = None,
) -> ZerodhaSnapshotBundle:
    """Capture one read-only Zerodha bundle and preserve it exactly once."""
    from execution.reconciliation.zerodha_snapshot import capture_zerodha_snapshot

    return store.record(capture_zerodha_snapshot(client, observed_at=observed_at))
