"""Durable, cross-process job store for QuantTerm market operations."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"
TERMINAL = frozenset({SUCCEEDED, FAILED, BLOCKED, CANCELLED})


class OperationStore:
    """SQLite-backed queue shared by the API and the market-operations worker.

    Every public method opens a short-lived SQLite connection. This avoids sharing
    connection objects across the API process, worker process and worker threads.
    WAL mode keeps dashboard reads responsive while progress is being written.

    Important: ``with sqlite3.connect(...)`` does **not** close the connection on
    modern CPython — only commit/rollback. Idle lane polling + progress writes
    previously leaked FDs until Errno 24 / "unable to open database file".
    """

    def __init__(self, path: str | Path = "logs/market_ops/jobs.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _raw_connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path), timeout=5.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA busy_timeout=5000")
        return con

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = self._raw_connect()
        try:
            yield con
            con.commit()
        except Exception:
            try:
                con.rollback()
            except Exception:
                pass
            raise
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    requested_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    updated_at REAL NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    worker_pid INTEGER,
                    stage TEXT NOT NULL DEFAULT '',
                    message TEXT NOT NULL DEFAULT '',
                    progress_current INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error_code TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT ''
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_operations_status_lane "
                "ON operations(status, lane, requested_at)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_operations_kind_requested "
                "ON operations(kind, requested_at DESC)"
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        for key in ("payload_json", "result_json"):
            raw = data.pop(key, "{}")
            try:
                data[key.removesuffix("_json")] = json.loads(raw or "{}")
            except Exception:
                data[key.removesuffix("_json")] = {}
        total = int(data.get("progress_total") or 0)
        current = int(data.get("progress_current") or 0)
        data["progress_pct"] = round((current / total) * 100, 1) if total > 0 else None
        return data

    def enqueue(
        self,
        kind: str,
        *,
        lane: str,
        requested_by: str = "terminal",
        payload: dict[str, Any] | None = None,
        deduplicate: bool = True,
        message: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Enqueue one operation and return ``(record, created)``.

        Repeated button clicks return the existing pending/running operation rather
        than creating queue spam. A new click after a terminal result creates a new run.
        """
        kind = str(kind).strip().upper()
        lane = str(lane).strip().lower()
        if not kind or not lane:
            raise ValueError("kind and lane are required")
        now = time.time()
        operation_id = uuid.uuid4().hex
        queue_message = str(
            message
            or "Queued and waiting for the dedicated market-operations worker"
        )
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if deduplicate:
                row = con.execute(
                    "SELECT * FROM operations WHERE kind=? AND status IN (?,?) "
                    "ORDER BY requested_at DESC LIMIT 1",
                    (kind, PENDING, RUNNING),
                ).fetchone()
                if row is not None:
                    # Refresh the visible queue message on re-click so OFFLINE/ONLINE
                    # transparency is not stuck on the bootstrap default text.
                    if message and str(row["status"]) == PENDING:
                        con.execute(
                            "UPDATE operations SET message=?, updated_at=?, requested_by=? "
                            "WHERE operation_id=? AND status=?",
                            (
                                queue_message,
                                now,
                                str(requested_by or "terminal"),
                                str(row["operation_id"]),
                                PENDING,
                            ),
                        )
                        row = con.execute(
                            "SELECT * FROM operations WHERE operation_id=?",
                            (str(row["operation_id"]),),
                        ).fetchone()
                    con.commit()
                    return self._decode(row) or {}, False
            con.execute(
                """
                INSERT INTO operations (
                    operation_id,kind,lane,status,requested_by,requested_at,updated_at,
                    payload_json,message
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    operation_id,
                    kind,
                    lane,
                    PENDING,
                    str(requested_by or "terminal"),
                    now,
                    now,
                    json.dumps(payload or {}, default=str),
                    queue_message,
                ),
            )
            row = con.execute(
                "SELECT * FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            con.commit()
        return self._decode(row) or {}, True

    def lease_next(self, lane: str, *, worker_pid: int) -> dict[str, Any] | None:
        now = time.time()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT * FROM operations WHERE lane=? AND status=? "
                "ORDER BY requested_at ASC LIMIT 1",
                (lane, PENDING),
            ).fetchone()
            if row is None:
                con.commit()
                return None
            operation_id = str(row["operation_id"])
            con.execute(
                "UPDATE operations SET status=?,started_at=COALESCE(started_at,?),"
                "updated_at=?,attempt=attempt+1,worker_pid=?,stage=?,message=? "
                "WHERE operation_id=? AND status=?",
                (
                    RUNNING,
                    now,
                    now,
                    int(worker_pid),
                    "ACCEPTED",
                    "Worker accepted the job — next stage updates will show live progress",
                    operation_id,
                    PENDING,
                ),
            )
            leased = con.execute(
                "SELECT * FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            con.commit()
        return self._decode(leased)

    def progress(
        self,
        operation_id: str,
        *,
        stage: str,
        message: str,
        current: int | None = None,
        total: int | None = None,
    ) -> None:
        fields = ["stage=?", "message=?", "updated_at=?"]
        values: list[Any] = [str(stage), str(message), time.time()]
        if current is not None:
            fields.append("progress_current=?")
            values.append(max(0, int(current)))
        if total is not None:
            fields.append("progress_total=?")
            values.append(max(0, int(total)))
        values.append(operation_id)
        with self._connect() as con:
            con.execute(
                f"UPDATE operations SET {','.join(fields)} WHERE operation_id=? AND status=?",
                (*values, RUNNING),
            )

    def finish(
        self,
        operation_id: str,
        *,
        status: str,
        message: str,
        result: dict[str, Any] | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> None:
        if status not in TERMINAL:
            raise ValueError(f"invalid terminal status: {status}")
        now = time.time()
        with self._connect() as con:
            con.execute(
                """
                UPDATE operations SET status=?,finished_at=?,updated_at=?,stage=?,message=?,
                    result_json=?,error_code=?,error_message=?,
                    progress_current=CASE WHEN progress_total>0 AND ?=? THEN progress_total
                                          ELSE progress_current END
                WHERE operation_id=?
                """,
                (
                    status,
                    now,
                    now,
                    status,
                    str(message),
                    json.dumps(result or {}, default=str),
                    str(error_code or ""),
                    str(error_message or ""),
                    status,
                    SUCCEEDED,
                    operation_id,
                ),
            )

    def recover_orphans(self) -> int:
        """Requeue operations left RUNNING by a crashed worker process."""
        now = time.time()
        with self._connect() as con:
            cur = con.execute(
                "UPDATE operations SET status=?,updated_at=?,stage=?,message=?,worker_pid=NULL "
                "WHERE status=?",
                (
                    PENDING,
                    now,
                    "RECOVERED",
                    "Previous worker stopped before completion; operation requeued",
                    RUNNING,
                ),
            )
            return int(cur.rowcount or 0)

    def get(self, operation_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        return self._decode(row)

    def latest(self, kind: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM operations WHERE kind=? ORDER BY requested_at DESC LIMIT 1",
                (str(kind).upper(),),
            ).fetchone()
        return self._decode(row)

    def latest_by_kinds(self, kinds: Iterable[str]) -> dict[str, dict[str, Any]]:
        """One connection for many kind lookups — keeps /api/dashboard light."""
        wanted = [str(k).upper() for k in kinds if str(k).strip()]
        if not wanted:
            return {}
        out: dict[str, dict[str, Any]] = {}
        with self._connect() as con:
            for kind in wanted:
                row = con.execute(
                    "SELECT * FROM operations WHERE kind=? ORDER BY requested_at DESC LIMIT 1",
                    (kind,),
                ).fetchone()
                item = self._decode(row)
                if item:
                    out[kind] = item
        return out

    def dashboard_snapshot(
        self,
        *,
        kinds: Iterable[str],
        recent_limit: int = 40,
    ) -> dict[str, Any]:
        """Single-connection ops snapshot for the Terminal dashboard hot path."""
        wanted = [str(k).upper() for k in kinds if str(k).strip()]
        limit = max(10, min(int(recent_limit), 100))
        latest: dict[str, dict[str, Any]] = {}
        with self._connect() as con:
            for kind in wanted:
                row = con.execute(
                    "SELECT * FROM operations WHERE kind=? ORDER BY requested_at DESC LIMIT 1",
                    (kind,),
                ).fetchone()
                item = self._decode(row)
                if item:
                    latest[kind] = item
            recent_rows = con.execute(
                "SELECT * FROM operations ORDER BY requested_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            active_rows = con.execute(
                "SELECT * FROM operations WHERE status IN (?,?) "
                "ORDER BY CASE status WHEN ? THEN 0 ELSE 1 END, requested_at ASC",
                (RUNNING, PENDING, RUNNING),
            ).fetchall()
            count_rows = con.execute(
                "SELECT status,COUNT(*) AS n FROM operations GROUP BY status"
            ).fetchall()
        return {
            "latest": latest,
            "recent": [self._decode(row) or {} for row in recent_rows],
            "active": [self._decode(row) or {} for row in active_rows],
            "counts": {str(row["status"]): int(row["n"]) for row in count_rows},
        }

    def recent(self, limit: int = 80) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM operations ORDER BY requested_at DESC LIMIT ?",
                (max(1, min(int(limit), 250)),),
            ).fetchall()
        return [self._decode(row) or {} for row in rows]

    def active(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM operations WHERE status IN (?,?) "
                "ORDER BY CASE status WHEN ? THEN 0 ELSE 1 END, requested_at ASC",
                (RUNNING, PENDING, RUNNING),
            ).fetchall()
        return [self._decode(row) or {} for row in rows]

    def counts(self) -> dict[str, int]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT status,COUNT(*) AS n FROM operations GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["n"]) for row in rows}

    def cancel_pending(self, kinds: Iterable[str]) -> int:
        normalised = tuple(sorted({str(kind).upper() for kind in kinds if str(kind).strip()}))
        if not normalised:
            return 0
        placeholders = ",".join("?" for _ in normalised)
        now = time.time()
        with self._connect() as con:
            cur = con.execute(
                f"UPDATE operations SET status=?,finished_at=?,updated_at=?,stage=?,message=? "
                f"WHERE status=? AND kind IN ({placeholders})",
                (
                    CANCELLED,
                    now,
                    now,
                    CANCELLED,
                    "Stopped by user before execution",
                    PENDING,
                    *normalised,
                ),
            )
            return int(cur.rowcount or 0)

    def request_cancel(self, operation_id: str) -> dict[str, Any] | None:
        """Cancel PENDING immediately; ask RUNNING ops to stop cooperatively."""
        op_id = str(operation_id or "").strip()
        if not op_id:
            return None
        now = time.time()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT * FROM operations WHERE operation_id=?",
                (op_id,),
            ).fetchone()
            if row is None:
                con.commit()
                return None
            status = str(row["status"] or "")
            if status in TERMINAL:
                con.commit()
                return self._decode(row)
            if status == PENDING:
                con.execute(
                    "UPDATE operations SET status=?,finished_at=?,updated_at=?,stage=?,message=? "
                    "WHERE operation_id=? AND status=?",
                    (
                        CANCELLED,
                        now,
                        now,
                        CANCELLED,
                        "Stopped by user before execution",
                        op_id,
                        PENDING,
                    ),
                )
            else:
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except Exception:
                    payload = {}
                if not isinstance(payload, dict):
                    payload = {}
                payload["cancel_requested"] = True
                payload["cancel_requested_at"] = now
                con.execute(
                    "UPDATE operations SET payload_json=?,updated_at=?,message=? "
                    "WHERE operation_id=? AND status=?",
                    (
                        json.dumps(payload, default=str),
                        now,
                        "Stop requested — finishing current batch…",
                        op_id,
                        RUNNING,
                    ),
                )
            row = con.execute(
                "SELECT * FROM operations WHERE operation_id=?",
                (op_id,),
            ).fetchone()
            con.commit()
        return self._decode(row)

    def request_cancel_kinds(self, kinds: Iterable[str]) -> dict[str, Any]:
        """Stop queued + request cancel on running ops for the given kinds."""
        normalised = sorted({str(kind).upper() for kind in kinds if str(kind).strip()})
        pending_n = self.cancel_pending(normalised)
        running_ids: list[str] = []
        with self._connect() as con:
            if normalised:
                placeholders = ",".join("?" for _ in normalised)
                rows = con.execute(
                    f"SELECT operation_id FROM operations WHERE status=? AND kind IN ({placeholders})",
                    (RUNNING, *normalised),
                ).fetchall()
                running_ids = [str(r["operation_id"]) for r in rows]
        running_n = 0
        for op_id in running_ids:
            if self.request_cancel(op_id) is not None:
                running_n += 1
        return {
            "kinds": normalised,
            "cancelled_pending": pending_n,
            "cancel_requested_running": running_n,
            "operation_ids": running_ids,
        }

    def is_cancel_requested(self, operation_id: str) -> bool:
        op = self.get(operation_id)
        if not op:
            return False
        if str(op.get("status") or "") == CANCELLED:
            return True
        payload = op.get("payload") if isinstance(op.get("payload"), dict) else {}
        return bool(payload.get("cancel_requested"))
