"""Durable, cross-process job store for QuantTerm market operations."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"
TERMINAL = frozenset({SUCCEEDED, FAILED, BLOCKED, CANCELLED})
DEFAULT_RUNNING_LEASE_S = 30 * 60
KIND_RUNNING_LEASE_S = {
    "DUE_DILIGENCE_ACQUIRE": 15 * 60,
}

# Status/history reads are on the hot path for Home, Dashboard, News and readiness.
# Keep them metadata-only so a 700-800KB MARKET_SCAN result can never turn a cheap
# status request into a multi-second blob read + JSON decode. Full payload/result
# remains available through get(operation_id) and the explicit full-history method.
_SUMMARY_COLUMNS = (
    "operation_id",
    "kind",
    "lane",
    "status",
    "requested_by",
    "requested_at",
    "started_at",
    "finished_at",
    "updated_at",
    "attempt",
    "worker_pid",
    "stage",
    "message",
    "progress_current",
    "progress_total",
    "error_code",
    "error_message",
    "priority",
)
_SUMMARY_SELECT = ",".join(_SUMMARY_COLUMNS)
_MARKET_SCAN_ARTIFACT = "logs/product/latest_momentum_scan.json"


def _result_for_storage(kind: str, status: str, result: dict[str, Any] | None) -> dict[str, Any]:
    """Keep operation history compact while preserving the canonical result artifact.

    Successful market scans already persist their complete payload in the canonical
    product scan store. Duplicating that payload into every operation row made a
    status/history request read and decode tens of megabytes. Store only the compact
    execution summary/reference here; explicit non-scan results remain unchanged.
    Existing historical rows are intentionally left readable for compatibility.
    """
    data = dict(result or {})
    if str(kind).upper() == "MARKET_SCAN" and status == SUCCEEDED and "payload" in data:
        data.pop("payload", None)
        data["artifact"] = {
            "type": "canonical_scan_store",
            "path": _MARKET_SCAN_ARTIFACT,
        }
        data["result_compacted"] = True
    return data


def pid_is_alive(pid: int | None) -> bool:
    """True when this OS still has a process for ``pid``."""
    try:
        value = int(pid or 0)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def read_pid_file(path: str | Path) -> int:
    """Read a single PID from a lock/runtime pid file. 0 when missing or invalid."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip().split()[0]
        value = int(text)
    except Exception:
        return 0
    return value if value > 1 else 0


def live_lock_owner_pid(lock_path: str | Path) -> int:
    """PID written into the flock file when that process is still alive."""
    pid = read_pid_file(lock_path)
    return pid if pid_is_alive(pid) else 0


class _BorrowedConnection:
    """Keep one SQLite connection per worker thread instead of opening a new FD each poll."""

    def __init__(self, con: sqlite3.Connection) -> None:
        self._con = con

    def __enter__(self) -> sqlite3.Connection:
        return self._con

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            try:
                self._con.commit()
            except Exception:
                pass
        else:
            try:
                self._con.rollback()
            except Exception:
                pass
        return False


class OperationStore:
    """SQLite-backed queue shared by the API and the market-operations worker.

    Connections are cached per worker thread so polling does not continually open
    new file descriptors. WAL keeps readers and the dedicated operation worker from
    blocking each other during normal status/progress traffic.
    """

    def __init__(self, path: str | Path = "logs/market_ops/jobs.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cached = getattr(self._local, "con", None)
        if cached is not None and getattr(self._local, "path", None) == str(self.path):
            try:
                cached.execute("SELECT 1")
                return _BorrowedConnection(cached)
            except Exception:
                try:
                    cached.close()
                except Exception:
                    pass
                self._local.con = None
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                con = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
                con.row_factory = sqlite3.Row
                con.execute("PRAGMA journal_mode=WAL")
                con.execute("PRAGMA synchronous=NORMAL")
                con.execute("PRAGMA busy_timeout=30000")
                self._local.con = con
                self._local.path = str(self.path)
                return _BorrowedConnection(con)
            except sqlite3.OperationalError as exc:
                last_exc = exc
                self._drop_cached()
                time.sleep(0.05)
                self.path.parent.mkdir(parents=True, exist_ok=True)
        raise last_exc if last_exc is not None else sqlite3.OperationalError("unable to open database file")

    def _drop_cached(self) -> None:
        cached = getattr(self._local, "con", None)
        self._local.con = None
        if cached is None:
            return
        try:
            cached.close()
        except Exception:
            pass

    def _begin_immediate(self, con: sqlite3.Connection, *, attempts: int = 10) -> None:
        delay = 0.05
        last: sqlite3.OperationalError | None = None
        for _ in range(max(1, int(attempts))):
            try:
                con.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as exc:
                last = exc
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    self._drop_cached()
                    raise
                try:
                    con.rollback()
                except Exception:
                    pass
                time.sleep(delay)
                delay = min(1.0, delay * 2)
        self._drop_cached()
        raise last if last is not None else sqlite3.OperationalError("database is locked")

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
                    error_message TEXT NOT NULL DEFAULT '',
                    priority INTEGER NOT NULL DEFAULT 0
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
            try:
                con.execute(
                    "ALTER TABLE operations ADD COLUMN priority INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_operations_lease "
                "ON operations(status, lane, priority, requested_at)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_operations_requested_at "
                "ON operations(requested_at DESC)"
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

    @staticmethod
    def _decode_summary(row: sqlite3.Row | None) -> dict[str, Any] | None:
        """Decode metadata-only rows without ever touching payload/result JSON."""
        if row is None:
            return None
        data = dict(row)
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
        priority: int = 0,
    ) -> tuple[dict[str, Any], bool]:
        """Enqueue one operation and return ``(record, created)``.

        Repeated button clicks return the existing pending/running operation rather
        than creating queue spam. A new click after a terminal result creates a new run.
        A higher ``priority`` promotes a still-pending job so a user click jumps
        ahead of pipeline work.
        """
        kind = str(kind).strip().upper()
        lane = str(lane).strip().lower()
        if not kind or not lane:
            raise ValueError("kind and lane are required")
        try:
            priority_value = int(priority or 0)
        except (TypeError, ValueError):
            priority_value = 0
        now = time.time()
        operation_id = uuid.uuid4().hex
        with self._connect() as con:
            self._begin_immediate(con)
            if deduplicate:
                row = con.execute(
                    "SELECT * FROM operations WHERE kind=? AND status IN (?,?) "
                    "ORDER BY priority DESC, requested_at DESC LIMIT 1",
                    (kind, PENDING, RUNNING),
                ).fetchone()
                if row is not None:
                    if str(row["status"]) == PENDING and priority_value > int(row["priority"] or 0):
                        con.execute(
                            "UPDATE operations SET priority=?, requested_by=?, updated_at=?, "
                            "message=? WHERE operation_id=? AND status=?",
                            (
                                priority_value,
                                str(requested_by or "terminal"),
                                now,
                                "User click jumped the queue",
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
                    payload_json,message,priority
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
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
                    "Queued and waiting for the dedicated market-operations worker",
                    priority_value,
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
            self._begin_immediate(con)
            row = con.execute(
                "SELECT * FROM operations WHERE lane=? AND status=? "
                "ORDER BY priority DESC, requested_at ASC LIMIT 1",
                (lane, PENDING),
            ).fetchone()
            if row is None:
                con.commit()
                return None
            operation_id = str(row["operation_id"])
            con.execute(
                "UPDATE operations SET status=?,started_at=?,finished_at=NULL,updated_at=?,"
                "attempt=attempt+1,worker_pid=?,stage=?,message=?,error_code='',error_message='' "
                "WHERE operation_id=? AND status=?",
                (
                    RUNNING,
                    now,
                    now,
                    int(worker_pid),
                    "STARTING",
                    "Worker accepted the operation",
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
            row = con.execute(
                "SELECT kind FROM operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            kind = str(row["kind"] or "") if row is not None else ""
            stored_result = _result_for_storage(kind, status, result)
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
                    json.dumps(stored_result, default=str),
                    str(error_code or ""),
                    str(error_message or ""),
                    status,
                    SUCCEEDED,
                    operation_id,
                ),
            )

    def recover_orphans(self) -> int:
        """Requeue operations left RUNNING by a stopped worker with a fresh next lease clock."""
        now = time.time()
        with self._connect() as con:
            cur = con.execute(
                "UPDATE operations SET status=?,started_at=NULL,finished_at=NULL,updated_at=?,"
                "stage=?,message=?,worker_pid=NULL WHERE status=?",
                (
                    PENDING,
                    now,
                    "RECOVERED",
                    "Previous worker stopped before completion; operation requeued",
                    RUNNING,
                ),
            )
        return int(cur.rowcount or 0)

    def recover_dead_running(self, *, keep_pid: int | None = None) -> int:
        """Requeue RUNNING jobs whose worker process is gone. Leave ``keep_pid`` alone."""
        now = time.time()
        recovered = 0
        with self._connect() as con:
            rows = list(
                con.execute(
                    "SELECT operation_id, worker_pid FROM operations WHERE status=?",
                    (RUNNING,),
                )
            )
            for row in rows:
                pid = row["worker_pid"]
                try:
                    pid_i = int(pid) if pid is not None else 0
                except (TypeError, ValueError):
                    pid_i = 0
                if keep_pid is not None and pid_i == int(keep_pid):
                    continue
                if pid_i and pid_is_alive(pid_i):
                    continue
                con.execute(
                    "UPDATE operations SET status=?,started_at=NULL,finished_at=NULL,updated_at=?,"
                    "stage=?,message=?,worker_pid=NULL WHERE operation_id=? AND status=?",
                    (
                        PENDING,
                        now,
                        "RECOVERED",
                        "Scan worker process died; operation requeued",
                        row["operation_id"],
                        RUNNING,
                    ),
                )
                recovered += 1
        return recovered

    def recover_stale_running(
        self,
        *,
        now: float | None = None,
        deadlines: dict[str, float] | None = None,
    ) -> int:
        """Fail only abandoned RUNNING rows that exceeded their lease.

        Never mark an operation terminal while its recorded worker PID is alive:
        changing SQLite state cannot stop the Python lane that is still executing,
        and doing so can create a second PENDING operation behind an invisible live
        scan. Operation-specific code owns real execution deadlines for live work.
        """
        clock = time.time() if now is None else float(now)
        limits = dict(KIND_RUNNING_LEASE_S)
        if deadlines:
            limits.update({str(k): float(v) for k, v in deadlines.items()})
        recovered = 0
        with self._connect() as con:
            rows = list(
                con.execute(
                    "SELECT operation_id,kind,worker_pid,started_at,updated_at "
                    "FROM operations WHERE status=?",
                    (RUNNING,),
                )
            )
            for row in rows:
                try:
                    worker_pid = int(row["worker_pid"] or 0)
                except (TypeError, ValueError):
                    worker_pid = 0
                if worker_pid and pid_is_alive(worker_pid):
                    continue
                last_activity = max(
                    float(row["started_at"] or 0),
                    float(row["updated_at"] or 0),
                )
                kind = str(row["kind"] or "")
                limit = float(limits.get(kind, DEFAULT_RUNNING_LEASE_S))
                if last_activity <= 0 or (clock - last_activity) < limit:
                    continue
                age = clock - last_activity
                con.execute(
                    """
                    UPDATE operations SET status=?,finished_at=?,updated_at=?,stage=?,message=?,
                        error_code=?,error_message=?
                    WHERE operation_id=? AND status=?
                    """,
                    (
                        FAILED,
                        clock,
                        clock,
                        FAILED,
                        f"{kind} abandoned its {int(limit)}s lease after {int(age)}s without a live worker",
                        "DEADLINE_EXCEEDED",
                        f"Operation had no live worker and no activity for {int(age)}s",
                        str(row["operation_id"]),
                        RUNNING,
                    ),
                )
                recovered += 1
        return recovered

    def oldest_running(self) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM operations WHERE status=? ORDER BY started_at ASC LIMIT 1",
                (RUNNING,),
            ).fetchone()
        return self._decode(row)

    def get(self, operation_id: str) -> dict[str, Any] | None:
        """Return one full operation, including payload/result JSON."""
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        return self._decode(row)

    def latest(self, kind: str) -> dict[str, Any] | None:
        """Return the full latest operation for callers that explicitly need details."""
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM operations WHERE kind=? ORDER BY requested_at DESC LIMIT 1",
                (str(kind).upper(),),
            ).fetchone()
        return self._decode(row)

    def latest_summary(self, kind: str) -> dict[str, Any] | None:
        """Latest operation metadata without payload/result blob I/O or JSON decode."""
        with self._connect() as con:
            row = con.execute(
                f"SELECT {_SUMMARY_SELECT} FROM operations "
                "WHERE kind=? ORDER BY requested_at DESC LIMIT 1",
                (str(kind).upper(),),
            ).fetchone()
        return self._decode_summary(row)

    def recent(self, limit: int = 80) -> list[dict[str, Any]]:
        """Return lightweight recent status rows; never fetch historical result blobs.

        This method is intentionally the safe default because it feeds desk/status
        projections. Call ``recent_full`` or ``get`` only on explicit detail paths.
        """
        return self.recent_summary(limit)

    def recent_full(self, limit: int = 80) -> list[dict[str, Any]]:
        """Return full recent operations for explicit detail/history consumers."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM operations ORDER BY requested_at DESC LIMIT ?",
                (max(1, min(int(limit), 250)),),
            ).fetchall()
        return [self._decode(row) or {} for row in rows]

    def recent_summary(self, limit: int = 80) -> list[dict[str, Any]]:
        """Return recent operation metadata only; safe for latency-sensitive status APIs."""
        with self._connect() as con:
            rows = con.execute(
                f"SELECT {_SUMMARY_SELECT} FROM operations "
                "ORDER BY requested_at DESC LIMIT ?",
                (max(1, min(int(limit), 250)),),
            ).fetchall()
        return [self._decode_summary(row) or {} for row in rows]

    def active(self) -> list[dict[str, Any]]:
        """Return full active operations for explicit detail consumers."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM operations WHERE status IN (?,?) "
                "ORDER BY CASE status WHEN ? THEN 0 ELSE 1 END, requested_at ASC",
                (RUNNING, PENDING, RUNNING),
            ).fetchall()
        return [self._decode(row) or {} for row in rows]

    def active_summary(self) -> list[dict[str, Any]]:
        """Return active operation metadata only for UI/status hot paths."""
        with self._connect() as con:
            rows = con.execute(
                f"SELECT {_SUMMARY_SELECT} FROM operations WHERE status IN (?,?) "
                "ORDER BY CASE status WHEN ? THEN 0 ELSE 1 END, requested_at ASC",
                (RUNNING, PENDING, RUNNING),
            ).fetchall()
        return [self._decode_summary(row) or {} for row in rows]

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
                    "Cancelled before execution",
                    PENDING,
                    *normalised,
                ),
            )
        return int(cur.rowcount or 0)
