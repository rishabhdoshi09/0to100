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
    "MARKET_SCAN": 20 * 60,
    "LONG_TERM_SCAN": 20 * 60,
    "LONG_TERM_REFRESH": 20 * 60,
    "NEWS_REFRESH": 10 * 60,
    "MARKET_REPORT": 15 * 60,
    "FNO_REFRESH": 10 * 60,
    "DATA_PREPARE": 30 * 60,
}


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

    Every public method opens a short-lived SQLite connection. This avoids sharing
    connection objects across the API process, worker process and worker threads.
    WAL mode keeps dashboard reads responsive while progress is being written.
    """

    def __init__(
        self,
        path: str | Path = "logs/market_ops/jobs.db",
        *,
        migrate: bool = True,
        timeout_s: float = 30.0,
        busy_timeout_ms: int = 30_000,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._timeout_s = float(timeout_s)
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._migrate = bool(migrate)
        self._connect_attempts = 3 if migrate else 1
        if migrate:
            self._init_schema()

    @classmethod
    def reader(cls, path: str | Path) -> "OperationStore":
        """Read-only-ish opener: no schema migration, tiny lock wait."""
        return cls(path, migrate=False, timeout_s=0.2, busy_timeout_ms=200)

    def _connect(self):
        if not self._migrate and not self.path.exists():
            raise sqlite3.OperationalError("operations database is unavailable")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cache_key = (str(self.path), self._timeout_s, self._busy_timeout_ms)
        cached = getattr(self._local, "con", None)
        if cached is not None and getattr(self._local, "cache_key", None) == cache_key:
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
        attempts = max(1, int(getattr(self, "_connect_attempts", 3)))
        for attempt in range(attempts):
            try:
                con = sqlite3.connect(str(self.path), timeout=self._timeout_s, isolation_level=None)
                con.row_factory = sqlite3.Row
                if self._migrate:
                    con.execute("PRAGMA journal_mode=WAL")
                    con.execute("PRAGMA synchronous=NORMAL")
                con.execute("PRAGMA query_only=ON" if not self._migrate else "PRAGMA query_only=OFF")
                con.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
                self._local.con = con
                self._local.cache_key = cache_key
                return _BorrowedConnection(con)
            except sqlite3.OperationalError as exc:
                last_exc = exc
                self._drop_cached()
                if not self._migrate or attempt + 1 >= attempts:
                    raise
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
                    priority INTEGER NOT NULL DEFAULT 0,
                    first_started_at REAL,
                    attempt_started_at REAL
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
            try:
                con.execute("ALTER TABLE operations ADD COLUMN first_started_at REAL")
            except sqlite3.OperationalError:
                pass
            try:
                con.execute("ALTER TABLE operations ADD COLUMN attempt_started_at REAL")
            except sqlite3.OperationalError:
                pass
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_operations_lease "
                "ON operations(status, lane, priority, requested_at)"
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
                "UPDATE operations SET status=?,"
                "started_at=?,"
                "attempt_started_at=?,"
                "first_started_at=COALESCE(first_started_at, ?),"
                "updated_at=?,attempt=attempt+1,worker_pid=?,stage=?,message=? "
                "WHERE operation_id=? AND status=?",
                (
                    RUNNING,
                    now,
                    now,
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
                    "UPDATE operations SET status=?,updated_at=?,stage=?,message=?,worker_pid=NULL "
                    "WHERE operation_id=? AND status=?",
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

    def attempt_start_epoch(self, row: dict[str, Any] | sqlite3.Row | None) -> float:
        """Current-attempt start. Never use first_started_at for deadlines."""
        if row is None:
            return 0.0
        try:
            data = dict(row)
        except Exception:
            data = row if isinstance(row, dict) else {}
        for key in ("attempt_started_at", "started_at"):
            try:
                value = float(data.get(key) or 0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
        return 0.0

    def overdue_running(
        self,
        *,
        now: float | None = None,
        deadlines: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """RUNNING rows whose *current attempt* exceeded its lease.

        Does not mark them FAILED. The owning worker must cancel the actual
        execution first, then finish(). Marking FAILED while the child is
        still running is the ghost-execution bug.
        """
        clock = time.time() if now is None else float(now)
        limits = dict(KIND_RUNNING_LEASE_S)
        if deadlines:
            limits.update({str(k): float(v) for k, v in deadlines.items()})
        overdue: list[dict[str, Any]] = []
        with self._connect() as con:
            rows = list(con.execute("SELECT * FROM operations WHERE status=?", (RUNNING,)))
        for row in rows:
            started = self.attempt_start_epoch(row)
            kind = str(row["kind"] or "")
            limit = float(limits.get(kind, DEFAULT_RUNNING_LEASE_S))
            if started <= 0 or (clock - started) < limit:
                continue
            item = self._decode(row) or {}
            item["elapsed_s"] = clock - started
            item["lease_s"] = limit
            overdue.append(item)
        return overdue

    def recover_stale_running(
        self,
        *,
        now: float | None = None,
        deadlines: dict[str, float] | None = None,
        keep_pid: int | None = None,
    ) -> int:
        """Fail only orphaned RUNNING rows whose current-attempt lease elapsed.

        A live local PID is never marked FAILED here. That would leave the
        worker child still executing while the DB said terminal. The owning
        worker cancels first, then finish().
        """
        clock = time.time() if now is None else float(now)
        recovered = 0
        my_pid = os.getpid()
        keep = int(keep_pid) if keep_pid is not None else None
        for row in self.overdue_running(now=clock, deadlines=deadlines):
            try:
                pid_i = int(row.get("worker_pid") or 0)
            except (TypeError, ValueError):
                pid_i = 0
            if keep is not None and pid_i == keep:
                continue
            if pid_i and pid_i == my_pid:
                continue
            if pid_i and pid_is_alive(pid_i):
                continue
            kind = str(row.get("kind") or "")
            age = float(row.get("elapsed_s") or 0)
            limit = float(row.get("lease_s") or DEFAULT_RUNNING_LEASE_S)
            with self._connect() as con:
                con.execute(
                    """
                    UPDATE operations SET status=?,finished_at=?,updated_at=?,stage=?,message=?,
                        error_code=?,error_message=?,worker_pid=NULL
                    WHERE operation_id=? AND status=?
                    """,
                    (
                        FAILED,
                        clock,
                        clock,
                        FAILED,
                        f"{kind} exceeded its {int(limit)}s deadline after {int(age)}s",
                        "DEADLINE_EXCEEDED",
                        f"Operation stayed RUNNING for {int(age)}s; orphan worker",
                        str(row["operation_id"]),
                        RUNNING,
                    ),
                )
            recovered += 1
        return recovered

    def compact_status(
        self,
        *,
        runtime: dict[str, Any] | None = None,
        kinds: Iterable[str] | None = None,
        freshness: str = "CURRENT",
    ) -> dict[str, Any]:
        """Worker-persisted compact dashboard. Cheap for the HTTP GET path."""
        runtime = runtime or {}
        recent = self.recent(80)
        active = self.active()
        latest: dict[str, Any] = {}
        kind_set = {str(kind).upper() for kind in (kinds or ()) if str(kind).strip()}
        if not kind_set:
            kind_set = {str(row.get("kind") or "").upper() for row in recent + active}
            kind_set.discard("")
        for kind in kind_set:
            item = self.latest(kind)
            if item:
                latest[kind] = item
        from operations.status_snapshot import slim_operations_status

        return slim_operations_status(
            {
                "available": True,
                "freshness": freshness,
                "generated_at": time.time(),
                "running": bool(runtime.get("process_running") or runtime.get("running")),
                "worker_pid": runtime.get("worker_pid"),
                "heartbeat": runtime.get("heartbeat", ""),
                "active_lanes": dict(runtime.get("active") or {}),
                "counts": self.counts(),
                "active": active,
                "recent": recent,
                "latest": latest,
                "fd_count": runtime.get("fd_count"),
                "overdue": self.overdue_running(),
            }
        )

    def oldest_running(self) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM operations WHERE status=? ORDER BY started_at ASC LIMIT 1",
                (RUNNING,),
            ).fetchone()
        return self._decode(row)

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
                    "Cancelled before execution",
                    PENDING,
                    *normalised,
                ),
            )
        return int(cur.rowcount or 0)
