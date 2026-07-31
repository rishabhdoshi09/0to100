"""
🗂️ Durable job ledger for the autonomy supervisor.

A persisted, restart-safe job queue (SQLite) with leases and idempotency keys — the scheduler of
record, replacing uncontrolled daemon threads. A job interrupted by process death is recoverable when
its lease expires; re-running with the same idempotency key never duplicates work. Deterministic and
network-free: the clock is injectable so tests control time exactly.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

# statuses
PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
BLOCKED = "BLOCKED"
RETRYABLE_FAILED = "RETRYABLE_FAILED"
PERMANENT_FAILED = "PERMANENT_FAILED"
SKIPPED_IDEMPOTENT = "SKIPPED_IDEMPOTENT"
CANCELLED = "CANCELLED"

_TERMINAL = {SUCCEEDED, PERMANENT_FAILED, SKIPPED_IDEMPOTENT, CANCELLED}
_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    scheduled_for REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_expires_at REAL,
    idempotency_key TEXT,
    input_snapshot_id TEXT,
    output_snapshot_id TEXT,
    result_summary TEXT,
    error_code TEXT,
    error_message TEXT,
    next_retry_at REAL,
    critical INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status);
CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_idem ON jobs(idempotency_key)
    WHERE idempotency_key IS NOT NULL;
"""


@dataclass
class Job:
    job_id: str
    job_type: str
    scheduled_for: float
    status: str
    attempt: int
    idempotency_key: str | None = None
    input_snapshot_id: str | None = None
    output_snapshot_id: str | None = None
    result_summary: str = ""
    error_code: str = ""
    error_message: str = ""
    next_retry_at: float | None = None
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    critical: bool = False


def _row_to_job(r) -> Job:
    return Job(job_id=r["job_id"], job_type=r["job_type"], scheduled_for=r["scheduled_for"],
               status=r["status"], attempt=r["attempt"], idempotency_key=r["idempotency_key"],
               input_snapshot_id=r["input_snapshot_id"], output_snapshot_id=r["output_snapshot_id"],
               result_summary=r["result_summary"] or "", error_code=r["error_code"] or "",
               error_message=r["error_message"] or "", next_retry_at=r["next_retry_at"],
               lease_owner=r["lease_owner"], lease_expires_at=r["lease_expires_at"],
               started_at=r["started_at"], finished_at=r["finished_at"],
               critical=bool(r["critical"]))


class JobStore:
    def __init__(self, db_path, *, clock=None):
        import time
        self.clock = clock or time.time
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_DDL)
        self._db.commit()

    # ── enqueue (idempotent) ─────────────────────────────────────────────────────
    def enqueue(self, job_type: str, *, scheduled_for: float | None = None,
                idempotency_key: str | None = None, input_snapshot_id: str | None = None,
                critical: bool = False) -> Job:
        now = self.clock()
        sched = now if scheduled_for is None else scheduled_for
        with self._lock:
            if idempotency_key is not None:
                r = self._db.execute("SELECT * FROM jobs WHERE idempotency_key=?",
                                     (idempotency_key,)).fetchone()
                if r is not None:
                    return _row_to_job(r)          # already known → never duplicate
            jid = uuid.uuid4().hex[:16]
            self._db.execute(
                "INSERT INTO jobs(job_id,job_type,scheduled_for,status,attempt,idempotency_key,"
                "input_snapshot_id,critical,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (jid, job_type, sched, PENDING, 0, idempotency_key, input_snapshot_id,
                 1 if critical else 0, now))
            self._db.commit()
            return _row_to_job(self._db.execute("SELECT * FROM jobs WHERE job_id=?", (jid,)).fetchone())

    # ── lease one due job (reclaims expired leases) ──────────────────────────────
    def lease_due(self, owner: str, *, lease_seconds: float = 300.0) -> Job | None:
        now = self.clock()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                # reclaim a RUNNING job whose lease died (crash recovery), oldest first
                r = self._db.execute(
                    "SELECT * FROM jobs WHERE status=? AND lease_expires_at IS NOT NULL "
                    "AND lease_expires_at < ? ORDER BY scheduled_for LIMIT 1", (RUNNING, now)).fetchone()
                if r is None:
                    r = self._db.execute(
                        "SELECT * FROM jobs WHERE status=? AND scheduled_for<=? "
                        "ORDER BY critical DESC, scheduled_for LIMIT 1", (PENDING, now)).fetchone()
                if r is None:
                    self._db.execute("COMMIT")
                    return None
                self._db.execute(
                    "UPDATE jobs SET status=?, lease_owner=?, lease_expires_at=?, started_at=?, "
                    "attempt=attempt+1 WHERE job_id=?",
                    (RUNNING, owner, now + lease_seconds, now, r["job_id"]))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
            return _row_to_job(self._db.execute("SELECT * FROM jobs WHERE job_id=?",
                                                (r["job_id"],)).fetchone())

    # ── complete a leased job ────────────────────────────────────────────────────
    def complete(self, job_id: str, status: str, *, result_summary: str = "",
                 output_snapshot_id: str | None = None, error_code: str = "",
                 error_message: str = "", next_retry_at: float | None = None) -> None:
        now = self.clock()
        with self._lock:
            finished = None if status == PENDING else now
            self._db.execute(
                "UPDATE jobs SET status=?, finished_at=?, result_summary=?, output_snapshot_id=?, "
                "error_code=?, error_message=?, next_retry_at=?, lease_owner=NULL, "
                "lease_expires_at=NULL WHERE job_id=?",
                (status, finished, result_summary, output_snapshot_id, error_code, error_message,
                 next_retry_at, job_id))
            self._db.commit()

    def reschedule_retry(self, job_id: str, *, when: float, error_code: str = "",
                         error_message: str = "") -> None:
        """A RETRYABLE_FAILED job goes back to PENDING at `when` (bounded backoff decided by caller)."""
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, scheduled_for=?, next_retry_at=?, error_code=?, "
                "error_message=?, lease_owner=NULL, lease_expires_at=NULL, finished_at=NULL "
                "WHERE job_id=?",
                (PENDING, when, when, error_code, error_message, job_id))
            self._db.commit()

    # ── reads ────────────────────────────────────────────────────────────────────
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _row_to_job(r) if r else None

    def list(self, *, status: str | None = None, limit: int = 200) -> list:
        with self._lock:
            if status:
                rows = self._db.execute("SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC "
                                        "LIMIT ?", (status, limit)).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                                        (limit,)).fetchall()
        return [_row_to_job(r) for r in rows]

    def overdue_critical(self, *, grace_seconds: float = 0.0) -> list:
        """Critical jobs whose scheduled_for is past due beyond the grace and not yet running/done."""
        now = self.clock()
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM jobs WHERE critical=1 AND status=? AND scheduled_for < ?",
                (PENDING, now - grace_seconds)).fetchall()
        return [_row_to_job(r) for r in rows]

    def reclaim_expired(self) -> int:
        """Explicitly return crashed RUNNING jobs (dead lease) to PENDING. Returns count."""
        now = self.clock()
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status=?, lease_owner=NULL WHERE status=? AND "
                "lease_expires_at IS NOT NULL AND lease_expires_at < ?", (PENDING, RUNNING, now))
            self._db.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()
