"""Durable, dependency-aware job ledger for the autonomy supervisor.

SQLite is the scheduler of record. Jobs have leases, idempotency keys and explicit BLOCKED
dependencies. Existing databases are migrated in place; a login or data recovery can requeue the
same logical job rather than creating a duplicate or leaving it blocked forever.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
BLOCKED = "BLOCKED"
RETRYABLE_FAILED = "RETRYABLE_FAILED"
PERMANENT_FAILED = "PERMANENT_FAILED"
SKIPPED_IDEMPOTENT = "SKIPPED_IDEMPOTENT"
CANCELLED = "CANCELLED"

_TERMINAL = {SUCCEEDED, PERMANENT_FAILED, SKIPPED_IDEMPOTENT, CANCELLED}
# Re-assert a leftover PENDING intent after this long so an evening restart
# does not treat this morning's row as a live outage.
_STALE_PENDING_S = 3600.0
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
    created_at REAL NOT NULL,
    blocked_on TEXT,
    blocked_reason TEXT,
    blocked_at REAL,
    unblocked_at REAL,
    dependency_version TEXT
);
CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS ix_jobs_blocked_on ON jobs(blocked_on, status);
CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_idem ON jobs(idempotency_key)
    WHERE idempotency_key IS NOT NULL;
"""
_MIGRATION_COLUMNS = {
    "blocked_on": "TEXT",
    "blocked_reason": "TEXT",
    "blocked_at": "REAL",
    "unblocked_at": "REAL",
    "dependency_version": "TEXT",
}


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
    blocked_on: str | None = None
    blocked_reason: str = ""
    blocked_at: float | None = None
    unblocked_at: float | None = None
    dependency_version: str | None = None


def _value(row, name, default=None):
    try:
        return row[name]
    except (KeyError, IndexError):
        return default


def _row_to_job(r) -> Job:
    return Job(
        job_id=r["job_id"], job_type=r["job_type"], scheduled_for=r["scheduled_for"],
        status=r["status"], attempt=r["attempt"], idempotency_key=r["idempotency_key"],
        input_snapshot_id=r["input_snapshot_id"], output_snapshot_id=r["output_snapshot_id"],
        result_summary=r["result_summary"] or "", error_code=r["error_code"] or "",
        error_message=r["error_message"] or "", next_retry_at=r["next_retry_at"],
        lease_owner=r["lease_owner"], lease_expires_at=r["lease_expires_at"],
        started_at=r["started_at"], finished_at=r["finished_at"], critical=bool(r["critical"]),
        blocked_on=_value(r, "blocked_on"), blocked_reason=_value(r, "blocked_reason", "") or "",
        blocked_at=_value(r, "blocked_at"), unblocked_at=_value(r, "unblocked_at"),
        dependency_version=_value(r, "dependency_version"),
    )


class JobStore:
    def __init__(self, db_path, *, clock=None):
        import time
        self.clock = clock or time.time
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_DDL)
        self._migrate()
        self._db.commit()

    def _migrate(self) -> None:
        present = {r[1] for r in self._db.execute("PRAGMA table_info(jobs)").fetchall()}
        for name, sql_type in _MIGRATION_COLUMNS.items():
            if name not in present:
                self._db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}")
        self._db.execute("CREATE INDEX IF NOT EXISTS ix_jobs_blocked_on ON jobs(blocked_on, status)")

    def enqueue(self, job_type: str, *, scheduled_for: float | None = None,
                idempotency_key: str | None = None, input_snapshot_id: str | None = None,
                critical: bool = False) -> Job:
        now = self.clock()
        sched = now if scheduled_for is None else scheduled_for
        with self._lock:
            if idempotency_key is not None:
                row = self._db.execute("SELECT * FROM jobs WHERE idempotency_key=?",
                                       (idempotency_key,)).fetchone()
                if row is not None:
                    job = _row_to_job(row)
                    if (
                        job.status == PENDING
                        and (now - float(job.scheduled_for or 0.0)) >= _STALE_PENDING_S
                    ):
                        crit = 1 if critical else (1 if job.critical else 0)
                        self._db.execute(
                            "UPDATE jobs SET scheduled_for=?, critical=? "
                            "WHERE job_id=? AND status=?",
                            (sched, crit, job.job_id, PENDING),
                        )
                        self._db.commit()
                        return self.get(job.job_id, _locked=True)
                    return job
            jid = uuid.uuid4().hex[:16]
            self._db.execute(
                "INSERT INTO jobs(job_id,job_type,scheduled_for,status,attempt,idempotency_key,"
                "input_snapshot_id,critical,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (jid, job_type, sched, PENDING, 0, idempotency_key, input_snapshot_id,
                 1 if critical else 0, now))
            self._db.commit()
            return self.get(jid, _locked=True)

    def lease_due(self, owner: str, *, lease_seconds: float = 300.0) -> Job | None:
        now = self.clock()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT * FROM jobs WHERE status=? AND lease_expires_at IS NOT NULL "
                    "AND lease_expires_at < ? ORDER BY scheduled_for LIMIT 1", (RUNNING, now)).fetchone()
                if row is None:
                    # Fresh market work is time-sensitive. A long post-market outcome/learning
                    # job must not get the first lease merely because it is marked critical and
                    # then make scan/news wait minutes. This is ordering only: it does not skip,
                    # duplicate, or weaken any critical job.
                    # Poll-wait jobs remain lowest priority so background polling cannot starve
                    # real work either.
                    row = self._db.execute(
                        "SELECT * FROM jobs WHERE status=? AND scheduled_for<=? "
                        "ORDER BY CASE job_type "
                        "WHEN 'market_scan' THEN 40 WHEN 'news_refresh' THEN 30 "
                        "WHEN 'paper_cycle' THEN 20 ELSE 0 END DESC, "
                        "CASE WHEN error_code IN "
                        "('DATA_REFRESH_IN_PROGRESS','MARKET_OP_IN_PROGRESS','LONG_TERM_OP_IN_PROGRESS') "
                        "THEN 0 ELSE 1 END DESC, critical DESC, scheduled_for, created_at LIMIT 1",
                        (PENDING, now)).fetchone()
                if row is None:
                    self._db.execute("COMMIT")
                    return None
                self._db.execute(
                    "UPDATE jobs SET status=?, lease_owner=?, lease_expires_at=?, started_at=?, "
                    "attempt=attempt+1 WHERE job_id=?",
                    (RUNNING, owner, now + lease_seconds, now, row["job_id"]))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
            return self.get(row["job_id"], _locked=True)

    def renew_lease(self, job_id: str, owner: str, *, lease_seconds: float = 300.0) -> bool:
        """Extend one live worker lease without changing job state or attempt count."""
        now = self.clock()
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET lease_expires_at=? WHERE job_id=? AND status=? AND lease_owner=?",
                (now + lease_seconds, job_id, RUNNING, owner),
            )
            self._db.commit()
            return cur.rowcount == 1

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

    def block(self, job_id: str, *, dependency: str, reason: str,
              dependency_version: str | None = None, result_summary: str = "") -> None:
        now = self.clock()
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, finished_at=?, result_summary=?, blocked_on=?, "
                "blocked_reason=?, blocked_at=?, dependency_version=?, lease_owner=NULL, "
                "lease_expires_at=NULL WHERE job_id=?",
                (BLOCKED, now, result_summary or reason, dependency, reason, now,
                 dependency_version, job_id))
            self._db.commit()

    def unblock_dependency(self, dependency: str, *, dependency_version: str | None = None,
                           scheduled_for: float | None = None) -> int:
        """Requeue jobs blocked on one dependency without changing their idempotency identity."""
        now = self.clock()
        sched = now if scheduled_for is None else scheduled_for
        with self._lock:
            if dependency_version is None:
                cur = self._db.execute(
                    "UPDATE jobs SET status=?, scheduled_for=?, finished_at=NULL, unblocked_at=?, "
                    "blocked_on=NULL, blocked_reason='', dependency_version=NULL WHERE status=? "
                    "AND blocked_on=?", (PENDING, sched, now, BLOCKED, dependency))
            else:
                cur = self._db.execute(
                    "UPDATE jobs SET status=?, scheduled_for=?, finished_at=NULL, unblocked_at=?, "
                    "blocked_on=NULL, blocked_reason='', dependency_version=? WHERE status=? "
                    "AND blocked_on=?", (PENDING, sched, now, dependency_version, BLOCKED, dependency))
            self._db.commit()
            return cur.rowcount

    def requeue(self, job_id: str, *, scheduled_for: float | None = None) -> None:
        now = self.clock()
        sched = now if scheduled_for is None else scheduled_for
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, scheduled_for=?, finished_at=NULL, next_retry_at=NULL, "
                "lease_owner=NULL, lease_expires_at=NULL WHERE job_id=?",
                (PENDING, sched, job_id))
            self._db.commit()

    def reschedule_retry(self, job_id: str, *, when: float, error_code: str = "",
                         error_message: str = "") -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, scheduled_for=?, next_retry_at=?, error_code=?, "
                "error_message=?, lease_owner=NULL, lease_expires_at=NULL, finished_at=NULL "
                "WHERE job_id=?", (PENDING, when, when, error_code, error_message, job_id))
            self._db.commit()

    def get(self, job_id: str, _locked: bool = False) -> Job | None:
        if _locked:
            row = self._db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        else:
            with self._lock:
                row = self._db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def find_by_type_and_key(self, job_type: str, idempotency_key: str) -> Job | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM jobs WHERE job_type=? AND idempotency_key=?",
                (job_type, idempotency_key)).fetchone()
        return _row_to_job(row) if row else None

    def list(self, *, status: str | None = None, job_type: str | None = None,
             limit: int = 200) -> list[Job]:
        clauses, params = [], []
        if status:
            clauses.append("status=?"); params.append(status)
        if job_type:
            clauses.append("job_type=?"); params.append(job_type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC LIMIT ?",
                (*params, limit)).fetchall()
        return [_row_to_job(r) for r in rows]

    def cancel_superseded_pending(self, job_type: str, *, keep: int = 1) -> int:
        """Retire old recurring rows while preserving the newest pending intent."""
        keep = max(0, int(keep))
        now = self.clock()
        with self._lock:
            rows = self._db.execute(
                "SELECT job_id FROM jobs WHERE job_type=? AND status=? "
                "ORDER BY scheduled_for DESC, created_at DESC",
                (job_type, PENDING),
            ).fetchall()
            stale = [row["job_id"] for row in rows[keep:]]
            if not stale:
                return 0
            placeholders = ",".join("?" for _ in stale)
            cur = self._db.execute(
                f"UPDATE jobs SET status=?, finished_at=?, result_summary=?, lease_owner=NULL, "
                f"lease_expires_at=NULL WHERE job_id IN ({placeholders})",
                (CANCELLED, now, "superseded by a newer recurring job", *stale),
            )
            self._db.commit()
            return cur.rowcount

    def overdue_critical(self, *, grace_seconds: float = 0.0) -> list[Job]:
        """Return current critical work, not historical recurring backlog.

        Only the newest pending row for each job type is considered. A job type
        already running is not overdue. Scheduled paper cycles are opportunity/
        position-management events and remain visible in the ledger without
        degrading the organisation as an infrastructure outage. Synthetic or
        manually requested critical rows still exercise the generic safety path.
        """
        now = self.clock()
        with self._lock:
            rows = self._db.execute(
                "SELECT j.* FROM jobs j "
                "WHERE j.critical=1 AND j.status=? AND j.scheduled_for < ? "
                "AND NOT (j.job_type='paper_cycle' AND j.idempotency_key LIKE 'paper_cycle:%') "
                "AND NOT EXISTS (SELECT 1 FROM jobs r WHERE r.job_type=j.job_type AND r.status=?) "
                "AND j.scheduled_for=(SELECT MAX(p.scheduled_for) FROM jobs p "
                "WHERE p.job_type=j.job_type AND p.status=?)",
                (PENDING, now - grace_seconds, RUNNING, PENDING),
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def reclaim_expired(self) -> int:
        now = self.clock()
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status=?, lease_owner=NULL, lease_expires_at=NULL WHERE status=? "
                "AND lease_expires_at IS NOT NULL AND lease_expires_at < ?",
                (PENDING, RUNNING, now))
            self._db.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()
