"""
🎛️ The autonomy supervisor — one durable, Streamlit-independent process.

Holds a single-instance lock, drives the persisted job ledger with leases + idempotency, runs the
explicit operational state machine, emits a heartbeat and an append-only incident trail, and writes a
read-only status snapshot for the retail UI. It never places a broker order and never enables live
capital. `tick()` is a deterministic single step (for tests); `run()` is the production loop.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST
from research.autonomy import health as H
from research.autonomy import jobs as JOBS
from research.autonomy.dialogue import DialogueLog, Record, OPERATIONAL_INCIDENT

_MAX_ATTEMPTS = 5
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 300.0


class SingleInstanceLock:
    """POSIX flock single-instance guard (degrades to a best-effort pid file elsewhere)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def acquire(self) -> bool:
        try:
            import fcntl
            self._fh = open(self.path, "w")
            try:
                fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._fh.close(); self._fh = None
                return False
            self._fh.write(str(os.getpid())); self._fh.flush()
            return True
        except Exception:
            # non-posix fallback: create-exclusive pid file
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode()); os.close(fd)
                return True
            except FileExistsError:
                return False

    def release(self) -> None:
        try:
            if self._fh is not None:
                import fcntl
                fcntl.flock(self._fh, fcntl.LOCK_UN)
                self._fh.close()
            if self.path.exists():
                self.path.unlink()
        except Exception:
            pass


class Supervisor:
    def __init__(self, root, *, deps=None, clock=None, owner="supervisor"):
        import time
        self.clock = clock or time.time
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = owner
        self.deps = deps or JOBS.Deps()
        self.jobs = JS.JobStore(self.root / "jobs.db", clock=self.clock)
        self._state_persist = ST.StatePersistence(self.root / "state.json")
        self.state = self._state_persist.load()
        self.dialogue = DialogueLog(self.root / "dialogue.jsonl")
        self.lock = SingleInstanceLock(self.root / "supervisor.lock")
        self._status_path = self.root / "status.json"
        self._failures_path = self.root / "failures.json"
        self.failures = self._load_failures()
        self._stop = False

    # ── failures set persistence ─────────────────────────────────────────────────
    def _load_failures(self) -> set:
        try:
            return set(json.loads(self._failures_path.read_text(encoding="utf-8")))
        except Exception:
            return set()

    def _save_failures(self) -> None:
        tmp = self._failures_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.failures)), encoding="utf-8")
        os.replace(tmp, self._failures_path)

    # ── lifecycle ────────────────────────────────────────────────────────────────
    def start(self) -> bool:
        if not self.lock.acquire():
            return False
        self._transition(ST.STARTING, "boot", "Supervisor acquired the single-instance lock.", "start")
        return True

    def _transition(self, to_state, reason, explanation, trigger, snapshot_id=None):
        self.state.transition(to_state, reason_code=reason, explanation=explanation, trigger=trigger,
                              snapshot_id=snapshot_id)
        self._state_persist.save(self.state)
        self._write_status()

    def heartbeat(self):
        self._write_status()

    def _write_status(self):
        caps = H.capabilities(self.failures)
        crit = self.jobs.overdue_critical(grace_seconds=0.0)
        d = self.state.as_dict()
        d.update({"heartbeat_ist": ST._now_ist_iso(), "active_failures": sorted(self.failures),
                  "capabilities": caps,
                  "overdue_critical": [j.job_type for j in crit],
                  "jobs": self._job_counts()})
        tmp = self._status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self._status_path)

    def _job_counts(self) -> dict:
        counts: dict = {}
        for j in self.jobs.list(limit=500):
            counts[j.status] = counts.get(j.status, 0) + 1
        return counts

    # ── enqueue due jobs (idempotent) ────────────────────────────────────────────
    def enqueue_due(self, now_ist=None):
        now_ist = now_ist or self.deps.now_ist()
        session_date = now_ist.date().isoformat()
        # auth health — always ensure one pending health probe exists
        self.jobs.enqueue(SCH.AUTH_HEALTH, idempotency_key=f"auth:{session_date}:{now_ist.hour}",
                          critical=True)
        # data refresh — one per session (idempotent); only meaningful once authed
        self.jobs.enqueue(SCH.DATA_REFRESH, idempotency_key=SCH.data_refresh_key(session_date),
                          critical=True)
        snap = self.deps.active_snapshot_id() or "none"
        # scan + paper cycle per slot when the market calendar allows
        if SCH.scan_due(now_ist, None, self.deps.holidays()):
            slot = SCH.scan_slot(now_ist)
            self.jobs.enqueue(SCH.MARKET_SCAN, idempotency_key=SCH.scan_key(snap, slot))
            self.jobs.enqueue(SCH.PAPER_CYCLE,
                              idempotency_key=SCH.paper_cycle_key(snap, f"{session_date}:{slot}"),
                              critical=True)

    # ── one deterministic step ───────────────────────────────────────────────────
    def tick(self, now_ist=None):
        self.jobs.reclaim_expired()
        self.enqueue_due(now_ist)
        self._check_overdue()
        job = self.jobs.lease_due(self.owner, lease_seconds=300.0)
        if job is None:
            self.heartbeat()
            return None
        self._execute(job)
        self.heartbeat()
        return job

    def _execute(self, job):
        handler = JOBS.HANDLERS.get(job.job_type)
        if handler is None:
            self.jobs.complete(job.job_id, JS.PERMANENT_FAILED, error_code="NO_HANDLER",
                               error_message=f"no handler for {job.job_type}")
            self._incident("NO_HANDLER", f"No handler for job {job.job_type}", job)
            return
        ctx = JOBS._Ctx(self.deps)
        try:
            result = handler(ctx)
        except Exception as exc:                        # a bad job never kills the organisation
            self._retry_or_fail(job, error_code="HANDLER_EXCEPTION", error_message=str(exc))
            self._incident("HANDLER_EXCEPTION", f"{job.job_type}: {exc}", job)
            return

        # apply health deltas
        self.failures |= set(result.failures)
        self.failures -= set(result.clears)
        self._save_failures()

        if result.status == JS.RETRYABLE_FAILED:
            self._retry_or_fail(job, error_code=result.error_code, error_message=result.error_message,
                                summary=result.summary)
            self._incident(result.error_code or "RETRYABLE", result.summary or result.error_message, job)
        else:
            self.jobs.complete(job.job_id, result.status, result_summary=result.summary,
                               output_snapshot_id=result.output_snapshot_id,
                               error_code=result.error_code, error_message=result.error_message)
            if result.status in (JS.BLOCKED,):
                self._incident("BLOCKED", result.summary, job)

        # drive the operational state machine — but gating failures are AUTHORITATIVE over an
        # optimistic hint (a scan must not promote the state out of AUTH_REQUIRED / DATA_BLOCKED)
        target = self._gated_state(result.state_hint)
        if target and target != self.state.state:
            self._transition(target, reason=job.job_type,
                             explanation=result.summary or job.job_type, trigger=job.job_id,
                             snapshot_id=result.output_snapshot_id)

    def _gated_state(self, hint):
        if H.AUTH_MISSING in self.failures:
            return ST.AUTH_REQUIRED
        if H.SNAPSHOT_STALE in self.failures:
            return ST.DATA_BLOCKED
        return hint

    def _retry_or_fail(self, job, *, error_code, error_message, summary=""):
        if job.attempt >= _MAX_ATTEMPTS:
            self.jobs.complete(job.job_id, JS.PERMANENT_FAILED, error_code=error_code,
                               error_message=error_message, result_summary=summary)
            return
        backoff = min(_MAX_BACKOFF_S, _BASE_BACKOFF_S * (2 ** job.attempt))
        self.jobs.reschedule_retry(job.job_id, when=self.clock() + backoff, error_code=error_code,
                                   error_message=error_message)

    def _check_overdue(self):
        overdue = self.jobs.overdue_critical(grace_seconds=3600.0)
        if overdue and self.state.state not in (ST.DEGRADED, ST.HALTED):
            names = ", ".join(sorted({j.job_type for j in overdue}))
            self._incident("CRITICAL_OVERDUE", f"Critical jobs overdue: {names}", overdue[0])
            self._transition(ST.DEGRADED, "critical_overdue",
                             f"A critical job is overdue: {names}.", "overdue_check")

    def _incident(self, code, message, job=None):
        self.dialogue.append(Record(record_type=OPERATIONAL_INCIDENT, producer="supervisor",
                                    claim=message, evidence={"error_code": code,
                                    "job_type": getattr(job, "job_type", ""),
                                    "job_id": getattr(job, "job_id", "")}, decision=code))

    # ── production loop ──────────────────────────────────────────────────────────
    def run(self, *, interval_s=15.0, sleep_fn=None, max_iterations=None):
        import time
        sleep_fn = sleep_fn or time.sleep
        i = 0
        while not self._stop:
            self.tick()
            i += 1
            if max_iterations is not None and i >= max_iterations:
                break
            sleep_fn(interval_s)

    def stop(self):
        self._stop = True

    def shutdown(self):
        """Graceful stop: persist state, write a final status, release the lock. State is preserved."""
        self._stop = True
        self._state_persist.save(self.state)
        self._write_status()
        self.jobs.close()
        self.lock.release()


class _Ctx:
    def __init__(self, deps):
        self.deps = deps
