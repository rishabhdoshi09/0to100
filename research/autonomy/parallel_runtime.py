"""Parallel runtime bridge for heavy read/data work.

The autonomy supervisor remains the single deterministic mutation owner for paper
portfolio/risk state. Heavy market scans are executed only by the dedicated
market-operations worker, which already has isolated lanes and an internally
parallel scanner. Corporate-action acquisition runs in one bounded background
I/O lane and only updates the verified CA data files.

This module deliberately patches orchestration, not strategy logic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OPS_DB = ROOT / "logs" / "market_ops" / "jobs.db"
OPS_RUNTIME = ROOT / "logs" / "market_ops" / "runtime.json"
CA_RUNTIME = ROOT / "logs" / "ca_refresh_runtime.json"

_BRIDGE_PENDING = {"MARKET_OP_IN_PROGRESS", "LONG_TERM_OP_IN_PROGRESS"}
_SCAN_REUSE_S = 120.0
_LONG_TERM_REUSE_S = 10 * 60.0

_install_lock = threading.Lock()
_installed = False
_ops_process: subprocess.Popen | None = None
_ca_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qt-ca-backfill")
_ca_lock = threading.Lock()
_ca_future: Future | None = None
_ca_last_start = 0.0


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _ops_store():
    from operations.store import OperationStore
    return OperationStore(OPS_DB)


def _ops_worker_healthy() -> bool:
    from operations.store import pid_is_alive
    try:
        raw = json.loads(OPS_RUNTIME.read_text(encoding="utf-8"))
        heartbeat = float(raw.get("heartbeat_epoch") or 0.0)
        return (
            bool(raw.get("process_running"))
            and 0 <= time.time() - heartbeat <= 10.0
            and pid_is_alive(raw.get("worker_pid"))
        )
    except Exception:
        return False


def _ensure_ops_worker() -> None:
    """Best-effort worker bootstrap for autonomy-only launches; lock prevents duplicates."""
    global _ops_process
    if _ops_worker_healthy():
        return
    if _ops_process is not None and _ops_process.poll() is None:
        return
    env = os.environ.copy()
    existing = str(env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([existing] if existing else []))
    try:
        _ops_process = subprocess.Popen(
            [sys.executable, "-u", "-m", "operations.market_ops"],
            cwd=str(ROOT),
            env=env,
        )
    except Exception:
        _ops_process = None


def _queue_operation(kind: str, *, requested_by: str = "autonomy", priority: int = 40,
                     reuse_s: float = 0.0) -> dict[str, Any]:
    from operations import market_ops as MOPS
    from operations.store import PENDING, RUNNING, SUCCEEDED

    store = _ops_store()
    latest = store.latest(kind)
    now = time.time()
    if latest:
        status = str(latest.get("status") or "")
        if status in {PENDING, RUNNING}:
            _ensure_ops_worker()
            return latest
        if status == SUCCEEDED and reuse_s > 0:
            finished = float(latest.get("finished_at") or latest.get("updated_at") or 0.0)
            if finished and 0 <= now - finished <= float(reuse_s):
                return latest
    operation, _created = store.enqueue(
        kind,
        lane=MOPS.LANES[kind],
        requested_by=requested_by,
        priority=int(priority),
    )
    _ensure_ops_worker()
    return operation


def ensure_market_scan_started(*, requested_by: str = "autonomy") -> dict[str, Any]:
    from operations.market_ops import MARKET_SCAN
    return _queue_operation(
        MARKET_SCAN,
        requested_by=requested_by,
        priority=60 if requested_by == "autonomy" else 100,
        reuse_s=_SCAN_REUSE_S,
    )


def _operation_result(operation: dict[str, Any]) -> dict[str, Any]:
    if not operation:
        return {}
    op_id = str(operation.get("operation_id") or "")
    if not op_id:
        return operation
    try:
        return _ops_store().get(op_id) or operation
    except Exception:
        return operation


def _saved_scan_payload() -> dict[str, Any]:
    try:
        from product.scan_store import load_scan
        return dict(load_scan() or {})
    except Exception:
        return {}


def _delegated_market_scan(ctx):
    """Observe one dedicated market-ops scan instead of running a duplicate scan here."""
    from operations.store import BLOCKED, FAILED, PENDING, RUNNING, SUCCEEDED
    from research.autonomy import health as H
    from research.autonomy import job_store as JS
    from research.autonomy import jobs as JOBS
    from research.autonomy import supervisor_state as ST

    try:
        operation = ensure_market_scan_started(requested_by="autonomy")
        operation = _operation_result(operation)
    except Exception as exc:
        return JOBS.JobResult(
            JS.RETRYABLE_FAILED,
            "dedicated market scan could not be queued",
            error_code="MARKET_OP_QUEUE_ERROR",
            error_message=str(exc),
            state_hint=ST.OBSERVING,
        )
    status = str(operation.get("status") or "")
    metadata = {
        "operation_id": operation.get("operation_id"),
        "operation_status": status,
        "stage": operation.get("stage", ""),
        "progress_current": int(operation.get("progress_current") or 0),
        "progress_total": int(operation.get("progress_total") or 0),
        "execution_plane": "market_ops",
    }
    if status in {PENDING, RUNNING}:
        return JOBS.JobResult(
            JS.RETRYABLE_FAILED,
            "dedicated market scan is running in parallel",
            error_code="MARKET_OP_IN_PROGRESS",
            error_message="market-ops owns the full-universe scan; autonomy will observe completion",
            state_hint=ST.OBSERVING,
            metadata=metadata,
        )
    if status == SUCCEEDED:
        payload = _saved_scan_payload()
        summary = dict(payload.get("summary") or {})
        if not summary:
            summary = dict((operation.get("result") or {}).get("summary") or {})
        n = int(summary.get("with_any_setup") or summary.get("qualified") or 0)
        return JOBS.JobResult(
            JS.SUCCEEDED,
            f"shared market scan complete · {n} setups · market-ops",
            state_hint=ST.OBSERVING,
            unblocks=(JOBS.DEP_SCAN,),
            metadata={**summary, **metadata},
        )
    if status == BLOCKED:
        return JOBS.JobResult(
            JS.RETRYABLE_FAILED,
            "dedicated market scan blocked",
            error_code=str(operation.get("error_code") or "MARKET_OP_BLOCKED"),
            error_message=str(operation.get("error_message") or operation.get("message") or "market scan blocked"),
            state_hint=ST.OBSERVING,
            metadata=metadata,
        )
    if status == FAILED:
        return JOBS.JobResult(
            JS.RETRYABLE_FAILED,
            "dedicated market scan failed",
            error_code=str(operation.get("error_code") or "MARKET_OP_FAILED"),
            error_message=str(operation.get("error_message") or operation.get("message") or "market scan failed"),
            state_hint=ST.OBSERVING,
            metadata=metadata,
        )
    return JOBS.JobResult(
        JS.RETRYABLE_FAILED,
        "dedicated market scan state unavailable",
        error_code="MARKET_OP_STATE_UNKNOWN",
        error_message=status or "unknown operation status",
        state_hint=ST.OBSERVING,
        failures={H.PROVIDER_UNAVAILABLE} if not _ops_worker_healthy() else set(),
        metadata=metadata,
    )


def _delegated_long_term(ctx, *, refresh: bool):
    from operations.market_ops import LONG_TERM_REFRESH, LONG_TERM_SCAN
    from operations.store import BLOCKED, FAILED, PENDING, RUNNING, SUCCEEDED
    from research.autonomy import job_store as JS
    from research.autonomy import jobs as JOBS
    from research.autonomy import supervisor_state as ST

    kind = LONG_TERM_REFRESH if refresh else LONG_TERM_SCAN
    try:
        operation = _queue_operation(kind, requested_by="autonomy", priority=35, reuse_s=_LONG_TERM_REUSE_S)
        operation = _operation_result(operation)
    except Exception as exc:
        return JOBS.JobResult(
            JS.RETRYABLE_FAILED,
            "long-term operation could not be queued",
            error_code="LONG_TERM_OP_QUEUE_ERROR",
            error_message=str(exc),
        )
    status = str(operation.get("status") or "")
    metadata = {
        "operation_id": operation.get("operation_id"),
        "operation_status": status,
        "execution_plane": "market_ops",
        **dict(operation.get("result") or {}),
    }
    if status in {PENDING, RUNNING}:
        return JOBS.JobResult(
            JS.RETRYABLE_FAILED,
            "long-term scan is running in its isolated lane",
            error_code="LONG_TERM_OP_IN_PROGRESS",
            error_message="market-ops long_term lane owns this scan",
            state_hint=ST.OBSERVING,
            metadata=metadata,
        )
    if status == SUCCEEDED:
        return JOBS.JobResult(
            JS.SUCCEEDED,
            "long-term scan complete · market-ops",
            state_hint=ST.OBSERVING,
            metadata=metadata,
        )
    return JOBS.JobResult(
        JS.RETRYABLE_FAILED,
        "long-term scan unavailable",
        error_code=str(operation.get("error_code") or ("LONG_TERM_OP_BLOCKED" if status == BLOCKED else "LONG_TERM_OP_FAILED")),
        error_message=str(operation.get("error_message") or operation.get("message") or status),
        metadata=metadata,
    )


def _run_ca_refresh() -> dict[str, Any]:
    from data.corporate_actions_resilient import refresh_events_resilient

    _atomic_json(CA_RUNTIME, {
        "running": True,
        "started_at": time.time(),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
    })
    try:
        result = refresh_events_resilient(years=5, budget_s=90.0)
        _atomic_json(CA_RUNTIME, {
            "running": False,
            "finished_at": time.time(),
            "finished_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
            "result": result,
        })
        return result
    except Exception as exc:
        _atomic_json(CA_RUNTIME, {
            "running": False,
            "finished_at": time.time(),
            "pid": os.getpid(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def _ensure_ca_background(*, force: bool = False) -> Future | None:
    global _ca_future, _ca_last_start
    now = time.time()
    with _ca_lock:
        if _ca_future is not None and not _ca_future.done():
            return _ca_future
        # Failed/partial sources should retry, but not on every 15-second heartbeat.
        if not force and _ca_last_start and now - _ca_last_start < 5 * 60:
            return _ca_future
        _ca_last_start = now
        _ca_future = _ca_executor.submit(_run_ca_refresh)
        return _ca_future


def _ca_status() -> dict[str, Any]:
    try:
        from data.corporate_actions_resilient import coverage_status
        return dict(coverage_status(years=5) or {})
    except Exception as exc:
        return {
            "available": False,
            "coverage_complete": False,
            "refresh_due": True,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _background_corporate_actions(ctx):
    """Schedule CA I/O and return immediately; historical capability stays honest."""
    from research.autonomy import health as H
    from research.autonomy import job_store as JS
    from research.autonomy import jobs as JOBS

    status = _ca_status()
    if status.get("refresh_due") or not status.get("coverage_complete"):
        _ensure_ca_background()
    running = bool(_ca_future is not None and not _ca_future.done())
    failures = set() if status.get("coverage_complete") else {H.CA_INCOMPLETE}
    clears = {H.CA_INCOMPLETE, H.OPTIONS_HISTORY_INCOMPLETE} if status.get("coverage_complete") else set()
    return JOBS.JobResult(
        JS.SUCCEEDED,
        (
            f"corporate actions background={'running' if running else 'idle'} · "
            f"coverage {int(status.get('windows_complete') or 0)}/{int(status.get('windows_total') or 0)} · "
            f"events {int(status.get('n_events') or 0)}"
        ),
        failures=failures,
        clears=clears,
        metadata={**status, "background_running": running},
    )


def _reconcile_ca_failure(supervisor) -> None:
    """Update only the capability flag; CA completeness never blocks cash scanning."""
    from research.autonomy import health as H

    status = _ca_status()
    if status.get("refresh_due") or not status.get("coverage_complete"):
        _ensure_ca_background()
    before = set(supervisor.failures)
    if status.get("coverage_complete"):
        supervisor.failures.discard(H.CA_INCOMPLETE)
    else:
        supervisor.failures.add(H.CA_INCOMPLETE)
    if before != supervisor.failures:
        supervisor._save_failures()


def install_parallel_runtime() -> None:
    """Install idempotent orchestration patches before the Supervisor is constructed."""
    global _installed
    with _install_lock:
        if _installed:
            return
        from research.autonomy import jobs as JOBS
        from research.autonomy import schedules as SCH
        from research.autonomy.supervisor import Supervisor

        # One heavy-execution plane. Autonomy observes durable operation results.
        JOBS.HANDLERS[SCH.MARKET_SCAN] = _delegated_market_scan
        JOBS.HANDLERS[SCH.LONG_TERM_SCAN] = lambda ctx: _delegated_long_term(ctx, refresh=False)
        JOBS.HANDLERS[SCH.LONG_TERM_REFRESH] = lambda ctx: _delegated_long_term(ctx, refresh=True)
        JOBS.HANDLERS[SCH.CORPORATE_ACTIONS] = _background_corporate_actions

        if getattr(Supervisor, "_parallel_runtime_installed", False):
            _installed = True
            return

        original_enqueue_due = Supervisor.enqueue_due
        original_retry_or_fail = Supervisor._retry_or_fail
        original_incident = Supervisor._incident

        def enqueue_due_parallel(self, now_ist=None):
            result = original_enqueue_due(self, now_ist)
            current = now_ist or self.deps.now_ist()
            # Launch the market scan before the serial autonomy worker leases a
            # potentially multi-minute DATA_REFRESH. The dedicated lane can then
            # scan in parallel while snapshot reconciliation continues.
            try:
                if SCH.scan_slot(current, self.deps.holidays()):
                    ensure_market_scan_started(requested_by="autonomy")
            except Exception:
                pass
            try:
                _reconcile_ca_failure(self)
            except Exception:
                pass
            return result

        def retry_or_fail_parallel(self, job, *, error_code, error_message, summary=""):
            if error_code in _BRIDGE_PENDING:
                # Polling a durable external operation is not a failed attempt and
                # must never exhaust the autonomy retry budget.
                self.jobs.reschedule_retry(
                    job.job_id,
                    when=self.clock() + 1.0,
                    error_code=error_code,
                    error_message=error_message,
                )
                return
            return original_retry_or_fail(
                self,
                job,
                error_code=error_code,
                error_message=error_message,
                summary=summary,
            )

        def incident_parallel(self, code, message, job=None):
            if code in _BRIDGE_PENDING:
                return None
            return original_incident(self, code, message, job)

        Supervisor.enqueue_due = enqueue_due_parallel
        Supervisor._retry_or_fail = retry_or_fail_parallel
        Supervisor._incident = incident_parallel
        Supervisor._parallel_runtime_installed = True
        _installed = True
