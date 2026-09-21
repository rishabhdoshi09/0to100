"""Non-blocking DATA_REFRESH bridge for the autonomy supervisor.

DATA_REFRESH performs genuine network/disk snapshot work and can legitimately take
minutes on a catch-up day. It must not monopolise the single mutation-owner loop.
This module runs the existing canonical DATA_REFRESH handler in one daemon data
thread and makes the durable autonomy job poll that work.

The original handler remains the authority for auth, freshness, snapshot validation
and fail-safe semantics. This bridge changes execution placement only; it does not
weaken data gates or allow paper entries before fresh data succeeds.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable
from core.runtime_paths import logs_dir, logs_path

ROOT = Path(__file__).resolve().parents[2]
PROGRESS_PATH = logs_path("kite_history", "runtime_progress.json")
IN_PROGRESS = "DATA_REFRESH_IN_PROGRESS"
_REUSE_SUCCESS_S = 15 * 60.0
_STALL_WARN_S = 10 * 60.0
_STUCK_DEAD_THREAD_S = 15.0

_install_lock = threading.Lock()
_installed = False


def _progress_payload() -> dict[str, Any]:
    try:
        payload = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _required_date(ctx) -> str:
    return str(getattr(ctx, "required_session_date", "") or "")[:10]


def _latest_date(result) -> str:
    try:
        metadata = dict(getattr(result, "metadata", {}) or {})
    except Exception:
        metadata = {}
    return str(metadata.get("latest_date") or metadata.get("session_date") or "")[:10]


def _is_success(result) -> bool:
    from research.autonomy import job_store as JS
    return getattr(result, "status", "") == JS.SUCCEEDED


def _success_satisfies(result, required: str) -> bool:
    if not _is_success(result):
        return False
    if not required:
        return True
    latest = _latest_date(result)
    return bool(latest and latest >= required)


def make_parallel_data_refresh_handler(
    original_handler: Callable,
    *,
    clock: Callable[[], float] = time.time,
):
    """Return a polling handler around the existing synchronous DATA_REFRESH handler."""
    state_lock = threading.Lock()
    state: dict[str, Any] = {
        "running": False,
        "started_at": 0.0,
        "finished_at": 0.0,
        "required": "",
        "result": None,
        "thread": None,
    }

    def launch(ctx, required: str) -> None:
        state["running"] = True
        state["started_at"] = float(clock())
        state["finished_at"] = 0.0
        state["required"] = required
        state["result"] = None

        def worker() -> None:
            try:
                result = original_handler(ctx)
            except Exception as exc:
                from research.autonomy import job_store as JS
                from research.autonomy import jobs as JOBS
                from research.autonomy import health as H
                from research.autonomy import supervisor_state as ST

                result = JOBS.JobResult(
                    JS.RETRYABLE_FAILED,
                    "data refresh worker failed",
                    error_code="DATA_REFRESH_WORKER_ERROR",
                    error_message=f"{type(exc).__name__}: {exc}",
                    failures={H.SNAPSHOT_STALE},
                    state_hint=ST.DATA_BLOCKED,
                    new_entries_allowed=False,
                )
            with state_lock:
                state["result"] = result
                state["running"] = False
                state["finished_at"] = float(clock())

        thread = threading.Thread(target=worker, name="qt-data-refresh", daemon=True)
        state["thread"] = thread
        thread.start()

    def in_progress_result(required: str):
        from research.autonomy import job_store as JS
        from research.autonomy import jobs as JOBS
        from research.autonomy import supervisor_state as ST

        with state_lock:
            started = float(state.get("started_at") or clock())
            worker_required = str(state.get("required") or "")
        elapsed = max(0.0, float(clock()) - started)
        progress = _progress_payload()
        progress_started = float(progress.get("started_epoch") or 0.0)
        # Ignore telemetry left behind by a previous worker. A stale file must
        # never make a newly launched refresh look stalled.
        if not progress_started or progress_started + 1.0 < started:
            progress = {}
        stage = str(progress.get("stage") or "starting")
        current = int(progress.get("progress_current") or 0)
        total = int(progress.get("progress_total") or 0)
        pct = progress.get("percent_complete")
        rate = float(progress.get("symbols_per_sec") or 0.0)
        last_progress_epoch = float(progress.get("last_progress_epoch") or 0.0)
        progress_age = (
            max(0.0, float(clock()) - last_progress_epoch)
            if last_progress_epoch
            else None
        )
        stall_warning = bool(progress_age is not None and progress_age >= _STALL_WARN_S)
        parts = [f"data refresh running in background · {stage}"]
        if total > 0:
            pct_text = f"{float(pct):.1f}%" if pct is not None else f"{(100.0 * current / total):.1f}%"
            parts.append(f"{current}/{total} · {pct_text}")
        if rate > 0:
            parts.append(f"{rate:.2f} sym/s")
        parts.append(f"{elapsed:.0f}s")
        if stall_warning:
            parts.append(f"stall warning: no measurable progress for {progress_age:.0f}s")
        elif not progress:
            parts.append("progress telemetry starting")
        summary = " · ".join(parts)
        next_poll = 2.0
        return JOBS.JobResult(
            JS.RETRYABLE_FAILED,
            summary,
            error_code=IN_PROGRESS,
            error_message="snapshot refresh is still running; supervisor remains available",
            state_hint=ST.DATA_REFRESHING,
            new_entries_allowed=False,
            metadata={
                "execution_plane": "background_data",
                "elapsed_s": round(elapsed, 1),
                "required_date": required,
                "worker_required_date": worker_required,
                "stall_warning": stall_warning,
                "progress_age_s": round(progress_age, 1) if progress_age is not None else None,
                "progress_current": current,
                "progress_total": total,
                "percent_complete": pct,
                "symbols_per_sec": rate,
                "next_poll_at": round(float(clock()) + next_poll, 1),
                "progress": progress,
            },
        )

    def _dead_worker_result(required: str):
        from research.autonomy import health as H
        from research.autonomy import job_store as JS
        from research.autonomy import jobs as JOBS
        from research.autonomy import supervisor_state as ST

        return JOBS.JobResult(
            JS.RETRYABLE_FAILED,
            "data refresh worker stopped without a result",
            error_code="DATA_REFRESH_WORKER_STUCK",
            error_message="background data thread died; supervisor can lease other work",
            failures={H.SNAPSHOT_STALE},
            state_hint=ST.DATA_BLOCKED,
            new_entries_allowed=False,
            metadata={
                "execution_plane": "background_data",
                "required_date": required,
                "stall_warning": True,
            },
        )

    def handler(ctx):
        required = _required_date(ctx)
        now = float(clock())
        with state_lock:
            running = bool(state.get("running"))
            result = state.get("result")
            finished_at = float(state.get("finished_at") or 0.0)
            thread = state.get("thread")
            started = float(state.get("started_at") or 0.0)
            dead = bool(
                running
                and thread is not None
                and not thread.is_alive()
                and result is None
                and started
                and now - started >= _STUCK_DEAD_THREAD_S
            )
            if dead:
                state["running"] = False
                return _dead_worker_result(required)

            # A successful refresh may be reused only when it satisfies the date
            # required by THIS durable job. A yesterday-success is never returned
            # as success for a newer EOD requirement.
            if result is not None and _success_satisfies(result, required):
                if finished_at and 0 <= now - finished_at <= _REUSE_SUCCESS_S:
                    return result

            if result is not None and not running:
                if not _is_success(result):
                    # Deliver a genuine canonical failure exactly once. The normal
                    # durable retry policy decides when another attempt starts.
                    state["result"] = None
                    return result
                # Success exists but is stale/insufficient for the newly requested
                # session. Discard it and launch a fresh canonical refresh below.
                state["result"] = None

            if not running:
                launch(ctx, required)

        return in_progress_result(required)

    # Expose only read-only diagnostic state for tests / health projection.
    handler.runtime_state = state  # type: ignore[attr-defined]
    return handler


def install_parallel_data_refresh() -> None:
    """Install once, after the general parallel-runtime bridge."""
    global _installed
    with _install_lock:
        if _installed:
            return
        from research.autonomy import jobs as JOBS
        from research.autonomy import schedules as SCH
        from research.autonomy.supervisor import Supervisor

        original_handler = JOBS.HANDLERS.get(SCH.DATA_REFRESH)
        if original_handler is None:
            raise RuntimeError("DATA_REFRESH handler is not registered")
        JOBS.HANDLERS[SCH.DATA_REFRESH] = make_parallel_data_refresh_handler(original_handler)

        if not getattr(Supervisor, "_data_refresh_parallel_installed", False):
            original_retry = Supervisor._retry_or_fail
            original_incident = Supervisor._incident

            def retry_or_fail_nonblocking(self, job, *, error_code, error_message, summary=""):
                if error_code == IN_PROGRESS:
                    # Polling background I/O is not a failed attempt. The job-store
                    # poll reschedule reverses the lease increment so durable attempt
                    # counts represent real executions/failures, never heartbeat polls.
                    self.jobs.reschedule_poll(
                        job.job_id,
                        when=self.clock() + 2.0,
                        error_code=error_code,
                        error_message=error_message,
                        result_summary=summary or "data refresh running in background",
                    )
                    return
                return original_retry(
                    self,
                    job,
                    error_code=error_code,
                    error_message=error_message,
                    summary=summary,
                )

            def incident_nonblocking(self, code, message, job=None):
                if code == IN_PROGRESS:
                    return None
                return original_incident(self, code, message, job)

            Supervisor._retry_or_fail = retry_or_fail_nonblocking
            Supervisor._incident = incident_nonblocking
            Supervisor._data_refresh_parallel_installed = True

        _installed = True
