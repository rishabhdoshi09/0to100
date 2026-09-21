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
RUNTIME_PROGRESS_PATH = logs_path("kite_history", "runtime_progress.json")
IN_PROGRESS = "DATA_REFRESH_IN_PROGRESS"
_REUSE_SUCCESS_S = 15 * 60.0
_STALL_WARN_S = 10 * 60.0
_STUCK_DEAD_THREAD_S = 15.0

_install_lock = threading.Lock()
_installed = False


def _progress_payload() -> dict[str, Any]:
    try:
        payload = json.loads(RUNTIME_PROGRESS_PATH.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _progress_token(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    keys = (
        "stage", "status", "current", "total", "symbol", "cid",
        "candles_fetched", "updated_epoch", "error",
    )
    return json.dumps([payload.get(key) for key in keys], separators=(",", ":"), default=str)


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return ""
    seconds_i = int(round(seconds))
    if seconds_i < 60:
        return f"{seconds_i}s"
    minutes, sec = divmod(seconds_i, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minute = divmod(minutes, 60)
    return f"{hours}h{minute:02d}m"


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
        "baseline_progress_token": "",
        "last_progress_token": "",
        "last_progress_at": 0.0,
        "last_progress_current": -1,
        "last_progress_observed_at": 0.0,
    }

    def launch(ctx, required: str) -> None:
        started = float(clock())
        baseline = _progress_token(_progress_payload())
        state["running"] = True
        state["started_at"] = started
        state["finished_at"] = 0.0
        state["required"] = required
        state["result"] = None
        state["baseline_progress_token"] = baseline
        state["last_progress_token"] = ""
        state["last_progress_at"] = started
        state["last_progress_current"] = -1
        state["last_progress_observed_at"] = started

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

        now = float(clock())
        progress = _progress_payload()
        token = _progress_token(progress)
        with state_lock:
            started = float(state.get("started_at") or now)
            worker_required = str(state.get("required") or "")
            baseline = str(state.get("baseline_progress_token") or "")
            last_token = str(state.get("last_progress_token") or "")
            progress_is_current = bool(progress and token and token != baseline)
            if progress_is_current and token != last_token:
                state["last_progress_token"] = token
                state["last_progress_at"] = now
                try:
                    state["last_progress_current"] = int(progress.get("current") or 0)
                except Exception:
                    state["last_progress_current"] = -1
                state["last_progress_observed_at"] = now
            last_progress_at = float(state.get("last_progress_at") or started)

        elapsed = max(0.0, now - started)
        no_progress_s = max(0.0, now - last_progress_at)
        stalled = bool(elapsed >= _STALL_WARN_S and no_progress_s >= _STALL_WARN_S)

        stage = "starting"
        current = total = 0
        pct = 0.0
        rate = 0.0
        eta_s = None
        symbol = ""
        if progress_is_current:
            stage = str(progress.get("stage") or progress.get("status") or "historical_sync")
            try:
                current = max(0, int(progress.get("current") or 0))
                total = max(0, int(progress.get("total") or 0))
            except Exception:
                current = total = 0
            symbol = str(progress.get("symbol") or "")
            if total:
                pct = min(100.0, max(0.0, (100.0 * current / total)))
            if current > 0 and elapsed > 0:
                rate = current / elapsed
                if total > current and rate > 0:
                    eta_s = (total - current) / rate

        detail = stage
        if total:
            detail += f" · {current}/{total} ({pct:.1f}%)"
        if rate > 0:
            detail += f" · {rate:.2f} sym/s"
        eta_text = _format_eta(eta_s)
        if eta_text:
            detail += f" · eta {eta_text}"
        if symbol:
            detail += f" · {symbol}"
        if stalled:
            detail += f" · stalled {no_progress_s:.0f}s without measurable progress"

        next_poll = 2.0
        return JOBS.JobResult(
            JS.RETRYABLE_FAILED,
            f"data refresh running in background · {detail}",
            error_code=IN_PROGRESS,
            error_message="snapshot refresh is still running; supervisor remains available",
            state_hint=ST.DATA_REFRESHING,
            new_entries_allowed=False,
            metadata={
                "execution_plane": "background_data",
                "elapsed_s": round(elapsed, 1),
                "required_date": required,
                "worker_required_date": worker_required,
                "stall_warning": stalled,
                "stall_basis": "NO_MEASURABLE_PROGRESS" if stalled else "",
                "no_progress_s": round(no_progress_s, 1),
                "progress_current": current,
                "progress_total": total,
                "progress_pct": round(pct, 2),
                "throughput_symbols_per_s": round(rate, 4),
                "eta_s": round(float(eta_s), 1) if eta_s is not None else None,
                "next_poll_at": round(now + next_poll, 1),
                "progress": progress if progress_is_current else {},
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
