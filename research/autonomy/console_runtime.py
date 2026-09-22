"""Visible, resilient console driver for the QuantTerm autonomy supervisor.

The durable Supervisor remains the scheduler and mutation owner. This module only drives its ticks,
prints operator-readable activity, and prevents one unexpected tick exception from silently killing
the process. It also owns a lightweight runtime heartbeat that remains fresh while a long blocking
job is executing; durable job/state truth still belongs to the Supervisor.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
import threading
import time
import traceback
from typing import Callable

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST


_POLL_ERROR_CODES = {
    "DATA_REFRESH_IN_PROGRESS",
    "MARKET_OP_IN_PROGRESS",
    "LONG_TERM_OP_IN_PROGRESS",
    "HISTORICAL_PAPER_IN_PROGRESS",
}


_HEARTBEAT_CONSOLE_LOCK = threading.Lock()
_HEARTBEAT_CONSOLE_STATE: dict[int, dict[str, object]] = {}
_DURATION_TOKEN = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?s\b")


def _normalise_progress_text(value: str) -> str:
    """Ignore elapsed-time churn while preserving material stage/progress changes."""
    return _DURATION_TOKEN.sub("<elapsed>", str(value or "").strip())


def _is_background_poll(job) -> bool:
    return str(getattr(job, "error_code", "") or "") in _POLL_ERROR_CODES


def _stamp() -> str:
    from core.market_clock import console_stamp

    try:
        return console_stamp()
    except Exception:
        return time.strftime("%H:%M:%S")


def _emit(kind: str, message: str) -> None:
    print(f"[{_stamp()}] {kind:<11} {message}", flush=True)


def _next_job(supervisor) -> str:
    try:
        pending = supervisor.jobs.list(status=JS.PENDING, limit=1)
        if pending:
            job = pending[0]
            if _is_background_poll(job):
                detail = str(job.result_summary or job.error_message or "background work in progress")
                return f"{job.job_type} · {detail}"
            return f"{job.job_type} (attempt {job.attempt})"
    except Exception:
        pass
    return "none due"


def _phase(supervisor) -> str:
    try:
        now = supervisor.deps.now_ist()
        return str(SCH.session_phase(now, supervisor.deps.holidays()))
    except Exception:
        return "unknown"


def _runtime_path(supervisor) -> Path:
    return Path(supervisor.root) / "runtime.json"


def _write_runtime_status(
    supervisor,
    *,
    process_running: bool,
    active_job: dict | None = None,
) -> None:
    """Write process liveness without touching the supervisor's durable status snapshot.

    This intentionally avoids ``Supervisor.heartbeat()`` because a worker heartbeat may run while
    the main thread is mutating job/state records. The runtime file contains only ephemeral process
    liveness and the currently executing job; it is never a scheduling or trading source of truth.
    """
    path = _runtime_path(supervisor)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "heartbeat_ist": ST._now_ist_iso(),
        "process_running": bool(process_running),
        "scheduler_owner_pid": os.getpid(),
        "state": str(getattr(supervisor.state, "state", "UNKNOWN")),
        "active_job": dict(active_job or {}),
    }
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _heartbeat(
    supervisor,
    *,
    force: bool,
    last_at: float,
    every_s: float,
    active_job: dict | None = None,
) -> float:
    """Emit operator heartbeat only on material change or long proof-of-life.

    runtime.json is written independently at a much faster cadence, so suppressing
    identical console lines never weakens liveness detection.
    """
    now = time.monotonic()
    if not force and every_s > 0 and now - last_at < every_s:
        return last_at
    try:
        counts = supervisor._job_counts()
    except Exception:
        counts = {}
    failures = sorted(getattr(supervisor, "failures", set()) or set())
    failure_text = ",".join(failures[:4]) if failures else "none"
    next_text = _next_job(supervisor)
    active_text = ""
    active_signature: tuple = ()
    if active_job:
        started = float(active_job.get("started_monotonic", now) or now)
        elapsed = max(0.0, now - started)
        active_text = (
            f" · active={active_job.get('job_type', 'unknown')} "
            f"({elapsed:.0f}s, attempt {active_job.get('attempt', 0)})"
        )
        active_signature = (
            str(active_job.get("job_id") or ""),
            str(active_job.get("job_type") or ""),
            int(active_job.get("attempt") or 0),
            bool(active_job.get("background_poll")),
        )
    try:
        activity = str((supervisor._activity_truth() or {}).get("activity") or "UNKNOWN")
    except Exception:
        activity = "UNKNOWN"

    state = str(getattr(supervisor.state, "state", "UNKNOWN"))
    phase = _phase(supervisor)
    signature = (
        state,
        phase,
        activity,
        int(counts.get(JS.PENDING, 0)),
        int(counts.get(JS.RUNNING, 0)),
        int(counts.get(JS.BLOCKED, 0)),
        int(counts.get(JS.PERMANENT_FAILED, 0)),
        _normalise_progress_text(next_text),
        active_signature,
        tuple(failures),
    )
    max_silence_s = max(300.0, float(every_s or 0.0) * 10.0)
    key = id(supervisor)
    with _HEARTBEAT_CONSOLE_LOCK:
        prior = dict(_HEARTBEAT_CONSOLE_STATE.get(key) or {})
        changed = prior.get("signature") != signature
        last_emit = float(prior.get("last_emit") or 0.0)
        should_emit = bool(changed or not last_emit or now - last_emit >= max_silence_s)
        _HEARTBEAT_CONSOLE_STATE[key] = {
            "signature": signature,
            "last_emit": now if should_emit else last_emit,
        }

    if should_emit:
        _emit(
            "HEARTBEAT",
            (
                f"pid={os.getpid()} · state={state} · activity={activity} · "
                f"phase={phase} · jobs "
                f"P:{counts.get(JS.PENDING, 0)} R:{counts.get(JS.RUNNING, 0)} "
                f"B:{counts.get(JS.BLOCKED, 0)} F:{counts.get(JS.PERMANENT_FAILED, 0)} · "
                f"next={next_text}{active_text} · failures={failure_text}"
            ),
        )
    # Returning 'now' throttles callers even when an identical console line was
    # deliberately suppressed; the independent runtime pulse still advances.
    return now


def run_visible_loop(
    supervisor,
    *,
    interval_s: float = 15.0,
    max_iterations: int | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    heartbeat_s: float = 30.0,
) -> None:
    """Drive one Supervisor with visible activity and per-tick fault containment."""
    sleep_fn = sleep_fn or time.sleep
    count = 0
    consecutive_errors = 0
    active_lock = threading.Lock()
    active: dict = {}
    runtime_stop = threading.Event()

    def active_snapshot() -> dict:
        with active_lock:
            return dict(active)

    _write_runtime_status(supervisor, process_running=True, active_job={})
    last_heartbeat = _heartbeat(
        supervisor,
        force=True,
        last_at=0.0,
        every_s=heartbeat_s,
        active_job={},
    )

    # Independent process-liveness pulse. It continues while ``tick()`` is blocked inside a long
    # scan/data refresh, which prevents the web terminal from falsely declaring autonomy offline.
    runtime_interval = max(1.0, min(15.0, float(heartbeat_s or 30.0) / 2.0))

    def runtime_worker() -> None:
        next_console = time.monotonic() + max(1.0, float(heartbeat_s or 30.0))
        while not runtime_stop.wait(runtime_interval):
            current = active_snapshot()
            if current.get("job_id"):
                try:
                    renewed = supervisor.jobs.renew_lease(
                        str(current["job_id"]),
                        supervisor.owner,
                        lease_seconds=300.0,
                    )
                    if not renewed:
                        _emit(
                            "LEASE WARN",
                            f"could not renew {current.get('job_type') or 'job'} "
                            f"id={current.get('job_id')}",
                        )
                except Exception as exc:
                    _emit(
                        "LEASE WARN",
                        f"lease renewal failed: {type(exc).__name__}: {exc}",
                    )
            try:
                _write_runtime_status(
                    supervisor,
                    process_running=True,
                    active_job=current,
                )
            except Exception as exc:
                _emit("HB ERROR", f"runtime heartbeat write failed: {type(exc).__name__}: {exc}")
            now = time.monotonic()
            if now >= next_console:
                _heartbeat(
                    supervisor,
                    force=True,
                    last_at=0.0,
                    every_s=heartbeat_s,
                    active_job=current,
                )
                next_console = now + max(1.0, float(heartbeat_s or 30.0))

    runtime_thread = threading.Thread(
        target=runtime_worker,
        name="quantterm-runtime-heartbeat",
        daemon=True,
    )
    runtime_thread.start()

    original_execute = supervisor._execute
    elapsed_by_job: dict[str, float] = {}
    polled_jobs: set[str] = set()
    progress_signature_by_job: dict[str, str] = {}

    def visible_execute(job):
        started = time.monotonic()
        background_poll = _is_background_poll(job)
        current = {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "attempt": job.attempt,
            "critical": bool(getattr(job, "critical", False)),
            "background_poll": background_poll,
            "started_ist": ST._now_ist_iso(),
            "started_monotonic": started,
        }
        with active_lock:
            active.clear()
            active.update(current)
        try:
            _write_runtime_status(supervisor, process_running=True, active_job=current)
        except Exception:
            pass
        if background_poll:
            if job.job_id not in polled_jobs:
                _emit(
                    "JOB POLL",
                    f"{job.job_type} · id={job.job_id} · checking existing background worker",
                )
                polled_jobs.add(job.job_id)
        else:
            _emit(
                "JOB START",
                f"{job.job_type} · id={job.job_id} · attempt={job.attempt}"
                + (" · critical" if getattr(job, "critical", False) else ""),
            )
        try:
            return original_execute(job)
        finally:
            elapsed_by_job[job.job_id] = time.monotonic() - started
            with active_lock:
                active.clear()
            try:
                _write_runtime_status(supervisor, process_running=True, active_job={})
            except Exception:
                pass

    supervisor._execute = visible_execute
    try:
        while not supervisor._stop:
            job = None
            try:
                before_state = getattr(supervisor.state, "state", "UNKNOWN")
                job = supervisor.tick()
                consecutive_errors = 0
                after_state = getattr(supervisor.state, "state", "UNKNOWN")
                if after_state != before_state:
                    _emit("STATE", f"{before_state} → {after_state} · {supervisor.state.explanation}")

                if job is not None:
                    final = supervisor.jobs.get(job.job_id)
                    final = final or job
                    summary = final.result_summary or final.error_message or "no summary"
                    elapsed = elapsed_by_job.pop(job.job_id, 0.0)
                    if final.status == JS.PENDING and _is_background_poll(final):
                        progress_signature = _normalise_progress_text(summary)
                        if progress_signature_by_job.get(final.job_id) != progress_signature:
                            _emit(
                                "JOB PROGRESS",
                                f"{final.job_type} · {summary}",
                            )
                            progress_signature_by_job[final.job_id] = progress_signature
                    else:
                        progress_signature_by_job.pop(final.job_id, None)
                        polled_jobs.discard(final.job_id)
                        _emit(
                            "JOB DONE",
                            f"{final.job_type} → {final.status} · {elapsed:.1f}s · "
                            f"attempt={final.attempt} · {summary}",
                        )
                    last_heartbeat = _heartbeat(
                        supervisor,
                        force=True,
                        last_at=last_heartbeat,
                        every_s=heartbeat_s,
                        active_job={},
                    )
                else:
                    last_heartbeat = _heartbeat(
                        supervisor,
                        force=False,
                        last_at=last_heartbeat,
                        every_s=heartbeat_s,
                        active_job={},
                    )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                consecutive_errors += 1
                _emit(
                    "LOOP ERROR",
                    f"{type(exc).__name__}: {exc} · retrying (consecutive={consecutive_errors})",
                )
                traceback.print_exc()
                try:
                    supervisor._incident(
                        "SUPERVISOR_TICK_EXCEPTION",
                        f"{type(exc).__name__}: {exc}",
                    )
                except Exception:
                    pass
                try:
                    if consecutive_errors >= 3 and getattr(supervisor.state, "state", "") != ST.HALTED:
                        supervisor._transition(
                            ST.DEGRADED,
                            "tick_exception",
                            f"Autonomy loop recovered from {consecutive_errors} consecutive tick errors.",
                            "console_runtime",
                        )
                    supervisor.heartbeat()
                except Exception:
                    pass
                last_heartbeat = _heartbeat(
                    supervisor,
                    force=True,
                    last_at=last_heartbeat,
                    every_s=heartbeat_s,
                    active_job=active_snapshot(),
                )

            count += 1
            if max_iterations is not None and count >= max_iterations:
                break

            if consecutive_errors:
                delay = min(60.0, max(1.0, 2.0 ** min(consecutive_errors, 5)))
            else:
                delay = max(0.1, float(interval_s))
            sleep_fn(delay)
    finally:
        supervisor._execute = original_execute
        runtime_stop.set()
        runtime_thread.join(timeout=max(2.0, runtime_interval + 0.5))
        try:
            _write_runtime_status(
                supervisor,
                process_running=False,
                active_job=active_snapshot(),
            )
        except Exception:
            pass
        with _HEARTBEAT_CONSOLE_LOCK:
            _HEARTBEAT_CONSOLE_STATE.pop(id(supervisor), None)
