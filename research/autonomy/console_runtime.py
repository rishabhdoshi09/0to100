"""Visible, resilient console driver for the QuantTerm autonomy supervisor.

The durable Supervisor remains the scheduler and mutation owner. This module only drives its ticks,
prints operator-readable activity, and prevents one unexpected tick exception from silently killing
the process. It also owns a lightweight runtime heartbeat that remains fresh while a long blocking
job is executing; durable job/state truth still belongs to the Supervisor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Callable

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST


def _stamp() -> str:
    try:
        value = ST._now_ist_iso()
        return value[11:19] if len(value) >= 19 else value
    except Exception:
        return time.strftime("%H:%M:%S")


def _emit(kind: str, message: str) -> None:
    print(f"[{_stamp()}] {kind:<11} {message}", flush=True)


def _next_job(supervisor) -> str:
    try:
        pending = supervisor.jobs.list(status=JS.PENDING, limit=1)
        if pending:
            job = pending[0]
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
    now = time.monotonic()
    if not force and now - last_at < every_s:
        return last_at
    try:
        counts = supervisor._job_counts()
    except Exception:
        counts = {}
    failures = sorted(getattr(supervisor, "failures", set()) or set())
    failure_text = ",".join(failures[:4]) if failures else "none"
    active_text = ""
    if active_job:
        started = float(active_job.get("started_monotonic", now) or now)
        elapsed = max(0.0, now - started)
        active_text = (
            f" · active={active_job.get('job_type', 'unknown')} "
            f"({elapsed:.0f}s, attempt {active_job.get('attempt', 0)})"
        )
    _emit(
        "HEARTBEAT",
        (
            f"pid={os.getpid()} · state={getattr(supervisor.state, 'state', 'UNKNOWN')} · "
            f"phase={_phase(supervisor)} · jobs "
            f"P:{counts.get(JS.PENDING, 0)} R:{counts.get(JS.RUNNING, 0)} "
            f"B:{counts.get(JS.BLOCKED, 0)} F:{counts.get(JS.PERMANENT_FAILED, 0)} · "
            f"next={_next_job(supervisor)}{active_text} · failures={failure_text}"
        ),
    )
    return now


def _primary_active(active_map: dict) -> dict:
    if not active_map:
        return {}
    for job in active_map.values():
        if job.get("job_type") == SCH.MARKET_SCAN:
            return dict(job)
    return dict(next(iter(active_map.values())))


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
            return _primary_active(active)

    original_execute = supervisor._execute
    elapsed_by_job: dict[str, float] = {}

    def visible_execute(job):
        started = time.monotonic()
        current = {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "attempt": job.attempt,
            "critical": bool(getattr(job, "critical", False)),
            "started_ist": ST._now_ist_iso(),
            "started_monotonic": started,
        }
        with active_lock:
            active[job.job_id] = current
            snapshot = _primary_active(active)
        try:
            _write_runtime_status(supervisor, process_running=True, active_job=snapshot)
        except Exception:
            pass
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
                active.pop(job.job_id, None)
                snapshot = _primary_active(active)
            try:
                _write_runtime_status(supervisor, process_running=True, active_job=snapshot)
            except Exception:
                pass

    supervisor._execute = visible_execute

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
    # It also force-starts an owner Scan Now click so news cannot hold the button hostage.
    runtime_interval = max(0.4, min(2.0, float(heartbeat_s or 30.0) / 8.0))

    def runtime_worker() -> None:
        next_console = time.monotonic() + max(1.0, float(heartbeat_s or 30.0))
        while not runtime_stop.wait(runtime_interval):
            try:
                clicked = supervisor.pop_clicked_scan()
            except Exception:
                clicked = None
            if clicked is not None:
                threading.Thread(
                    target=visible_execute,
                    args=(clicked,),
                    name="autonomy-owner-scan",
                    daemon=True,
                ).start()
            current = active_snapshot()
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
                    if getattr(final, "status", None) != JS.RUNNING:
                        summary = final.result_summary or final.error_message or "no summary"
                        elapsed = elapsed_by_job.pop(job.job_id, 0.0)
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
                        active_job=active_snapshot(),
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
            elif hasattr(supervisor, "has_urgent_work") and supervisor.has_urgent_work():
                delay = 0.2
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
