"""Visible, resilient console driver for the QuantTerm autonomy supervisor.

The durable Supervisor remains the scheduler and mutation owner.  This module only drives its ticks,
prints operator-readable activity, and prevents one unexpected tick exception from silently killing
the process.  It is used by ``python main.py autonomy``; tests may still call ``Supervisor.run``
directly for deterministic loop behaviour.
"""
from __future__ import annotations

import os
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


def _heartbeat(supervisor, *, force: bool, last_at: float, every_s: float) -> float:
    now = time.monotonic()
    if not force and now - last_at < every_s:
        return last_at
    try:
        counts = supervisor._job_counts()
    except Exception:
        counts = {}
    failures = sorted(getattr(supervisor, "failures", set()) or set())
    failure_text = ",".join(failures[:4]) if failures else "none"
    _emit(
        "HEARTBEAT",
        (
            f"pid={os.getpid()} · state={getattr(supervisor.state, 'state', 'UNKNOWN')} · "
            f"phase={_phase(supervisor)} · jobs "
            f"P:{counts.get(JS.PENDING, 0)} R:{counts.get(JS.RUNNING, 0)} "
            f"B:{counts.get(JS.BLOCKED, 0)} F:{counts.get(JS.PERMANENT_FAILED, 0)} · "
            f"next={_next_job(supervisor)} · failures={failure_text}"
        ),
    )
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
    last_heartbeat = _heartbeat(supervisor, force=True, last_at=0.0, every_s=heartbeat_s)

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
                _emit(
                    "JOB",
                    f"{final.job_type} → {final.status} · attempt={final.attempt} · {summary}",
                )
                last_heartbeat = _heartbeat(
                    supervisor, force=True, last_at=last_heartbeat, every_s=heartbeat_s
                )
            else:
                last_heartbeat = _heartbeat(
                    supervisor, force=False, last_at=last_heartbeat, every_s=heartbeat_s
                )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # one bad tick must not silently kill the operating loop
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
                supervisor, force=True, last_at=last_heartbeat, every_s=heartbeat_s
            )

        count += 1
        if max_iterations is not None and count >= max_iterations:
            break

        if consecutive_errors:
            delay = min(60.0, max(1.0, 2.0 ** min(consecutive_errors, 5)))
        else:
            delay = max(0.1, float(interval_s))
        sleep_fn(delay)
