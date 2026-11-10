"""Fresh-host bootstrap for durable QuantTerm data stores.

The normal repair ladder assumes there is already a canonical history store to
repair.  That is the correct behaviour during a session (a request for three
missing symbols must not unexpectedly download hundreds of sessions), but a
brand-new installed host needs one explicit owner for the *first* build.

This module is that owner.  It is called once by the installed host supervisor
before user-facing services start.  It never fabricates data and it does not
turn an unavailable official archive into a green result: failure is persisted
and the rest of the desk may still start in a truthful DEGRADED state so its
normal fallback machinery can operate.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.runtime_paths import runtime_path

STATUS_PATH = Path("state") / "host_bootstrap.json"
MIN_WHOLE_MARKET_SYMBOLS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from core.atomic_io import atomic_write_text
    atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def bootstrap_host_state(*, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Prepare durable stores and attempt the first canonical history build.

    Returns a truthful report rather than raising on a market-data outage.  A
    safety/bootstrap programming error is recorded as FAILED; an unavailable
    official history source is DEGRADED so the normal scan/data fallbacks still
    get a chance after the desk starts.
    """
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": _now(),
        "state": "STARTING",
        "sqlite": {},
        "history_before": {},
        "history_after": {},
        "history_build_attempted": False,
        "error": "",
    }
    sqlite_failed = False
    try:
        try:
            from product.sqlite_runtime import bootstrap_product_stores

            report["sqlite"] = bootstrap_product_stores() or {}
        except Exception as exc:
            # Durable product stores are required for an unattended desk.  A
            # history fallback cannot compensate for a state store we cannot
            # initialise, so keep the evidence and fail the supervisor closed.
            sqlite_failed = True
            report["sqlite"] = {"state": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:300]}

        from data.bhavcopy_runtime import status as history_status

        before = history_status(load_cache=True)
        report["history_before"] = before
        ready = bool(before.get("ready")) and int(before.get("sessions") or 0) >= int(before.get("minimum_sessions") or 60)
        if not ready:
            report["history_build_attempted"] = True
            try:
                from data.bhavcopy_store import DEFAULT_DAYS, HistoryAcquisitionCancelled, build_store

                build_store(days=DEFAULT_DAYS, should_stop=should_stop)
            except HistoryAcquisitionCancelled:
                report["state"] = "CANCELLED"
                report["error"] = "history bootstrap cancelled during shutdown"
                report["finished_at"] = _now()
                _atomic_json(runtime_path(STATUS_PATH), report)
                return report
            except Exception as exc:
                # Do not hide the outage; scanner/provider fallbacks may still be
                # useful after startup, so this is DEGRADED rather than a fatal
                # host-supervisor exception.
                report["error"] = f"{type(exc).__name__}: {exc}"[:300]

        after = history_status(load_cache=True)
        report["history_after"] = after
        symbols = int(after.get("symbols") or 0)
        sessions = int(after.get("sessions") or 0)
        min_sessions = int(after.get("minimum_sessions") or 60)
        if sqlite_failed:
            report["state"] = "FAILED"
            if not report["error"]:
                report["error"] = str(report["sqlite"].get("error") or "durable product-store bootstrap failed")
        elif bool(after.get("ready")) and symbols >= MIN_WHOLE_MARKET_SYMBOLS and sessions >= min_sessions:
            report["state"] = "READY"
        else:
            report["state"] = "DEGRADED"
            if not report["error"]:
                report["error"] = (
                    f"canonical official history is not ready: symbols={symbols}, "
                    f"sessions={sessions}, minimum_sessions={min_sessions}"
                )
    except Exception as exc:
        report["state"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"[:300]
    report["finished_at"] = _now()
    try:
        _atomic_json(runtime_path(STATUS_PATH), report)
    except Exception:
        pass
    return report


# ── Process isolation ──────────────────────────────────────────────────────────
# Verifying official history means unpickling the canonical store (183 MB on
# disk, 3,463 symbols on the owner's Mac). Run in-process, that left ~300 MB
# resident in the long-lived host supervisor for the life of the host on an
# 8 GB machine already deep in swap; clearing the store afterwards recovers only
# part of it (pandas and allocator fragmentation stay). The supervisor therefore
# runs the bootstrap in a child interpreter: the memory is returned when the
# child exits, and a crash in the history path cannot take the supervisor down.

_REPORT_PREFIX = "QT_HOST_BOOTSTRAP_REPORT "


def _final_report(state: str, error: str, started_at: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_at,
        "state": state,
        "sqlite": {},
        "history_before": {},
        "history_after": {},
        "history_build_attempted": False,
        "error": error[:300],
        "isolated": True,
        "finished_at": _now(),
    }
    try:
        _atomic_json(runtime_path(STATUS_PATH), report)
    except Exception:
        pass
    return report


def bootstrap_host_state_isolated(
    *,
    should_stop: Callable[[], bool] | None = None,
    poll_s: float = 0.5,
    term_grace_s: float = 30.0,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """``bootstrap_host_state`` in a child interpreter; same report contract.

    The child's log lines are forwarded to this process's stdout unchanged. A
    stop request is delivered as SIGTERM, which the child turns into the normal
    cooperative cancellation (``should_stop``); a child that ignores it is
    killed after ``term_grace_s``. A child that dies without a report is FAILED,
    never READY.
    """
    import os
    import subprocess
    import sys
    import threading
    import time

    from core.runtime_paths import REPO_ROOT

    started_at = _now()
    argv = list(command or [sys.executable, "-u", "-m", "product.host_bootstrap", "--child"])
    try:
        proc = subprocess.Popen(
            argv, cwd=str(REPO_ROOT), env=dict(os.environ),
            stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1,
        )
    except Exception as exc:
        return _final_report("FAILED", f"bootstrap child could not start: {type(exc).__name__}: {exc}", started_at)

    captured: dict[str, Any] = {}

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.startswith(_REPORT_PREFIX):
                try:
                    payload = json.loads(line[len(_REPORT_PREFIX):])
                    if isinstance(payload, dict):
                        captured["report"] = payload
                except ValueError:
                    pass
                continue
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass

    reader = threading.Thread(target=pump, name="quantterm-bootstrap-log", daemon=True)
    reader.start()
    cancel_sent_at: float | None = None
    while True:
        try:
            proc.wait(timeout=poll_s)
            break
        except subprocess.TimeoutExpired:
            pass
        if cancel_sent_at is None and should_stop is not None and should_stop():
            cancel_sent_at = time.monotonic()
            try:
                proc.terminate()
            except OSError:
                pass
        elif cancel_sent_at is not None and time.monotonic() - cancel_sent_at > term_grace_s:
            try:
                proc.kill()
            except OSError:
                pass
    reader.join(timeout=5.0)

    report = captured.get("report")
    if isinstance(report, dict) and report.get("state"):
        report["isolated"] = True
        return report
    if cancel_sent_at is not None:
        return _final_report("CANCELLED", "history bootstrap cancelled during shutdown", started_at)
    return _final_report(
        "FAILED",
        f"bootstrap child exited rc={proc.returncode} without a report",
        started_at,
    )


def _child_main() -> int:
    import signal
    import sys

    stop = {"requested": False}

    def _request_stop(_signum, _frame) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGTERM, _request_stop)
    report = bootstrap_host_state(should_stop=lambda: stop["requested"])
    sys.stdout.write(_REPORT_PREFIX + json.dumps(report, default=str) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    import sys

    if "--child" in sys.argv[1:]:
        raise SystemExit(_child_main())
    print(json.dumps(bootstrap_host_state(), indent=2, default=str))
