"""Installed-host entrypoint for the QuantTerm paper/shadow desk.

The service manager starts this module, not the interactive launcher. It loads an
optional operator-owned environment file, guards the already-adopted durable
runtime, starts one low-frequency post-session report scheduler, then hands child
ownership to the canonical host supervisor.

Runtime loss is fail-closed: the storage guard signals the supervisor to stop all
children, report generation pauses, and the entrypoint waits without creating a
replacement path. If the same runtime returns, the canonical supervisor is
started again and re-verifies the exact SHA and live-money interlock.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

REPORT_POLL_SECONDS = 15 * 60
REPORT_SCHEDULER_REL = Path("state") / "daily_report_scheduler.json"


def load_env_file(path: str | os.PathLike[str] | None) -> list[str]:
    if not path:
        return []
    target = Path(path).expanduser()
    if not target.exists():
        raise RuntimeError(f"configured QT_HOST_ENV_FILE does not exist: {target}")
    loaded: list[str] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not (key[0].isalpha() or key[0] == "_"):
            continue
        if not all(ch.isalnum() or ch == "_" for ch in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _read_scheduler_status() -> dict[str, Any]:
    from core.runtime_paths import runtime_path
    path = runtime_path(REPORT_SCHEDULER_REL)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _write_scheduler_status(payload: dict[str, Any]) -> None:
    from core.runtime_paths import runtime_path
    path = runtime_path(REPORT_SCHEDULER_REL)
    # An installed runtime already owns state/. Never recreate parent paths here:
    # if external storage vanished, mkdir could silently create split state on the
    # system disk before the storage watchdog fires.
    if not path.parent.is_dir():
        raise RuntimeError(f"runtime state directory is unavailable: {path.parent}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _report_iteration() -> dict[str, Any]:
    from core.market_clock import now_ist
    now = now_ist()
    previous = _read_scheduler_status()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "last_attempt_ist": now.isoformat(),
        "ist_date": now.date().isoformat(),
        "state": "OK",
        "last_error": "",
        "alerted_date": str(previous.get("alerted_date") or ""),
    }
    try:
        from product.host_report_job import run_once
        result = run_once(now=now)
        payload["report_result"] = result
        payload["state"] = "RAN" if result.get("ran") else "NOT_DUE"
    except Exception as exc:
        payload["state"] = "FAILED"
        payload["last_error"] = f"{type(exc).__name__}: report scheduler iteration failed"
        if payload["alerted_date"] != now.date().isoformat():
            try:
                from product.host_alerts import send_operational_alert
                alert = send_operational_alert(
                    "QuantTerm daily operating report scheduler failed\n"
                    f"IST date: {now.date().isoformat()}\n"
                    f"Failure class: {type(exc).__name__}\n"
                    "Read state/daily_report_scheduler.json and service logs."
                ).to_dict()
                payload["alert"] = alert
                if alert.get("attempted"):
                    payload["alerted_date"] = now.date().isoformat()
            except Exception as alert_exc:
                payload["alert"] = {
                    "attempted": True, "delivered": False,
                    "detail": f"{type(alert_exc).__name__}: alert delivery failed",
                }
    _write_scheduler_status(payload)
    return payload


def _report_loop(storage_guard) -> None:
    while not storage_guard.shutdown.is_set():
        if storage_guard.check():
            try:
                _report_iteration()
            except Exception:
                # Scheduler failures are persisted by _report_iteration when the
                # runtime is available. Storage loss is owned by the guard and
                # must not create a fallback state tree here.
                pass
        storage_guard.shutdown.wait(REPORT_POLL_SECONDS)


def main() -> int:
    load_env_file(os.environ.get("QT_HOST_ENV_FILE"))
    if not os.environ.get("QT_RUNTIME_ROOT", "").strip():
        raise RuntimeError("installed QuantTerm requires QT_RUNTIME_ROOT")
    if not os.environ.get("QT_BUILD_SHA", "").strip():
        raise RuntimeError("installed QuantTerm requires QT_BUILD_SHA")

    # This invariant is inherited by every child. Once the installed service is
    # running, a vanished configured runtime is an error at path resolution
    # time; helpers such as ensure_logs_path are never allowed to mkdir a fresh
    # replacement tree on another filesystem while the watchdog is reacting.
    os.environ["QT_RUNTIME_ROOT_REQUIRE_EXISTING"] = "1"

    from product.runtime_storage_guard import RuntimeStorageGuard

    guard_interval = float(os.environ.get("QT_RUNTIME_STORAGE_WATCH_S", "15") or 15)
    storage_guard = RuntimeStorageGuard(interval_s=guard_interval)
    outer_shutdown = threading.Event()
    loss_reason = {"detail": ""}

    def storage_lost(reason: str) -> None:
        loss_reason["detail"] = str(reason or "runtime storage unavailable")[:500]
        print(
            f"[HOST STORAGE] LOST: {loss_reason['detail']} · stopping all QuantTerm children",
            file=sys.stderr,
            flush=True,
        )
        # host_supervisor.main owns SIGTERM while it is running. Its handler only
        # requests shutdown, so the supervisor's finally block still reaps every
        # child and releases the machine lock cleanly.
        os.kill(os.getpid(), signal.SIGTERM)

    def outer_stop(_signum, _frame) -> None:
        outer_shutdown.set()
        storage_guard.close()

    storage_guard.start(storage_lost)
    threading.Thread(
        target=_report_loop,
        args=(storage_guard,),
        name="quantterm-post-session-report-scheduler",
        daemon=True,
    ).start()

    try:
        from product.host_supervisor import main as supervisor_main

        while not outer_shutdown.is_set():
            rc = int(supervisor_main())
            if not storage_guard.lost.is_set():
                return rc

            # supervisor_main installed its own signal handlers. While children
            # are down and storage is absent, restore an outer handler so a real
            # service stop does not get mistaken for storage recovery.
            signal.signal(signal.SIGTERM, outer_stop)
            signal.signal(signal.SIGINT, outer_stop)
            print(
                "[HOST STORAGE] QuantTerm is paused. Waiting for the same durable runtime to return; no fallback directory will be created.",
                file=sys.stderr,
                flush=True,
            )
            if not storage_guard.wait_until_recovered(should_stop=outer_shutdown.is_set):
                return 0
            print(
                "[HOST STORAGE] Runtime recovered. Re-starting the canonical supervisor; safety and exact-SHA checks will run again.",
                file=sys.stderr,
                flush=True,
            )
            loss_reason["detail"] = ""
        return 0
    finally:
        storage_guard.close()


if __name__ == "__main__":
    raise SystemExit(main())
