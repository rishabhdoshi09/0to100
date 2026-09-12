"""Installed-host entrypoint for the QuantTerm paper/shadow desk.

The service manager starts this module, not the interactive launcher. It loads an
optional operator-owned environment file, starts one low-frequency post-session
report scheduler, then hands process ownership to the canonical host supervisor.
Scheduler failures are persisted and alert at most once per IST date instead of
being silently swallowed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _report_loop() -> None:
    while True:
        _report_iteration()
        time.sleep(REPORT_POLL_SECONDS)


def main() -> int:
    load_env_file(os.environ.get("QT_HOST_ENV_FILE"))
    if not os.environ.get("QT_RUNTIME_ROOT", "").strip():
        raise RuntimeError("installed QuantTerm requires QT_RUNTIME_ROOT")
    if not os.environ.get("QT_BUILD_SHA", "").strip():
        raise RuntimeError("installed QuantTerm requires QT_BUILD_SHA")

    threading.Thread(
        target=_report_loop,
        name="quantterm-post-session-report-scheduler",
        daemon=True,
    ).start()

    from product.host_supervisor import main as supervisor_main
    return int(supervisor_main())


if __name__ == "__main__":
    raise SystemExit(main())
