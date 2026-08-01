"""Lifecycle and read-only API projection for the scheduled Zerodha observer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import terminal_api as core

RUNTIME_PATH = core.ROOT / "logs" / "reconciliation" / "observer_runtime.json"
SNAPSHOT_DB = core.ROOT / "logs" / "reconciliation" / "broker_snapshots.db"

_observer_process: subprocess.Popen | None = None
_installed = False


def _json_file(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _fresh(value: Any, max_age_seconds: float = 30.0) -> bool:
    try:
        age = time.time() - float(value)
        return 0 <= age <= max_age_seconds
    except Exception:
        return False


def observer_enabled() -> bool:
    value = os.getenv("QT_ENABLE_ZERODHA_OBSERVER", "1").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def observer_payload() -> dict[str, Any]:
    runtime = dict(_json_file(RUNTIME_PATH, {}) or {})
    running = bool(runtime.get("process_running")) and _fresh(runtime.get("heartbeat_epoch"))
    snapshots: dict[str, Any] = {
        "available": False,
        "summary": {
            "snapshots": 0,
            "account_complete_snapshots": 0,
            "protection_complete_snapshots": 0,
            "complete_snapshots": 0,
            "latest_snapshot_id": "",
            "latest_complete_snapshot_id": "",
        },
        "latest": {},
    }
    try:
        if SNAPSHOT_DB.exists():
            from execution.reconciliation.snapshot_store import BrokerSnapshotStore

            store = BrokerSnapshotStore(SNAPSHOT_DB)
            latest = store.latest()
            snapshots = {
                "available": latest is not None,
                "summary": store.summary(),
                "latest": latest or {},
            }
    except Exception as exc:
        snapshots["error"] = str(exc)

    return {
        "enabled": observer_enabled(),
        "running": running,
        "process_running": bool(runtime.get("process_running")),
        "heartbeat": runtime.get("heartbeat", ""),
        "phase": runtime.get("phase", "OFFLINE"),
        "broker_mutations_enabled": False,
        "last_result": dict(runtime.get("last_result", {}) or {}),
        "last_error": str(runtime.get("last_error", "") or ""),
        "snapshots": snapshots,
        "message": (
            "Scheduled Zerodha observation is read-only and cannot place, modify or cancel "
            "orders or GTTs."
        ),
    }


def ensure_observer_worker() -> dict[str, Any]:
    global _observer_process
    payload = observer_payload()
    if not observer_enabled() or payload.get("running"):
        return payload
    if _observer_process is not None and _observer_process.poll() is None:
        return payload
    _observer_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "operations.zerodha_observer"],
        cwd=str(core.ROOT),
        env=os.environ.copy(),
    )
    deadline = time.time() + 2.5
    while time.time() < deadline:
        time.sleep(0.1)
        payload = observer_payload()
        if payload.get("running"):
            break
        if _observer_process.poll() is not None:
            break
    return payload


def stop_observer_worker() -> None:
    global _observer_process
    if _observer_process is not None and _observer_process.poll() is None:
        _observer_process.terminate()
        try:
            _observer_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _observer_process.kill()
    _observer_process = None


def install(app) -> None:
    """Install one endpoint and lifecycle hooks on the existing terminal app."""
    global _installed
    if _installed:
        return
    _installed = True
    router = getattr(app, "router", None)
    if router is None or not hasattr(router, "add_event_handler"):
        raise RuntimeError("terminal FastAPI router does not support lifecycle handlers")
    router.add_event_handler("startup", ensure_observer_worker)
    router.add_event_handler("shutdown", stop_observer_worker)
    app.add_api_route(
        "/api/broker-observer",
        observer_payload,
        methods=["GET"],
        name="broker_observer_status",
    )
