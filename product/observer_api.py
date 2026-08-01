"""Lifecycle and read-only API projections installed on the terminal app."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from fastapi import HTTPException

import terminal_api as core
from product.workspace import SCANNER_MODES, build_command_center_state, scanner_rows

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


def command_center_workspace() -> dict[str, Any]:
    """Project authoritative persisted state into one coherent command surface."""
    market = core._market_payload()
    scan = core._scan_payload()
    long_term = core._long_term_payload()
    state = build_command_center_state(
        scan_payload=scan,
        long_term_payload=long_term,
        paper=core._paper_payload(),
        autonomy=core._autonomy_payload(),
        market=market,
    )
    return {"generated_at": datetime.now(timezone.utc).isoformat(), **state}


def scanner_workspace(mode: str) -> dict[str, Any]:
    """Return one server-ranked scanner mode without duplicating scan calculations."""
    requested = mode.strip().replace("_", "-")
    canonical = next(
        (item for item in SCANNER_MODES if item.lower() == requested.lower()),
        None,
    )
    if canonical is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scanner mode. Choose one of: {', '.join(SCANNER_MODES)}",
        )
    market = core._market_payload()
    scan = core._scan_payload()
    long_term = core._long_term_payload()
    rows = scanner_rows(
        canonical,
        scan_payload=scan,
        long_term_payload=long_term,
        conviction_rows=core._conviction(scan, market),
    )
    source = (
        "long_term"
        if canonical == "Long-Term"
        else "conviction"
        if canonical == "Conviction"
        else "market_scan"
    )
    scanned_at = (
        long_term.get("scanned_at", "")
        if canonical == "Long-Term"
        else scan.get("scanned_at", "")
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": canonical,
        "source": source,
        "scanned_at": scanned_at,
        "universe_size": int(scan.get("universe_size", 0) or 0),
        "rows": rows,
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
    """Install read-only workspace routes and observer lifecycle once."""
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
    app.add_api_route(
        "/api/command-center-workspace",
        command_center_workspace,
        methods=["GET"],
        name="command_center_workspace",
    )
    app.add_api_route(
        "/api/scanner-workspace/{mode}",
        scanner_workspace,
        methods=["GET"],
        name="scanner_workspace",
    )
