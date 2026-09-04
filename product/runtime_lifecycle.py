"""Desk lifecycle from real process and job state.

Cheap: file reads, PID checks, port probes. No network scrapes, no scans.
"""
from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTING = "STARTING"
READY = "READY"
DEGRADED = "DEGRADED"
FAILED = "FAILED"
RECOVERING = "RECOVERING"
LIFECYCLES = (STARTING, READY, DEGRADED, FAILED, RECOVERING)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid or 0)
    except (TypeError, ValueError):
        return False
    if value <= 1:
        return False
    try:
        os.kill(value, 0)
        return True
    except OSError:
        return False


def _port_open(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False
    finally:
        sock.close()


def _component(name: str, status: str, *, detail: str = "", pid: Any = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "pid": int(pid) if _pid_alive(pid) else None,
    }


def inspect_runtime(*, api_serving: bool = True) -> dict[str, Any]:
    """Return STARTING / READY / DEGRADED / FAILED / RECOVERING from live checks."""
    ops_path = Path(os.environ.get("QT_MARKET_OPS_RUNTIME") or (ROOT / "logs" / "market_ops" / "runtime.json"))
    auto_status = Path(os.environ.get("QT_AUTONOMY_STATUS") or (ROOT / "logs" / "autonomy" / "status.json"))
    auto_runtime = Path(os.environ.get("QT_AUTONOMY_RUNTIME") or (ROOT / "logs" / "autonomy" / "runtime.json"))
    ops = _read_json(ops_path)
    autonomy = _read_json(auto_status)
    if not autonomy:
        autonomy = _read_json(auto_runtime)

    ops_pid = ops.get("worker_pid") or ops.get("pid")
    ops_hb = float(ops.get("heartbeat_epoch") or 0)
    ops_fresh = bool(ops_hb and (time.time() - ops_hb) <= 12)
    ops_alive = bool(ops.get("process_running") or ops.get("running")) and _pid_alive(ops_pid) and ops_fresh
    ops_known_dead = bool(ops) and (
        (ops.get("process_running") is False and not _pid_alive(ops_pid))
        or (bool(ops_pid) and not _pid_alive(ops_pid) and bool(ops_hb))
    )

    auto_pid = autonomy.get("scheduler_owner_pid") or autonomy.get("worker_pid")
    auto_alive = bool(autonomy.get("running") or autonomy.get("process_running")) and _pid_alive(auto_pid)

    api_listen = _port_open(8765)
    ui_listen = _port_open(5173)
    report_listen = _port_open(8766)

    history = {}
    try:
        from data.bhavcopy_runtime import official_history_freshness, status as history_status

        history = dict(official_history_freshness(history_status(load_cache=True)))
    except Exception as exc:
        history = {"current": False, "ready": False, "reason_code": "HISTORY_PROBE_FAILED", "error": str(exc)[:200]}

    scan = _read_json(ROOT / "logs" / "product" / "latest_momentum_scan.json")
    scan_ok = bool(scan.get("records") or scan.get("scanned_at"))

    components = [
        _component(
            "api",
            READY if (api_serving or api_listen) else FAILED,
            detail="Terminal API is serving" if api_serving or api_listen else "Port 8765 is not listening",
        ),
        _component(
            "desk_ui",
            READY if ui_listen else STARTING,
            detail="Vite desk is listening" if ui_listen else "Desk UI is not on :5173 yet",
        ),
        _component(
            "market_ops",
            READY if ops_alive else (
                RECOVERING if _pid_alive(ops_pid) and not ops_fresh
                else (FAILED if ops_known_dead else STARTING)
            ),
            detail=(
                f"heartbeat {int(time.time() - ops_hb)}s ago" if ops_hb else "no heartbeat file"
            ),
            pid=ops_pid,
        ),
        _component(
            "autonomy",
            READY if auto_alive else STARTING,
            detail=str(autonomy.get("plain_state") or autonomy.get("state") or "not running"),
            pid=auto_pid,
        ),
        _component(
            "reports",
            READY if report_listen else STARTING,
            detail="Research-report API on :8766" if report_listen else "Report API not listening",
        ),
        _component(
            "official_history",
            READY if history.get("current") else (STARTING if history.get("ready") or history.get("sessions") else FAILED),
            detail=(
                f"as of {history.get('available_session') or history.get('latest_date') or 'unknown'}"
                if history.get("current")
                else str(history.get("reason_code") or history.get("error") or "official history not current")
            ),
        ),
        _component(
            "scan_artifact",
            READY if scan_ok else STARTING,
            detail=str(scan.get("scanned_at") or "no saved whole-market scan"),
        ),
    ]

    by_name = {row["name"]: row["status"] for row in components}
    reasons: list[str] = []
    if by_name["api"] == FAILED:
        reasons.append("Terminal API is not listening on :8765")
    if by_name["market_ops"] == FAILED:
        reasons.append("Market-operations worker is not alive")
    elif by_name["market_ops"] == RECOVERING:
        reasons.append("Market-operations heartbeat is stale; supervisor should restart it")
    if by_name["official_history"] == FAILED:
        reasons.append(str(history.get("reason_code") or "Official NSE history is not ready"))
    if not scan_ok and history.get("current"):
        reasons.append("Official history is current but no scan artifact exists yet")

    if by_name["api"] == FAILED:
        lifecycle = FAILED
    elif by_name["market_ops"] == RECOVERING:
        lifecycle = RECOVERING
    elif by_name["market_ops"] == FAILED:
        lifecycle = FAILED if api_serving or api_listen else FAILED
    elif by_name["api"] == READY and by_name["market_ops"] == READY and by_name["official_history"] == READY and scan_ok:
        lifecycle = READY
    elif by_name["official_history"] == FAILED and (api_serving or api_listen):
        lifecycle = DEGRADED
    elif by_name["api"] == READY and by_name["market_ops"] in {STARTING, READY}:
        lifecycle = STARTING
    else:
        lifecycle = STARTING

    if lifecycle == READY and (not auto_alive or not report_listen):
        lifecycle = DEGRADED
        if not auto_alive:
            reasons.append("Autonomy supervisor is not running")
        if not report_listen:
            reasons.append("Research-report API is not listening")

    oldest_running = None
    active_age = None
    for lane, row in (ops.get("active") or {}).items():
        if not isinstance(row, dict):
            continue
        try:
            started = float(row.get("started_at") or 0)
        except (TypeError, ValueError):
            started = 0.0
        if started <= 0:
            continue
        age = time.time() - started
        if active_age is None or age > active_age:
            active_age = age
            oldest_running = {
                "lane": lane,
                "kind": row.get("kind"),
                "operation_id": row.get("operation_id"),
                "age_s": round(age, 1),
            }

    resources = {}
    try:
        from product.process_resources import RESOURCE_EXHAUSTED, RESOURCE_PRESSURE, resource_diagnostics

        resources = resource_diagnostics(
            api_pid=os.getpid() if api_serving else None,
            market_ops_pid=int(ops_pid) if ops_pid else None,
            oldest_running=oldest_running,
            active_operation_age_s=None if active_age is None else round(active_age, 1),
        )
        state = str(resources.get("state") or "")
        if state == RESOURCE_EXHAUSTED:
            lifecycle = FAILED if lifecycle != FAILED else lifecycle
            reasons.insert(0, str(resources.get("reason") or "Resource exhausted"))
        elif state == RESOURCE_PRESSURE and lifecycle == READY:
            lifecycle = DEGRADED
            reasons.append(str(resources.get("reason") or "Resource pressure"))
    except Exception as exc:
        resources = {"state": "UNKNOWN", "reason": f"resource probe failed: {exc}"[:200]}

    return {
        "schema_version": 1,
        "lifecycle": lifecycle,
        "checked_at": _now(),
        "reason": reasons[0] if reasons else (
            "Required services are alive and official history is current"
            if lifecycle == READY
            else "Desk is still coming up"
        ),
        "reasons": reasons,
        "components": components,
        "history": {
            "current": bool(history.get("current")),
            "available_session": history.get("available_session") or history.get("latest_date") or "",
            "expected_session": history.get("expected_latest_completed_session") or "",
            "reason_code": history.get("reason_code") or "",
        },
        "resources": resources,
        "live_locked": True,
    }
