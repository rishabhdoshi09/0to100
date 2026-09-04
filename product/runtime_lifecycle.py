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
OPERATION_STATE_DIVERGED = "OPERATION_STATE_DIVERGED"
OPERATION_DEADLINE_EXCEEDED = "OPERATION_DEADLINE_EXCEEDED"
OPERATIONS_UNAVAILABLE = "OPERATIONS_UNAVAILABLE"
INTEGRITY_OK = "OK"


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
    try:
        from operations.store import DEFAULT_RUNNING_LEASE_S, KIND_RUNNING_LEASE_S, TERMINAL
    except Exception:
        KIND_RUNNING_LEASE_S = {}
        DEFAULT_RUNNING_LEASE_S = 30 * 60
        TERMINAL = frozenset({"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"})

    overdue_active: list[dict[str, Any]] = []
    for lane, row in (ops.get("active") or {}).items():
        if not isinstance(row, dict):
            continue
        try:
            started = float(row.get("attempt_started_at") or row.get("started_at") or 0)
        except (TypeError, ValueError):
            started = 0.0
        if started <= 0:
            continue
        age = time.time() - started
        kind = str(row.get("kind") or "")
        lease_s = float(KIND_RUNNING_LEASE_S.get(kind, DEFAULT_RUNNING_LEASE_S))
        if age > lease_s:
            overdue_active.append(
                {
                    "lane": lane,
                    "kind": kind,
                    "operation_id": row.get("operation_id"),
                    "age_s": round(age, 1),
                    "lease_s": lease_s,
                }
            )
        if active_age is None or age > active_age:
            active_age = age
            oldest_running = {
                "lane": lane,
                "kind": kind,
                "operation_id": row.get("operation_id"),
                "age_s": round(age, 1),
            }

    persisted_ops_fd = None
    try:
        persisted_ops_fd = int(ops["fd_count"]) if ops.get("fd_count") is not None else None
    except (TypeError, ValueError):
        persisted_ops_fd = None

    resources = {}
    try:
        from product.process_resources import (
            RESOURCE_EXHAUSTED,
            RESOURCE_PRESSURE,
            RESOURCE_UNKNOWN,
            resource_diagnostics,
        )

        resources = resource_diagnostics(
            api_pid=os.getpid() if api_serving else None,
            market_ops_pid=int(ops_pid) if ops_pid else None,
            market_ops_fd_count=persisted_ops_fd,
            oldest_running=oldest_running,
            active_operation_age_s=None if active_age is None else round(active_age, 1),
        )
        state = str(resources.get("state") or "")
        if state == RESOURCE_EXHAUSTED:
            lifecycle = FAILED
            reasons.insert(0, str(resources.get("reason") or "Resource exhausted"))
        elif state == RESOURCE_UNKNOWN:
            if lifecycle == READY:
                lifecycle = DEGRADED
            reasons.append(str(resources.get("reason") or "File-descriptor usage could not be measured"))
        elif state == RESOURCE_PRESSURE and lifecycle == READY:
            lifecycle = DEGRADED
            reasons.append(str(resources.get("reason") or "Resource pressure"))
    except Exception as exc:
        resources = {"state": "RESOURCE_UNKNOWN", "reason": f"resource probe failed: {exc}"[:200]}
        if lifecycle == READY:
            lifecycle = DEGRADED
        reasons.append(str(resources["reason"]))

    integrity = {
        "state": INTEGRITY_OK,
        "detail": "",
        "diverged": [],
        "overdue": overdue_active,
    }
    snapshot: dict[str, Any] | None = None
    snap_freshness = "UNAVAILABLE"
    try:
        from operations.status_snapshot import load_operations_snapshot

        snapshot, snap_freshness = load_operations_snapshot()
    except Exception:
        snapshot, snap_freshness = None, "UNAVAILABLE"

    db_rows: dict[str, dict[str, Any]] = {}
    if snapshot:
        for bucket in ("active", "recent"):
            for item in snapshot.get(bucket) or []:
                if isinstance(item, dict) and item.get("operation_id"):
                    db_rows[str(item["operation_id"])] = item
        for item in (snapshot.get("latest") or {}).values():
            if isinstance(item, dict) and item.get("operation_id"):
                db_rows.setdefault(str(item["operation_id"]), item)

    if not snapshot and (ops.get("active") or overdue_active):
        try:
            from operations.store import OperationStore

            jobs_db = Path(os.environ.get("QT_OPS_DB") or (ROOT / "logs" / "market_ops" / "jobs.db"))
            store = OperationStore.reader(jobs_db)
            for lane, row in (ops.get("active") or {}).items():
                if not isinstance(row, dict) or not row.get("operation_id"):
                    continue
                rec = store.get(str(row["operation_id"]))
                if rec:
                    db_rows[str(row["operation_id"])] = rec
            snap_freshness = "CURRENT"
        except Exception:
            snap_freshness = "UNAVAILABLE"

    diverged: list[dict[str, Any]] = []
    for lane, row in (ops.get("active") or {}).items():
        if not isinstance(row, dict):
            continue
        op_id = str(row.get("operation_id") or "")
        if not op_id:
            continue
        rec = db_rows.get(op_id)
        if rec and str(rec.get("status") or "") in TERMINAL:
            diverged.append(
                {
                    "lane": lane,
                    "operation_id": op_id,
                    "kind": row.get("kind"),
                    "runtime_status": "ACTIVE",
                    "db_status": rec.get("status"),
                }
            )

    if diverged:
        integrity["state"] = OPERATION_STATE_DIVERGED
        integrity["diverged"] = diverged
        integrity["detail"] = "runtime.active contains an operation the store already marked terminal"
        reasons.insert(0, OPERATION_STATE_DIVERGED)
        lifecycle = DEGRADED if lifecycle == READY else lifecycle
        if lifecycle == STARTING and ops_alive:
            lifecycle = DEGRADED
    elif overdue_active:
        integrity["state"] = OPERATION_DEADLINE_EXCEEDED
        integrity["detail"] = "an active operation exceeded its declared attempt deadline"
        reasons.insert(0, OPERATION_DEADLINE_EXCEEDED)
        lifecycle = DEGRADED if lifecycle == READY else lifecycle
        if lifecycle == STARTING and ops_alive:
            lifecycle = DEGRADED
    elif snap_freshness == "UNAVAILABLE" and ops_alive:
        integrity["state"] = OPERATIONS_UNAVAILABLE
        integrity["detail"] = "operations status store is unavailable beyond a bounded probe"
        reasons.append(OPERATIONS_UNAVAILABLE)
        if lifecycle == READY:
            lifecycle = DEGRADED

    if lifecycle == READY and (
        integrity["state"] != INTEGRITY_OK
        or str(resources.get("state") or "") in {"RESOURCE_EXHAUSTED", "RESOURCE_UNKNOWN", "UNKNOWN"}
    ):
        lifecycle = DEGRADED

    components.append(
        _component(
            "operations_integrity",
            READY if integrity["state"] == INTEGRITY_OK else (
                FAILED if integrity["state"] == OPERATION_STATE_DIVERGED else DEGRADED
            ),
            detail=integrity.get("detail") or integrity["state"],
        )
    )

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
        "integrity": integrity,
        "live_locked": True,
    }
