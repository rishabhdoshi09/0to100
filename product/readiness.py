"""Per-job readiness. Kite login is not a global DATA_READY switch."""
from __future__ import annotations

from typing import Any, Mapping

OFFICIAL_MARKET_DATA_READY = "OFFICIAL_MARKET_DATA_READY"
BROKER_LIVE_DATA_READY = "BROKER_LIVE_DATA_READY"
RESEARCH_DATA_READY = "RESEARCH_DATA_READY"
OUTCOME_DATA_READY = "OUTCOME_DATA_READY"
EXECUTION_READY = "EXECUTION_READY"

REQUIRES = {
    "MARKET_SCAN_COMPLETED_SESSION": (OFFICIAL_MARKET_DATA_READY,),
    "RESEARCH_ACQUIRE": (RESEARCH_DATA_READY,),
    "HISTORICAL_REPLAY": (OFFICIAL_MARKET_DATA_READY,),
    "OUTCOME_RESOLUTION": (OUTCOME_DATA_READY,),
    "PAPER_ENTRY": (BROKER_LIVE_DATA_READY, EXECUTION_READY),
    "BROKER_PORTFOLIO_SYNC": (BROKER_LIVE_DATA_READY,),
    "LEARNING": (OUTCOME_DATA_READY,),
}


def official_history() -> dict[str, Any]:
    try:
        from data.bhavcopy_runtime import official_history_freshness

        return dict(official_history_freshness(load_cache=True) or {})
    except Exception as exc:
        return {"current": False, "reason_code": "HISTORY_PROBE_FAILED", "error": str(exc)[:200]}


def broker_live() -> dict[str, Any]:
    try:
        from research.autonomy.auth import SESSION_VALID, probe_auth

        health = probe_auth()
        valid = str(getattr(health, "status", "") or "") == SESSION_VALID
        return {
            "ready": bool(valid),
            "status": str(getattr(health, "status", "") or "UNKNOWN"),
            "reason": str(getattr(health, "reason", "") or ""),
            "error_code": str(getattr(health, "error_code", "") or ""),
        }
    except Exception as exc:
        return {
            "ready": False,
            "status": "PROBE_FAILED",
            "reason": str(exc)[:200],
            "error_code": type(exc).__name__.upper(),
        }


def kite_snapshot_id() -> str:
    try:
        from research.intelligence.data.snapshot_store import SnapshotStore
        from pathlib import Path

        logs = Path(__file__).resolve().parents[1] / "logs"
        return str(SnapshotStore(logs / "snapshots").get_active_snapshot() or "")
    except Exception:
        return ""


def broker_status() -> dict[str, Any]:
    """Canonical broker/live-data projection for APIs and execution gates.

    Authentication, live-data readiness and execution readiness are intentionally
    separate from the autonomy supervisor's health. A missing Zerodha login can
    block live execution while official-data research continues normally.
    """
    auth = broker_live()
    snapshot = kite_snapshot_id()
    auth_ready = bool(auth.get("ready"))
    live_data_ready = bool(auth_ready and snapshot)
    auth_status = str(auth.get("status") or "UNKNOWN")

    if live_data_ready:
        state = "READY"
        detail = "Zerodha session is valid and the active broker snapshot is available."
    elif auth_status in {"TOKEN_MISSING", "SESSION_EXPIRED"}:
        state = "LOGIN_REQUIRED"
        detail = str(auth.get("reason") or "Daily Zerodha login is required.")
    elif auth_ready and not snapshot:
        state = "SNAPSHOT_REQUIRED"
        detail = "Zerodha session is valid, but no active broker snapshot is available yet."
    elif auth_status in {"PROVIDER_UNAVAILABLE", "PROBE_FAILED"}:
        state = "UNAVAILABLE"
        detail = str(auth.get("reason") or "Zerodha could not be reached.")
    elif auth_status == "CONFIG_INVALID":
        state = "CONFIG_REQUIRED"
        detail = str(auth.get("reason") or "Zerodha configuration is incomplete.")
    else:
        state = "NOT_READY"
        detail = str(auth.get("reason") or "Zerodha live-data readiness is not established.")

    return {
        "state": state,
        "ready": live_data_ready,
        "live_data_ready": live_data_ready,
        "execution_ready": live_data_ready,
        "auth_ready": auth_ready,
        "login_required": state == "LOGIN_REQUIRED",
        "auth_status": auth_status,
        "reason_code": str(auth.get("error_code") or auth_status or ""),
        "detail": detail,
        "snapshot_id": snapshot,
    }


def inspect_readiness(*, now=None) -> dict[str, Any]:
    """Truthful capability matrix. Official work does not wait on Kite."""
    history = official_history()
    broker = broker_status()
    official_ok = bool(history.get("current"))
    available = str(history.get("available_session") or history.get("latest_date") or "")[:10]
    expected = str(history.get("expected_latest_completed_session") or "")[:10]
    outcome_ok = official_ok or (bool(available) and (not expected or available >= expected))
    broker_ok = bool(broker.get("live_data_ready"))
    scan_ok = False
    try:
        from product.scan_store import default_scan_path, load_scan

        scan = load_scan(default_scan_path()) or {}
        scan_ok = bool(scan.get("records") or scan.get("scanned_at"))
    except Exception:
        scan_ok = False
    capabilities = {
        OFFICIAL_MARKET_DATA_READY: official_ok,
        BROKER_LIVE_DATA_READY: broker_ok,
        RESEARCH_DATA_READY: True,
        OUTCOME_DATA_READY: outcome_ok,
        EXECUTION_READY: bool(broker.get("execution_ready")),
    }
    return {
        "schema_version": 1,
        "official_history": {
            "ready": official_ok,
            "available_session": available,
            "expected_session": expected,
            "reason_code": history.get("reason_code") or "",
        },
        "broker": broker,
        "scan_artifact": scan_ok,
        "capabilities": capabilities,
        "job_requires": {name: list(deps) for name, deps in REQUIRES.items()},
        "blocked_without_kite": [
            name
            for name, deps in REQUIRES.items()
            if BROKER_LIVE_DATA_READY in deps or EXECUTION_READY in deps
        ],
        "allowed_without_kite": [
            name
            for name, deps in REQUIRES.items()
            if BROKER_LIVE_DATA_READY not in deps and EXECUTION_READY not in deps
        ],
    }


def missing_for(job: str, readiness: Mapping[str, Any] | None = None) -> list[str]:
    matrix = readiness or inspect_readiness()
    caps = dict(matrix.get("capabilities") or {})
    return [dep for dep in REQUIRES.get(job, ()) if not caps.get(dep)]
