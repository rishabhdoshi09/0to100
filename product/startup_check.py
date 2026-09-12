"""Bounded read-only startup self-check. Telegram absence is not fatal.

Operational readiness and evidence readiness are independent truths.
A process may be operating while evidence is still missing, stale, building,
or incomplete. Evidence-dependent claims fail closed.
"""
from __future__ import annotations

import os
import socket
import urllib.request
from typing import Any
from core.runtime_paths import logs_dir

SCHEMA_VERSION = 3
_OPERATIONAL_READY_STATUSES = {"READY", "RUNNING", "LOCKED", "HEALTHY"}
_EVIDENCE_READY_STATUSES = {"READY", "HEALTHY", "CURRENT"}
_EVIDENCE_BUILDING_STATUSES = {"BUILDING", "COLLECTING", "WAITING", "NOT_STARTED", "INCOMPLETE"}
_EVIDENCE_STALE_STATUSES = {"STALE"}
_EVIDENCE_MISSING_STATUSES = {"MISSING"}
SCAN_EVIDENCE_MAX_AGE_S = 6 * 60 * 60
DOMAIN_OPERATIONAL = "operational"
DOMAIN_EVIDENCE = "evidence"
DOMAIN_CAPABILITY = "capability"


def _port_open(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.4)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False
    finally:
        sock.close()


def _url_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return int(response.status) == 200
    except Exception:
        return False


def _lane(name: str, status: str, detail: str = "", *, required: bool = False,
          domain: str = DOMAIN_OPERATIONAL) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail,
            "required": required, "domain": domain}


def _history_readiness() -> tuple[str, str]:
    try:
        from data.bhavcopy_runtime import official_history_freshness
        freshness = official_history_freshness(load_cache=True)
    except Exception as exc:
        return "MISSING", f"Official NSE history unavailable: {str(exc)[:160]}"
    current = bool(freshness.get("current"))
    available = str(freshness.get("available_session") or "").strip()
    expected = str(freshness.get("expected_latest_completed_session") or "").strip()
    reason = str(freshness.get("reason_code") or "").strip()
    if current:
        return "READY", f"Current through {available}" if available else "Official NSE history is current"
    parts = [reason or "HISTORY_NOT_READY"]
    if available:
        parts.append(f"available {available}")
    if expected:
        parts.append(f"expected {expected}")
    detail = " · ".join(parts)
    return ("STALE", detail) if available else ("MISSING", detail)


def _scan_evidence_status(payload: dict[str, Any] | None) -> tuple[str, str]:
    if not payload or not isinstance(payload, dict):
        return "MISSING", "No saved whole-market scan"
    records = payload.get("records")
    if not isinstance(records, list):
        records = []
    scanned_at = str(payload.get("scanned_at") or "").strip()
    if not scanned_at and not records:
        return "MISSING", "Scan artifact is empty"
    try:
        from product.scan_store import scan_age_hours
        age_h = scan_age_hours(payload)
    except Exception:
        age_h = None
    if age_h is None:
        return ("INCOMPLETE", "Scan records exist but scanned_at is unusable") if records else ("MISSING", "Scan artifact has no usable timestamp")
    if age_h * 3600.0 > float(SCAN_EVIDENCE_MAX_AGE_S):
        return "STALE", f"Scan is {age_h:.1f}h old · scanned_at {scanned_at}"
    if not records:
        return "INCOMPLETE", f"Scan has no records · scanned_at {scanned_at}"
    return "READY", f"{len(records)} records · scanned_at {scanned_at}"


def _paper_readiness() -> tuple[str, str]:
    try:
        from product.paper_status import read_paper_status
        paper = read_paper_status()
    except Exception as exc:
        return "WAITING", f"Paper status unavailable: {str(exc)[:160]}"
    if not bool(paper.supervisor_running):
        return "WAITING", "Paper supervisor is not running"
    if bool(paper.enabled):
        return "READY", "Paper supervisor running"
    return "READY", "Paper supervisor running · new paper entries paused"


def _live_lock_readiness() -> tuple[bool, bool, str, dict[str, Any]]:
    """Read the same canonical state enforced at the broker mutation boundary.

    Verification failure is deliberately distinct from LOCKED.  It keeps startup
    NOT READY instead of turning an exception into positive safety evidence.
    """
    try:
        from product.live_execution_interlock import get_live_execution_state
        state = get_live_execution_state()
        payload = state.as_dict()
        verified = bool(state.verified)
        locked = bool(state.locked and not state.authorized)
        if not verified:
            return locked, False, "Canonical live interlock reported unverified state", payload
        if not locked:
            return False, True, "Canonical live interlock is not locked", payload
        return True, True, state.reason, payload
    except Exception as exc:
        return True, False, f"Canonical live interlock could not be verified: {str(exc)[:160]}", {}


def _required_waiting(lanes: list[dict[str, Any]], *, domain: str | None = None,
                      ready_statuses: set[str] | None = None) -> list[dict[str, Any]]:
    ready = ready_statuses or _OPERATIONAL_READY_STATUSES
    out = []
    for lane in lanes:
        if not lane.get("required"):
            continue
        if domain is not None and str(lane.get("domain") or "") != domain:
            continue
        if str(lane.get("status") or "") not in ready:
            out.append(lane)
    return out


def _aggregate_operational(lanes: list[dict[str, Any]], *, live_locked: bool,
                           live_lock_verified: bool = True) -> dict[str, Any]:
    waiting = _required_waiting(lanes, domain=DOMAIN_OPERATIONAL,
                                ready_statuses=_OPERATIONAL_READY_STATUSES)
    blockers = [str(lane.get("name") or "") for lane in waiting]
    if not live_lock_verified:
        if "LIVE MONEY" not in blockers:
            blockers.append("LIVE MONEY")
        return {"ready": False, "status": "FAILED", "blockers": blockers,
                "reasons": ["Live-money interlock could not be verified — startup fails closed"]}
    if not live_locked:
        if "LIVE MONEY" not in blockers:
            blockers.append("LIVE MONEY")
        return {"ready": False, "status": "FAILED", "blockers": blockers,
                "reasons": ["Live money is unlocked — fail-closed contract broken"]}
    if not waiting:
        return {"ready": True, "status": "READY", "blockers": [], "reasons": []}
    names = {str(lane.get("name") or "") for lane in waiting}
    statuses = {str(lane.get("status") or "") for lane in waiting}
    if statuses & {"FAILED", "UNLOCKED", "UNVERIFIED", "BROKEN"}:
        status = "FAILED"
    elif names & {"UI", "API", "AUTONOMY", "MARKET OPERATIONS", "PAPER BOT"}:
        status = "NOT_READY"
    else:
        status = "DEGRADED"
    reasons = [f"{lane.get('name')}: {lane.get('status')}" +
               (f" · {lane.get('detail')}" if lane.get("detail") else "") for lane in waiting]
    return {"ready": False, "status": status, "blockers": blockers, "reasons": reasons}


def _aggregate_evidence(lanes: list[dict[str, Any]]) -> dict[str, Any]:
    required = [lane for lane in lanes if lane.get("required") and str(lane.get("domain") or "") == DOMAIN_EVIDENCE]
    blockers = [str(lane.get("name") or "") for lane in required if str(lane.get("status") or "") not in _EVIDENCE_READY_STATUSES]
    statuses = [str(lane.get("status") or "") for lane in required]
    reasons = [f"{lane.get('name')}: {lane.get('status')}" +
               (f" · {lane.get('detail')}" if lane.get("detail") else "")
               for lane in required if str(lane.get("status") or "") not in _EVIDENCE_READY_STATUSES]
    if not required or all(status in _EVIDENCE_READY_STATUSES for status in statuses):
        return {"ready": True, "status": "READY", "blockers": [], "reasons": []}
    if any(status in _EVIDENCE_MISSING_STATUSES for status in statuses):
        status = "DEGRADED" if any(status in (_EVIDENCE_READY_STATUSES | _EVIDENCE_STALE_STATUSES | _EVIDENCE_BUILDING_STATUSES) for status in statuses) else "MISSING"
    elif any(status in _EVIDENCE_STALE_STATUSES for status in statuses):
        status = "STALE"
    elif any(status in _EVIDENCE_BUILDING_STATUSES for status in statuses):
        status = "BUILDING"
    else:
        status = "NOT_READY"
    return {"ready": False, "status": status, "blockers": blockers, "reasons": reasons}


def build_startup_check(*, probe_network: bool = True) -> dict[str, Any]:
    ui = _url_ok("http://127.0.0.1:5173/") if probe_network else _port_open(5173)
    api = _url_ok("http://127.0.0.1:8765/api/health") if probe_network else _port_open(8765)
    reports = _url_ok("http://127.0.0.1:8766/health") if probe_network else _port_open(8766)

    try:
        from product.autonomy_status import read_autonomy_status
        autonomy_running = bool(read_autonomy_status().get("running"))
    except Exception:
        autonomy_running = False

    try:
        import json
        from pathlib import Path
        runtime = json.loads((logs_dir() / "market_ops" / "runtime.json").read_text())
        ops_running = bool(runtime.get("running") or runtime.get("process_running"))
    except Exception:
        ops_running = False

    data_status, data_detail = _history_readiness()
    try:
        from product.scan_store import default_scan_path, load_scan
        scan_status, scan_detail = _scan_evidence_status(load_scan(default_scan_path()))
    except Exception as exc:
        scan_status, scan_detail = "MISSING", f"Scan artifact unreadable: {str(exc)[:160]}"
    paper_status, paper_detail = _paper_readiness()

    try:
        from product.forward_soak import persist_soak_verification, soak_status as read_soak
        persist_soak_verification(min_interval_s=120)
        soak_status = str(read_soak().get("status") or "NOT_STARTED")
    except Exception:
        soak_status = "UNKNOWN"

    try:
        from data.kite_client import _fresh_env
        kite_ok = bool(_fresh_env("KITE_ACCESS_TOKEN"))
    except Exception:
        kite_ok = False

    live_locked, live_lock_verified, live_detail, live_interlock = _live_lock_readiness()
    live_status = "LOCKED" if live_lock_verified and live_locked else ("UNVERIFIED" if not live_lock_verified else "UNLOCKED")

    lanes = [
        _lane("UI", "READY" if ui else "WAITING", "http://127.0.0.1:5173", required=True),
        _lane("API", "READY" if api else "WAITING", "http://127.0.0.1:8765/api/health", required=True),
        _lane("REPORTS", "READY" if reports else "WAITING", "optional research reports", domain=DOMAIN_CAPABILITY),
        _lane("AUTONOMY", "RUNNING" if autonomy_running else "WAITING", required=True),
        _lane("MARKET OPERATIONS", "RUNNING" if ops_running else "WAITING", required=True),
        _lane("DATA", data_status, data_detail, required=True, domain=DOMAIN_EVIDENCE),
        _lane("SCAN PIPELINE", scan_status, scan_detail, required=True, domain=DOMAIN_EVIDENCE),
        _lane("PAPER BOT", paper_status, paper_detail, required=True),
        _lane("FORWARD EVIDENCE", soak_status, domain=DOMAIN_EVIDENCE),
        _lane("ZERODHA", "READY" if kite_ok else "LOGIN NEEDED", required=False, domain=DOMAIN_CAPABILITY),
        _lane("LIVE MONEY", live_status, live_detail, required=True),
    ]
    operational = _aggregate_operational(lanes, live_locked=live_locked,
                                         live_lock_verified=live_lock_verified)
    evidence = _aggregate_evidence(lanes)
    operational_waiting = _required_waiting(lanes, domain=DOMAIN_OPERATIONAL,
                                            ready_statuses=_OPERATIONAL_READY_STATUSES)
    evidence_waiting = _required_waiting(lanes, domain=DOMAIN_EVIDENCE,
                                         ready_statuses=_EVIDENCE_READY_STATUSES)
    fully_ready = bool(operational["ready"] and evidence["ready"] and live_locked and live_lock_verified)
    return {
        "schema_version": SCHEMA_VERSION,
        "ready": fully_ready,
        "operational_ready": bool(operational["ready"]),
        "evidence_ready": bool(evidence["ready"]),
        "operational": operational,
        "evidence": evidence,
        "home_url": "http://127.0.0.1:5173",
        "lanes": lanes,
        "live_locked": live_locked,
        "live_lock_verified": live_lock_verified,
        "live_interlock": live_interlock,
        "required_waiting": [str(lane.get("name") or "") for lane in operational_waiting + evidence_waiting],
        "operational_waiting": [str(lane.get("name") or "") for lane in operational_waiting],
        "evidence_waiting": [str(lane.get("name") or "") for lane in evidence_waiting],
        "note": "Telegram absence is not a product failure. Operational readiness is not evidence readiness.",
    }


def print_startup_summary(*, probe_network: bool = True) -> int:
    payload = build_startup_check(probe_network=probe_network)
    by = {lane["name"]: lane for lane in payload["lanes"]}
    operational = dict(payload.get("operational") or {})
    evidence = dict(payload.get("evidence") or {})
    print(f"Operational runtime: {operational.get('status') or 'UNKNOWN'}")
    if operational.get("blockers"):
        print("  blockers: " + ", ".join(operational["blockers"]))
    print(f"Evidence: {evidence.get('status') or 'UNKNOWN'}")
    if evidence.get("blockers"):
        print("  blockers: " + ", ".join(evidence["blockers"]))
    if operational.get("ready") and evidence.get("ready"):
        print("QuantTerm is ready.")
    elif operational.get("ready"):
        print("QuantTerm is operating; evidence-dependent claims are not fully ready.")
    else:
        print("QuantTerm operational runtime is not ready.")
    print(f"Home: {payload['home_url']}")
    print()
    print(f"Data: {by['DATA']['status']}" + (f" · {by['DATA']['detail']}" if by['DATA'].get("detail") else ""))
    print(f"Scan: {by['SCAN PIPELINE']['status']}" + (f" · {by['SCAN PIPELINE']['detail']}" if by['SCAN PIPELINE'].get("detail") else ""))
    print(f"Automation: {by['AUTONOMY']['status']}")
    print(f"Paper bot: {by['PAPER BOT']['status']}" + (f" · {by['PAPER BOT']['detail']}" if by['PAPER BOT'].get("detail") else ""))
    print(f"Forward evidence: {by['FORWARD EVIDENCE']['status']}")
    print(f"Zerodha: {by['ZERODHA']['status']}")
    print(f"Live money: {by['LIVE MONEY']['status']}")
    if payload.get("operational_waiting"):
        print("Operational still preparing: " + ", ".join(payload["operational_waiting"]))
    if payload.get("evidence_waiting"):
        print("Evidence still preparing: " + ", ".join(payload["evidence_waiting"]))
    if not payload.get("live_lock_verified"):
        print("LIVE MONEY INTERLOCK UNVERIFIED — startup not ready")
        return 2
    if not payload["live_locked"]:
        print("LIVE MONEY UNLOCKED — fail-closed contract broken")
        return 2
    return 0


def maybe_open_home_browser() -> bool:
    if os.environ.get("QT_NONINTERACTIVE") == "1" or os.environ.get("QT_NO_BROWSER") == "1":
        return False
    if not os.environ.get("DISPLAY") and os.uname().sysname == "Linux":
        if not os.environ.get("WAYLAND_DISPLAY"):
            return False
    if not _url_ok("http://127.0.0.1:5173/") or not _url_ok("http://127.0.0.1:8765/api/health"):
        return False
    try:
        import webbrowser
        webbrowser.open("http://127.0.0.1:5173", new=1, autoraise=True)
        return True
    except Exception:
        return False
