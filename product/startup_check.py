"""Bounded read-only startup self-check. Telegram absence is not fatal."""

from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request
from typing import Any

SCHEMA_VERSION = 2


_READY_REQUIRED_STATUSES = {"READY", "RUNNING", "LOCKED", "COLLECTING", "HEALTHY"}


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


def _lane(name: str, status: str, detail: str = "", *, required: bool = False) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "required": required}


def _history_readiness() -> tuple[bool, str]:
    """Use the canonical NSE history freshness contract, not scan-file existence."""
    try:
        from data.bhavcopy_runtime import official_history_freshness

        freshness = official_history_freshness(load_cache=True)
    except Exception as exc:
        return False, f"Official NSE history unavailable: {str(exc)[:160]}"

    current = bool(freshness.get("current"))
    available = str(freshness.get("available_session") or "").strip()
    expected = str(freshness.get("expected_latest_completed_session") or "").strip()
    reason = str(freshness.get("reason_code") or "").strip()
    if current:
        return True, f"Current through {available}" if available else "Official NSE history is current"

    parts = [reason or "HISTORY_NOT_READY"]
    if available:
        parts.append(f"available {available}")
    if expected:
        parts.append(f"expected {expected}")
    return False, " · ".join(parts)


def _paper_readiness() -> tuple[bool, str]:
    """Paper capability is ready when its supervising process is actually alive."""
    try:
        from product.paper_status import read_paper_status

        paper = read_paper_status()
    except Exception as exc:
        return False, f"Paper status unavailable: {str(exc)[:160]}"

    if not bool(paper.supervisor_running):
        return False, "Paper supervisor is not running"
    if bool(paper.enabled):
        return True, "Paper supervisor running"
    return True, "Paper supervisor running · new paper entries paused"


def _required_waiting(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        lane for lane in lanes
        if lane.get("required") and str(lane.get("status") or "") not in _READY_REQUIRED_STATUSES
    ]


def build_startup_check(*, probe_network: bool = True) -> dict[str, Any]:
    ui = _url_ok("http://127.0.0.1:5173/") if probe_network else _port_open(5173)
    api = _url_ok("http://127.0.0.1:8765/api/health") if probe_network else _port_open(8765)
    reports = _url_ok("http://127.0.0.1:8766/health") if probe_network else _port_open(8766)

    autonomy_running = False
    try:
        from product.autonomy_status import read_autonomy_status
        autonomy_running = bool(read_autonomy_status().get("running"))
    except Exception:
        autonomy_running = False

    ops_running = False
    try:
        import json
        from pathlib import Path
        runtime = json.loads((Path(__file__).resolve().parents[1] / "logs" / "market_ops" / "runtime.json").read_text())
        ops_running = bool(runtime.get("running") or runtime.get("process_running"))
    except Exception:
        ops_running = False

    data_ready, data_detail = _history_readiness()

    scan_ok = False
    try:
        from product.scan_store import default_scan_path
        from product.forward_soak import _read_json
        payload = _read_json(default_scan_path())
        scan_ok = bool(payload.get("records") or payload.get("available") or payload.get("scanned_at"))
    except Exception:
        scan_ok = False

    paper_ready, paper_detail = _paper_readiness()

    soak_status = "NOT_STARTED"
    try:
        from product.forward_soak import persist_soak_verification, soak_status as read_soak
        persist_soak_verification(min_interval_s=120)
        soak_status = str(read_soak().get("status") or "NOT_STARTED")
    except Exception:
        soak_status = "UNKNOWN"

    kite_ok = True
    try:
        from data.kite_client import _fresh_env
        kite_ok = bool(_fresh_env("KITE_ACCESS_TOKEN"))
    except Exception:
        kite_ok = False

    live_locked = True
    try:
        from product.execution_adapter import LiveExecutionAdapter, LiveMoneyLocked
        try:
            LiveExecutionAdapter().submit(object())
            live_locked = False
        except LiveMoneyLocked:
            live_locked = True
    except Exception:
        live_locked = True

    lanes = [
        _lane("UI", "READY" if ui else "WAITING", "http://127.0.0.1:5173", required=True),
        _lane("API", "READY" if api else "WAITING", "http://127.0.0.1:8765", required=True),
        _lane("REPORTS", "READY" if reports else "WAITING", "optional research reports", required=False),
        _lane("AUTONOMY", "RUNNING" if autonomy_running else "WAITING", required=True),
        _lane("MARKET OPERATIONS", "RUNNING" if ops_running else "WAITING", required=True),
        _lane("DATA", "READY" if data_ready else "WAITING", data_detail, required=True),
        _lane("SCAN PIPELINE", "READY" if scan_ok else "WAITING", required=False),
        _lane("PAPER BOT", "READY" if paper_ready else "WAITING", paper_detail, required=True),
        _lane("FORWARD EVIDENCE", soak_status, required=False),
        _lane("ZERODHA", "READY" if kite_ok else "LOGIN NEEDED", required=False),
        _lane("LIVE MONEY", "LOCKED" if live_locked else "UNLOCKED", required=True),
    ]
    required_down = _required_waiting(lanes)
    ready = bool(live_locked and not required_down)
    return {
        "schema_version": SCHEMA_VERSION,
        "ready": ready,
        "home_url": "http://127.0.0.1:5173",
        "lanes": lanes,
        "live_locked": live_locked,
        "required_waiting": [str(lane.get("name") or "") for lane in required_down],
        "note": "Telegram absence is not a product failure.",
    }


def print_startup_summary(*, probe_network: bool = True) -> int:
    payload = build_startup_check(probe_network=probe_network)
    by = {l["name"]: l for l in payload["lanes"]}
    print("QuantTerm is ready." if payload["ready"] else "QuantTerm is running; required lanes are still preparing.")
    print(f"Home: {payload['home_url']}")
    print()
    print(f"Data: {by['DATA']['status']}" + (f" · {by['DATA']['detail']}" if by['DATA'].get('detail') else ""))
    print(f"Automation: {by['AUTONOMY']['status']}")
    print(f"Paper bot: {by['PAPER BOT']['status']}" + (f" · {by['PAPER BOT']['detail']}" if by['PAPER BOT'].get('detail') else ""))
    print(f"Forward evidence: {by['FORWARD EVIDENCE']['status']}")
    print(f"Zerodha: {by['ZERODHA']['status']}")
    print(f"Live money: {by['LIVE MONEY']['status']}")
    if payload.get("required_waiting"):
        print("Still preparing: " + ", ".join(payload["required_waiting"]))
    if not payload["live_locked"]:
        print("LIVE MONEY UNLOCKED — fail-closed contract broken")
        return 2
    return 0


def maybe_open_home_browser() -> bool:
    """Open Home once. Never from non-interactive or no-browser mode."""
    if os.environ.get("QT_NONINTERACTIVE") == "1" or os.environ.get("QT_NO_BROWSER") == "1":
        return False
    if not os.environ.get("DISPLAY") and os.uname().sysname == "Linux":
        # Headless Linux: do not spawn a browser.
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
