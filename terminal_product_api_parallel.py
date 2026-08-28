"""Canonical terminal API with user-first operation routing and recovery.

The launcher normally owns Market Operations. The API remains an observer while
that worker is healthy, but an explicit user action must never disappear behind a
stale worker. If the launcher-owned worker is dead/stale, this product wrapper
cleans up only the verified stale ``operations.market_ops`` PID, invokes the base
bounded recovery path, and refuses the command if a healthy worker still does not
appear. This gives the one-terminal product a second safety net without creating a
second market-data or scan architecture.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time

import terminal_api as core
import terminal_product_api as product
from operations.store import pid_is_alive
from product.operator_health import enrich_autonomy_payload

# The base API intentionally keeps the control mapping in one mutable registry.
core._OPERATION_CONTROLS["RUN_LONG_TERM_SCAN_NOW"] = "LONG_TERM_SCAN"
_base_ensure_ops_worker = core._ensure_ops_worker


def _healthy_runtime() -> dict:
    runtime = core._ops_runtime_payload()
    if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
        return runtime
    return runtime


def _market_ops_command(pid: int) -> str:
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=0.5,
        ).strip()
    except Exception:
        return ""


def _stop_stale_owner(runtime: dict) -> bool:
    """Terminate only a verified stale market-ops process.

    A stale runtime file alone is never enough to kill an arbitrary PID. The
    command line must still identify ``operations.market_ops``. TERM is bounded;
    KILL is the final cleanup only when the same verified process survives.
    """
    try:
        pid = int(runtime.get("worker_pid") or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 1 or not pid_is_alive(pid):
        return False
    command = _market_ops_command(pid)
    if "operations.market_ops" not in command:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return True
    deadline = time.time() + 1.5
    while time.time() < deadline:
        if not pid_is_alive(pid):
            return True
        time.sleep(0.05)
    if "operations.market_ops" in _market_ops_command(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return True


def _ensure_ops_worker_strict(*, wait: bool = True) -> dict:
    """Return only when Market Operations is healthy or fail loudly.

    Healthy launcher ownership is reused. Unhealthy ownership is verified and
    cleaned up, then the existing bounded base recovery starts/reuses exactly the
    canonical ``operations.market_ops`` worker. The user never receives a ghost
    ``accepted: true`` solely because a queue row was written.
    """
    runtime = _healthy_runtime()
    if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
        return runtime

    _stop_stale_owner(runtime)
    recovered = _base_ensure_ops_worker(wait=True)
    if recovered.get("running") and pid_is_alive(recovered.get("worker_pid")):
        return recovered

    # One final bounded observation covers the launcher restarting concurrently.
    deadline = time.time() + (1.5 if wait else 0.5)
    while time.time() < deadline:
        time.sleep(0.1)
        recovered = _healthy_runtime()
        if recovered.get("running") and pid_is_alive(recovered.get("worker_pid")):
            return recovered

    raise RuntimeError(
        "Market operations worker did not become ready; the command was not "
        "silently accepted. Check System Health for the worker blocker."
    )


# Startup and the base control endpoint resolve this global at runtime.
core._ensure_ops_worker = _ensure_ops_worker_strict

# The durable scheduler ledger contains historical failures by design. Enrich
# only the product-facing projection so the dashboard answers "what is wrong
# now?" while retaining the full audit rows under jobs_recent.
_base_autonomy_payload = core._autonomy_payload


def _operator_autonomy_payload() -> dict:
    return enrich_autonomy_payload(_base_autonomy_payload())


core._autonomy_payload = _operator_autonomy_payload


@product.app.get("/api/operator-health")
def operator_health() -> dict:
    """Current-session health plus historical-ledger separation for diagnostics."""
    return core._autonomy_payload()


@product.app.get("/api/recommendations-workspace")
def recommendations_workspace() -> dict:
    """Canonical Recommendations projection from the saved scan and long-term evidence.

    This route intentionally performs no network crawl and no hidden scan. The React
    page owns the user-visible scan trigger; this GET only projects the latest durable
    evidence into the multi-method recommendation desk. Empty high-conviction is a
    valid result, not an exception.
    """
    from product.recommendations_workspace import (
        build_recommendations_workspace,
        slim_workspace_for_desk,
    )

    payload = build_recommendations_workspace(
        scan_payload=core._scan_payload(),
        long_term_payload=core._long_term_payload(),
        refresh_technicals=False,
        settle_cases=False,
        deep_confirm=False,
        persist_ledger=False,
    )
    return slim_workspace_for_desk(payload)


@product.app.get("/api/market-reports-workspace")
def market_reports_workspace() -> dict:
    """Canonical read projection for Market Reports.

    Missing/stale report state is returned honestly through ``needs_refresh``; the
    frontend then queues the durable MARKET_REPORT operation. This GET never fakes a
    headline and never performs an unbounded network refresh in the request thread.
    """
    from product.recommendations_workspace import build_market_reports_workspace

    return build_market_reports_workspace(
        persist_today=True,
        news_payload=core._news_payload(),
        scan_payload=core._scan_payload(),
        rebuild=False,
    )


def _registered_paths() -> set[str]:
    return {
        str(getattr(route, "path", ""))
        for route in product.app.routes
        if getattr(route, "path", None)
    }


@product.app.get("/api/product-contract")
def product_contract() -> dict:
    """Machine-readable proof that the primary desk surfaces are actually wired.

    This is a wiring/availability contract, not a claim that market data exists. It
    distinguishes route registration, trigger availability and current durable data.
    """
    paths = _registered_paths()
    scan = core._scan_payload()
    long_term = core._long_term_payload()
    operations = core._operations_payload()
    autonomy = core._autonomy_payload()
    checks = {
        "market_scan": {
            "route_registered": "/api/controls/{control_name}" in paths,
            "trigger": "RUN_SCAN_NOW",
            "worker_running": bool(operations.get("running")),
            "data_available": bool(scan.get("available")),
        },
        "recommendations": {
            "route_registered": "/api/recommendations-workspace" in paths,
            "depends_on": ["market_scan", "long_term_scan"],
            "data_available": bool(scan.get("available") or long_term.get("available")),
        },
        "market_reports": {
            "route_registered": "/api/market-reports-workspace" in paths,
            "trigger": "REFRESH_MARKET_REPORT_NOW",
            "worker_running": bool(operations.get("running")),
        },
        "stock_intelligence": {
            "route_registered": "/api/stock-intelligence/{symbol}" in paths,
            "acquire_route_registered": "/api/due-diligence/{symbol}/acquire" in paths,
        },
        "learning": {
            "operator_health_route_registered": "/api/operator-health" in paths,
            "status": str(autonomy.get("learning_status") or "UNKNOWN"),
            "supervisor_running": bool(autonomy.get("running")),
        },
    }
    wired = all(
        bool(item.get("route_registered", item.get("operator_health_route_registered", False)))
        for item in checks.values()
    )
    return {
        "wired": wired,
        "checks": checks,
        "note": (
            "wired=true proves the canonical API paths and triggers are registered. "
            "Data availability and provider health are reported separately and are never fabricated."
        ),
    }


app = product.app
