"""Canonical terminal API with performance-safe, user-first operation routing.

The React desk is an interactive control plane. In the canonical one-terminal
runtime the launcher is the *single owner* of the Market Operations process.
The API may observe that worker and refuse a user command when it is unhealthy,
but it must never spawn, terminate, or replace the launcher-owned process.

Running ``terminal_product_api:app`` directly still keeps the legacy standalone
worker fallback from :mod:`terminal_api`; this wrapper is intentionally the
launcher-supervised product entry point.
"""
from __future__ import annotations

import time

import terminal_api as core
import terminal_product_api as product
from operations.store import pid_is_alive
from product.operator_health import enrich_autonomy_payload

# The base API intentionally keeps the control mapping in one mutable registry.
core._OPERATION_CONTROLS["RUN_LONG_TERM_SCAN_NOW"] = "LONG_TERM_SCAN"


def _healthy_runtime() -> dict:
    runtime = core._ops_runtime_payload()
    if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
        return runtime
    return runtime


def _ensure_ops_worker_strict(*, wait: bool = True) -> dict:
    """Observe the launcher-owned worker without ever taking lifecycle ownership.

    ``terminal_api`` calls this with ``wait=True`` during FastAPI startup. Startup
    is allowed to continue while the launcher watchdog is bringing Market Ops up,
    so the API itself never enters a competing restart loop.

    User controls call this with ``wait=False``. In that path an unhealthy worker
    fails loudly instead of returning a ghost ``accepted: true`` queue entry.
    """
    runtime = _healthy_runtime()
    if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
        return runtime

    if wait:
        deadline = time.time() + 3.0
        while time.time() < deadline:
            time.sleep(0.1)
            runtime = _healthy_runtime()
            if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
                return runtime
        # Startup remains non-owning: the launcher watchdog is the only component
        # allowed to recover Market Ops. Returning degraded state keeps the API up
        # so /api/health and the UI can show the blocker while recovery proceeds.
        return runtime

    raise RuntimeError(
        "Market operations worker is not healthy yet; the launcher watchdog owns "
        "recovery. The user command was not silently accepted."
    )


# The existing startup hook and control endpoint resolve this global at runtime.
# Replacing it therefore removes API-side worker lifecycle ownership without
# duplicating routes or changing the standalone terminal_api implementation.
core._ensure_ops_worker = _ensure_ops_worker_strict

# The durable scheduler ledger contains historical failures by design.  Enrich
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
