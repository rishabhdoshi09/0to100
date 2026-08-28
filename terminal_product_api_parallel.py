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


app = product.app
