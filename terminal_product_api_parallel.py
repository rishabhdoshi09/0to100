"""Canonical terminal API with performance-safe, user-first operation routing.

The React desk is an interactive control plane. A user click must never be
acknowledged as a ghost queue entry with no healthy consumer. This wrapper keeps
the canonical product API, corrects long-term routing, and hardens the market-ops
worker handshake used by all terminal controls.
"""
from __future__ import annotations

import os
import signal
import time

import terminal_api as core
import terminal_product_api as product
from operations.store import pid_is_alive

# The base API intentionally keeps the control mapping in one mutable registry.
core._OPERATION_CONTROLS["RUN_LONG_TERM_SCAN_NOW"] = "LONG_TERM_SCAN"

_base_ensure_ops_worker = core._ensure_ops_worker


def _stop_stale_owner(runtime: dict) -> None:
    """Terminate a stale worker owner before replacement; PID existence is not health."""
    candidates: list[int] = []
    try:
        runtime_pid = int(runtime.get("worker_pid") or 0)
    except (TypeError, ValueError):
        runtime_pid = 0
    if runtime_pid > 1:
        candidates.append(runtime_pid)
    proc = getattr(core, "_ops_process", None)
    if proc is not None and proc.poll() is None and proc.pid > 1:
        candidates.append(int(proc.pid))
    for pid in dict.fromkeys(candidates):
        if not pid_is_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
        deadline = time.time() + 0.8
        while time.time() < deadline and pid_is_alive(pid):
            time.sleep(0.04)
        if pid_is_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    proc = getattr(core, "_ops_process", None)
    if proc is not None and proc.poll() is not None:
        core._ops_process = None


def _ensure_ops_worker_strict(*, wait: bool = True) -> dict:
    """Return only with a heartbeat-healthy worker, or fail the user request loudly."""
    runtime = core._ops_runtime_payload()
    if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
        return runtime

    # A prior child can be alive but wedged before/after taking the worker lock.
    # Kill that stale ownership before asking the canonical starter to replace it.
    _stop_stale_owner(runtime)
    runtime = _base_ensure_ops_worker(wait=True)
    if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
        return runtime
    raise RuntimeError(
        "Market operations worker did not become ready; Scan Now was not silently accepted. "
        "The launcher watchdog will retry it."
    )


# The existing control endpoint resolves this global at request time, so every
# market operation gets the readiness handshake without duplicating API routes.
core._ensure_ops_worker = _ensure_ops_worker_strict

app = product.app
