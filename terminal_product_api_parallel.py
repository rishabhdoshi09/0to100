"""Canonical QuantTerm API façade with fail-closed live-safety projection.

The established API implementation lives in ``_terminal_product_api_parallel_core``.
This small boundary module preserves its routing/recovery behaviour while replacing
operator-facing routes that historically stamped or could retain ``live_locked=True``.
Every exposed safety value comes from the canonical live-execution interlock.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_CORE_PATH = Path(__file__).with_name("_terminal_product_api_parallel_core.py")
_SPEC = importlib.util.spec_from_file_location("_quantterm_terminal_product_api_parallel_core", _CORE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to load canonical terminal API core from {_CORE_PATH}")
_core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_core)

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

app = _core.app


def _with_live_safety(payload) -> dict:
    """Overlay canonical broker-boundary truth on an existing API projection."""
    from product.live_safety import live_safety_projection

    out = dict(payload or {})
    out.update(live_safety_projection())
    return out


def paper_autopilot() -> dict:
    """Existing Paper Autopilot projection plus canonical broker-boundary truth."""
    return _with_live_safety(_core.paper_autopilot())


def decision_simulator_get(
    symbol: str = "",
    as_of: str = "",
    alternative: str = "",
    decision_id: str = "",
) -> dict:
    """Decision-simulator read projection with canonical broker-boundary truth."""
    return _with_live_safety(
        _core.decision_simulator_get(
            symbol=symbol,
            as_of=as_of,
            alternative=alternative,
            decision_id=decision_id,
        )
    )


def decision_simulator_run(
    symbol: str = "",
    as_of: str = "",
    alternative: str = "",
    decision_id: str = "",
) -> dict:
    """Decision-simulator trigger/result with canonical broker-boundary truth."""
    return _with_live_safety(
        _core.decision_simulator_run(
            symbol=symbol,
            as_of=as_of,
            alternative=alternative,
            decision_id=decision_id,
        )
    )


def _replace_route(path: str, endpoint, *, method: str, name: str) -> None:
    """Replace one method/path pair; duplicates would leave the legacy route first."""
    wanted = method.upper()
    app.router.routes[:] = [
        route
        for route in app.router.routes
        if not (
            getattr(route, "path", None) == path
            and wanted in (getattr(route, "methods", set()) or set())
        )
    ]
    app.add_api_route(path, endpoint, methods=[wanted], name=name)


_replace_route("/api/paper-autopilot", paper_autopilot, method="GET", name="paper_autopilot")
_replace_route("/api/decision-simulator", decision_simulator_get, method="GET", name="decision_simulator_get")
_replace_route("/api/decision-simulator", decision_simulator_run, method="POST", name="decision_simulator_run")
