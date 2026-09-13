"""Canonical QuantTerm API façade with fail-closed live-safety projection.

The established API implementation lives in ``_terminal_product_api_parallel_core``.
This small boundary module replaces only the Paper Autopilot route that previously
stamped ``live_locked=True``. All other routing/recovery behaviour is preserved.
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


def paper_autopilot() -> dict:
    """Existing Paper Autopilot projection plus canonical broker-boundary truth."""
    from product.live_safety import live_safety_projection

    payload = dict(_core.paper_autopilot() or {})
    payload.update(live_safety_projection())
    return payload


def _replace_get_route(path: str, endpoint, name: str) -> None:
    # Remove the previous route before adding the truthful projection. FastAPI
    # resolves in registration order, so merely adding a duplicate would leave
    # the old hard-coded endpoint reachable first.
    app.router.routes[:] = [
        route
        for route in app.router.routes
        if not (
            getattr(route, "path", None) == path
            and "GET" in (getattr(route, "methods", set()) or set())
        )
    ]
    app.add_api_route(path, endpoint, methods=["GET"], name=name)


_replace_get_route("/api/paper-autopilot", paper_autopilot, "paper_autopilot")
