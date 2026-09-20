"""Canonical QuantTerm FastAPI application.

This module is the public backend boundary. Runtime recovery/routing lives in
`api.runtime`; callers launch `api.app:app` and do not depend on historical
module names.
"""
from __future__ import annotations

from . import runtime as _core

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


def decision_simulation_gate() -> dict:
    """Startup discovery/approval truth plus the current canonical best trades."""
    from product.decision_simulation_gate import status
    return _with_live_safety(status())


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
    """One approval action; autonomy owns the actual present/historical loops.

    Starting Decision Simulation must not launch a second legacy batch replay in
    parallel with the supervisor's historical-paper loop. The no-argument POST
    is therefore approval only. After approval, the autonomy supervisor runs
    snapshot-bound PAPER_FORWARD decisions when entries are allowed and
    HISTORICAL_REPLAY virtual-paper batches while the cash market is closed.

    Addressed requests remain explicit one-decision counterfactual inspections,
    but they cannot implicitly approve the production thesis.
    """
    addressed = bool(str(symbol or "").strip() or str(decision_id or "").strip())

    if addressed:
        # An explicitly addressed counterfactual is an inspection tool, not the
        # switch that starts autonomous Decision Simulation. It must remain
        # available before approval so operators/tests can inspect a historical
        # decision and PIT integrity without implicitly authorising any new
        # PAPER_FORWARD or autonomous HISTORICAL_REPLAY work.
        result = dict(_core.decision_simulator_run(
            symbol=symbol,
            as_of=as_of,
            alternative=alternative,
            decision_id=decision_id,
        ) or {})
        try:
            from product.decision_simulation_gate import status
            approval = status()
        except Exception:
            approval = {}
        result["approval"] = approval
        result["autonomy_approved"] = bool(approval.get("approved"))
        result["thesis_hash"] = str(approval.get("thesis_hash") or "")
        result["simulation_scope"] = ["EXPLICIT_COUNTERFACTUAL_ONLY"]
        result["simulation_authority"] = "operator-inspection"
        return _with_live_safety(result)

    from product.decision_simulation_gate import approve

    approval = approve()
    if not approval.get("accepted"):
        return _with_live_safety({
            "status": "WAITING_FOR_BEST_TRADES",
            "accepted": False,
            "approval": approval,
            "message": approval.get("message") or "Best-trade discovery is not ready.",
            "provenance": "HISTORICAL_REPLAY",
        })

    # Do not call _core.decision_simulator_run() here. That legacy endpoint
    # starts an independent batch replay and would duplicate the supervisor's
    # canonical historical-paper engine after this approval unlocks it.
    return _with_live_safety({
        "status": "APPROVED",
        "accepted": True,
        "approval": approval,
        "thesis_hash": str(approval.get("thesis_hash") or ""),
        "simulation_scope": ["PAPER_FORWARD", "HISTORICAL_REPLAY"],
        "simulation_authority": "quantterm-autonomy",
        "legacy_batch_started": False,
        "message": (
            "Decision Simulation approved. QuantTerm autonomy now owns the "
            "present PAPER_FORWARD pass and historical PIT virtual-paper loop."
        ),
    })


def market_reports_workspace() -> dict:
    """Canonical Market Reports projection using the builder's current contract.

    The legacy core route still supplied removed ``long_term_payload`` and
    ``market_payload`` kwargs.  Keep the public route at this façade and pass only
    sourced inputs that the current builder actually accepts.  Market context is
    resolved by the builder itself; no compatibility filler is invented here.
    """
    from product.recommendations_workspace import build_market_reports_workspace

    payload = build_market_reports_workspace(
        persist_today=True,
        news_payload=_core.core._news_payload(),
        scan_payload=_core.core._scan_payload(),
        rebuild=False,
    )
    if payload.get("needs_refresh") and not payload.get("empty_detail"):
        payload["empty_detail"] = (
            "Today's sourced market report is incomplete. Missing scan/news evidence "
            "stays empty; QuantTerm does not invent headlines, prices, or market facts."
        )
    return payload


def product_contract() -> dict:
    """Extend the canonical contract with the operator surfaces required for FINAL."""
    payload = dict(_core.product_contract() or {})
    checks = dict(payload.get("checks") or {})
    paths = {
        str(getattr(route, "path", ""))
        for route in app.routes
        if getattr(route, "path", None)
    }

    required_controls = {
        "REFRESH_DATA_NOW",
        "RUN_SCAN_NOW",
        "REFRESH_NEWS_NOW",
        "REFRESH_LONG_TERM_NOW",
        "REFRESH_FNO_NOW",
        "REFRESH_MARKET_REPORT_NOW",
        "RUN_CYCLE_NOW",
    }
    allowed_controls = {str(item).upper() for item in getattr(_core.core, "_ALLOWED_CONTROLS", set())}
    forbidden_fragments = ("LIVE", "BUY", "SELL", "UNLOCK", "BROKER")
    exposed_live_controls = sorted(
        control
        for control in allowed_controls
        if any(fragment in control for fragment in forbidden_fragments)
    )

    checks["operator_control_center"] = {
        "route_registered": "/api/controls/{control_name}" in paths,
        "required_controls": sorted(required_controls),
        "required_controls_available": sorted(required_controls & allowed_controls),
        "all_required_controls_available": required_controls <= allowed_controls,
        "live_money_controls_exposed": exposed_live_controls,
    }
    checks["forward_evidence_console"] = {
        "forward_evidence_route_registered": "/api/forward-evidence" in paths,
        "forward_soak_route_registered": "/api/forward-soak" in paths,
        "decision_simulator_route_registered": "/api/decision-simulator" in paths,
        "decision_simulation_gate_route_registered": "/api/decision-simulation-gate" in paths,
        "simulation_evidence_class": "HISTORICAL_REPLAY",
        "forward_evidence_class": "PAPER_FORWARD",
    }

    operator_ok = (
        checks["operator_control_center"]["route_registered"]
        and checks["operator_control_center"]["all_required_controls_available"]
        and not checks["operator_control_center"]["live_money_controls_exposed"]
    )
    evidence_ok = all(
        bool(value)
        for key, value in checks["forward_evidence_console"].items()
        if key.endswith("route_registered")
    )
    payload["checks"] = checks
    payload["wired"] = bool(payload.get("wired") and operator_ok and evidence_ok)
    payload["note"] = (
        "wired=true now also proves the operator Control Center and Forward Evidence console are registered, "
        "their safe controls are backed by the canonical allow-list, and no live-money mutation control is exposed."
    )
    return payload


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
_replace_route("/api/decision-simulation-gate", decision_simulation_gate, method="GET", name="decision_simulation_gate")
_replace_route("/api/decision-simulator", decision_simulator_get, method="GET", name="decision_simulator_get")
_replace_route("/api/decision-simulator", decision_simulator_run, method="POST", name="decision_simulator_run")
_replace_route("/api/market-reports-workspace", market_reports_workspace, method="GET", name="market_reports_workspace")
_replace_route("/api/product-contract", product_contract, method="GET", name="product_contract")
