from __future__ import annotations

import terminal_product_api_parallel as api


PRIMARY_PATHS = {
    "/api/dashboard",
    "/api/operations/{operation_id}",
    "/api/controls/{control_name}",
    "/api/recommendations-workspace",
    "/api/market-reports-workspace",
    "/api/stock-intelligence/{symbol}",
    "/api/due-diligence/{symbol}",
    "/api/due-diligence/{symbol}/acquire",
    "/api/operator-health",
    "/api/product-contract",
    "/api/strategy-catalog",
    "/api/research-status",
    "/api/system-health-contract",
    "/api/scan-audit",
    "/api/decision-journal",
    "/api/paper-autopilot",
    "/api/why-no-trade",
    "/api/learning-policies",
    "/api/learning-dashboard",
    "/api/forward-soak",
    "/api/decision-simulator",
    "/api/decision-simulation-gate",
}


def _paths() -> set[str]:
    return {
        str(getattr(route, "path", ""))
        for route in api.app.routes
        if getattr(route, "path", None)
    }


def test_primary_react_product_routes_are_registered():
    missing = PRIMARY_PATHS - _paths()
    assert not missing, f"React product calls missing API route(s): {sorted(missing)}"


def test_product_contract_separates_wiring_from_data_availability(monkeypatch):
    monkeypatch.setattr(api.core, "_scan_payload", lambda: {"available": False, "records": []})
    monkeypatch.setattr(api.core, "_long_term_payload", lambda: {"available": False, "records": []})
    monkeypatch.setattr(api.core, "_operations_payload", lambda: {"running": False})
    monkeypatch.setattr(
        api.core,
        "_autonomy_payload",
        lambda: {"running": True, "learning_status": "WAITING_FOR_FRESH_EOD_DATA"},
    )

    payload = api.product_contract()
    assert payload["wired"] is True
    assert payload["checks"]["recommendations"]["route_registered"] is True
    assert payload["checks"]["recommendations"]["data_available"] is False
    assert payload["checks"]["market_scan"]["worker_running"] is False
    assert payload["checks"]["learning"]["status"] == "WAITING_FOR_FRESH_EOD_DATA"
    assert payload["checks"]["learning"]["dashboard_route_registered"] is True
    assert payload["checks"]["strategies"]["catalog_route_registered"] is True
    assert payload["checks"]["system_health"]["health_contract_route_registered"] is True


def test_recommendations_route_returns_honest_empty_workspace(monkeypatch):
    monkeypatch.setattr(
        api.core,
        "_scan_payload",
        lambda: {"available": False, "records": [], "scanned_at": "", "records_status": ""},
    )
    monkeypatch.setattr(
        api.core,
        "_long_term_payload",
        lambda: {"available": False, "records": [], "scanned_at": ""},
    )

    payload = api.recommendations_workspace()
    assert payload["schema_version"] >= 1
    assert isinstance(payload["categories"], list)
    assert payload["ensemble"]["high_conviction_count"] == 0
    assert payload["from_saved_market_scan"] is True


def test_market_reports_route_returns_structured_empty_state(monkeypatch, tmp_path):
    monkeypatch.setattr(api.core, "_scan_payload", lambda: {"available": False, "records": []})
    monkeypatch.setattr(api.core, "_news_payload", lambda: {"available": False, "articles": []})

    import product.recommendations_workspace as rw
    monkeypatch.setattr(rw, "REPORTS_DIR", tmp_path / "market_reports")

    payload = api.market_reports_workspace()
    assert payload["schema_version"] >= 1
    assert isinstance(payload["reports"], list)
    assert isinstance(payload["missing_lanes"], list)
    assert payload["needs_refresh"] is True
    assert "invent" in (payload.get("empty_detail") or "").lower()


def test_decision_simulation_approval_does_not_start_legacy_batch(monkeypatch):
    import product.decision_simulation_gate as gate

    approval = {
        "accepted": True,
        "phase": "APPROVED",
        "thesis_hash": "thesis-1",
        "simulation_scope": ["PAPER_FORWARD", "HISTORICAL_REPLAY"],
    }
    monkeypatch.setattr(gate, "approve", lambda: dict(approval))
    monkeypatch.setattr(api, "_with_live_safety", lambda payload: dict(payload))

    def forbidden(*args, **kwargs):
        raise AssertionError("approval must not launch the legacy batch simulator")

    monkeypatch.setattr(api._core, "decision_simulator_run", forbidden)
    payload = api.decision_simulator_run()

    assert payload["accepted"] is True
    assert payload["status"] == "APPROVED"
    assert payload["simulation_authority"] == "quantterm-autonomy"
    assert payload["legacy_batch_started"] is False
    assert payload["thesis_hash"] == "thesis-1"


def test_addressed_counterfactual_is_inspection_only_and_does_not_approve(monkeypatch):
    import product.decision_simulation_gate as gate

    monkeypatch.setattr(
        gate,
        "status",
        lambda: {
            "phase": "AWAITING_APPROVAL",
            "approved": False,
            "thesis_hash": "thesis-1",
        },
    )
    monkeypatch.setattr(api, "_with_live_safety", lambda payload: dict(payload))
    monkeypatch.setattr(
        api._core,
        "decision_simulator_run",
        lambda **kwargs: {
            "status": "SUCCEEDED",
            "kind": "PAST_DECISION_SIMULATION",
            "symbol": kwargs.get("symbol"),
            "fingerprint": "fp-1",
        },
    )

    payload = api.decision_simulator_run(symbol="RELIANCE")

    assert payload["status"] == "SUCCEEDED"
    assert payload["fingerprint"] == "fp-1"
    assert payload["autonomy_approved"] is False
    assert payload["simulation_authority"] == "operator-inspection"
    assert payload["simulation_scope"] == ["EXPLICIT_COUNTERFACTUAL_ONLY"]

