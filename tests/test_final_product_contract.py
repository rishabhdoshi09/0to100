from __future__ import annotations


def test_final_product_contract_requires_operator_and_forward_evidence_surfaces() -> None:
    import api.app as terminal_api

    payload = terminal_api.product_contract()
    checks = payload["checks"]

    operator = checks["operator_control_center"]
    assert operator["route_registered"] is True
    assert operator["all_required_controls_available"] is True
    assert operator["live_money_controls_exposed"] == []
    assert set(operator["required_controls"]) == {
        "REFRESH_DATA_NOW",
        "RUN_SCAN_NOW",
        "REFRESH_NEWS_NOW",
        "REFRESH_LONG_TERM_NOW",
        "REFRESH_FNO_NOW",
        "REFRESH_MARKET_REPORT_NOW",
        "RUN_CYCLE_NOW",
    }

    evidence = checks["forward_evidence_console"]
    assert evidence["forward_evidence_route_registered"] is True
    assert evidence["forward_soak_route_registered"] is True
    assert evidence["decision_simulator_route_registered"] is True
    assert evidence["simulation_evidence_class"] == "HISTORICAL_REPLAY"
    assert evidence["forward_evidence_class"] == "PAPER_FORWARD"

    # The extended contract is an acceptance gate, not decoration: any missing
    # operator/evidence route or unsafe control must make wired=false.
    assert payload["wired"] is True


def test_product_contract_route_is_unique_and_points_to_final_wrapper() -> None:
    import api.app as terminal_api

    routes = [
        route
        for route in terminal_api.app.router.routes
        if getattr(route, "path", None) == "/api/product-contract"
        and "GET" in (getattr(route, "methods", set()) or set())
    ]
    assert len(routes) == 1
    assert routes[0].endpoint is terminal_api.product_contract
