from __future__ import annotations


def _unverified_projection():
    return {
        "live_locked": None,
        "live_lock_verified": False,
        "live_execution_authorized": None,
        "live_lock_status": "UNVERIFIED",
        "live_lock_reason": "test probe unavailable",
        "live_lock_source": "product.live_execution_interlock",
    }


def test_learning_dashboard_overrides_legacy_positive_safety(monkeypatch):
    import product.live_safety as live_safety
    import product.paper_learning_loop as learning

    monkeypatch.setattr(
        learning._core,
        "learning_dashboard",
        lambda policy_path=None: {
            "schema_version": 1,
            "live_locked": True,
            "policies": [],
            "counterfactuals": [],
            "forward_soak": {"error": "probe failed", "live_locked": True},
        },
    )
    monkeypatch.setattr(live_safety, "live_safety_projection", _unverified_projection)

    payload = learning.learning_dashboard()
    assert payload["live_locked"] is None
    assert payload["live_lock_verified"] is False
    assert payload["live_execution_authorized"] is None
    assert payload["forward_soak"]["live_locked"] is None
    assert payload["forward_soak"]["live_lock_verified"] is False


def test_paper_autopilot_route_overrides_legacy_positive_safety(monkeypatch):
    import product.live_safety as live_safety
    import api.app as terminal_api

    monkeypatch.setattr(
        terminal_api._core,
        "paper_autopilot",
        lambda: {
            "schema_version": 1,
            "paper": {"open_positions": []},
            "latest": {},
            "live_locked": True,
        },
    )
    monkeypatch.setattr(live_safety, "live_safety_projection", _unverified_projection)

    payload = terminal_api.paper_autopilot()
    assert payload["live_locked"] is None
    assert payload["live_lock_verified"] is False
    assert payload["live_execution_authorized"] is None

    routes = [
        route
        for route in terminal_api.app.router.routes
        if getattr(route, "path", None) == "/api/paper-autopilot"
        and "GET" in (getattr(route, "methods", set()) or set())
    ]
    assert len(routes) == 1
    assert routes[0].endpoint is terminal_api.paper_autopilot


def test_decision_simulator_routes_override_legacy_positive_safety(monkeypatch):
    import product.live_safety as live_safety
    import api.app as terminal_api

    monkeypatch.setattr(
        terminal_api._core,
        "decision_simulator_get",
        lambda **kwargs: {
            "available": False,
            "provenance": "BACKTEST",
            "live_locked": True,
            "cache_hit": True,
        },
    )
    monkeypatch.setattr(
        terminal_api._core,
        "decision_simulator_run",
        lambda **kwargs: {"status": "RUNNING", "live_locked": True},
    )
    monkeypatch.setattr(live_safety, "live_safety_projection", _unverified_projection)

    read_payload = terminal_api.decision_simulator_get(symbol="TCS", as_of="2024-01-15")
    run_payload = terminal_api.decision_simulator_run()
    for payload in (read_payload, run_payload):
        assert payload["live_locked"] is None
        assert payload["live_lock_verified"] is False
        assert payload["live_execution_authorized"] is None

    get_routes = [
        route
        for route in terminal_api.app.router.routes
        if getattr(route, "path", None) == "/api/decision-simulator"
        and "GET" in (getattr(route, "methods", set()) or set())
    ]
    post_routes = [
        route
        for route in terminal_api.app.router.routes
        if getattr(route, "path", None) == "/api/decision-simulator"
        and "POST" in (getattr(route, "methods", set()) or set())
    ]
    assert len(get_routes) == 1
    assert get_routes[0].endpoint is terminal_api.decision_simulator_get
    assert len(post_routes) == 1
    assert post_routes[0].endpoint is terminal_api.decision_simulator_run
