"""Health diagnostics expose FD pressure without lying about readiness."""
from __future__ import annotations

from product.process_resources import (
    RESOURCE_EXHAUSTED,
    RESOURCE_OK,
    RESOURCE_PRESSURE,
    classify_fd_pressure,
    resource_diagnostics,
)
from product.runtime_lifecycle import inspect_runtime


def test_classify_fd_pressure_bands():
    assert classify_fd_pressure(10, 100) == RESOURCE_OK
    assert classify_fd_pressure(75, 100) == RESOURCE_PRESSURE
    assert classify_fd_pressure(95, 100) == RESOURCE_EXHAUSTED


def test_resource_exhausted_is_honest():
    payload = resource_diagnostics(api_pid=None, market_ops_pid=None)
    assert payload["state"] in {RESOURCE_OK, RESOURCE_PRESSURE, RESOURCE_EXHAUSTED}
    assert payload["api"]["pid"]
    assert "api" in payload
    assert "market_ops" in payload


def test_inspect_runtime_includes_resources():
    runtime = inspect_runtime(api_serving=True)
    assert "resources" in runtime
    assert runtime["resources"]["api"]["pid"]
    assert runtime["resources"]["state"] in {RESOURCE_OK, RESOURCE_PRESSURE, RESOURCE_EXHAUSTED, "UNKNOWN"}


def test_health_surfaces_resource_exhausted(monkeypatch):
    import terminal_api as api
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        "product.process_resources.resource_diagnostics",
        lambda **_k: {
            "state": RESOURCE_EXHAUSTED,
            "reason": "Process file-descriptor usage is exhausted.",
            "api": {"pid": 1, "fd_count": 950, "fd_soft_limit": 1024, "fd_used_pct": 92.8, "state": RESOURCE_EXHAUSTED},
            "market_ops": {"pid": 2, "fd_count": 40, "fd_soft_limit": 1024, "fd_used_pct": 3.9, "state": RESOURCE_OK},
            "active_operation_age_s": 4000,
            "oldest_running_operation": {"kind": "DUE_DILIGENCE_ACQUIRE", "age_s": 4000},
        },
    )
    client = TestClient(api.app)
    payload = client.get("/api/health").json()
    assert payload["resources"]["state"] == RESOURCE_EXHAUSTED
    assert payload["lifecycle"] == "FAILED"
    assert "exhausted" in str(payload["reason"]).lower() or "exhausted" in " ".join(payload.get("reasons") or []).lower()
