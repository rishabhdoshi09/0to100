"""Health diagnostics expose FD pressure without lying about readiness."""
from __future__ import annotations

import ctypes

from product.process_resources import (
    RESOURCE_EXHAUSTED,
    RESOURCE_OK,
    RESOURCE_PRESSURE,
    _darwin_open_fd_count,
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


def test_darwin_fd_counter_uses_libproc_without_spawning_lsof(monkeypatch):
    import product.process_resources as resources

    class FakeProcPidInfo:
        argtypes = None
        restype = None

        def __call__(self, _pid, _flavour, _arg, buffer, _size):
            if buffer is None:
                return 40  # capacity request: 5 proc_fdinfo records
            return 32  # actual result: 4 proc_fdinfo records

    class FakeLibProc:
        proc_pidinfo = FakeProcPidInfo()

    monkeypatch.setattr(resources.sys, "platform", "darwin")
    monkeypatch.setattr(ctypes, "CDLL", lambda *_a, **_k: FakeLibProc())

    assert _darwin_open_fd_count(4242) == 4


def test_darwin_fd_counter_is_disabled_on_non_macos(monkeypatch):
    import product.process_resources as resources

    monkeypatch.setattr(resources.sys, "platform", "linux")
    assert _darwin_open_fd_count(4242) is None


def test_inspect_runtime_does_not_conflate_operational_and_evidence(monkeypatch):
    import os
    import time
    from datetime import datetime, timezone, timedelta

    import product.runtime_lifecycle as RL

    pid = os.getpid()
    now = time.time()
    stale_scan = {
        "schema_version": 1,
        "scanned_at": (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat(),
        "records": [{"symbol": "OLD"}],
    }

    def fake_read_json(path):
        text = str(path)
        if "market_ops" in text:
            return {
                "worker_pid": pid,
                "pid": pid,
                "heartbeat_epoch": now,
                "process_running": True,
                "running": True,
                "active": {},
            }
        if "autonomy" in text:
            return {
                "scheduler_owner_pid": pid,
                "process_running": True,
                "running": True,
                "state": "OBSERVING",
            }
        if "latest_momentum_scan" in text:
            return stale_scan
        return {}

    monkeypatch.setattr(RL, "_read_json", fake_read_json)
    monkeypatch.setattr(RL, "_port_open", lambda port: True)
    monkeypatch.setattr(RL, "_pid_alive", lambda value: int(value or 0) == pid)
    monkeypatch.setattr(
        "data.bhavcopy_runtime.official_history_freshness",
        lambda *_a, **_k: {
            "current": False,
            "ready": True,
            "sessions": 100,
            "reason_code": "HISTORY_STALE",
            "available_session": "2026-09-01",
            "expected_latest_completed_session": "2026-09-05",
        },
    )
    monkeypatch.setattr(
        "data.bhavcopy_runtime.status",
        lambda **_k: {"ready": True, "sessions": 100, "latest_date": "2026-09-01"},
    )
    monkeypatch.setattr(
        "product.process_resources.resource_diagnostics",
        lambda **_k: {
            "state": RESOURCE_OK,
            "reason": "",
            "api": {"pid": pid, "state": RESOURCE_OK},
            "market_ops": {"pid": pid, "state": RESOURCE_OK},
        },
    )

    runtime = inspect_runtime(api_serving=True)
    assert runtime["operational_ready"] is True
    assert runtime["evidence_ready"] is False
    assert runtime["operational"]["status"] in {"READY", "DEGRADED"}
    assert runtime["lifecycle"] == "DEGRADED"
    assert runtime["lifecycle"] != "STARTING"
    assert runtime["lifecycle"] != "READY"
