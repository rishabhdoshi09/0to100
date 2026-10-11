"""Nonblocking browser dashboard reads preserve history but not stale authority."""
from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from product.dashboard_snapshot_cache import DashboardSnapshotCache


def snapshot():
    return {
        "generated_at": "2026-10-11T00:00:00Z",
        "scan": {"available": True, "scanned_at": "2026-10-10", "records": [{"symbol": "INFY"}]},
        "long_term": {"available": True, "records": [{"symbol": "INFY"}]},
        "news": {"available": True, "articles": [{"title": "Article"}]},
        "paper": {"available": True, "enabled": True, "supervisor_running": True,
                  "open_positions": [{"symbol": "INFY"}]},
        "autonomy": {"available": True, "running": True, "process_running": True,
                     "new_paper_entries": True, "existing_exits": True,
                     "research_enabled": True, "new_entry_capability": "allowed",
                     "existing_exit_capability": "allowed", "research_capability": "allowed",
                     "broker": {"state": "CONNECTED", "ready": True}},
        "operations": {"available": True, "running": True,
                       "active": [{"kind": "MARKET_SCAN"}],
                       "active_lanes": {"scan": "busy"}},
        "fno": {"available": True,
                "directional": {"available": True, "status": "READY",
                                "candidates": [{"symbol": "INFY"}]},
                "desk": {"status": "READY", "candidates": [{"symbol": "INFY"}],
                         "candidate_count": 1, "paper_available": True}},
        "data": {"ready": True, "blockers": []},
        "conviction": [{"symbol": "INFY"}],
        "daily_wrap": [{"text": "Today's note"}],
    }


def _wait_for_completion(cache):
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        with cache._lock:
            if not cache._refreshing:
                return
        time.sleep(0.01)
    raise AssertionError("background dashboard refresh did not complete")


def test_cold_browser_polls_return_quickly_and_spawn_only_one_loader():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def slow_loader():
        calls.append(True)
        entered.set()
        assert release.wait(timeout=3)
        return snapshot()

    cache = DashboardSnapshotCache(slow_loader, snapshot, ttl_seconds=45)
    try:
        start = time.monotonic()
        first = cache.read()
        assert time.monotonic() - start < 1.0
        assert first["dashboard_cache"]["status"] == "BOOTSTRAPPING"
        assert first["paper"]["enabled"] is False
        assert entered.wait(1)
        again = cache.read()
        assert again["dashboard_cache"]["status"] == "BOOTSTRAPPING"
        assert len(calls) == 1
    finally:
        release.set()
    _wait_for_completion(cache)
    ready = cache.read()
    assert ready["dashboard_cache"]["status"] == "FRESH"
    assert ready["paper"]["enabled"] is True
    assert ready["scan"]["records"][0]["symbol"] == "INFY"


def test_expired_snapshot_returns_historical_rows_but_blocks_authority():
    release = threading.Event()

    def loader():
        assert release.wait(3)
        return snapshot()

    cache = DashboardSnapshotCache(loader, snapshot, ttl_seconds=45)
    with cache._lock:
        cache._snapshot = snapshot()
        cache._completed_at = time.monotonic() - 46

    try:
        result = cache.read()
        assert result["dashboard_cache"]["status"] == "STALE"
        assert result["scan"]["records"][0]["symbol"] == "INFY"
        assert result["scan"]["available"] is False
        assert result["paper"]["open_positions"][0]["symbol"] == "INFY"
        assert result["paper"]["enabled"] is False
        assert result["paper"]["supervisor_running"] is False
        assert result["autonomy"]["new_paper_entries"] is False
        assert result["autonomy"]["existing_exits"] is False
        assert result["autonomy"]["broker"]["ready"] is False
        assert result["operations"]["running"] is False
        assert result["data"]["ready"] is False
        assert result["fno"]["directional"]["candidates"] == []
        assert result["fno"]["desk"]["candidates"] == []
        assert result["conviction"] == []
        assert result["dashboard_cache"]["age_seconds"] >= 45
    finally:
        release.set()
    _wait_for_completion(cache)


def test_failed_refresh_keeps_last_good_but_never_marks_it_current():
    cache = DashboardSnapshotCache(
        lambda: (_ for _ in ()).throw(RuntimeError("simulated")),
        snapshot,
        ttl_seconds=45,
    )
    with cache._lock:
        cache._snapshot = snapshot()
        cache._completed_at = time.monotonic() - 46

    stale = cache.read()
    assert stale["dashboard_cache"]["status"] == "STALE"
    _wait_for_completion(cache)
    failed = cache.read()
    assert failed["dashboard_cache"]["status"] == "STALE"
    assert failed["dashboard_cache"]["error"].startswith("Dashboard builder failed")
    assert failed["paper"]["enabled"] is False
    assert failed["scan"]["records"][0]["symbol"] == "INFY"


def test_http_dashboard_uses_cache_without_changing_internal_diagnostic_contract(monkeypatch):
    import terminal_api

    class FakeCache:
        def read(self):
            return {"marker": "cached_http"}

    monkeypatch.setattr(terminal_api, "_dashboard_http_cache", FakeCache())
    client = TestClient(terminal_api.app)
    result = client.get("/api/dashboard")
    assert result.status_code == 200
    assert result.json() == {"marker": "cached_http"}


def test_gate_stale_snapshot_never_grants_approval_or_active_trades():
    from api.app import _gate_http_stale, _gate_http_cold

    fake = _gate_http_cold()
    fake.update(
        approved=True, approval_required=False, discovery_ready=True,
        scan_fresh=True, actionable=9, research_actionable=10,
        best_trades=[{"symbol": "INFY"}], live_execution_authorized=True,
        current_thesis_hash="old", current_evolution_policy_fingerprint="old",
    )
    _gate_http_stale(fake)
    assert fake["approved"] is False
    assert fake["approval_required"] is True
    assert fake["discovery_ready"] is False
    assert fake["scan_fresh"] is False
    assert fake["best_trades"] == []
    assert fake["actionable"] == 0
    assert fake["live_execution_authorized"] is False
    assert fake["current_thesis_hash"] == ""


def test_gate_http_get_nonblocking_even_when_original_compute_is_slow(monkeypatch):
    import api.app as canonical

    entered = threading.Event()
    release = threading.Event()

    def slow_build():
        entered.set()
        assert release.wait(3)
        return {
            **canonical._gate_http_cold(),
            "approved": True,
            "discovery_ready": True,
            "scan_fresh": True,
            "best_trades": [{"symbol": "INFY"}],
        }

    cache = DashboardSnapshotCache(
        slow_build, canonical._gate_http_cold,
        ttl_seconds=15, stale_transform=canonical._gate_http_stale,
        metadata_key="status_cache",
    )
    monkeypatch.setattr(canonical, "_gate_http_cache", cache)
    try:
        client = TestClient(canonical.app)
        started = time.monotonic()
        response = client.get("/api/decision-simulation-gate")
        assert time.monotonic() - started < 1.0
        assert response.status_code == 200
        body = response.json()
        assert body["approved"] is False
        assert body["discovery_ready"] is False
        assert body["status_cache"]["status"] == "BOOTSTRAPPING"
        assert entered.wait(1)
        client.get("/api/decision-simulation-gate")
    finally:
        release.set()
    _wait_for_completion(cache)
    ready = client.get("/api/decision-simulation-gate").json()
    assert ready["approved"] is True
    assert ready["status_cache"]["status"] == "FRESH"


def test_direct_gate_status_remains_uncached_and_authoritative(monkeypatch):
    import api.app as canonical

    monkeypatch.setattr(
        canonical, "_compute_gate_http_status",
        lambda: {"approved": True, "discovery_ready": True},
    )
    monkeypatch.setattr(
        canonical, "_gate_http_cache",
        type("OnlyHTTP", (), {"read": lambda self: (_ for _ in ()).throw(
            AssertionError("direct status unexpectedly used HTTP cache"))})(),
    )
    assert canonical.decision_simulation_gate()["approved"] is True
