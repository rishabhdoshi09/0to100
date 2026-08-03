"""Dashboard must not block on Yahoo regime fetch or huge scan payloads."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest


def test_compute_regime_cache_only_does_not_fetch(monkeypatch):
    import core.regime_engine as RE

    sentinel = SimpleNamespace(summary_line=lambda: "cached")
    RE._CACHE["regime_state"] = sentinel
    RE._CACHE["timestamp"] = time.time() - 10_000  # expired vs TTL

    def boom(*a, **k):
        raise AssertionError("network fetch must not run on allow_network=False")

    monkeypatch.setattr(RE, "_fetch_ohlcv", boom)
    out = RE.compute_regime(allow_network=False)
    assert out is sentinel


def test_compute_regime_cache_only_raises_when_empty(monkeypatch):
    import core.regime_engine as RE

    RE._CACHE.clear()

    def boom(*a, **k):
        raise AssertionError("network fetch must not run on allow_network=False")

    monkeypatch.setattr(RE, "_fetch_ohlcv", boom)
    with pytest.raises(RuntimeError, match="deferred"):
        RE.compute_regime(allow_network=False)


def test_scan_payload_truncates_large_results(monkeypatch):
    import terminal_api as api

    records = []
    for i in range(300):
        records.append(
            {
                "symbol": f"S{i:03d}",
                "status": "Watch for breakout" if i % 2 else "Watch",
                "verdict": "WATCH",
                "signals": ["PRE_BREAKOUT"] if i % 2 else [],
                "score": float(300 - i),
                "entry": 100.0,
                "stop": 90.0,
                "target": 120.0,
                "price": 98.0,
                "chase_risk": False,
                "reasons": ["x"],
            }
        )
    payload = {
        "schema_version": 1,
        "scanned_at": "2026-08-03T00:00:00+00:00",
        "universe_size": 300,
        "summary": {"with_any_setup": 300},
        "records": records,
    }
    monkeypatch.setattr("product.scan_store.load_scan", lambda path=None: payload)
    out = api._scan_payload(record_limit=40)
    assert out["available"] is True
    assert out["records_total"] == 300
    assert out["records_truncated"] is True
    assert len(out["records"]) <= 40


def test_ensure_ops_worker_wait_zero_does_not_sleep(monkeypatch):
    import terminal_api as api

    monkeypatch.setattr(api, "OPS_RUNTIME", api.Path("/tmp/qt-missing-runtime.json"))
    monkeypatch.setattr(api, "_ops_process", None)
    api._ops_ensure_last_attempt = 0.0
    sleeps: list[float] = []

    class FakePopen:
        def __init__(self, *a, **k):
            self.pid = 4242

        def poll(self):
            return None  # still starting

    monkeypatch.setattr(api.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(api, "_reclaim_stale_ops_lock", lambda **k: "ok")
    monkeypatch.setattr(
        api,
        "_ops_runtime_payload",
        lambda: {"running": False, "process_running": False, "active": {}},
    )
    real_sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    out = api._ensure_ops_worker(wait_s=0.0, force=True)
    assert out.get("ensure_attempted") is True
    assert sleeps == []
    # restore unused
    assert real_sleep
