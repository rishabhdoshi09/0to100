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


def test_institutional_dashboard_path_is_cache_only(monkeypatch):
    import terminal_api as api

    calls: list[bool] = []

    def fake_workspace(*, days=30, include_nifty_options=True, allow_network=True):
        calls.append(bool(allow_network))
        return {"available": True, "network_used": allow_network, "cash": {}}

    monkeypatch.setattr("data.fii_dii_store.workspace_payload", fake_workspace)
    out = api._institutional_payload(allow_network=False)
    assert out["network_used"] is False
    assert calls == [False]


def test_workspace_payload_allow_network_false_skips_nse(monkeypatch):
    import data.fii_dii_store as store

    monkeypatch.setattr(store, "summarize", lambda days, auto_refresh=True: {
        "available": True,
        "note": "",
        "sessions": 1,
        "auto_refresh": auto_refresh,
    })

    def boom(*a, **k):
        raise AssertionError("NSE path must not run when allow_network=False")

    monkeypatch.setattr("data.fii_dii.get_fii_derivative_stats_uncached", boom)
    monkeypatch.setattr("data.institutional_flows.get_flows", boom)
    monkeypatch.setattr("options.chain_fetch.chain_workspace_cached", boom)
    monkeypatch.setattr(
        "data.fii_dii.get_fii_derivative_stats_cached_only",
        lambda: {"available": False},
    )
    monkeypatch.setattr(
        "data.institutional_flows.get_flows_cached_only",
        lambda: {"bulk_deals": [], "bulk_buys": []},
    )
    monkeypatch.setattr(
        "options.chain_fetch.chain_workspace_memory_only",
        lambda _sym="NIFTY": {"available": False},
    )
    out = store.workspace_payload(allow_network=False)
    assert out["network_used"] is False
    assert out["available"] is True


def test_operations_snapshot_uses_one_connection(tmp_path):
    from operations.store import OperationStore, SUCCEEDED

    db = tmp_path / "ops.db"
    store = OperationStore(db)
    op, _ = store.enqueue("MARKET_SCAN", lane="market_scan")
    store.finish(op["operation_id"], status=SUCCEEDED, message="done", result={"ok": True})
    snap = store.dashboard_snapshot(kinds=["MARKET_SCAN", "NEWS_REFRESH"], recent_limit=10)
    assert "MARKET_SCAN" in snap["latest"]
    assert snap["counts"].get(SUCCEEDED) == 1
    assert len(snap["recent"]) == 1


def test_dashboard_micro_cache_avoids_rebuild(monkeypatch):
    import terminal_api as api

    api._dashboard_cache = {"ts": 0.0, "payload": None}
    builds: list[int] = []

    monkeypatch.setattr(api, "_ensure_ops_worker", lambda **k: {"running": True})
    monkeypatch.setattr(api, "_schedule_regime_refresh", lambda: None)
    monkeypatch.setattr(api, "_schedule_institutional_refresh", lambda: None)
    monkeypatch.setattr(api, "_market_payload", lambda **k: {"available": True, "health": "ok", "summary": "", "trade_stance": "", "breadth": "—", "leaders": [], "laggards": [], "nifty_change_1d": None, "nifty_change_5d": None, "vix": None, "technical_details": {}})
    monkeypatch.setattr(api, "_scan_payload", lambda **k: (builds.append(1) or {"available": False, "universe_size": 0, "summary": {}, "records": []}))
    monkeypatch.setattr(api, "_long_term_payload", lambda **k: {"available": False, "summary": {}, "records": [], "job": {}})
    monkeypatch.setattr(api, "_paper_payload", lambda: {"available": False, "enabled": False, "supervisor_running": False, "capital": 0, "equity": 0, "equity_curve": [], "open_risk": 0, "risk_per_trade_pct": 0, "max_positions": 0, "open_positions": [], "closed_trades": [], "refusals": [], "last_cycle": {}, "last_error": ""})
    monkeypatch.setattr(api, "_autonomy_payload", lambda: {"available": False, "running": False, "state": "UNKNOWN", "plain_state": "", "explanation": "", "heartbeat_ist": "", "new_paper_entries": False, "recent_dialogue": [], "jobs": {}})
    monkeypatch.setattr(api, "_operations_payload", lambda **k: {"available": True, "running": True, "latest": {}, "active": [], "recent": [], "counts": {}, "active_lanes": {}})
    monkeypatch.setattr(api, "_news_payload", lambda **k: {"available": False, "stats": {}, "articles": [], "source_health": [], "latest_refresh": {}})
    monkeypatch.setattr(api, "_fno_payload", lambda: {"available": False, "underlyings": [], "exclusions": [], "mapped_underlyings": 0, "source": "x"})
    monkeypatch.setattr(api, "_data_payload", lambda *a, **k: {"ready": False, "bhavcopy": {}, "blockers": []})
    monkeypatch.setattr(api, "_institutional_payload", lambda **k: {"available": False})
    monkeypatch.setattr(api, "_conviction", lambda *a, **k: [])

    first = api.dashboard()
    second = api.dashboard()
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert len(builds) == 1
