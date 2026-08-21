"""Full-universe signal backtest — 100% bhav stocks, never places orders."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_run_backtest_none_uses_all_symbols(monkeypatch):
    from scan import signal_backtest as SB

    available = [f"S{i:03d}" for i in range(50)]
    monkeypatch.setattr("data.bhavcopy_store.store_symbols", lambda: available)

    captured = {}

    def fake_inner(sc, stats, symbols, sample_step, lookback_sessions, horizon, t0, on_trade=None):
        captured["symbols"] = list(symbols)
        return {
            "generated_at": "test",
            "symbols": len(symbols),
            "horizon_days": horizon,
            "signals": {},
            "elapsed_s": 0.1,
        }

    monkeypatch.setattr(SB, "_run_backtest_inner", fake_inner)
    monkeypatch.setattr(SB, "UnifiedScanner", lambda: object(), raising=False)
    # UnifiedScanner imported inside run_backtest — patch the module attr used after import
    import scan.unified_scanner as US

    monkeypatch.setattr(US, "UnifiedScanner", lambda: object())

    out = SB.run_backtest(max_symbols=None)
    assert captured["symbols"] == available
    assert out["universe"]["truncated"] is False
    assert out["universe"]["run"] == 50


def test_run_backtest_cap_marks_truncated(monkeypatch):
    from scan import signal_backtest as SB
    import scan.unified_scanner as US

    available = [f"S{i:03d}" for i in range(30)]
    monkeypatch.setattr("data.bhavcopy_store.store_symbols", lambda: available)
    monkeypatch.setattr(US, "UnifiedScanner", lambda: object())

    def fake_inner(sc, stats, symbols, sample_step, lookback_sessions, horizon, t0, on_trade=None):
        return {
            "generated_at": "test",
            "symbols": len(symbols),
            "horizon_days": horizon,
            "signals": {},
            "elapsed_s": 0.1,
        }

    monkeypatch.setattr(SB, "_run_backtest_inner", fake_inner)
    out = SB.run_backtest(max_symbols=10)
    assert out["universe"]["run"] == 10
    assert out["universe"]["truncated"] is True


def test_resolve_full_scope(monkeypatch):
    from product import full_universe_backtest as FUB

    monkeypatch.setattr("data.bhavcopy_store.store_symbols", lambda: ["AAA", "BBB", "CCC"])
    uni = FUB.resolve_backtest_universe("full")
    assert uni["count"] == 3
    assert uni["symbols"] == ["AAA", "BBB", "CCC"]


def test_full_universe_backtest_wrapper(monkeypatch):
    from product import full_universe_backtest as FUB

    monkeypatch.setattr(FUB, "resolve_backtest_universe", lambda scope="full": {
        "scope": "full",
        "source": "bhavcopy_store",
        "available_in_store": 3,
        "symbols": ["AAA", "BBB", "CCC"],
        "count": 3,
    })

    def fake_run_backtest(**kwargs):
        assert kwargs.get("max_symbols") is None
        assert kwargs.get("symbols") == ["AAA", "BBB", "CCC"]
        return {
            "generated_at": "t",
            "symbols": 3,
            "signals": {"MOMENTUM": {"trades": 1}},
            "universe": {"run": 3, "truncated": False},
            "elapsed_s": 1.2,
        }

    monkeypatch.setattr("scan.signal_backtest.run_backtest", fake_run_backtest)
    report = FUB.run_full_universe_backtest(scope="full")
    assert report["ok"] is True
    assert report["places_orders"] is False
    assert report["live_locked"] is True
    assert report["universe"]["requested"] == 3


def test_control_enqueues_full_universe_backtest(monkeypatch):
    import terminal_api as api
    from operations.market_ops import FULL_UNIVERSE_BACKTEST, LANES

    calls = {}

    class FakeStore:
        def enqueue(self, kind, lane=None, requested_by=None, message=None):
            calls["kind"] = kind
            calls["lane"] = lane
            calls["requested_by"] = requested_by
            calls["message"] = message
            return {"operation_id": "op-1", "status": "PENDING"}, True

    monkeypatch.setattr(
        api,
        "_ensure_ops_worker",
        lambda **_k: {"running": True, "ensure_ok": True, "worker_pid": 1},
    )
    monkeypatch.setattr("operations.store.OperationStore", lambda *a, **k: FakeStore())

    client = TestClient(api.app)
    r = client.post("/api/controls/RUN_FULL_UNIVERSE_BACKTEST_NOW")
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True
    assert calls["kind"] == FULL_UNIVERSE_BACKTEST
    assert calls["lane"] == LANES[FULL_UNIVERSE_BACKTEST]
