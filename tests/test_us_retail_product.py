"""US retail product plane — honest data, durable scan, no invented prices."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _frame(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [2_000_000.0] * n,
        },
        index=idx,
    )


def test_us_history_store_persists_and_loads(tmp_path: Path, monkeypatch):
    from data import us_history_store as hist

    monkeypatch.setattr(hist, "STORE_DIR", tmp_path / "us_bhav")
    monkeypatch.setattr(hist, "META_PATH", tmp_path / "us_bhav" / "meta.json")
    assert hist.save_symbol("AAPL", _frame())
    loaded = hist.load_symbol("AAPL")
    assert loaded is not None
    assert len(loaded) >= 60
    status = hist.status()
    assert status["symbols"] == 1
    assert status["source"] == "yfinance"


def test_us_history_prepare_uses_batch(monkeypatch, tmp_path: Path):
    from data import us_history_store as hist

    monkeypatch.setattr(hist, "STORE_DIR", tmp_path / "us_bhav")
    monkeypatch.setattr(hist, "META_PATH", tmp_path / "us_bhav" / "meta.json")
    monkeypatch.setattr(
        "data.us_data.get_us_daily_batch",
        lambda symbols, lookback_days=400: {s: _frame() for s in symbols},
    )
    events = []
    result = hist.prepare_history(
        ["AAPL", "MSFT"],
        progress=lambda c, t: events.append((c, t)),
        scope="S&P 500",
    )
    assert result["prepared_count"] == 2
    assert result["ready"] is False or result["symbols"] == 2  # ready needs >=40
    assert events


def test_us_market_scan_service_persists_payload(monkeypatch, tmp_path: Path):
    from scan import us_market_scan_service as USS
    from scan.unified_scanner import StockSignal

    monkeypatch.setattr(USS, "US_SCAN_PATH", tmp_path / "latest_us_scan.json")
    monkeypatch.setattr(
        USS,
        "_scope_universe",
        lambda scope: ({"AAA": "Alpha", "BBB": "Beta"}, "S&P 500 (test)"),
    )
    monkeypatch.setattr("data.us_data.sp500_return_30d", lambda: 1.2)
    monkeypatch.setattr(
        "data.us_data.get_us_daily_batch",
        lambda symbols, lookback_days=400: {s: _frame() for s in symbols},
    )
    monkeypatch.setattr("data.us_history_store.load_symbol", lambda _s: _frame())
    monkeypatch.setattr("data.us_history_store.save_symbol", lambda *_a, **_k: True)

    sig = StockSignal(
        symbol="AAA",
        price=150.0,
        change_pct=2.0,
        momentum_5d=5.0,
        rsi=55.0,
        volume_ratio=2.0,
        signals=["BREAKOUT", "MOMENTUM"],
        reasons=["volume confirmed breakout"],
        score=80.0,
        verdict="BUY",
        entry=151.0,
        stop=145.0,
        target=165.0,
        avg_vol20=3_000_000.0,
    )

    class FakeScanner:
        def __init__(self, max_workers=8):
            self._nifty_ret30 = 0.0
            self._max_workers = max_workers

        def _analyze(self, symbol, df):
            return sig if symbol == "AAA" else None

    monkeypatch.setattr("scan.unified_scanner.UnifiedScanner", FakeScanner)
    monkeypatch.setattr("execution.us_autopilot.on_setups", lambda *_a, **_k: None)
    monkeypatch.setattr("execution.us_autopilot.review_cycle", lambda: None)

    report = USS.run_us_market_scan(scope="S&P 500", save=True, use_disk_cache=True)
    assert report.ok
    assert report.payload["market"] == "US"
    assert report.payload["records"]
    assert all(r.get("fno_available") is False for r in report.payload["records"])
    loaded = USS.load_us_scan()
    assert loaded is not None
    assert loaded["scope"].startswith("S&P 500")


def test_us_retail_readiness_and_dashboard(monkeypatch):
    from product import us_retail

    monkeypatch.setattr(
        "data.us_history_store.status",
        lambda: {
            "ready": True,
            "symbols": 120,
            "latest_date": "2026-08-01",
            "source": "yfinance",
        },
    )
    monkeypatch.setattr(
        "data.us_universe.get_us_universe_with_names",
        lambda: {"AAPL": "Apple", "MSFT": "Microsoft"},
    )
    monkeypatch.setattr(
        us_retail,
        "load_us_scan",
        lambda: {
            "scanned_at": "2026-08-01T00:00:00+00:00",
            "universe_size": 2,
            "scope": "S&P 500",
            "summary": {"with_any_setup": 1},
            "records": [{"symbol": "AAPL", "verdict": "BUY", "status": "Ready to trade"}],
        },
    )
    monkeypatch.setattr(
        "data.us_data.us_live_prices",
        lambda symbols: {s: {"price": 100.0} for s in symbols},
    )
    monkeypatch.setattr(
        "execution.us_autopilot.get_status",
        lambda: {"armed": False, "open_positions": 0},
    )

    ready = us_retail.readiness()
    assert ready["market"] == "US"
    assert ready["places_orders"] is False
    assert ready["state"] in ("READY", "PARTIAL")
    dash = us_retail.dashboard()
    assert dash["scan"]["available"] is True
    assert dash["overview"]["currency"] == "USD"
    assert "Yahoo" in dash["honesty"] or "yfinance" in dash["honesty"].lower() or "Yahoo" in str(dash)


def test_us_stock_workspace_never_invents_fundamentals(monkeypatch):
    from product import us_retail

    monkeypatch.setattr(
        "data.us_universe.get_us_universe_with_names",
        lambda: {"AAPL": "Apple Inc"},
    )
    monkeypatch.setattr(
        "data.us_history_store.get_ohlcv",
        lambda symbol, allow_network=True: _frame(),
    )
    monkeypatch.setattr(
        us_retail,
        "scan_payload",
        lambda: {"records": [{"symbol": "AAPL", "verdict": "BUY", "price": 150}]},
    )
    ws = us_retail.stock_workspace("AAPL")
    assert ws["available"] is True
    assert ws["fundamentals"]["available"] is False
    assert ws["options"]["available"] is False
    assert ws["places_orders"] is False
    assert len(ws["bars"]) > 0


def test_us_api_routes_registered():
    import terminal_api as api

    paths = {getattr(route, "path", None) for route in api.app.routes}
    assert "/api/us/dashboard" in paths
    assert "/api/us/scan" in paths
    assert "/api/us/stock/{symbol}" in paths
    assert "/api/us/chart/{symbol}" in paths
    assert "RUN_US_SCAN_NOW" in api._OPERATION_CONTROLS
