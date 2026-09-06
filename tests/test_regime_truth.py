"""Missing index data must not become a bullish demo regime."""
from __future__ import annotations

from product.market_view import build_market_view
from product.recommendations_workspace import build_market_reports_workspace


def test_compute_regime_is_unavailable_when_nifty_series_missing(monkeypatch):
    import core.regime_engine as re

    re._CACHE.clear()
    monkeypatch.setattr(re, "_fetch_ohlcv", lambda *_a, **_k: None)
    state = re.compute_regime()
    assert state.data_available is False
    assert state.data_source == "unavailable"
    assert state.market_regime == "UNAVAILABLE"
    assert state.quality_multiplier == 1.0
    assert state.nifty_price == 0.0
    assert state.leading_sectors == []
    assert "demo" not in (state.unavailable_reason or "").lower()
    view = build_market_view(state)
    assert view.available is False
    assert view.health == "Unavailable"
    assert "demo" not in view.summary.lower()
    assert "do not infer" in view.trade_stance.lower()


def test_market_view_explicit_unavailable_is_not_healthy():
    view = build_market_view({
        "data_available": False,
        "data_source": "unavailable",
        "unavailable_reason": "Nifty series missing",
        "market_regime": "UNAVAILABLE",
        "regime_score": 76,
        "risk_mode": "NORMAL",
        "breakout_environment": "FAVORABLE",
        "breadth_label": "STRONG",
        "breadth_strength": 72,
        "leading_sectors": ["BANK"],
        "nifty_price": 24387.5,
        "nifty_change_1d": 0.68,
    })
    assert view.available is False
    assert view.health == "Unavailable"
    assert view.leaders == ()
    assert view.nifty_price == 0.0


def test_market_payload_marks_cached_unavailable_regime(monkeypatch):
    import terminal_api

    view = build_market_view({
        "data_available": False,
        "data_source": "unavailable",
        "unavailable_reason": "Nifty series missing",
        "market_regime": "UNAVAILABLE",
    })
    monkeypatch.setattr(terminal_api, "_warm_regime", lambda: None)
    monkeypatch.setattr("product.market_view.peek_cached_market_view", lambda: view)
    payload = terminal_api._market_payload()
    assert payload["available"] is False
    assert payload["health"] == "Unavailable"
    assert payload["leaders"] == []
    assert payload["nifty_price"] is None


def test_report_workspace_does_not_invent_regime_on_read(tmp_path, monkeypatch):
    import product.recommendations_workspace as rw

    monkeypatch.setattr(rw, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr("product.market_view.peek_cached_market_view", lambda: None)
    payload = build_market_reports_workspace(
        persist_today=False,
        rebuild=False,
        scan_payload={"records": []},
        news_payload={"available": False, "articles": []},
    )
    assert "regime" in payload["missing_lanes"]
    assert not (payload.get("market") or {}).get("available")


def test_quality_engine_rs_does_not_use_demo_regime(monkeypatch):
    import numpy as np

    from scan.quality_engine import QualityEngine

    monkeypatch.setattr("data.index_store.recent_index_closes", lambda *_a, **_k: [])

    class _Boom:
        def history(self, **_k):
            raise RuntimeError("yahoo blocked")

    monkeypatch.setattr("yfinance.Ticker", lambda *_a, **_k: _Boom())
    engine = QualityEngine()
    close = np.array([100.0 + i for i in range(30)], dtype=float)
    assert engine._rs_vs_nifty("HDFCBANK", close) is None
