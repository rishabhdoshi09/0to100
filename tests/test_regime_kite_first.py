"""India regime fetch must be Kite-first; Yahoo only with QT_YAHOO_FALLBACK."""
from __future__ import annotations

import pandas as pd
import pytest


def _bars(n: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = pd.Series(range(100, 100 + n), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def test_fetch_ohlcv_prefers_kite(monkeypatch):
    import core.regime_engine as RE

    monkeypatch.delenv("QT_YAHOO_FALLBACK", raising=False)
    calls = {"kite": 0, "store": 0, "yahoo": 0}

    def kite(ticker, days=365):
        calls["kite"] += 1
        return _bars()

    def store(ticker):
        calls["store"] += 1
        raise AssertionError("index_store should not run when Kite hits")

    monkeypatch.setattr(RE, "_fetch_ohlcv_kite", kite)
    monkeypatch.setattr("data.index_store.get_index_ohlcv", store)
    out = RE._fetch_ohlcv("^NSEI")
    assert out is not None
    assert calls["kite"] == 1
    assert calls["store"] == 0
    assert RE._LAST_SOURCES["^NSEI"] == "kite"


def test_fetch_ohlcv_skips_yahoo_by_default(monkeypatch):
    import core.regime_engine as RE

    monkeypatch.delenv("QT_YAHOO_FALLBACK", raising=False)
    monkeypatch.setattr(RE, "_fetch_ohlcv_kite", lambda *a, **k: None)
    monkeypatch.setattr("data.index_store.get_index_ohlcv", lambda *a, **k: None)

    def boom(*a, **k):
        raise AssertionError("yfinance must stay off by default")

    monkeypatch.setitem(__import__("sys").modules, "yfinance", type("Y", (), {"download": staticmethod(boom)})())
    out = RE._fetch_ohlcv("^NSEI")
    assert out is None
    assert RE._LAST_SOURCES["^NSEI"] == "unavailable"


def test_fetch_ohlcv_yahoo_opt_in(monkeypatch):
    import core.regime_engine as RE

    monkeypatch.setenv("QT_YAHOO_FALLBACK", "1")
    monkeypatch.setattr(RE, "_fetch_ohlcv_kite", lambda *a, **k: None)
    monkeypatch.setattr("data.index_store.get_index_ohlcv", lambda *a, **k: None)

    class FakeYF:
        @staticmethod
        def download(*a, **k):
            return _bars(40)

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    out = RE._fetch_ohlcv("^NSEI")
    assert out is not None
    assert RE._LAST_SOURCES["^NSEI"] == "yfinance"


def test_compute_regime_does_not_invent_demo(monkeypatch):
    import core.regime_engine as RE

    RE._CACHE.clear()
    monkeypatch.setattr(RE, "_fetch_ohlcv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Kite/NSE"):
        RE.compute_regime(allow_network=True)


def test_breadth_unavailable_when_no_sector_frames():
    import core.regime_engine as RE

    score, label, _ = RE._compute_breadth({"IT": None, "BANK": None})
    assert score == 0
    assert label == "UNAVAILABLE"
