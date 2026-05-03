"""Tests for the screener engine."""
import numpy as np
import pandas as pd

from sq_ai.backend.screener_engine import (
    _technical_features,
    matches,
    run_screener,
)


def _df(trend=0.001, n=300, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.012, n)
    close = 1000 * np.cumprod(1 + rets)
    return pd.DataFrame({
        "open": close * 1.001,
        "high": close * 1.005,
        "low":  close * 0.995,
        "close": close,
        "volume": rng.uniform(1e5, 5e5, n),
    }, index=pd.date_range("2023-01-01", periods=n, freq="B"))


def test_technical_features_keys():
    f = _technical_features(_df())
    for k in ("price", "sma_20", "sma_50", "rsi", "atr", "atr_pct",
              "vol_ratio", "macd_state", "high_52w", "ret_1m", "ret_3m"):
        assert k in f


def test_matches_rsi_range():
    feats = {"price": 100, "sma_20": 99, "sma_50": 98, "sma_200": 95,
             "rsi": 55, "atr_pct": 0.02, "vol_ratio": 1.2,
             "macd_state": "bullish", "from_52w_high_pct": -0.1,
             "ret_1w": 0.01, "ret_1m": 0.05, "ret_3m": 0.10}
    assert matches(feats, {"rsi": {"min": 50, "max": 60}})
    assert not matches(feats, {"rsi": {"min": 70, "max": 80}})


def test_matches_macd_and_volume():
    feats = {"price": 100, "sma_20": 99, "sma_50": 98, "sma_200": 95,
             "rsi": 50, "atr_pct": 0.02, "vol_ratio": 0.8,
             "macd_state": "bearish", "from_52w_high_pct": 0,
             "ret_1w": 0, "ret_1m": 0, "ret_3m": 0}
    assert not matches(feats, {"macd": "bullish"})
    assert not matches(feats, {"volume": "above_avg"})
    assert matches(feats, {"volume": "below_avg"})


def test_matches_price_vs_sma():
    feats = {"price": 110, "sma_20": 100, "sma_50": 95, "sma_200": 90,
             "rsi": 50, "atr_pct": 0.02, "vol_ratio": 1.0,
             "macd_state": "bullish", "from_52w_high_pct": 0,
             "ret_1w": 0, "ret_1m": 0, "ret_3m": 0}
    assert matches(feats, {"price_vs_sma20": "above"})
    assert not matches(feats, {"price_vs_sma20": "below"})


def test_run_screener_returns_top_n_with_score():
    df = _df()
    fetch = lambda s: df  # noqa: E731

    res = run_screener(["A.NS", "B.NS"], filters={}, fetch_fn=fetch,
                       max_results=10)
    assert len(res) == 2
    for r in res:
        assert "score" in r
        assert "symbol" in r


def test_run_screener_applies_filters():
    df = _df()
    fetch = lambda s: df  # noqa: E731
    # require RSI >= 99 → no rows pass
    res = run_screener(["A.NS"], filters={"rsi": {"min": 99}}, fetch_fn=fetch)
    assert res == []
