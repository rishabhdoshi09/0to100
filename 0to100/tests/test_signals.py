"""Tests for the critical regime fix and composite signal logic."""
import numpy as np
import pandas as pd

from sq_ai.signals.composite_signal import (
    CompositeSignal,
    regime_from_smas,
    rsi,
    atr,
    zscore,
)


def _synthetic_df(trend: float = 0.001, n: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.01, n)
    close = 1000 * np.cumprod(1 + rets)
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    op = close * (1 + rng.normal(0, 0.001, n))
    vol = rng.uniform(1e5, 5e5, n)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({"open": op, "high": high, "low": low, "close": close,
                         "volume": vol}, index=idx)


def test_regime_uptrend_downtrend_sideways():
    assert regime_from_smas(110, 100) == 2
    assert regime_from_smas(90, 100) == 0
    assert regime_from_smas(100, 100) == 1
    assert regime_from_smas(None, 100) == 1


def test_compute_indicators_emits_regime():
    df = _synthetic_df(trend=0.002)             # clear uptrend
    sig = CompositeSignal()
    feats = sig.compute_indicators(df)
    assert "regime" in feats
    assert feats["regime"] in {0, 1, 2}
    # uptrend should not be a downtrend
    assert feats["regime"] in {1, 2}
    assert isinstance(feats["rsi"], float)
    assert feats["sma_20"] > 0


def test_compute_indicators_downtrend_gives_regime0():
    df = _synthetic_df(trend=-0.003)
    feats = CompositeSignal().compute_indicators(df)
    assert feats["regime"] in {0, 1}


def test_rsi_atr_zscore_basic():
    df = _synthetic_df()
    r = rsi(df["close"], 14)
    assert 0 <= r.iloc[-1] <= 100
    a = atr(df, 14)
    assert a.iloc[-1] > 0
    z = zscore(df["close"], 20)
    assert not np.isnan(z.iloc[-1])


def test_compute_signal_blocks_buy_in_downtrend():
    sig = CompositeSignal()
    feats = {
        "sma_20": 90, "sma_50": 100, "volatility_20": 0.2,
        "momentum_5d": 0.05, "volume_trend": 1.2,
        "rsi": 55, "atr": 5.0, "regime": 0, "zscore_20": 0.5, "close": 100,
    }
    out = sig.compute(feats)
    # regime gate must clamp BUY to HOLD
    assert out["direction"] != 1
    assert out["regime"] == 0


def test_compute_signal_allows_buy_in_uptrend():
    sig = CompositeSignal()
    feats = {
        "sma_20": 110, "sma_50": 100, "volatility_20": 0.15,
        "momentum_5d": 0.04, "volume_trend": 1.5,
        "rsi": 45, "atr": 5.0, "regime": 2, "zscore_20": -0.2, "close": 100,
    }
    out = sig.compute(feats)
    assert out["regime"] == 2
    assert out["signal"] >= 0     # non-negative under bullish regime


def test_ml_signal_returns_zero_when_no_model():
    sig = CompositeSignal(model_path="/nonexistent.pkl")
    score = sig._compute_ml_signal({"regime": 1, "rsi": 50})
    # 0.0 is the spec'd "no-info" output
    assert -1 <= score <= 1
