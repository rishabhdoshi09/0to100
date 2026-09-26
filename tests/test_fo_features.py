import numpy as np
import pandas as pd

from product.fo_features import build_fo_features
from product.fo_setup import score_fo_setup


def _bars():
    n = 90
    x = np.arange(n, dtype=float)
    close = 100.0 + x * 0.18 + np.sin(x / 2.5) * 1.8
    high = close + 1.1
    low = close - 1.1
    volume = np.full(n, 1000.0)
    # A clean final breakout with strong volume and a close near the high.
    prior_high = float(high[-21:-1].max())
    close[-1] = prior_high + 1.2
    high[-1] = close[-1] + 0.25
    low[-1] = close[-1] - 1.5
    volume[-1] = 2600.0
    return pd.DataFrame({
        "open": close - 0.2,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_feature_builder_is_point_in_time_and_creates_required_fields():
    bars = _bars()
    features = build_fo_features(
        symbol="RELIANCE",
        daily_bars=bars,
        direction="LONG",
        intraday_vwap=float(bars["close"].iloc[-1] - 0.5),
        benchmark_return_pct=0.6,
        sector_relative_strength_pct=1.0,
        futures_price_change_pct=1.1,
        futures_oi_change_pct=6.0,
        is_fo=True,
        underlying_spread_bps=3.0,
    )
    assert features["symbol"] == "RELIANCE"
    assert features["breakout_level"] < features["price"]
    assert features["rvol"] > 2.0
    assert features["vwap"] > 0
    assert features["ema20"] > 0 and features["ema50"] > 0
    assert features["adx"] > 0
    assert features["atr_pct"] > 0
    assert features["feature_provenance"]["daily"] == "POINT_IN_TIME_SUPPLIED_BARS"


def test_missing_futures_oi_evidence_fails_closed():
    bars = _bars()
    features = build_fo_features(
        symbol="RELIANCE",
        daily_bars=bars,
        direction="LONG",
        intraday_vwap=float(bars["close"].iloc[-1] - 0.5),
        benchmark_return_pct=0.6,
        sector_relative_strength_pct=1.0,
        futures_price_change_pct=1.1,
        futures_oi_change_pct=None,
        is_fo=True,
    )
    result = score_fo_setup(features, "LONG")
    assert result["tradable"] is False
    assert "MISSING_FUTURES_OI_CHANGE_PCT" in result["blockers"]


def test_insufficient_history_is_refused_instead_of_filled():
    short = _bars().iloc[-30:]
    try:
        build_fo_features(
            symbol="RELIANCE",
            daily_bars=short,
            direction="LONG",
            intraday_vwap=100,
            benchmark_return_pct=0,
            sector_relative_strength_pct=0,
            futures_price_change_pct=0,
            futures_oi_change_pct=0,
            is_fo=True,
        )
    except ValueError as exc:
        assert "60" in str(exc)
    else:
        raise AssertionError("short history must fail closed")
