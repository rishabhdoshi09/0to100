"""Point-in-time feature builder for the NSE F&O directional lane."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise ValueError(f"missing required column: {name}")
    return pd.to_numeric(df[name], errors="coerce").astype(float)


def _last_finite(series: pd.Series, default: float = 0.0) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.iloc[-1]) if not clean.empty else default


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    if avg_loss.iloc[-1] == 0 and avg_gain.iloc[-1] > 0:
        return 100.0
    return _last_finite(rsi, 50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _atr(high, low, close, period)
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return _last_finite(adx, 0.0)


def build_fo_features(
    *,
    symbol: str,
    daily_bars: pd.DataFrame,
    direction: str,
    intraday_vwap: float,
    benchmark_return_pct: float,
    sector_relative_strength_pct: float,
    futures_price_change_pct: float | None,
    futures_oi_change_pct: float | None,
    is_fo: bool,
    underlying_spread_bps: float = 0.0,
    breakout_lookback: int = 20,
) -> dict[str, Any]:
    """Build only from data available through the current bar.

    daily_bars must already be point-in-time sliced; this function never
    reaches beyond the supplied frame.
    """
    if daily_bars is None or len(daily_bars) < max(60, breakout_lookback + 2):
        raise ValueError("at least 60 point-in-time daily bars are required")

    direction = str(direction or "").upper()
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction must be LONG or SHORT")

    close = _series(daily_bars, "close")
    high = _series(daily_bars, "high")
    low = _series(daily_bars, "low")
    volume = _series(daily_bars, "volume")
    if close.isna().iloc[-1] or close.iloc[-1] <= 0:
        raise ValueError("latest close is invalid")

    price = float(close.iloc[-1])
    ema20 = _last_finite(close.ewm(span=20, adjust=False).mean())
    ema50 = _last_finite(close.ewm(span=50, adjust=False).mean())
    rsi = _rsi(close)
    adx = _adx(high, low, close)
    atr_series = _atr(high, low, close)
    atr = _last_finite(atr_series)
    atr_pct = atr / price * 100.0 if price > 0 and atr > 0 else 0.0

    prior_vol = volume.iloc[-21:-1]
    avg_vol20 = float(prior_vol.mean()) if len(prior_vol) else 0.0
    rvol = float(volume.iloc[-1] / avg_vol20) if avg_vol20 > 0 else 0.0

    prior_high = float(high.iloc[-(breakout_lookback + 1):-1].max())
    prior_low = float(low.iloc[-(breakout_lookback + 1):-1].min())

    lookback = min(20, len(close) - 1)
    stock_return = (
        (price / float(close.iloc[-(lookback + 1)]) - 1.0) * 100.0
        if lookback > 0 and close.iloc[-(lookback + 1)] > 0
        else 0.0
    )
    relative_strength = stock_return - float(benchmark_return_pct)

    turnover = (close * volume).iloc[-20:]
    avg_turnover_crore = float(turnover.mean() / 1e7) if len(turnover) else 0.0

    trigger = prior_high if direction == "LONG" else prior_low
    directional_clearance = (
        price - trigger if direction == "LONG" else trigger - price
    )
    chase_distance_atr = max(0.0, directional_clearance) / atr if atr > 0 else 0.0

    latest_range = max(float(high.iloc[-1] - low.iloc[-1]), 1e-9)
    close_location = (
        (price - float(low.iloc[-1])) / latest_range
        if direction == "LONG"
        else (float(high.iloc[-1]) - price) / latest_range
    )
    crossed = directional_clearance > 0
    false_breakout = bool(crossed and (rvol < 1.2 or close_location < 0.50))

    return {
        "symbol": str(symbol or "").upper(),
        "is_fo": bool(is_fo),
        "price": round(price, 4),
        "breakout_level": round(prior_high, 4),
        "breakdown_level": round(prior_low, 4),
        "vwap": float(intraday_vwap),
        "ema20": round(ema20, 4),
        "ema50": round(ema50, 4),
        "rsi": round(rsi, 3),
        "adx": round(adx, 3),
        "rvol": round(rvol, 4),
        "relative_strength_pct": round(relative_strength, 4),
        "sector_strength_pct": round(float(sector_relative_strength_pct), 4),
        "nifty_change_pct": round(float(benchmark_return_pct), 4),
        "futures_price_change_pct": (
            round(float(futures_price_change_pct), 4)
            if futures_price_change_pct is not None else None
        ),
        "futures_oi_change_pct": (
            round(float(futures_oi_change_pct), 4)
            if futures_oi_change_pct is not None else None
        ),
        "atr_pct": round(atr_pct, 4),
        "avg_turnover_crore": round(avg_turnover_crore, 4),
        "underlying_spread_bps": round(float(underlying_spread_bps), 4),
        "chase_distance_atr": round(chase_distance_atr, 4),
        "false_breakout": false_breakout,
        "close_location": round(close_location, 4),
        "feature_provenance": {
            "daily": "POINT_IN_TIME_SUPPLIED_BARS",
            "vwap": "INTRADAY_REQUIRED",
            "futures_oi": "POINT_IN_TIME_REQUIRED",
        },
    }
