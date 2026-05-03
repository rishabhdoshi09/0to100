"""
Technical indicator engine.

All indicators computed here are PURE functions of historical price data.
No future-looking calculations. No side effects.
Returns a snapshot dict that is injected into the LLM context.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)


class IndicatorEngine:
    """Compute a rich set of indicators from an OHLCV DataFrame."""

    def compute(self, df: pd.DataFrame, symbol: str = "") -> Dict[str, Any]:
        """
        Compute all indicators.

        df must have columns: open, high, low, close, volume
        and a DatetimeIndex.
        Returns a flat dict of indicator name → scalar value.
        None means not enough data.
        """
        if df is None or df.empty or len(df) < 2:
            log.warning("insufficient_data_for_indicators", symbol=symbol)
            return {"error": "insufficient_data"}

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        result: Dict[str, Any] = {
            "symbol": symbol,
            "last_close": self._scalar(close.iloc[-1]),
            "last_volume": self._scalar(volume.iloc[-1]),
        }

        result.update(self._moving_averages(close))
        result.update(self._zscore(close))
        result.update(self._volatility(close))
        result.update(self._momentum(close))
        result.update(self._rsi(close))
        result.update(self._atr(high, low, close))
        result.update(self._volume_analysis(close, volume))
        result.update(self._trend(close))
        result.update(self._regime(close))
        result.update(self._ml_feature_aliases(result))

        return result

    # ── Moving Averages ────────────────────────────────────────────────────

    def _moving_averages(self, close: pd.Series) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for period in (5, 10, 20, 50, 200):
            if len(close) >= period:
                out[f"sma_{period}"] = self._scalar(close.rolling(period).mean().iloc[-1])
                out[f"ema_{period}"] = self._scalar(close.ewm(span=period, adjust=False).mean().iloc[-1])
            else:
                out[f"sma_{period}"] = None
                out[f"ema_{period}"] = None

        # Golden / death cross signal
        sma50 = out.get("sma_50")
        sma200 = out.get("sma_200")
        if sma50 and sma200:
            out["golden_cross"] = sma50 > sma200
        else:
            out["golden_cross"] = None

        # Price vs key MAs
        last = close.iloc[-1]
        if out.get("sma_20"):
            out["pct_above_sma20"] = round((last / out["sma_20"] - 1) * 100, 3)
        if out.get("sma_50"):
            out["pct_above_sma50"] = round((last / out["sma_50"] - 1) * 100, 3)

        return out

    # ── Z-Score Mean Reversion ─────────────────────────────────────────────

    def _zscore(self, close: pd.Series, window: int = 20) -> Dict[str, Any]:
        if len(close) < window:
            return {"zscore_20": None}
        roll = close.rolling(window)
        mean = roll.mean()
        std = roll.std()
        zscore = (close - mean) / std
        return {"zscore_20": self._scalar(zscore.iloc[-1])}

    # ── Volatility ────────────────────────────────────────────────────────

    def _volatility(self, close: pd.Series) -> Dict[str, Any]:
        if len(close) < 2:
            return {}
        log_ret = np.log(close / close.shift(1)).dropna()
        out: Dict[str, Any] = {}
        for period in (10, 20):
            if len(log_ret) >= period:
                vol = log_ret.rolling(period).std().iloc[-1]
                out[f"vol_{period}d_ann"] = self._scalar(vol * np.sqrt(252))
        return out

    # ── Momentum ──────────────────────────────────────────────────────────

    def _momentum(self, close: pd.Series) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for period in (5, 10, 20):
            if len(close) > period:
                ret = (close.iloc[-1] / close.iloc[-1 - period] - 1) * 100
                out[f"momentum_{period}d_pct"] = self._scalar(ret)
        return out

    # ── RSI ───────────────────────────────────────────────────────────────

    def _rsi(self, close: pd.Series, period: int = 14) -> Dict[str, Any]:
        if len(close) <= period:
            return {"rsi_14": None}
        delta = close.diff().dropna()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        val = self._scalar(rsi.iloc[-1])
        overbought = val > 70 if val is not None else None
        oversold = val < 30 if val is not None else None
        return {"rsi_14": val, "rsi_overbought": overbought, "rsi_oversold": oversold}

    # ── ATR ───────────────────────────────────────────────────────────────

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, Any]:
        if len(close) <= period:
            return {"atr_14": None}
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        # ATR as % of price
        atr_pct = (atr / close.iloc[-1]) * 100
        return {"atr_14": self._scalar(atr), "atr_pct": self._scalar(atr_pct)}

    # ── Volume Analysis ───────────────────────────────────────────────────

    def _volume_analysis(self, close: pd.Series, volume: pd.Series) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if len(volume) >= 20:
            avg_vol = volume.rolling(20).mean().iloc[-1]
            out["avg_volume_20d"] = self._scalar(avg_vol)
            out["volume_ratio"] = self._scalar(volume.iloc[-1] / avg_vol) if avg_vol else None
        return out

    # ── Trend State ───────────────────────────────────────────────────────

    def _trend(self, close: pd.Series) -> Dict[str, Any]:
        if len(close) < 5:
            return {"trend_5d": None}
        slope = float(np.polyfit(range(5), close.iloc[-5:].values, 1)[0])
        trend = "up" if slope > 0 else "down"
        return {"trend_5d": trend, "trend_slope_5d": self._scalar(slope)}

    # ── Regime ────────────────────────────────────────────────────────────

    def _regime(self, close: pd.Series) -> Dict[str, Any]:
        """
        Numeric market regime:
          2 = bull  (SMA20 > SMA50 > SMA200, or SMA20 > SMA50 with SMA50 rising)
          1 = neutral
          0 = bear  (SMA20 < SMA50 and SMA50 declining)
        Also returns regime_signal: +1 (bull), 0 (neutral), -1 (bear) for
        backwards compat with any code that still reads regime_signal.
        """
        if len(close) < 50:
            return {"regime": 1, "regime_signal": 0}

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        # SMA50 trend over last 5 bars
        sma50_5ago = close.rolling(50).mean().iloc[-6] if len(close) >= 55 else sma50

        if sma20 > sma50 and sma50 >= sma50_5ago:
            regime = 2
            regime_signal = 1
        elif sma20 < sma50 and sma50 <= sma50_5ago:
            regime = 0
            regime_signal = -1
        else:
            regime = 1
            regime_signal = 0

        return {"regime": regime, "regime_signal": self._scalar(regime_signal)}

    # ── ML feature name aliases ────────────────────────────────────────────

    @staticmethod
    def _ml_feature_aliases(ind: Dict[str, Any]) -> Dict[str, Any]:
        """
        The trained LightGBM model may expect different names than the indicator
        engine uses internally.  Return all known alias variants so that
        whichever name the model was trained with will be present.
        """
        return {
            # volatility aliases
            "volatility_20": ind.get("vol_20d_ann"),
            "volatility_10": ind.get("vol_10d_ann"),
            # momentum aliases
            "momentum_5d": ind.get("momentum_5d_pct"),
            "momentum_10d": ind.get("momentum_10d_pct"),
            "momentum_20d": ind.get("momentum_20d_pct"),
            # RSI alias (some models trained with bare 'rsi')
            "rsi": ind.get("rsi_14"),
            # ATR alias
            "atr": ind.get("atr_14"),
            # volume trend: 5d avg volume / 20d avg volume
            "volume_trend": (
                (ind.get("volume_ratio") or 1.0)  # volume_ratio = last_vol / avg20
            ),
        }

    # ── Helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _scalar(val) -> Optional[float]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), 5)
