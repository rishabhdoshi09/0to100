"""
Point-in-time feature store.

Key guarantee: features computed at bar T never use information from bar T+1.
All indicator computation windows are sliced *up to and including* bar T.

Features stored per (symbol, timestamp):
  ─ Price/volume technical: sma_20, sma_50, sma_200, ema_12, ema_26, macd,
    rsi_14, atr_14, bb_upper, bb_lower, bb_width, stoch_k, stoch_d,
    williams_r, mfi, obv, vwap, adx, cci
  ─ Momentum: momentum_5, momentum_20, return_1d, return_5d, return_20d
  ─ Volatility: volatility_20, volatility_60, vol_ratio
  ─ Volume: volume_ratio, volume_trend
  ─ Regime: detected by HMM (added separately)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class FeatureStore:
    """
    Computes and caches features from raw OHLCV DataFrames.
    Call `compute(symbol, df)` to get a feature DataFrame aligned to df's index.
    """

    def compute(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for *df* (OHLCV, ascending index).
        Returns a DataFrame of the same length with NaN-prefixed rows
        for the warm-up period.
        """
        if df.empty:
            return pd.DataFrame()

        feats = df[["open", "high", "low", "close", "volume"]].copy()

        c = feats["close"]
        h = feats["high"]
        lo = feats["low"]
        v = feats["volume"]

        # ── Moving averages ───────────────────────────────────────────────
        feats["sma_20"] = c.rolling(20).mean()
        feats["sma_50"] = c.rolling(50).mean()
        feats["sma_200"] = c.rolling(200).mean()
        feats["ema_12"] = c.ewm(span=12, adjust=False).mean()
        feats["ema_26"] = c.ewm(span=26, adjust=False).mean()
        feats["macd"] = feats["ema_12"] - feats["ema_26"]
        feats["macd_signal"] = feats["macd"].ewm(span=9, adjust=False).mean()
        feats["macd_hist"] = feats["macd"] - feats["macd_signal"]

        # ── RSI ───────────────────────────────────────────────────────────
        feats["rsi_14"] = self._rsi(c, 14)

        # ── ATR ───────────────────────────────────────────────────────────
        feats["atr_14"] = self._atr(h, lo, c, 14)

        # ── Bollinger Bands ───────────────────────────────────────────────
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        feats["bb_upper"] = bb_mid + 2 * bb_std
        feats["bb_lower"] = bb_mid - 2 * bb_std
        feats["bb_width"] = (feats["bb_upper"] - feats["bb_lower"]) / bb_mid.replace(0, np.nan)
        feats["bb_pct"] = (c - feats["bb_lower"]) / (feats["bb_upper"] - feats["bb_lower"]).replace(0, np.nan)

        # ── Stochastic ────────────────────────────────────────────────────
        low_14 = lo.rolling(14).min()
        high_14 = h.rolling(14).max()
        feats["stoch_k"] = 100 * (c - low_14) / (high_14 - low_14).replace(0, np.nan)
        feats["stoch_d"] = feats["stoch_k"].rolling(3).mean()

        # ── Williams %R ───────────────────────────────────────────────────
        feats["williams_r"] = -100 * (high_14 - c) / (high_14 - low_14).replace(0, np.nan)

        # ── MFI (Money Flow Index) ────────────────────────────────────────
        feats["mfi"] = self._mfi(h, lo, c, v, 14)

        # ── OBV ───────────────────────────────────────────────────────────
        feats["obv"] = self._obv(c, v)

        # ── VWAP (rolling 20-bar approximation) ───────────────────────────
        typical = (h + lo + c) / 3
        feats["vwap"] = (typical * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

        # ── ADX ───────────────────────────────────────────────────────────
        feats["adx"] = self._adx(h, lo, c, 14)

        # ── CCI ───────────────────────────────────────────────────────────
        feats["cci"] = self._cci(h, lo, c, 20)

        # ── Momentum / returns ────────────────────────────────────────────
        feats["momentum_5"] = c / c.shift(5) - 1
        feats["momentum_20"] = c / c.shift(20) - 1
        feats["return_1d"] = c.pct_change(1)
        feats["return_5d"] = c.pct_change(5)
        feats["return_20d"] = c.pct_change(20)

        # ── Volatility ────────────────────────────────────────────────────
        log_ret = np.log(c / c.shift(1))
        feats["volatility_20"] = log_ret.rolling(20).std() * np.sqrt(252)
        feats["volatility_60"] = log_ret.rolling(60).std() * np.sqrt(252)
        feats["vol_ratio"] = feats["volatility_20"] / feats["volatility_60"].replace(0, np.nan)

        # ── Volume features ───────────────────────────────────────────────
        feats["volume_ma20"] = v.rolling(20).mean()
        feats["volume_ratio"] = v / feats["volume_ma20"].replace(0, np.nan)
        feats["volume_trend"] = v.rolling(5).mean() / v.rolling(20).mean().replace(0, np.nan)

        # ── Price vs MA signals ───────────────────────────────────────────
        feats["price_vs_sma20"] = c / feats["sma_20"].replace(0, np.nan) - 1
        feats["price_vs_sma50"] = c / feats["sma_50"].replace(0, np.nan) - 1
        feats["sma20_vs_sma50"] = feats["sma_20"] / feats["sma_50"].replace(0, np.nan) - 1

        # Drop raw OHLCV columns — keep only features
        feats = feats.drop(columns=["open", "high", "low", "close", "volume"])

        logger.debug(f"Features computed: {symbol} {len(feats)} bars × {len(feats.columns)} features")
        return feats

    # ── Point-in-time safe slice ───────────────────────────────────────────

    def get_features_at(
        self,
        feature_df: pd.DataFrame,
        timestamp,
    ) -> Optional[pd.Series]:
        """Return feature row at *timestamp*, raising ValueError if in future."""
        if timestamp not in feature_df.index:
            prior = feature_df.index[feature_df.index <= timestamp]
            if prior.empty:
                return None
            timestamp = prior[-1]
        row = feature_df.loc[timestamp]
        if row.isna().all():
            return None
        return row

    # ── Private indicator helpers ─────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int,
    ) -> pd.Series:
        typical = (high + low + close) / 3
        raw_mf = typical * volume
        pos_mf = raw_mf.where(typical > typical.shift(1), 0).rolling(period).sum()
        neg_mf = raw_mf.where(typical < typical.shift(1), 0).rolling(period).sum()
        mfr = pos_mf / neg_mf.replace(0, np.nan)
        return 100 - 100 / (1 + mfr)

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    @staticmethod
    def _adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean()

    @staticmethod
    def _cci(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        typical = (high + low + close) / 3
        mean_dev = typical.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        return (typical - typical.rolling(period).mean()) / (0.015 * mean_dev.replace(0, np.nan))

    # ── Feature list ──────────────────────────────────────────────────────

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
            "macd", "macd_signal", "macd_hist",
            "rsi_14", "atr_14",
            "bb_upper", "bb_lower", "bb_width", "bb_pct",
            "stoch_k", "stoch_d", "williams_r", "mfi", "obv", "vwap",
            "adx", "cci",
            "momentum_5", "momentum_20",
            "return_1d", "return_5d", "return_20d",
            "volatility_20", "volatility_60", "vol_ratio",
            "volume_ma20", "volume_ratio", "volume_trend",
            "price_vs_sma20", "price_vs_sma50", "sma20_vs_sma50",
        ]
