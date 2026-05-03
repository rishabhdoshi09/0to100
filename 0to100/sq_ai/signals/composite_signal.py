"""Composite signal – merges ML, regime, sentiment.

CRITICAL FIX: ``_compute_indicators`` now emits ``regime`` (0/1/2) and
``_compute_ml_signal`` consumes it from the feature vector instead of
silently defaulting to 0.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from sq_ai.signals.ml_model import MLModel


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function indicator helpers (used by both live signals and backtester)
# ─────────────────────────────────────────────────────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = -delta.clip(upper=0).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low_, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [(high - low_), (high - close.shift()).abs(), (low_ - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mu = series.rolling(period).mean()
    sd = series.rolling(period).std(ddof=0)
    return ((series - mu) / sd.replace(0, np.nan)).fillna(0.0)


def regime_from_smas(sma20: float | None, sma50: float | None) -> int:
    """0 = downtrend, 1 = sideways, 2 = uptrend."""
    if sma20 is None or sma50 is None or pd.isna(sma20) or pd.isna(sma50):
        return 1
    if sma20 > sma50 * 1.005:
        return 2
    if sma20 < sma50 * 0.995:
        return 0
    return 1


# ─────────────────────────────────────────────────────────────────────────────
class CompositeSignal:
    """Single source of truth for trading signals (live AND backtest)."""

    DEFAULT_FEATURES = [
        "sma_20", "sma_50", "volatility_20", "momentum_5d",
        "volume_trend", "rsi", "atr", "regime",
    ]

    def __init__(self, model_path: str | None = None,
                 feature_names_path: str | None = None) -> None:
        self.ml = MLModel(model_path=model_path, feature_names_path=feature_names_path)
        self._weights = {"ml": 0.55, "regime": 0.20, "factor": 0.15, "llm": 0.10}

    # ---------------------------------------------------------------- public
    def compute_indicators(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute the full feature dict for the LAST bar in ``df``.

        ``df`` must have columns: open, high, low, close, volume and at
        least 50 rows for SMAs to be valid.
        """
        if len(df) < 50:
            raise ValueError(f"need >=50 bars, got {len(df)}")

        close = df["close"]
        sma_20 = float(close.rolling(20).mean().iloc[-1])
        sma_50 = float(close.rolling(50).mean().iloc[-1])
        ret_1d = close.pct_change()
        vol_20 = float(ret_1d.rolling(20).std(ddof=0).iloc[-1] * np.sqrt(252))
        mom_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(df) >= 6 else 0.0
        vol_ratio = float(df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]) \
            if df["volume"].rolling(20).mean().iloc[-1] else 1.0
        rsi_14 = float(rsi(close, 14).iloc[-1])
        atr_14 = float(atr(df, 14).iloc[-1])
        regime = regime_from_smas(sma_20, sma_50)
        z_20 = float(zscore(close, 20).iloc[-1])

        return {
            "sma_20": sma_20,
            "sma_50": sma_50,
            "volatility_20": vol_20,
            "momentum_5d": mom_5d,
            "volume_trend": vol_ratio,
            "rsi": rsi_14,
            "atr": atr_14,
            "regime": regime,        # ← THE FIX
            "zscore_20": z_20,
            "close": float(close.iloc[-1]),
        }

    # alias kept for backwards compat with original repo
    _compute_indicators = compute_indicators

    def compute(self, features: dict, llm_signal: dict | None = None) -> dict:
        """Combine ML + regime + factors + (optional) LLM into a single signal."""
        ml_score = self._compute_ml_signal(features)
        regime_score = self._compute_regime_signal(features)
        factor_score = self._compute_factor_signal(features)
        llm_score = self._parse_llm(llm_signal)

        combined = (
            self._weights["ml"] * ml_score
            + self._weights["regime"] * regime_score
            + self._weights["factor"] * factor_score
            + self._weights["llm"] * llm_score
        )
        combined = float(np.clip(combined, -1.0, 1.0))

        regime = int(features.get("regime", 1))
        # regime gate: never BUY in a downtrend
        if regime == 0 and combined > 0:
            combined = 0.0

        if combined > 0.05:
            direction = 1
        elif combined < -0.05:
            direction = -1
        else:
            direction = 0

        return {
            "signal": combined,
            "direction": direction,
            "confidence": abs(combined) * 100.0,
            "regime": regime,
            "attribution": {
                "ml": ml_score,
                "regime": regime_score,
                "factor": factor_score,
                "llm": llm_score,
            },
        }

    # ------------------------------------------------------------- internals
    def _compute_ml_signal(self, features: dict) -> float:
        if not self.ml.loaded:
            return 0.0
        order = self.ml.feature_names or self.DEFAULT_FEATURES
        x = []
        for name in order:
            v = features.get(name)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                # sane fallbacks – regime defaults to 1 (sideways), not 0
                v = 1 if name == "regime" else 0.0
            x.append(float(v))
        try:
            proba = self.ml.predict_proba_up(x)
            return float(np.clip((proba - 0.5) * 2, -1.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_regime_signal(features: dict) -> float:
        regime = int(features.get("regime", 1))
        return {0: -0.7, 1: 0.0, 2: 0.7}[regime]

    @staticmethod
    def _compute_factor_signal(features: dict) -> float:
        score = 0.0
        n = 0
        rsi_v = features.get("rsi", 50.0)
        if rsi_v < 30:
            score += 0.5
            n += 1
        elif rsi_v > 70:
            score -= 0.5
            n += 1
        z = features.get("zscore_20", 0.0)
        if z < -1.5:
            score += 0.3
            n += 1
        elif z > 1.5:
            score -= 0.3
            n += 1
        mom = features.get("momentum_5d", 0.0)
        if mom > 0.02:
            score += 0.2
            n += 1
        elif mom < -0.02:
            score -= 0.2
            n += 1
        return score / n if n else 0.0

    @staticmethod
    def _parse_llm(llm: Any | None) -> float:
        if not llm:
            return 0.0
        if isinstance(llm, dict):
            sent = llm.get("sentiment_score", 0.0)
            return float(np.clip(sent, -1.0, 1.0))
        return 0.0


__all__ = ["CompositeSignal", "regime_from_smas", "rsi", "atr", "zscore"]
