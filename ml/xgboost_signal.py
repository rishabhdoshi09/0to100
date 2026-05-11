"""
XGBoost signal generator.

Trains a multi-class classifier (BUY / HOLD / SELL) from historical
OHLCV data and IndicatorEngine features.  Outputs a signal dict
compatible with TradingSignal in llm/signal_validator.py.

No lookahead:
  - Lagged returns use .shift(1) so today's bar never sees tomorrow's data.
  - Forward labels use .shift(-5) and are dropped before training.
  - The most recent 5 rows (NaN forward label rows) are always excluded.

Model persistence:
  - Models saved to models/{symbol}_xgb.pkl via joblib.
  - Retrained when model is absent OR last train was > xgboost_retrain_days ago.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config import settings
from features.indicators import IndicatorEngine
from logger import get_logger

log = get_logger(__name__)

_MODELS_DIR = Path("models")
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

_MIN_TRAINING_SAMPLES = 100
_LABEL_BUY = 1
_LABEL_SELL = -1
_LABEL_HOLD = 0

# Internal XGBoost numeric classes (XGB requires 0-indexed)
_CLS_SELL = 0   # maps to label -1
_CLS_HOLD = 1   # maps to label  0
_CLS_BUY  = 2   # maps to label +1

_FEATURE_COLS = [
    "rsi_14",
    "momentum_5d_pct",
    "momentum_10d_pct",
    "momentum_20d_pct",
    "zscore_20",
    "atr_pct",
    "volume_ratio",
    "pct_above_sma20",
    "pct_above_sma50",
    "vol_10d_ann",
    "vol_20d_ann",
    "trend_slope_5d",
    "golden_cross",   # bool → int
    # lagged returns added dynamically in _build_features
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
]


class _ModelMeta:
    """Lightweight wrapper around a trained model + metadata."""

    def __init__(self, clf: XGBClassifier, trained_at: datetime) -> None:
        self.clf = clf
        self.trained_at = trained_at


class XGBoostSignalGenerator:
    """
    Trains per-symbol XGBoost classifiers and generates TradingSignal-compatible
    output dicts.
    """

    def __init__(self) -> None:
        self._indicators = IndicatorEngine()
        self._models: Dict[str, _ModelMeta] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Generate a signal for *symbol* from its OHLCV DataFrame *df*.

        Returns a dict whose keys exactly match TradingSignal fields:
          symbol, action, confidence, time_horizon, position_size,
          reasoning, risk_level.
        """
        if df is None or df.empty or len(df) < 20:
            log.warning("xgb_insufficient_data", symbol=symbol, rows=len(df) if df is not None else 0)
            return self._hold_signal(symbol, 0.5, "insufficient_data")

        self._maybe_retrain(df, symbol)

        meta = self._models.get(symbol)
        if meta is None:
            return self._hold_signal(symbol, 0.5, "model_not_available")

        features = self._build_features(df, symbol)
        if features is None or features.empty:
            return self._hold_signal(symbol, 0.5, "feature_build_failed")

        # ── Inject live news sentiment into the inference row ─────────────────
        news_sentiment = 0.0
        try:
            from news.semantic_index import SemanticNewsIndex
            from news.vader_scorer import batch_score
            articles = SemanticNewsIndex().search(symbol, top_k=5)
            if articles:
                scored = batch_score(articles)
                news_sentiment = sum(a.get("vader_score", 0.0) for a in scored) / len(scored)
                features.iloc[-1, features.columns.get_loc("news_sentiment_5d")] = news_sentiment
        except Exception:
            pass

        # Use the last row (most recent bar) for prediction
        last_row = features.iloc[[-1]][_FEATURE_COLS].copy()
        last_row = last_row.fillna(0.0)

        try:
            proba = meta.clf.predict_proba(last_row)[0]
        except Exception as exc:
            log.warning("xgb_predict_failed", symbol=symbol, error=str(exc))
            return self._hold_signal(symbol, 0.5, f"predict_error:{exc}")

        # proba indices: [_CLS_SELL=0, _CLS_HOLD=1, _CLS_BUY=2]
        predicted_cls = int(np.argmax(proba))
        confidence = float(np.max(proba))

        if predicted_cls == _CLS_BUY:
            action = "BUY"
        elif predicted_cls == _CLS_SELL:
            action = "SELL"
        else:
            action = "HOLD"

        # ── News sentiment alignment — nudge confidence ±3% max ──────────────
        if news_sentiment != 0.0:
            aligned = (action == "BUY" and news_sentiment > 0.05) or \
                      (action == "SELL" and news_sentiment < -0.05)
            opposed = (action == "BUY" and news_sentiment < -0.05) or \
                      (action == "SELL" and news_sentiment > 0.05)
            if aligned:
                confidence = min(confidence * 1.03, 0.99)
            elif opposed:
                confidence = max(confidence * 0.97, 0.01)

        reasoning = (
            f"xgb: sell_p={proba[_CLS_SELL]:.3f}, "
            f"hold_p={proba[_CLS_HOLD]:.3f}, "
            f"buy_p={proba[_CLS_BUY]:.3f}, "
            f"news_vader={news_sentiment:+.3f}"
        )

        log.info(
            "xgb_signal",
            symbol=symbol,
            action=action,
            confidence=round(confidence, 3),
        )

        return {
            "symbol": symbol,
            "action": action,
            "confidence": round(confidence, 4),
            "time_horizon": "swing",
            "position_size": settings.max_position_size_pct,
            "reasoning": reasoning,
            "risk_level": "medium",
        }

    # ── Retraining logic ───────────────────────────────────────────────────

    def _maybe_retrain(self, df: pd.DataFrame, symbol: str) -> None:
        """Retrain if no model or model is stale."""
        meta = self._models.get(symbol)
        pkl_path = _MODELS_DIR / f"{symbol}_xgb.pkl"

        # Load from disk if not in memory
        if meta is None and pkl_path.exists():
            try:
                meta = joblib.load(pkl_path)
                self._models[symbol] = meta
                log.debug("xgb_model_loaded_from_disk", symbol=symbol)
            except Exception as exc:
                log.warning("xgb_model_load_failed", symbol=symbol, error=str(exc))
                meta = None

        # Check if retraining needed
        needs_train = meta is None
        if meta is not None:
            days_old = (datetime.now(timezone.utc) - meta.trained_at).days
            if days_old >= settings.xgboost_retrain_days:
                needs_train = True
                log.info("xgb_retrain_triggered", symbol=symbol, days_old=days_old)

        if needs_train:
            self._train(df, symbol)

    def _train(self, df: pd.DataFrame, symbol: str) -> None:
        """Build features, generate labels, train XGBClassifier, persist."""
        train_window = df.iloc[-settings.xgboost_train_days:]

        features = self._build_features(train_window, symbol)
        if features is None or features.empty:
            log.warning("xgb_train_feature_build_failed", symbol=symbol)
            return

        labels = self._build_labels(train_window)
        if labels is None:
            log.warning("xgb_train_label_build_failed", symbol=symbol)
            return

        # Align features and labels
        combined = features.copy()
        combined["_label"] = labels

        # Drop HOLD rows (label == 0) and NaN rows
        combined = combined.dropna(subset=["_label"] + _FEATURE_COLS)
        combined = combined[combined["_label"] != _LABEL_HOLD]

        if len(combined) < _MIN_TRAINING_SAMPLES:
            log.warning(
                "xgb_insufficient_training_samples",
                symbol=symbol,
                samples=len(combined),
                min_required=_MIN_TRAINING_SAMPLES,
            )
            return

        X = combined[_FEATURE_COLS].fillna(0.0).values
        # Map labels to 0-indexed classes for XGBoost
        y_raw = combined["_label"].values
        y = np.where(y_raw == _LABEL_SELL, _CLS_SELL,
              np.where(y_raw == _LABEL_BUY, _CLS_BUY, _CLS_HOLD))

        clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )

        try:
            clf.fit(X, y)
        except Exception as exc:
            log.error("xgb_train_failed", symbol=symbol, error=str(exc))
            return

        meta = _ModelMeta(clf=clf, trained_at=datetime.now(timezone.utc))
        self._models[symbol] = meta

        pkl_path = _MODELS_DIR / f"{symbol}_xgb.pkl"
        try:
            joblib.dump(meta, pkl_path)
            log.info("xgb_model_saved", symbol=symbol, path=str(pkl_path), samples=len(X))
        except Exception as exc:
            log.warning("xgb_model_save_failed", symbol=symbol, error=str(exc))

    # ── Feature engineering ────────────────────────────────────────────────

    def _build_features(
        self, df: pd.DataFrame, symbol: str
    ) -> Optional[pd.DataFrame]:
        """
        Compute IndicatorEngine features row-by-row and add lagged returns.
        Returns a DataFrame indexed like df with columns = _FEATURE_COLS.
        """
        if df is None or len(df) < 20:
            return None

        close = df["close"].astype(float)

        # Lagged returns (shift(1) avoids lookahead)
        ret_1d = close.pct_change(1).shift(1)
        ret_2d = close.pct_change(2).shift(1)
        ret_3d = close.pct_change(3).shift(1)
        ret_5d = close.pct_change(5).shift(1)

        # Compute all indicator columns across the full window
        rows = []
        for i in range(len(df)):
            slice_df = df.iloc[:i + 1]
            ind = self._indicators.compute(slice_df, symbol)
            golden = ind.get("golden_cross")
            rows.append({
                "rsi_14":           ind.get("rsi_14"),
                "momentum_5d_pct":  ind.get("momentum_5d_pct"),
                "momentum_10d_pct": ind.get("momentum_10d_pct"),
                "momentum_20d_pct": ind.get("momentum_20d_pct"),
                "zscore_20":        ind.get("zscore_20"),
                "atr_pct":          ind.get("atr_pct"),
                "volume_ratio":     ind.get("volume_ratio"),
                "pct_above_sma20":  ind.get("pct_above_sma20"),
                "pct_above_sma50":  ind.get("pct_above_sma50"),
                "vol_10d_ann":      ind.get("vol_10d_ann"),
                "vol_20d_ann":      ind.get("vol_20d_ann"),
                "trend_slope_5d":   ind.get("trend_slope_5d"),
                "golden_cross":     int(golden) if golden is not None else None,
            })

        feat_df = pd.DataFrame(rows, index=df.index)
        feat_df["return_1d"] = ret_1d.values
        feat_df["return_2d"] = ret_2d.values
        feat_df["return_3d"] = ret_3d.values
        feat_df["return_5d"] = ret_5d.values

        # ── News sentiment (18th feature — inference-time only) ───────────────
        # Default 0.0 for all historical rows (VADER uses live retrieval).
        # generate_signal() overwrites the last row before prediction.
        feat_df["news_sentiment_5d"] = 0.0

        return feat_df

    @staticmethod
    def _build_labels(df: pd.DataFrame) -> Optional[pd.Series]:
        """
        forward_return_5d = close.pct_change(5).shift(-5)
        label = 1  if > 0.5%
        label = -1 if < -0.5%
        label = 0  otherwise
        Drop last 5 rows (NaN forward return).
        """
        close = df["close"].astype(float)
        fwd = close.pct_change(5).shift(-5)

        label = pd.Series(index=df.index, dtype=float)
        label[fwd > 0.005] = _LABEL_BUY
        label[fwd < -0.005] = _LABEL_SELL
        label[(fwd >= -0.005) & (fwd <= 0.005)] = _LABEL_HOLD
        label[fwd.isna()] = np.nan

        return label

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _hold_signal(symbol: str, confidence: float, reason: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "action": "HOLD",
            "confidence": confidence,
            "time_horizon": "swing",
            "position_size": 0.0,
            "reasoning": reason,
            "risk_level": "high",
        }
