"""
LightGBM signal generator — primary ML model.

Identical feature set and label definition as XGBoostSignalGenerator.
Trained per-symbol, persisted to models/{symbol}_lgb.pkl.

Label: forward 5d return > 0.5% → BUY, < -0.5% → SELL, else HOLD.
Retrain trigger: no model OR model age > LGBM_RETRAIN_DAYS days.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

from config import settings
from features.indicators import IndicatorEngine
from logger import get_logger
from ml.xgboost_signal import (
    XGBoostSignalGenerator,
    _FEATURE_COLS,
    _MIN_TRAINING_SAMPLES,
    _CLS_BUY,
    _CLS_SELL,
    _CLS_HOLD,
    _LABEL_BUY,
    _LABEL_SELL,
    _LABEL_HOLD,
    _MODELS_DIR,
)

log = get_logger(__name__)


class _LGBMeta:
    def __init__(self, clf, trained_at: datetime) -> None:
        self.clf = clf
        self.trained_at = trained_at


class LightGBMSignalGenerator:
    """
    Standalone LightGBM classifier — same features + labels as XGBoost.

    Usage
    -----
    gen = LightGBMSignalGenerator()
    signal = gen.generate_signal(df, "RELIANCE")
    """

    def __init__(self) -> None:
        self._indicators = IndicatorEngine()
        self._xgb = XGBoostSignalGenerator()   # reuse feature/label builders
        self._models: Dict[str, _LGBMeta] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if not _LGB_AVAILABLE:
            return self._hold(symbol, "lightgbm_not_installed")

        if df is None or df.empty or len(df) < 20:
            return self._hold(symbol, "insufficient_data")

        self._maybe_retrain(df, symbol)

        meta = self._models.get(symbol)
        if meta is None:
            return self._hold(symbol, "model_unavailable")

        features = self._xgb._build_features(df, symbol)
        if features is None or features.empty:
            return self._hold(symbol, "feature_build_failed")

        last_row = features.iloc[[-1]][_FEATURE_COLS].fillna(0.0)
        try:
            proba = meta.clf.predict_proba(last_row.values)[0]
        except Exception as exc:
            log.warning("lgbm_predict_failed", symbol=symbol, error=str(exc))
            return self._hold(symbol, f"predict_error:{exc}")

        predicted_cls = int(np.argmax(proba))
        confidence = float(np.max(proba))

        if predicted_cls == _CLS_BUY:
            action = "BUY"
        elif predicted_cls == _CLS_SELL:
            action = "SELL"
        else:
            action = "HOLD"

        reasoning = (
            f"lgbm: sell_p={proba[_CLS_SELL]:.3f}, "
            f"hold_p={proba[_CLS_HOLD]:.3f}, "
            f"buy_p={proba[_CLS_BUY]:.3f}"
        )
        log.info("lgbm_signal", symbol=symbol, action=action,
                 confidence=round(confidence, 3))

        return {
            "symbol":        symbol,
            "action":        action,
            "confidence":    round(confidence, 4),
            "time_horizon":  "swing",
            "position_size": settings.max_position_size_pct,
            "reasoning":     reasoning,
            "risk_level":    "medium",
        }

    # ── Retraining ─────────────────────────────────────────────────────────

    def _maybe_retrain(self, df: pd.DataFrame, symbol: str) -> None:
        meta = self._models.get(symbol)
        pkl_path = _MODELS_DIR / f"{symbol}_lgb.pkl"

        if meta is None and pkl_path.exists():
            try:
                meta = joblib.load(pkl_path)
                self._models[symbol] = meta
                log.debug("lgbm_model_loaded", symbol=symbol)
            except Exception as exc:
                log.warning("lgbm_load_failed", symbol=symbol, error=str(exc))
                meta = None

        needs_train = meta is None
        if meta is not None:
            days_old = (datetime.now(timezone.utc) - meta.trained_at).days
            if days_old >= settings.lgbm_retrain_days:
                needs_train = True
                log.info("lgbm_retrain_triggered", symbol=symbol, days_old=days_old)

        if needs_train:
            self._train(df, symbol)

    def _train(self, df: pd.DataFrame, symbol: str) -> None:
        if not _LGB_AVAILABLE:
            return

        train_window = df.iloc[-settings.lgbm_train_days:]
        features = self._xgb._build_features(train_window, symbol)
        labels = self._xgb._build_labels(train_window)

        if features is None or labels is None:
            return

        combined = features.copy()
        combined["_label"] = labels
        combined = combined.dropna(subset=["_label"] + _FEATURE_COLS)
        combined = combined[combined["_label"] != _LABEL_HOLD]

        if len(combined) < _MIN_TRAINING_SAMPLES:
            log.warning("lgbm_insufficient_samples", symbol=symbol,
                        samples=len(combined))
            return

        X = combined[_FEATURE_COLS].fillna(0.0).values
        y_raw = combined["_label"].values
        y = np.where(y_raw == _LABEL_SELL, _CLS_SELL,
              np.where(y_raw == _LABEL_BUY, _CLS_BUY, _CLS_HOLD))

        clf = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        try:
            clf.fit(X, y)
        except Exception as exc:
            log.error("lgbm_train_failed", symbol=symbol, error=str(exc))
            return

        meta = _LGBMeta(clf=clf, trained_at=datetime.now(timezone.utc))
        self._models[symbol] = meta

        pkl_path = _MODELS_DIR / f"{symbol}_lgb.pkl"
        try:
            joblib.dump(meta, pkl_path)
            log.info("lgbm_model_saved", symbol=symbol, path=str(pkl_path),
                     samples=len(X))
        except Exception as exc:
            log.warning("lgbm_save_failed", symbol=symbol, error=str(exc))

    @staticmethod
    def _hold(symbol: str, reason: str) -> Dict[str, Any]:
        return {
            "symbol":        symbol,
            "action":        "HOLD",
            "confidence":    0.5,
            "time_horizon":  "swing",
            "position_size": 0.0,
            "reasoning":     reason,
            "risk_level":    "high",
        }
