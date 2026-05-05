"""
Ensemble Signal Generator.

Combines XGBoost + LightGBM (+ optional Chronos-Bolt) into a single
weighted-vote signal. Output dict is compatible with TradingSignal fields.

Weights:
  XGBoost  : 0.4
  LightGBM : 0.4
  Chronos  : 0.2  (if available; weight redistributed to XGB/LGB when absent)

Consensus rule:
  - Final action = majority vote.
  - Confidence = weighted-average of per-model confidences.
  - If avg_confidence < 0.6 OR models disagree → HOLD.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

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
    _ModelMeta,
    _MODELS_DIR,
)

log = get_logger(__name__)

_XGB_WEIGHT = 0.4
_LGB_WEIGHT = 0.4
_CHRONOS_WEIGHT = 0.2
_CONFIDENCE_THRESHOLD = 0.6

# When Chronos absent, redistribute its weight equally
_XGB_WEIGHT_NO_CHRONOS = _XGB_WEIGHT + _CHRONOS_WEIGHT / 2
_LGB_WEIGHT_NO_CHRONOS = _LGB_WEIGHT + _CHRONOS_WEIGHT / 2

_LGB_AVAILABLE = False
try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    log.warning("lightgbm_not_installed", hint="pip install lightgbm")

_CHRONOS_AVAILABLE = False
# Chronos-Bolt requires HuggingFace transformers + torch; optional
try:
    from transformers import pipeline as hf_pipeline  # noqa: F401
    _CHRONOS_AVAILABLE = True
except ImportError:
    log.warning("chronos_not_available", hint="pip install transformers torch")


class _LGBMeta:
    def __init__(self, clf, trained_at: datetime) -> None:
        self.clf = clf
        self.trained_at = trained_at


class EnsembleSignalGenerator:
    """
    Multi-model ensemble signal generator. Thread-safe for single-process use.

    Usage
    -----
    gen = EnsembleSignalGenerator()
    signal = gen.generate_signal(df, "RELIANCE")
    """

    def __init__(self) -> None:
        self._indicators = IndicatorEngine()
        self._xgb = XGBoostSignalGenerator()
        self._lgb_models: Dict[str, _LGBMeta] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Generate ensemble signal from all available models.

        Returns a dict compatible with TradingSignal plus an extra
        ``ensemble_details`` key.
        """
        if df is None or df.empty or len(df) < 20:
            return self._hold(symbol, "insufficient_data")

        model_outputs: List[Tuple[str, str, float, float]] = []  # (name, action, confidence, weight)

        # ── XGBoost ────────────────────────────────────────────────────────
        try:
            xgb_sig = self._xgb.generate_signal(df, symbol)
            model_outputs.append(("xgboost", xgb_sig["action"], float(xgb_sig["confidence"]), _XGB_WEIGHT))
        except Exception as exc:
            log.warning("ensemble_xgb_failed", symbol=symbol, error=str(exc))

        # ── LightGBM ───────────────────────────────────────────────────────
        if _LGB_AVAILABLE:
            try:
                lgb_action, lgb_conf = self._lgb_signal(df, symbol)
                model_outputs.append(("lightgbm", lgb_action, lgb_conf, _LGB_WEIGHT))
            except Exception as exc:
                log.warning("ensemble_lgb_failed", symbol=symbol, error=str(exc))
        else:
            log.debug("ensemble_lgb_skipped", symbol=symbol)

        # ── Chronos-Bolt (optional) ────────────────────────────────────────
        if _CHRONOS_AVAILABLE:
            try:
                chron_action, chron_conf = self._chronos_signal(df, symbol)
                model_outputs.append(("chronos", chron_action, chron_conf, _CHRONOS_WEIGHT))
            except Exception as exc:
                log.warning("ensemble_chronos_failed", symbol=symbol, error=str(exc))

        if not model_outputs:
            return self._hold(symbol, "all_models_failed")

        # Redistribute weights if any model is absent
        total_weight = sum(w for _, _, _, w in model_outputs)
        if total_weight == 0:
            return self._hold(symbol, "zero_weight")

        # Weighted confidence per action
        buy_conf = sum(c * w for _, a, c, w in model_outputs if a == "BUY") / total_weight
        sell_conf = sum(c * w for _, a, c, w in model_outputs if a == "SELL") / total_weight
        hold_conf = sum(c * w for _, a, c, w in model_outputs if a == "HOLD") / total_weight
        weighted_conf = max(buy_conf, sell_conf, hold_conf)

        actions = [a for _, a, _, _ in model_outputs]
        buy_votes = actions.count("BUY")
        sell_votes = actions.count("SELL")
        n = len(actions)
        majority = n // 2 + 1

        if buy_votes >= majority:
            final_action = "BUY"
            final_conf = buy_conf
        elif sell_votes >= majority:
            final_action = "SELL"
            final_conf = sell_conf
        else:
            final_action = "HOLD"
            final_conf = hold_conf

        # Override with HOLD if confidence below threshold
        if weighted_conf < _CONFIDENCE_THRESHOLD:
            final_action = "HOLD"
            final_conf = weighted_conf

        ensemble_details = {
            m: {"action": a, "confidence": round(c, 4), "weight": w}
            for m, a, c, w in model_outputs
        }
        ensemble_details["weighted_conf"] = round(weighted_conf, 4)
        ensemble_details["buy_conf"] = round(buy_conf, 4)
        ensemble_details["sell_conf"] = round(sell_conf, 4)

        log.info(
            "ensemble_signal",
            symbol=symbol,
            action=final_action,
            confidence=round(final_conf, 3),
            models=len(model_outputs),
        )

        return {
            "symbol": symbol,
            "action": final_action,
            "confidence": round(float(final_conf), 4),
            "time_horizon": "swing",
            "position_size": settings.max_position_size_pct,
            "reasoning": f"ensemble({len(model_outputs)} models): {final_action} @ {round(final_conf*100, 1)}%",
            "risk_level": "medium",
            "ensemble_details": ensemble_details,
        }

    # ── LightGBM ───────────────────────────────────────────────────────────

    def _lgb_signal(self, df: pd.DataFrame, symbol: str) -> Tuple[str, float]:
        self._maybe_train_lgb(df, symbol)
        meta = self._lgb_models.get(symbol)
        if meta is None:
            return "HOLD", 0.5

        features = self._xgb._build_features(df, symbol)
        if features is None or features.empty:
            return "HOLD", 0.5

        last_row = features.iloc[[-1]][_FEATURE_COLS].fillna(0.0)

        try:
            proba = meta.clf.predict_proba(last_row.values)[0]
        except Exception as exc:
            log.warning("lgb_predict_failed", symbol=symbol, error=str(exc))
            return "HOLD", 0.5

        predicted_cls = int(np.argmax(proba))
        confidence = float(np.max(proba))

        if predicted_cls == _CLS_BUY:
            return "BUY", confidence
        elif predicted_cls == _CLS_SELL:
            return "SELL", confidence
        return "HOLD", confidence

    def _maybe_train_lgb(self, df: pd.DataFrame, symbol: str) -> None:
        from datetime import timezone as _tz
        meta = self._lgb_models.get(symbol)
        pkl_path = _MODELS_DIR / f"{symbol}_lgb.pkl"

        if meta is None and pkl_path.exists():
            try:
                meta = joblib.load(pkl_path)
                self._lgb_models[symbol] = meta
                log.debug("lgb_model_loaded", symbol=symbol)
            except Exception as exc:
                log.warning("lgb_model_load_failed", symbol=symbol, error=str(exc))
                meta = None

        needs_train = meta is None
        if meta is not None:
            days_old = (datetime.now(timezone.utc) - meta.trained_at).days
            if days_old >= settings.xgboost_retrain_days:
                needs_train = True

        if needs_train:
            self._train_lgb(df, symbol)

    def _train_lgb(self, df: pd.DataFrame, symbol: str) -> None:
        if not _LGB_AVAILABLE:
            return

        train_window = df.iloc[-settings.xgboost_train_days:]
        features = self._xgb._build_features(train_window, symbol)
        labels = self._xgb._build_labels(train_window)

        if features is None or labels is None:
            return

        combined = features.copy()
        combined["_label"] = labels
        combined = combined.dropna(subset=["_label"] + _FEATURE_COLS)
        combined = combined[combined["_label"] != _LABEL_HOLD]

        if len(combined) < _MIN_TRAINING_SAMPLES:
            log.warning("lgb_insufficient_samples", symbol=symbol, samples=len(combined))
            return

        X = combined[_FEATURE_COLS].fillna(0.0).values
        y_raw = combined["_label"].values
        y = np.where(y_raw == _LABEL_SELL, _CLS_SELL,
              np.where(y_raw == _LABEL_BUY, _CLS_BUY, _CLS_HOLD))

        import lightgbm as lgb
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
            log.error("lgb_train_failed", symbol=symbol, error=str(exc))
            return

        meta = _LGBMeta(clf=clf, trained_at=datetime.now(timezone.utc))
        self._lgb_models[symbol] = meta

        pkl_path = _MODELS_DIR / f"{symbol}_lgb.pkl"
        try:
            joblib.dump(meta, pkl_path)
            log.info("lgb_model_saved", symbol=symbol, path=str(pkl_path))
        except Exception as exc:
            log.warning("lgb_model_save_failed", symbol=symbol, error=str(exc))

    # ── Chronos-Bolt (optional) ────────────────────────────────────────────

    def _chronos_signal(self, df: pd.DataFrame, symbol: str) -> Tuple[str, float]:
        """
        Use Chronos-Bolt (zero-shot time-series forecasting) to predict
        5-day forward direction.  Falls back to HOLD if unavailable.
        """
        try:
            import torch
            from transformers import pipeline

            context_len = min(64, len(df))
            close = df["close"].astype(float).values[-context_len:]

            pipe = pipeline(
                "text-generation",
                model="amazon/chronos-bolt-tiny",
                trust_remote_code=True,
            )
            context_tensor = torch.tensor(close, dtype=torch.float32).unsqueeze(0)
            forecast = pipe(context_tensor, prediction_length=5)

            median_forecast = np.median(forecast[0], axis=0)
            fwd_return = (median_forecast[-1] - close[-1]) / close[-1]

            if fwd_return > 0.005:
                return "BUY", min(0.9, 0.6 + abs(fwd_return) * 5)
            elif fwd_return < -0.005:
                return "SELL", min(0.9, 0.6 + abs(fwd_return) * 5)
            return "HOLD", 0.5

        except Exception as exc:
            log.warning("chronos_signal_failed", symbol=symbol, error=str(exc))
            return "HOLD", 0.5

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _hold(symbol: str, reason: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "action": "HOLD",
            "confidence": 0.5,
            "time_horizon": "swing",
            "position_size": 0.0,
            "reasoning": reason,
            "risk_level": "high",
            "ensemble_details": {},
        }
