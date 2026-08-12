"""
Multi-horizon signal generator.

Trains three LightGBM models per symbol — 1d, 5d, 10d forward targets.
Consensus: if ≥ 2 of 3 horizons agree on BUY/SELL → that action.
Confidence = average across agreeing horizons.

Label thresholds:
  horizon_1d  : fwd_1d  > +0.25% → BUY, < -0.25% → SELL
  horizon_5d  : fwd_5d  > +0.50% → BUY, < -0.50% → SELL
  horizon_10d : fwd_10d > +1.00% → BUY, < -1.00% → SELL

Models saved to models/{symbol}_lgb_{horizon}.pkl
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

from config import settings
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

_HORIZONS = {
    "1d":  {"shift": 1,  "buy_thresh":  0.0025, "sell_thresh": -0.0025},
    "5d":  {"shift": 5,  "buy_thresh":  0.0050, "sell_thresh": -0.0050},
    "10d": {"shift": 10, "buy_thresh":  0.0100, "sell_thresh": -0.0100},
}


class _HorizonMeta:
    def __init__(self, clf, trained_at: datetime) -> None:
        self.clf = clf
        self.trained_at = trained_at


class MultiHorizonSignalGenerator:
    """
    Train and predict across 1d / 5d / 10d horizons using LightGBM.

    Usage
    -----
    gen = MultiHorizonSignalGenerator()
    result = gen.generate_signals(df, "RELIANCE")
    # result = {
    #   "horizon_1d":  {"action": "BUY",  "confidence": 0.72},
    #   "horizon_5d":  {"action": "BUY",  "confidence": 0.65},
    #   "horizon_10d": {"action": "HOLD", "confidence": 0.52},
    #   "consensus":   {"action": "BUY",  "confidence": 0.685, "agreement": 2},
    # }
    """

    def __init__(self) -> None:
        self._xgb = XGBoostSignalGenerator()
        # keyed by (symbol, horizon)
        self._models: Dict[Tuple[str, str], _HorizonMeta] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def generate_signals(
        self, df: pd.DataFrame, symbol: str
    ) -> Dict[str, Any]:
        if not _LGB_AVAILABLE:
            return self._all_hold(symbol, "lightgbm_not_installed")
        if df is None or df.empty or len(df) < 20:
            return self._all_hold(symbol, "insufficient_data")

        horizon_results: Dict[str, Dict] = {}

        for h_name, h_cfg in _HORIZONS.items():
            self._maybe_retrain(df, symbol, h_name, h_cfg)
            meta = self._models.get((symbol, h_name))
            if meta is None:
                horizon_results[f"horizon_{h_name}"] = {
                    "action": "HOLD", "confidence": 0.5, "reason": "no_model"
                }
                continue

            features = self._xgb._build_features(df, symbol)
            if features is None or features.empty:
                horizon_results[f"horizon_{h_name}"] = {
                    "action": "HOLD", "confidence": 0.5, "reason": "feature_failed"
                }
                continue

            last_row = features.iloc[[-1]][_FEATURE_COLS].fillna(0.0)
            try:
                proba = meta.clf.predict_proba(last_row.values)[0]
            except Exception as exc:
                log.warning("mh_predict_failed", symbol=symbol,
                            horizon=h_name, error=str(exc))
                horizon_results[f"horizon_{h_name}"] = {
                    "action": "HOLD", "confidence": 0.5, "reason": "predict_error"
                }
                continue

            predicted_cls = int(np.argmax(proba))
            confidence = float(np.max(proba))
            action = (
                "BUY"  if predicted_cls == _CLS_BUY else
                "SELL" if predicted_cls == _CLS_SELL else
                "HOLD"
            )
            horizon_results[f"horizon_{h_name}"] = {
                "action":     action,
                "confidence": round(confidence, 4),
                "sell_p":     round(float(proba[_CLS_SELL]), 4),
                "hold_p":     round(float(proba[_CLS_HOLD]), 4),
                "buy_p":      round(float(proba[_CLS_BUY]),  4),
            }

        consensus = self._compute_consensus(horizon_results)
        result = {**horizon_results, "consensus": consensus}
        log.info("multi_horizon_signal", symbol=symbol,
                 consensus_action=consensus["action"],
                 consensus_confidence=consensus["confidence"],
                 agreement=consensus["agreement"])
        return result

    # ── Consensus ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_consensus(horizon_results: Dict[str, Dict]) -> Dict[str, Any]:
        actions = [v["action"] for v in horizon_results.values()]
        buy_horizons  = [v for v in horizon_results.values() if v["action"] == "BUY"]
        sell_horizons = [v for v in horizon_results.values() if v["action"] == "SELL"]

        if len(buy_horizons) >= 2:
            action = "BUY"
            conf   = float(np.mean([v["confidence"] for v in buy_horizons]))
            agree  = len(buy_horizons)
        elif len(sell_horizons) >= 2:
            action = "SELL"
            conf   = float(np.mean([v["confidence"] for v in sell_horizons]))
            agree  = len(sell_horizons)
        else:
            action = "HOLD"
            conf   = float(np.mean([v["confidence"] for v in horizon_results.values()]))
            agree  = actions.count("HOLD")

        return {"action": action, "confidence": round(conf, 4), "agreement": agree}

    # ── Training ───────────────────────────────────────────────────────────

    def _maybe_retrain(
        self,
        df: pd.DataFrame,
        symbol: str,
        h_name: str,
        h_cfg: Dict,
    ) -> None:
        key = (symbol, h_name)
        meta = self._models.get(key)
        pkl_path = _MODELS_DIR / f"{symbol}_lgb_{h_name}.pkl"

        if meta is None and pkl_path.exists():
            try:
                meta = joblib.load(pkl_path)
                self._models[key] = meta
            except Exception as exc:
                log.warning("mh_load_failed", symbol=symbol,
                            horizon=h_name, error=str(exc))
                meta = None

        needs_train = meta is None
        if meta is not None:
            days_old = (datetime.now(timezone.utc) - meta.trained_at).days
            if days_old >= settings.lgbm_retrain_days:
                needs_train = True

        if needs_train:
            self._train(df, symbol, h_name, h_cfg)

    def _train(
        self,
        df: pd.DataFrame,
        symbol: str,
        h_name: str,
        h_cfg: Dict,
    ) -> None:
        if not _LGB_AVAILABLE:
            return

        train_window = df.iloc[-settings.lgbm_train_days:]
        features = self._xgb._build_features(train_window, symbol)
        if features is None:
            return

        close = train_window["close"].astype(float)
        shift = h_cfg["shift"]
        fwd = close.pct_change(shift).shift(-shift)

        labels = pd.Series(np.nan, index=train_window.index, dtype=float)
        labels[fwd >  h_cfg["buy_thresh"]]  = _LABEL_BUY
        labels[fwd <  h_cfg["sell_thresh"]] = _LABEL_SELL
        labels[(fwd >= h_cfg["sell_thresh"]) & (fwd <= h_cfg["buy_thresh"])] = _LABEL_HOLD
        labels[fwd.isna()] = np.nan

        combined = features.copy()
        combined["_label"] = labels
        combined = combined.dropna(subset=["_label"] + _FEATURE_COLS)
        combined = combined[combined["_label"] != _LABEL_HOLD]

        if len(combined) < _MIN_TRAINING_SAMPLES:
            log.warning("mh_insufficient_samples", symbol=symbol,
                        horizon=h_name, samples=len(combined))
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
            log.error("mh_train_failed", symbol=symbol,
                      horizon=h_name, error=str(exc))
            return

        key = (symbol, h_name)
        meta = _HorizonMeta(clf=clf, trained_at=datetime.now(timezone.utc))
        self._models[key] = meta

        pkl_path = _MODELS_DIR / f"{symbol}_lgb_{h_name}.pkl"
        try:
            joblib.dump(meta, pkl_path)
            log.info("mh_model_saved", symbol=symbol, horizon=h_name,
                     path=str(pkl_path), samples=len(X))
        except Exception as exc:
            log.warning("mh_save_failed", symbol=symbol,
                        horizon=h_name, error=str(exc))

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _all_hold(symbol: str, reason: str) -> Dict[str, Any]:
        hold = {"action": "HOLD", "confidence": 0.5, "reason": reason}
        return {
            "horizon_1d":  hold,
            "horizon_5d":  hold,
            "horizon_10d": hold,
            "consensus":   {"action": "HOLD", "confidence": 0.5,
                            "agreement": 0, "reason": reason},
        }
