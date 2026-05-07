"""
Ensemble Signal Generator — XGBoost (0.5) + LightGBM (0.5).

Majority vote; confidence = weighted average.
Falls back to HOLD when weighted confidence < 0.60.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config import settings
from logger import get_logger
from ml.xgboost_signal import XGBoostSignalGenerator
from ml.lgbm_signal import LightGBMSignalGenerator

log = get_logger(__name__)

_XGB_WEIGHT = 0.5
_LGB_WEIGHT = 0.5
_CONFIDENCE_THRESHOLD = 0.6


class EnsembleSignalGenerator:
    """
    Multi-model ensemble: XGBoost (0.5) + LightGBM (0.5).

    Usage
    -----
    gen = EnsembleSignalGenerator()
    signal = gen.generate_signal(df, "RELIANCE")
    """

    def __init__(self) -> None:
        self._xgb = XGBoostSignalGenerator()
        self._lgb = LightGBMSignalGenerator()

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if df is None or df.empty or len(df) < 20:
            return self._hold(symbol, "insufficient_data")

        model_outputs: List[Tuple[str, str, float, float]] = []

        try:
            xgb_sig = self._xgb.generate_signal(df, symbol)
            model_outputs.append(("xgboost", xgb_sig["action"],
                                   float(xgb_sig["confidence"]), _XGB_WEIGHT))
        except Exception as exc:
            log.warning("ensemble_xgb_failed", symbol=symbol, error=str(exc))

        try:
            lgb_sig = self._lgb.generate_signal(df, symbol)
            model_outputs.append(("lightgbm", lgb_sig["action"],
                                   float(lgb_sig["confidence"]), _LGB_WEIGHT))
        except Exception as exc:
            log.warning("ensemble_lgb_failed", symbol=symbol, error=str(exc))

        if not model_outputs:
            return self._hold(symbol, "all_models_failed")

        total_weight = sum(w for _, _, _, w in model_outputs)
        if total_weight == 0:
            return self._hold(symbol, "zero_weight")

        buy_conf  = sum(c * w for _, a, c, w in model_outputs if a == "BUY")  / total_weight
        sell_conf = sum(c * w for _, a, c, w in model_outputs if a == "SELL") / total_weight
        hold_conf = sum(c * w for _, a, c, w in model_outputs if a == "HOLD") / total_weight
        weighted_conf = max(buy_conf, sell_conf, hold_conf)

        actions   = [a for _, a, _, _ in model_outputs]
        majority  = len(actions) // 2 + 1

        if actions.count("BUY") >= majority:
            final_action, final_conf = "BUY", buy_conf
        elif actions.count("SELL") >= majority:
            final_action, final_conf = "SELL", sell_conf
        else:
            final_action, final_conf = "HOLD", hold_conf

        if weighted_conf < _CONFIDENCE_THRESHOLD:
            final_action, final_conf = "HOLD", weighted_conf

        ensemble_details = {
            m: {"action": a, "confidence": round(c, 4), "weight": w}
            for m, a, c, w in model_outputs
        }
        ensemble_details.update({
            "weighted_conf": round(weighted_conf, 4),
            "buy_conf":      round(buy_conf,  4),
            "sell_conf":     round(sell_conf, 4),
        })

        log.info("ensemble_signal", symbol=symbol, action=final_action,
                 confidence=round(final_conf, 3), models=len(model_outputs))

        return {
            "symbol":           symbol,
            "action":           final_action,
            "confidence":       round(float(final_conf), 4),
            "time_horizon":     "swing",
            "position_size":    settings.max_position_size_pct,
            "reasoning":        (f"ensemble({len(model_outputs)} models): "
                                 f"{final_action} @ {round(final_conf*100,1)}%"),
            "risk_level":       "medium",
            "ensemble_details": ensemble_details,
        }

    @staticmethod
    def _hold(symbol: str, reason: str) -> Dict[str, Any]:
        return {
            "symbol":           symbol,
            "action":           "HOLD",
            "confidence":       0.5,
            "time_horizon":     "swing",
            "position_size":    0.0,
            "reasoning":        reason,
            "risk_level":       "high",
            "ensemble_details": {},
        }
