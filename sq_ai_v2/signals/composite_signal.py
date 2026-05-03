"""
Composite signal — the brain of the trading system.

Signal architecture:
  ─── Technical/ML ensemble (60%) ───────────────────────────────────────────
    • LightGBM   }
    • CNN        } → MetaLearner → calibrated P(up)  (base 60%)
    • LSTM       }
    • GNN        }
  ─── Regime overlay (10%) ──────────────────────────────────────────────────
    • HMM regime multiplier applied to base signal
  ─── Sentiment (20%) ────────────────────────────────────────────────────────
    • FinBERT news sentiment
  ─── Fundamentals (10%) ─────────────────────────────────────────────────────
    • Earnings/revenue/macro surprise

Final signal = calibrated blended probability ∈ [0, 1].
Action thresholds: BUY if > 0.6, SELL if < 0.4, HOLD otherwise.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from data.feature_store.store import FeatureStore
from models.calibration import CalibrationWrapper
from models.ensemble.cnn_model import CNNWrapper
from models.ensemble.gnn_model import GNNWrapper
from models.ensemble.hmm_regime import HMMRegime
from models.ensemble.lightgbm_model import LightGBMModel
from models.ensemble.lstm_model import LSTMWrapper
from models.meta_learner import MetaLearner
from signals.feature_generation import FeatureGenerator
from signals.sentiment import SentimentAnalyser
from signals.fundamentals import FundamentalsSignal


@dataclass
class SignalResult:
    symbol: str
    action: str               # "BUY" | "SELL" | "HOLD"
    probability: float        # calibrated P(up) ∈ [0, 1]
    confidence: float         # abs distance from 0.5, scaled to [0, 1]
    base_prob: float          # ML ensemble raw probability
    sentiment_score: float
    fundamental_score: float
    regime: str               # "bull" | "chop" | "bear"
    regime_id: int
    component_probs: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


class CompositeSignal:
    """
    Orchestrates all sub-models and produces a final trading signal.
    Models are lazy-loaded on first use.
    """

    # Signal blending weights
    _W_ML = 0.60
    _W_SENTIMENT = 0.20
    _W_FUNDAMENTALS = 0.10
    _W_REGIME = 0.10

    # Decision thresholds
    _BUY_THRESHOLD = 0.60
    _SELL_THRESHOLD = 0.40

    def __init__(self) -> None:
        self._feature_gen = FeatureGenerator()
        self._sentiment = SentimentAnalyser()
        self._fundamentals = FundamentalsSignal()

        # Lazy-loaded models
        self._lgbm: Optional[LightGBMModel] = None
        self._cnn: Optional[CNNWrapper] = None
        self._lstm: Optional[LSTMWrapper] = None
        self._gnn: Optional[GNNWrapper] = None
        self._hmm: Optional[HMMRegime] = None
        self._meta: Optional[MetaLearner] = None
        self._calibrator: Optional[CalibrationWrapper] = None
        self._models_loaded = False

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_models(self) -> None:
        if self._models_loaded:
            return
        d = settings.model_dir
        try:
            self._lgbm = LightGBMModel(d / "lgbm.pkl")
        except Exception as exc:
            logger.warning(f"LightGBM load: {exc}")

        try:
            self._cnn = CNNWrapper(model_path=d / "cnn.pt")
        except Exception as exc:
            logger.warning(f"CNN load: {exc}")

        try:
            self._lstm = LSTMWrapper(model_path=d / "lstm.pt")
        except Exception as exc:
            logger.warning(f"LSTM load: {exc}")

        try:
            self._gnn = GNNWrapper(model_path=d / "gnn.pt")
        except Exception as exc:
            logger.warning(f"GNN load: {exc}")

        try:
            self._hmm = HMMRegime(d / "hmm.pkl")
        except Exception as exc:
            logger.warning(f"HMM load: {exc}")

        try:
            self._meta = MetaLearner(d / "meta_learner.pkl")
        except Exception as exc:
            logger.warning(f"MetaLearner load: {exc}")

        try:
            self._calibrator = CalibrationWrapper(model_path=d / "calibrator.pkl")
        except Exception as exc:
            logger.warning(f"Calibrator load: {exc}")

        self._models_loaded = True
        logger.info("CompositeSignal: all models loaded")

    # ── Main signal generation ─────────────────────────────────────────────

    def generate(
        self,
        symbol: str,
        df: pd.DataFrame,
        use_sentiment: bool = True,
        use_fundamentals: bool = True,
        gnn_feature_dict: Optional[Dict[str, pd.Series]] = None,
        gnn_return_history: Optional[Dict[str, np.ndarray]] = None,
    ) -> SignalResult:
        """
        Generate a composite trading signal for *symbol*.

        df: OHLCV DataFrame up to and including the current bar (no future data).
        """
        self._load_models()

        # ── 1. Feature engineering ────────────────────────────────────────
        feat_latest = self._feature_gen.compute_latest(symbol, df, normalise=True)
        if feat_latest is None:
            logger.warning(f"{symbol}: insufficient data for features")
            return self._hold(symbol, reason="insufficient_data")

        feat_df_norm = self._feature_gen.compute_normalised(symbol, df)
        feat_df_norm = feat_df_norm.replace([np.inf, -np.inf], np.nan).fillna(0)

        # ── 2. Base model probabilities ───────────────────────────────────
        component_probs: Dict[str, float] = {}

        lgbm_prob = self._get_lgbm_prob(feat_latest)
        component_probs["lgbm"] = lgbm_prob

        cnn_prob = self._get_cnn_prob(feat_df_norm)
        component_probs["cnn"] = cnn_prob

        lstm_prob = self._get_lstm_prob(feat_df_norm)
        component_probs["lstm"] = lstm_prob

        if gnn_feature_dict and gnn_return_history and self._gnn is not None:
            gnn_preds = self._gnn.predict_proba(gnn_feature_dict, gnn_return_history)
            gnn_prob = gnn_preds.get(symbol, 0.5)
        else:
            gnn_prob = 0.5
        component_probs["gnn"] = gnn_prob

        # ── 3. Meta-learner blend ─────────────────────────────────────────
        if self._meta is not None:
            base_prob = self._meta.predict_proba(component_probs)
        else:
            base_prob = float(np.mean(list(component_probs.values())))

        # ── 4. Calibration ────────────────────────────────────────────────
        if self._calibrator is not None:
            base_prob = self._calibrator.transform_scalar(base_prob)

        # ── 5. Regime overlay ─────────────────────────────────────────────
        regime_id, regime_probs, regime_name = (1, np.array([0, 1, 0]), "chop")
        if self._hmm is not None:
            regime_id, regime_probs, regime_name = self._hmm.predict(df)
        regime_signal = HMMRegime.regime_signal_multiplier(regime_id)
        # Regime adjusts base prob toward/away from 0.5
        regime_prob = 0.5 + (base_prob - 0.5) * regime_signal

        # ── 6. Sentiment ──────────────────────────────────────────────────
        sentiment_score = 0.5
        if use_sentiment:
            try:
                sentiment_score = self._sentiment.get_sentiment_signal(symbol)
            except Exception as exc:
                logger.debug(f"Sentiment failed: {exc}")

        # ── 7. Fundamentals ───────────────────────────────────────────────
        fundamental_score = 0.5
        if use_fundamentals:
            try:
                fundamental_score = self._fundamentals.get_combined_signal(symbol)
            except Exception as exc:
                logger.debug(f"Fundamentals failed: {exc}")

        # ── 8. Final blend ────────────────────────────────────────────────
        # Regime already baked into regime_prob; use it as the ML component
        ml_weight = self._W_ML + self._W_REGIME
        final_prob = (
            ml_weight * regime_prob
            + self._W_SENTIMENT * sentiment_score
            + self._W_FUNDAMENTALS * fundamental_score
        )
        final_prob = float(np.clip(final_prob, 0.0, 1.0))

        # ── 9. Action decision ────────────────────────────────────────────
        if final_prob >= self._BUY_THRESHOLD:
            action = "BUY"
        elif final_prob <= self._SELL_THRESHOLD:
            action = "SELL"
        else:
            action = "HOLD"

        confidence = abs(final_prob - 0.5) * 2  # 0 at boundary, 1 at extremes

        logger.info(
            f"Signal: {symbol} {action} prob={final_prob:.3f} "
            f"regime={regime_name} sentiment={sentiment_score:.2f}"
        )

        return SignalResult(
            symbol=symbol,
            action=action,
            probability=final_prob,
            confidence=confidence,
            base_prob=base_prob,
            sentiment_score=sentiment_score,
            fundamental_score=fundamental_score,
            regime=regime_name,
            regime_id=regime_id,
            component_probs=component_probs,
            metadata={
                "n_bars": len(df),
                "regime_probs": regime_probs.tolist(),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_lgbm_prob(self, feat_row: pd.Series) -> float:
        if self._lgbm is None:
            return 0.5
        try:
            return float(self._lgbm.predict_proba(feat_row.to_frame().T)[0])
        except Exception:
            return 0.5

    def _get_cnn_prob(self, feat_df: pd.DataFrame) -> float:
        if self._cnn is None:
            return 0.5
        try:
            probs = self._cnn.predict_proba(feat_df)
            return float(probs[-1]) if len(probs) > 0 else 0.5
        except Exception:
            return 0.5

    def _get_lstm_prob(self, feat_df: pd.DataFrame) -> float:
        if self._lstm is None:
            return 0.5
        try:
            probs = self._lstm.predict_proba(feat_df)
            return float(probs[-1]) if len(probs) > 0 else 0.5
        except Exception:
            return 0.5

    @staticmethod
    def _hold(symbol: str, reason: str = "") -> SignalResult:
        return SignalResult(
            symbol=symbol,
            action="HOLD",
            probability=0.5,
            confidence=0.0,
            base_prob=0.5,
            sentiment_score=0.5,
            fundamental_score=0.5,
            regime="chop",
            regime_id=1,
            metadata={"reason": reason},
        )
