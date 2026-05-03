"""
Meta-learner: combines base model probabilities into a final signal.

Training procedure:
  1. Collect out-of-sample (OOS) predictions from each base model.
  2. Stack into (T, n_models) matrix — the meta-features.
  3. Train a Logistic Regression on the OOS predictions vs actual labels.
  4. At inference, blend base probabilities through the learned weights.

Why Logistic Regression? It is:
  • Calibrated out-of-the-box (unlike tree ensembles).
  • Interpretable — weights reveal which models add value.
  • Fast to retrain daily.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config.settings import settings


_MODEL_NAMES = ["lgbm", "cnn", "lstm", "gnn"]


class MetaLearner:
    """
    Logistic Regression stacker over base model probability outputs.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model_path = model_path or (settings.model_dir / "meta_learner.pkl")
        self._lr: Optional[LogisticRegression] = None
        self._scaler = StandardScaler()
        self._model_names: List[str] = _MODEL_NAMES
        if self._model_path.exists():
            self.load()

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        oos_preds: Dict[str, np.ndarray],  # model_name → 1D array of OOS probabilities
        y: np.ndarray,                     # true binary labels, same length
    ) -> Dict:
        """
        oos_preds: out-of-sample predictions from each base model
        y: ground truth labels
        """
        self._model_names = list(oos_preds.keys())
        X = np.column_stack([oos_preds[name] for name in self._model_names])
        X = np.nan_to_num(X, nan=0.5)

        X_scaled = self._scaler.fit_transform(X)

        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, random_state=42)
        self._lr.fit(X_scaled, y)

        train_acc = self._lr.score(X_scaled, y)
        weights = dict(zip(self._model_names, self._lr.coef_[0]))
        logger.info(f"MetaLearner trained: accuracy={train_acc:.3f}, weights={weights}")
        return {"accuracy": train_acc, "weights": weights}

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_proba(self, base_probs: Dict[str, float]) -> float:
        """
        base_probs: {model_name: probability_scalar}
        Returns blended P(up) ∈ [0, 1].
        """
        if self._lr is None:
            # Fallback: simple average with preset weights
            return self._weighted_average(base_probs)

        try:
            x = np.array(
                [base_probs.get(name, 0.5) for name in self._model_names],
                dtype=np.float32,
            ).reshape(1, -1)
            x = np.nan_to_num(x, nan=0.5)
            x_scaled = self._scaler.transform(x)
            return float(self._lr.predict_proba(x_scaled)[0, 1])
        except Exception as exc:
            logger.warning(f"MetaLearner inference failed: {exc}")
            return self._weighted_average(base_probs)

    def predict_proba_batch(self, base_probs_df: pd.DataFrame) -> np.ndarray:
        """
        base_probs_df: DataFrame with one column per model, one row per bar.
        Returns array of P(up).
        """
        if self._lr is None:
            return base_probs_df.mean(axis=1).values

        try:
            X = base_probs_df[self._model_names].fillna(0.5).values
            X_scaled = self._scaler.transform(X)
            return self._lr.predict_proba(X_scaled)[:, 1]
        except Exception as exc:
            logger.warning(f"MetaLearner batch inference failed: {exc}")
            return base_probs_df.mean(axis=1).values

    # ── Fallback blending ─────────────────────────────────────────────────

    @staticmethod
    def _weighted_average(base_probs: Dict[str, float]) -> float:
        """Equal-weight average as fallback."""
        probs = [p for p in base_probs.values() if not np.isnan(p)]
        return float(np.mean(probs)) if probs else 0.5

    # ── Weights inspection ────────────────────────────────────────────────

    def get_weights(self) -> Dict[str, float]:
        if self._lr is None:
            return {}
        return dict(zip(self._model_names, self._lr.coef_[0]))

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"lr": self._lr, "scaler": self._scaler, "model_names": self._model_names},
                f,
            )
        logger.info(f"MetaLearner saved → {path}")

    def load(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self._lr = obj["lr"]
            self._scaler = obj["scaler"]
            self._model_names = obj.get("model_names", _MODEL_NAMES)
            logger.info(f"MetaLearner loaded ← {path}")
        except Exception as exc:
            logger.warning(f"MetaLearner load failed: {exc}")
