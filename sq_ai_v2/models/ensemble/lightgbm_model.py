"""
LightGBM directional model.
Predicts P(close[t+1] > close[t]) — a probability in [0, 1].
Supports incremental training (warm_start on new data).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit

from config.settings import settings


class LightGBMModel:
    """
    Binary classifier: up (1) vs down/flat (0) next-bar direction.
    """

    _PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42,
    }

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._booster: Optional[lgb.Booster] = None
        self._feature_names: List[str] = []
        self._model_path = model_path or (settings.model_dir / "lgbm.pkl")
        if self._model_path.exists():
            self.load(self._model_path)

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        num_rounds: int = 500,
    ) -> Dict:
        self._feature_names = list(X.columns)

        dtrain = lgb.Dataset(X.values, label=y.values, feature_name=self._feature_names)
        evals_result: Dict = {}

        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
            lgb.record_evaluation(evals_result),
        ]

        if eval_set is not None:
            X_val, y_val = eval_set
            dval = lgb.Dataset(X_val.values, label=y_val.values, reference=dtrain)
            valid_sets = [dtrain, dval]
            valid_names = ["train", "val"]
        else:
            valid_sets = [dtrain]
            valid_names = ["train"]

        self._booster = lgb.train(
            params=self._PARAMS,
            train_set=dtrain,
            num_boost_round=num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        best_iter = self._booster.best_iteration
        train_loss = evals_result.get("train", {}).get("binary_logloss", [np.nan])[-1]
        val_loss = evals_result.get("val", {}).get("binary_logloss", [np.nan])[-1]
        logger.info(
            f"LightGBM trained: best_iter={best_iter}, "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )
        return {"best_iteration": best_iter, "train_loss": train_loss, "val_loss": val_loss}

    def incremental_fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        init_model: Optional[lgb.Booster] = None,
        num_rounds: int = 50,
    ) -> None:
        """Warm-start on new data — suitable for daily retraining."""
        dtrain = lgb.Dataset(X.values, label=y.values, feature_name=list(X.columns))
        self._booster = lgb.train(
            params=self._PARAMS,
            train_set=dtrain,
            num_boost_round=num_rounds,
            init_model=init_model or self._booster,
            keep_training_booster=True,
            callbacks=[lgb.log_evaluation(-1)],
        )

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(up) for each row."""
        if self._booster is None:
            raise RuntimeError("Model not trained; call fit() first")
        X_aligned = self._align_features(X)
        return self._booster.predict(X_aligned.values)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Feature importance ────────────────────────────────────────────────

    def feature_importance(self) -> pd.Series:
        if self._booster is None:
            return pd.Series(dtype=float)
        imp = self._booster.feature_importance(importance_type="gain")
        names = self._booster.feature_name()
        return pd.Series(imp, index=names).sort_values(ascending=False)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"booster": self._booster, "feature_names": self._feature_names}, f)
        logger.info(f"LightGBM saved → {path}")

    def load(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self._booster = obj["booster"]
            self._feature_names = obj["feature_names"]
            logger.info(f"LightGBM loaded ← {path}")
        except Exception as exc:
            logger.warning(f"LightGBM load failed: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._feature_names:
            return X
        missing = set(self._feature_names) - set(X.columns)
        if missing:
            for col in missing:
                X = X.copy()
                X[col] = 0.0
        return X[self._feature_names]

    # ── Label creation ────────────────────────────────────────────────────

    @staticmethod
    def make_labels(close: pd.Series, forward_bars: int = 1) -> pd.Series:
        """Binary: 1 if close[t+k] > close[t], else 0."""
        return (close.shift(-forward_bars) > close).astype(int)
