"""Lightweight LightGBM inference wrapper.

Loads a pre-trained ``.pkl`` (trained on Colab, downloaded into ``models/``)
and exposes a single-row ``predict_proba_up`` method.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np


class MLModel:
    def __init__(self, model_path: str | None = None,
                 feature_names_path: str | None = None) -> None:
        self.model_path = model_path or os.environ.get(
            "SQ_MODEL_PATH", "./models/lgb_trading_model.pkl"
        )
        self.feature_names_path = feature_names_path or os.environ.get(
            "SQ_FEATURE_NAMES_PATH", "./models/feature_names.txt"
        )
        self._model = None
        self.feature_names: list[str] | None = None
        self._load()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if Path(self.model_path).exists():
            try:
                self._model = joblib.load(self.model_path)
                if hasattr(self._model, "feature_names_in_"):
                    self.feature_names = list(self._model.feature_names_in_)
                elif Path(self.feature_names_path).exists():
                    self.feature_names = Path(
                        self.feature_names_path
                    ).read_text().strip().splitlines()
            except Exception as exc:  # pragma: no cover
                print(f"[MLModel] load failed: {exc}")
                self._model = None

    def predict_proba_up(self, x: list[float]) -> float:
        """Return P(price-up) for a single feature row."""
        if self._model is None:
            return 0.5
        x_arr = np.asarray([x], dtype=float)
        if hasattr(self._model, "predict_proba"):
            return float(self._model.predict_proba(x_arr)[0][1])
        # raw lightgbm Booster
        return float(self._model.predict(x_arr)[0])
