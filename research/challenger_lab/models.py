"""Transparent baseline models for challenger bake-offs (no deep learning)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    model_version: str
    algorithm: str

    def key(self) -> str:
        return f"{self.model_id}@{self.model_version}"


class NaiveBaseline:
    """Predict the majority training label (classification) or zero (regression)."""

    algorithm = "naive_baseline"

    def __init__(self):
        self._pred: float = 0.0
        self._is_classification = True

    def fit(self, X, y) -> "NaiveBaseline":
        y = np.asarray(y, float)
        if y.size == 0:
            self._pred = 0.0
            return self
        # Heuristic: classification if labels look discrete small set
        uniq = np.unique(y[~np.isnan(y)])
        self._is_classification = uniq.size <= 10 and np.all(np.isclose(uniq, np.round(uniq)))
        if self._is_classification:
            # majority class
            vals, counts = np.unique(y[~np.isnan(y)], return_counts=True)
            self._pred = float(vals[int(np.argmax(counts))])
        else:
            self._pred = 0.0
        return self

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else np.asarray(X).shape[0]
        return np.full(n, self._pred, dtype=float)


class LogisticChallenger:
    """sklearn logistic regression — small transparent baseline challenger."""

    algorithm = "logistic_regression"

    def __init__(self, *, max_iter: int = 200, random_state: int = 42):
        self.max_iter = max_iter
        self.random_state = random_state
        self._clf = None
        self._classes = None

    def fit(self, X, y) -> "LogisticChallenger":
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, float)
        y = np.asarray(y, float)
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]
        # Drop HOLD/zero if binary-ish with {-1,0,1} — keep all classes present.
        self._clf = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._clf.fit(X, y)
        self._classes = self._clf.classes_
        return self

    def predict(self, X) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("LogisticChallenger not fit")
        return np.asarray(self._clf.predict(np.asarray(X, float)), float)


def fit_predict_oos(model, X_train, y_train, X_oos) -> np.ndarray:
    """Fit on train only, predict OOS — shared path so incumbent/challenger match."""
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_oos), float)


def identity_for(model: Any, *, model_id: str, model_version: str = "1") -> ModelIdentity:
    algo = getattr(model, "algorithm", model.__class__.__name__)
    return ModelIdentity(model_id=model_id, model_version=model_version, algorithm=str(algo))
