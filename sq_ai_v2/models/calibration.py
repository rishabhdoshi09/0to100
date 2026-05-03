"""
Probabilistic calibration.

A model that outputs 0.8 should be right ~80% of the time.
We apply Platt scaling (logistic) or Isotonic regression post-training
on a held-out validation fold.

Usage:
    cal = CalibrationWrapper(method="isotonic")
    cal.fit(raw_probs_val, y_val)
    calibrated = cal.transform(raw_probs_test)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from config.settings import settings


CalibMethod = Literal["isotonic", "platt", "none"]


class CalibrationWrapper:
    """
    Wraps Platt scaling (logistic) or Isotonic regression to calibrate
    raw model probabilities.
    """

    def __init__(
        self,
        method: CalibMethod = "isotonic",
        model_path: Optional[Path] = None,
    ) -> None:
        self.method = method
        self._model_path = model_path or (settings.model_dir / f"calibrator_{method}.pkl")
        self._calibrator = None
        if self._model_path.exists():
            self.load()

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        raw_probs: uncalibrated probabilities, shape (N,)
        y_true:    binary labels, shape (N,)
        """
        raw_probs = np.clip(raw_probs, 1e-6, 1 - 1e-6)

        if self.method == "platt":
            # Logistic regression on logit(p)
            logits = np.log(raw_probs / (1 - raw_probs)).reshape(-1, 1)
            self._calibrator = LogisticRegression(solver="lbfgs")
            self._calibrator.fit(logits, y_true)

        elif self.method == "isotonic":
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw_probs, y_true)

        else:
            self._calibrator = None  # identity

        # Reliability diagnostics
        stats = self._reliability_stats(raw_probs, y_true)
        logger.info(
            f"Calibration ({self.method}): "
            f"pre_ece={stats['pre_ece']:.4f}, "
            f"post_ece={stats['post_ece']:.4f}"
        )
        return stats

    # ── Transform ─────────────────────────────────────────────────────────

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        raw_probs = np.clip(np.asarray(raw_probs), 1e-6, 1 - 1e-6)
        if self._calibrator is None:
            return raw_probs

        if self.method == "platt":
            logits = np.log(raw_probs / (1 - raw_probs)).reshape(-1, 1)
            return self._calibrator.predict_proba(logits)[:, 1]

        elif self.method == "isotonic":
            return self._calibrator.predict(raw_probs)

        return raw_probs

    def transform_scalar(self, p: float) -> float:
        return float(self.transform(np.array([p]))[0])

    # ── Reliability ───────────────────────────────────────────────────────

    def _reliability_stats(
        self, raw_probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> dict:
        prob_true, prob_pred = calibration_curve(y_true, raw_probs, n_bins=n_bins)
        pre_ece = float(np.mean(np.abs(prob_true - prob_pred)))

        if self._calibrator is not None:
            cal_probs = self.transform(raw_probs)
            prob_true2, prob_pred2 = calibration_curve(y_true, cal_probs, n_bins=n_bins)
            post_ece = float(np.mean(np.abs(prob_true2 - prob_pred2)))
        else:
            post_ece = pre_ece

        return {"pre_ece": pre_ece, "post_ece": post_ece}

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"calibrator": self._calibrator, "method": self.method}, f)
        logger.info(f"Calibrator ({self.method}) saved → {path}")

    def load(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self._calibrator = obj["calibrator"]
            self.method = obj["method"]
            logger.info(f"Calibrator ({self.method}) loaded ← {path}")
        except Exception as exc:
            logger.warning(f"Calibrator load failed: {exc}")


# Type hint fix (Dict used without import in fit return type docstring)
from typing import Dict  # noqa: E402
