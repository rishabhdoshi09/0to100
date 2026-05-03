"""
Feature generation pipeline.

Wraps FeatureStore and adds:
  • Automated feature engineering via gplearn (genetic programming) — optional.
  • Population Stability Index (PSI) for drift detection.
  • Normalised feature matrix ready for model inference.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from config.settings import settings
from data.feature_store.store import FeatureStore

_SCALER_PATH = settings.model_dir / "feature_scaler.pkl"


class FeatureGenerator:
    """
    Computes, normalises, and optionally extends features with GP-generated ones.
    """

    def __init__(self) -> None:
        self._store = FeatureStore()
        self._scaler: Optional[StandardScaler] = None
        self._gp_transformer = None
        self._load_scaler()

    # ── Main API ──────────────────────────────────────────────────────────

    def compute_raw(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Return un-normalised feature DataFrame."""
        return self._store.compute(symbol, df)

    def compute_normalised(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        raw = self.compute_raw(symbol, df)
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        if self._scaler is None:
            return raw
        scaled = self._scaler.transform(raw)
        return pd.DataFrame(scaled, index=raw.index, columns=raw.columns)

    def compute_latest(
        self, symbol: str, df: pd.DataFrame, normalise: bool = True
    ) -> Optional[pd.Series]:
        """Return the most recent feature row (for live inference)."""
        fdf = self.compute_raw(symbol, df)
        fdf = fdf.replace([np.inf, -np.inf], np.nan)
        if fdf.empty:
            return None
        latest = fdf.iloc[-1]
        if normalise and self._scaler is not None:
            vals = self._scaler.transform(latest.fillna(0).values.reshape(1, -1))[0]
            return pd.Series(vals, index=latest.index)
        return latest.fillna(0)

    # ── GP feature engineering (optional) ────────────────────────────────

    def run_genetic_programming(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_components: int = 10,
        generations: int = 20,
    ) -> pd.DataFrame:
        """
        Use gplearn SymbolicTransformer to generate new features.
        Appends them to X and returns the augmented DataFrame.
        Skips silently if gplearn is not installed.
        """
        try:
            from gplearn.genetic import SymbolicTransformer
        except ImportError:
            logger.warning("gplearn not installed — skipping GP feature engineering")
            return X

        logger.info(f"Running genetic programming (generations={generations})...")
        gp = SymbolicTransformer(
            generations=generations,
            population_size=500,
            hall_of_fame=100,
            n_components=n_components,
            parsimony_coefficient=0.001,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        X_np = X.fillna(0).values
        gp.fit(X_np, y.values)
        new_feats = gp.transform(X_np)
        col_names = [f"gp_{i}" for i in range(new_feats.shape[1])]
        gp_df = pd.DataFrame(new_feats, index=X.index, columns=col_names)
        self._gp_transformer = gp
        logger.info(f"GP generated {n_components} new features")
        return pd.concat([X, gp_df], axis=1)

    # ── Drift detection (PSI) ─────────────────────────────────────────────

    @staticmethod
    def compute_psi(
        expected: pd.Series,
        actual: pd.Series,
        buckets: int = 10,
    ) -> float:
        """
        Population Stability Index for a single feature.
        PSI < 0.1: stable. 0.1–0.2: slight shift. >0.2: major drift.
        """
        def _bucket_counts(s: pd.Series, bins) -> np.ndarray:
            counts, _ = np.histogram(s.dropna(), bins=bins)
            counts = counts / max(len(s), 1)
            return np.where(counts == 0, 1e-4, counts)

        combined = pd.concat([expected, actual]).dropna()
        _, bins = np.histogram(combined, bins=buckets)

        exp_pct = _bucket_counts(expected, bins)
        act_pct = _bucket_counts(actual, bins)

        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi)

    def detect_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        threshold: float = 0.2,
    ) -> Dict[str, float]:
        """
        Compute PSI for all features. Returns {feature: psi} for drifting ones.
        """
        drifting = {}
        for col in reference.columns:
            if col in current.columns:
                psi = self.compute_psi(reference[col], current[col])
                if psi > threshold:
                    drifting[col] = psi
        if drifting:
            logger.warning(f"Feature drift detected: {drifting}")
        return drifting

    # ── Scaler ────────────────────────────────────────────────────────────

    def _load_scaler(self) -> None:
        if _SCALER_PATH.exists():
            try:
                with open(_SCALER_PATH, "rb") as f:
                    self._scaler = pickle.load(f)
                logger.debug("Feature scaler loaded")
            except Exception as exc:
                logger.warning(f"Scaler load failed: {exc}")
