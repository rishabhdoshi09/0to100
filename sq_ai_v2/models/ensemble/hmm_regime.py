"""
Hidden Markov Model for market regime detection.

Regimes:
  0 = bear  (low return, high vol)
  1 = chop  (near-zero return, moderate vol)
  2 = bull  (positive return, lower vol)

Features fed to HMM: [daily_return, log_volatility_20, volume_change]
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn import hmm
from loguru import logger

from config.settings import settings

_N_COMPONENTS = 3
_REGIME_NAMES = {0: "bear", 1: "chop", 2: "bull"}


class HMMRegime:
    """
    Gaussian HMM trained on multi-symbol returns to detect macro regime.
    At inference time, returns the current regime and its posterior probabilities.
    """

    def __init__(self, model_path: Optional[Path] = None, n_components: int = _N_COMPONENTS) -> None:
        self.n_components = n_components
        self._model_path = model_path or (settings.model_dir / "hmm.pkl")
        self._model: Optional[hmm.GaussianHMM] = None
        self._regime_map: Dict[int, int] = {}  # raw HMM state → 0/1/2
        if self._model_path.exists():
            self.load()

    # ── Feature extraction ─────────────────────────────────────────────────

    @staticmethod
    def extract_features(df: pd.DataFrame) -> np.ndarray:
        """
        df must have columns: close, volume (daily bars).
        Returns (T, 3) array: [daily_return, log_vol_20, vol_change].
        """
        c = df["close"]
        v = df["volume"]
        ret = c.pct_change().fillna(0)
        vol20 = ret.rolling(20).std().fillna(ret.std())
        log_vol = np.log(vol20 + 1e-8)
        vol_chg = v.pct_change().fillna(0).clip(-3, 3)
        return np.column_stack([ret.values, log_vol.values, vol_chg.values]).astype(np.float32)

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Train on all available symbols' daily OHLCV DataFrames.
        Concatenates feature sequences with length markers for hmmlearn.
        """
        all_feats: List[np.ndarray] = []
        lengths: List[int] = []

        for sym, df in data.items():
            feats = self.extract_features(df)
            all_feats.append(feats)
            lengths.append(len(feats))

        if not all_feats:
            logger.warning("HMM: no data provided")
            return

        X = np.concatenate(all_feats, axis=0)
        self._model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        self._model.fit(X, lengths=lengths)

        # Map HMM states to semantic labels by average return
        means = self._model.means_[:, 0]  # mean return per state
        order = np.argsort(means)  # bear < chop < bull
        self._regime_map = {int(order[i]): i for i in range(self.n_components)}

        logger.info(
            f"HMM trained: states={self.n_components}, "
            f"regime_map={self._regime_map}"
        )

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Tuple[int, np.ndarray, str]:
        """
        Returns:
          regime_id (0=bear, 1=chop, 2=bull),
          posterior probs (n_components,),
          regime_name str
        """
        if self._model is None:
            return 1, np.array([0.0, 1.0, 0.0]), "chop"

        feats = self.extract_features(df)
        if len(feats) < 20:
            return 1, np.array([0.0, 1.0, 0.0]), "chop"

        _, posteriors = self._model.score_samples(feats)
        last_posterior = posteriors[-1]  # probabilities for last bar

        raw_state = int(np.argmax(last_posterior))
        regime_id = self._regime_map.get(raw_state, 1)
        regime_name = _REGIME_NAMES[regime_id]

        # Remap posteriors to semantic order
        remapped = np.zeros(self.n_components)
        for raw, mapped in self._regime_map.items():
            remapped[mapped] = last_posterior[raw]

        return regime_id, remapped, regime_name

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """Return regime label (0/1/2) for every bar."""
        if self._model is None:
            return pd.Series(1, index=df.index)
        feats = self.extract_features(df)
        raw_states = self._model.predict(feats)
        mapped = np.array([self._regime_map.get(s, 1) for s in raw_states])
        return pd.Series(mapped, index=df.index)

    # ── Regime multipliers ────────────────────────────────────────────────

    @staticmethod
    def regime_signal_multiplier(regime_id: int) -> float:
        """Scale signal strength based on regime."""
        return {0: 0.5, 1: 0.8, 2: 1.2}.get(regime_id, 1.0)

    @staticmethod
    def regime_risk_multiplier(regime_id: int) -> float:
        """Scale position size based on regime."""
        return {0: 0.5, 1: 0.75, 2: 1.0}.get(regime_id, 1.0)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "regime_map": self._regime_map}, f)
        logger.info(f"HMM saved → {path}")

    def load(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self._model = obj["model"]
            self._regime_map = obj["regime_map"]
            logger.info(f"HMM loaded ← {path}")
        except Exception as exc:
            logger.warning(f"HMM load failed: {exc}")
