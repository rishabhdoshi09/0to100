"""
HMM-based regime detector.

Fits a 2-state Gaussian HMM on daily log returns + volume change
over the last 252 trading days.

States:
  0 = Ranging / mean-reverting (low volatility, no trend)
  1 = Trending (directional move, sustained momentum)

Automatically labels states by comparing return volatility per state —
the higher-volatility state is labeled TRENDING.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

_LOOKBACK = 252
_N_STATES = 2
_N_ITER = 100

try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    log.warning("hmmlearn_not_installed", hint="pip install hmmlearn")


class HMMRegimeDetector:
    """
    2-state Gaussian HMM regime detector.

    Usage
    -----
    det = HMMRegimeDetector()
    result = det.detect(df)   # df must have 'close' and 'volume' columns
    # result = {
    #     "regime":    0 or 1,
    #     "label":     "RANGING" or "TRENDING",
    #     "swap_prob": 0.03,     # probability of switching state tomorrow
    #     "hmm_key":   "RANGING_HMM" or "TRENDING_HMM",
    # }
    """

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._trending_state: int = 1   # updated after each fit

    def detect(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit HMM on df and return regime dict for the most recent bar."""
        fallback = self._fallback("hmm_unavailable")

        if not _HMM_AVAILABLE:
            return fallback

        if df is None or len(df) < 30:
            return self._fallback("insufficient_data")

        obs = self._build_observations(df)
        if obs is None or len(obs) < 20:
            return self._fallback("observation_build_failed")

        try:
            model = _GaussianHMM(
                n_components=_N_STATES,
                covariance_type="diag",
                n_iter=_N_ITER,
                random_state=42,
            )
            model.fit(obs)
            states = model.predict(obs)
        except Exception as exc:
            log.warning("hmm_fit_failed", error=str(exc))
            return self._fallback(f"fit_error:{exc}")

        # Identify which state is "trending" by comparing return volatility
        state_vols = [
            np.std(obs[states == s, 0]) if np.any(states == s) else 0.0
            for s in range(_N_STATES)
        ]
        trending_state = int(np.argmax(state_vols))
        current_state = int(states[-1])
        is_trending = current_state == trending_state

        # Probability of switching state tomorrow (from transition matrix)
        trans = model.transmat_
        swap_prob = float(trans[current_state, 1 - current_state])

        label = "TRENDING" if is_trending else "RANGING"
        hmm_key = f"{label}_HMM"

        log.info("hmm_regime_detected", regime=current_state, label=label,
                 swap_prob=round(swap_prob, 4))

        return {
            "regime":    current_state,
            "label":     label,
            "swap_prob": round(swap_prob, 4),
            "hmm_key":   hmm_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Feature construction ───────────────────────────────────────────────

    @staticmethod
    def _build_observations(df: pd.DataFrame) -> Optional[np.ndarray]:
        """Returns Nx2 array: [log_return, volume_change_pct]."""
        try:
            close  = df["close"].astype(float).values[-_LOOKBACK:]
            volume = df["volume"].astype(float).values[-_LOOKBACK:]

            log_ret    = np.diff(np.log(np.maximum(close, 1e-8)))
            vol_change = np.diff(volume) / np.maximum(volume[:-1], 1.0)

            n = min(len(log_ret), len(vol_change))
            if n < 10:
                return None

            obs = np.column_stack([log_ret[-n:], vol_change[-n:]])
            # Remove rows with NaN / inf
            obs = obs[np.isfinite(obs).all(axis=1)]
            return obs if len(obs) >= 10 else None
        except Exception as exc:
            log.warning("hmm_observation_failed", error=str(exc))
            return None

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback(reason: str) -> Dict[str, Any]:
        return {
            "regime":    -1,
            "label":     "UNKNOWN",
            "swap_prob": 0.5,
            "hmm_key":   "UNKNOWN_HMM",
            "reason":    reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
