"""
ConvictionScorer — deterministic 0-100 score from 6 components, 4 hard gates.

Components (all normalised to [0, 1] before weighting):
  trend     — price vs EMA alignment
  rsi       — distance from oversold/overbought
  momentum  — 5-day price momentum
  volume    — volume ratio vs 20-day average
  ml        — ML model probability
  regime    — macro regime (SMA crossover)

Gates (any single failure → score capped at 30, verdict forced to HOLD):
  1. Not in extreme overbought (RSI > 80)
  2. Volume confirmation (volume_ratio >= 0.8)
  3. Positive ML signal (ml_proba >= 0.45)
  4. Regime not strongly bearish (regime_score >= -0.3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from signals.profiles import TraderProfile, CONSERVATIVE


@dataclass
class ConvictionResult:
    score: float          # 0–100
    verdict: str          # BUY / SELL / HOLD
    components: Dict[str, float]
    gates_passed: bool
    gate_failures: list[str]


class ConvictionScorer:
    # Verdict thresholds
    BUY_THRESHOLD = 62
    SELL_THRESHOLD = 38

    def __init__(self, profile: TraderProfile = CONSERVATIVE) -> None:
        self.profile = profile

    def score(self, indicators: Dict) -> ConvictionResult:
        """
        Compute conviction score from indicator dict.

        Expected keys (from IndicatorEngine.compute):
          rsi_14, zscore_20, momentum_5d_pct, volume_ratio,
          sma_20, sma_50, ml_proba (optional)
        """
        components = self._compute_components(indicators)
        gate_failures = self._check_gates(indicators, components)
        gates_passed = len(gate_failures) == 0

        raw = sum(
            components[k] * self.profile.weights[k]
            for k in self.profile.weights
        )
        score = round(raw * 100, 1)

        if not gates_passed:
            score = min(score, 30.0)
            verdict = "HOLD"
        elif score >= self.BUY_THRESHOLD:
            verdict = "BUY"
        elif score <= self.SELL_THRESHOLD:
            verdict = "SELL"
        else:
            verdict = "HOLD"

        return ConvictionResult(
            score=score,
            verdict=verdict,
            components=components,
            gates_passed=gates_passed,
            gate_failures=gate_failures,
        )

    # ------------------------------------------------------------------
    def _compute_components(self, ind: Dict) -> Dict[str, float]:
        rsi = ind.get("rsi_14", 50.0)
        mom = ind.get("momentum_5d_pct", 0.0)
        vol_ratio = ind.get("volume_ratio", 1.0)
        sma20 = ind.get("sma_20", 0.0)
        sma50 = ind.get("sma_50", 0.0)
        ml_proba = ind.get("ml_proba", 0.5)

        # trend: 1.0 = sma20 well above sma50, 0.0 = well below
        if sma50 and sma50 > 0:
            trend_raw = (sma20 - sma50) / sma50
            trend = _clamp(0.5 + trend_raw * 10)
        else:
            trend = 0.5

        # rsi: oversold (30) → 1.0, overbought (70) → 0.0
        rsi_norm = _clamp(1.0 - (rsi - 30) / 40) if 30 <= rsi <= 70 else (1.0 if rsi < 30 else 0.0)

        # momentum: +5% → 1.0, -5% → 0.0
        mom_norm = _clamp(0.5 + mom * 10)

        # volume: ratio 2.0 → 1.0, 0.5 → 0.0
        vol_norm = _clamp((vol_ratio - 0.5) / 1.5)

        # ml: already a probability [0,1]
        ml_norm = _clamp(float(ml_proba))

        # regime: sma20 > sma50*1.005 → bullish
        if sma50 and sma50 > 0:
            regime_norm = 1.0 if sma20 > sma50 * 1.005 else (0.0 if sma20 < sma50 * 0.995 else 0.5)
        else:
            regime_norm = 0.5

        return {
            "trend": trend,
            "rsi": rsi_norm,
            "momentum": mom_norm,
            "volume": vol_norm,
            "ml": ml_norm,
            "regime": regime_norm,
        }

    def _check_gates(self, ind: Dict, comp: Dict) -> list[str]:
        failures = []
        rsi = ind.get("rsi_14", 50.0)
        vol_ratio = ind.get("volume_ratio", 1.0)
        ml_proba = float(ind.get("ml_proba", 0.5))
        regime = comp.get("regime", 0.5)

        if rsi > 80:
            failures.append(f"RSI overbought ({rsi:.1f} > 80)")
        if vol_ratio < 0.8:
            failures.append(f"Low volume ({vol_ratio:.2f}x < 0.8x avg)")
        if ml_proba < 0.45:
            failures.append(f"Weak ML signal (proba {ml_proba:.2f} < 0.45)")
        if regime < 0.1:
            failures.append("Regime strongly bearish (sma20 < sma50×0.995)")

        return failures


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))
