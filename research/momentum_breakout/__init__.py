"""
Institutional Momentum Breakout — a point-in-time, reproducible RESEARCH framework
(EXP-006) for studying whether stocks with prior leadership, a long contracting base,
a confirmed breakout, small structural risk and strong sector support produce positive
forward expectancy after realistic Indian cash-equity costs.

RESEARCH-ONLY. Nothing in this package is wired to autopilot, the broker, Telegram,
GTT, or strategy graduation. Candidate detection generates evidence, not orders.

The hypothesis is NOT assumed valid — the correct result may be PASS, FAIL, or
INCONCLUSIVE, decided by the existing gauntlet/harness evidence gate.
"""
from research.momentum_breakout.config import (
    MomentumBreakoutConfig, primary_config,
    DETECTOR_VERSION, FEATURES_VERSION, SCORING_VERSION,
)
from research.momentum_breakout.observation import (
    MomentumBreakoutObservation, ELIGIBLE, REJECTED,
)
from research.momentum_breakout.detector import BarSeries, consider, scan_symbol
from research.momentum_breakout import experiment

__all__ = [
    "MomentumBreakoutConfig", "primary_config", "MomentumBreakoutObservation",
    "ELIGIBLE", "REJECTED", "BarSeries", "consider", "scan_symbol", "experiment",
    "DETECTOR_VERSION", "FEATURES_VERSION", "SCORING_VERSION",
]
