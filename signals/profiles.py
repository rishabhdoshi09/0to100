"""
TraderProfile — risk/reward personality weights for conviction scoring.

Two built-in profiles; weights must sum to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class TraderProfile:
    name: str
    # Component weights for ConvictionScorer (6 components)
    weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Profile '{self.name}' weights sum to {total:.4f}, expected 1.0")


# --- Conservative: prioritises trend, RSI mean-reversion, and volume confirmation
CONSERVATIVE = TraderProfile(
    name="Conservative",
    weights={
        "trend": 0.25,
        "rsi": 0.20,
        "momentum": 0.10,
        "volume": 0.20,
        "ml": 0.15,
        "regime": 0.10,
    },
)

# --- Aggressive: momentum + ML heavy, less weight on volume/regime
AGGRESSIVE = TraderProfile(
    name="Aggressive",
    weights={
        "trend": 0.10,
        "rsi": 0.10,
        "momentum": 0.30,
        "volume": 0.10,
        "ml": 0.30,
        "regime": 0.10,
    },
)

PROFILES: Dict[str, TraderProfile] = {
    "Conservative": CONSERVATIVE,
    "Aggressive": AGGRESSIVE,
}
