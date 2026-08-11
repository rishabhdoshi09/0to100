"""Horizon and target specifications — research contracts, not live models."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OverlapPolicy(str, Enum):
    """How overlapping label windows are treated in train/test design."""

    PURGE = "purge"                 # drop train samples whose labels overlap test (default)
    DROP_OVERLAPPING = "drop_overlapping"  # alias intent: same as purge for CV
    ALLOW_WITH_WARNING = "allow_with_warning"  # research-only; never for promotion evidence


@dataclass(frozen=True)
class HorizonSpec:
    """A prediction horizon measured in trading bars (sessions)."""

    name: str
    bars: int
    entry: str = "close"            # price used at decision bar
    exit: str = "close"             # price used at horizon realisation
    calendar: str = "trading_bars"  # vs wall-clock — QuantTerm research uses sessions

    def __post_init__(self):
        if int(self.bars) < 1:
            raise ValueError(f"horizon bars must be >= 1, got {self.bars}")
        if not self.name:
            raise ValueError("horizon name required")


@dataclass(frozen=True)
class TargetSpec:
    """Full research target definition for one horizon.

    Capabilities (5d, 10d, 22d, …) are expressed as TargetSpec instances.
    They are *not* mandatory simultaneous live models.
    """

    horizon: HorizonSpec
    kind: str = "absolute_return"   # absolute_return | benchmark_relative | classification
    buy_thresh: float | None = None
    sell_thresh: float | None = None
    cost_pct_roundtrip: float = 0.0
    overlap_policy: OverlapPolicy = OverlapPolicy.PURGE
    purge_bars: int | None = None     # default = horizon.bars
    embargo_bars: int | None = None   # default = horizon.bars
    benchmark_relative: bool = False
    description: str = ""

    def __post_init__(self):
        if self.kind not in ("absolute_return", "benchmark_relative", "classification"):
            raise ValueError(f"unsupported target kind: {self.kind}")
        if self.cost_pct_roundtrip < 0:
            raise ValueError("cost_pct_roundtrip must be >= 0")
        if self.benchmark_relative and self.kind == "absolute_return":
            # Allow explicit flag to upgrade kind semantics without surprise.
            object.__setattr__(self, "kind", "benchmark_relative")
        if self.buy_thresh is not None and self.sell_thresh is not None:
            if self.sell_thresh > self.buy_thresh:
                raise ValueError("sell_thresh must be <= buy_thresh")

    @property
    def name(self) -> str:
        return self.horizon.name

    @property
    def bars(self) -> int:
        return int(self.horizon.bars)

    @property
    def effective_purge_bars(self) -> int:
        return int(self.bars if self.purge_bars is None else self.purge_bars)

    @property
    def effective_embargo_bars(self) -> int:
        return int(self.bars if self.embargo_bars is None else self.embargo_bars)

    def requires_purge(self) -> bool:
        return self.overlap_policy in (
            OverlapPolicy.PURGE,
            OverlapPolicy.DROP_OVERLAPPING,
        )
