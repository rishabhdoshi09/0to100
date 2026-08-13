"""Horizon capability catalog — expressible horizons, not mandatory models."""
from __future__ import annotations

from research.horizons.spec import HorizonSpec, TargetSpec, OverlapPolicy

# Required research *capabilities* (audit). Both 20 and 22 (month approx),
# 60/66, 120/130, 200/252 are available; callers choose which to activate.
_CAPABILITY_BARS = {
    "1d": 1,       # legacy ML path
    "5d": 5,
    "10d": 10,
    "20d": 20,
    "22d": 22,
    "60d": 60,
    "66d": 66,
    "120d": 120,
    "130d": 130,
    "200d": 200,
    "252d": 252,
}

CAPABILITY_HORIZONS: dict[str, HorizonSpec] = {
    name: HorizonSpec(name=name, bars=bars)
    for name, bars in _CAPABILITY_BARS.items()
}


def get_horizon(name: str) -> HorizonSpec:
    key = str(name).strip().lower()
    if key not in CAPABILITY_HORIZONS:
        raise KeyError(
            f"unknown horizon '{name}' — known: {sorted(CAPABILITY_HORIZONS)}"
        )
    return CAPABILITY_HORIZONS[key]


def capability_names() -> list[str]:
    return sorted(CAPABILITY_HORIZONS, key=lambda n: CAPABILITY_HORIZONS[n].bars)


# Exact reproduction of ml.multi_horizon._HORIZONS classification thresholds.
# Live MultiHorizonSignalGenerator remains authoritative for production inference;
# these targets exist so research can reproduce the same labels without forking.
LEGACY_MH_TARGETS: dict[str, TargetSpec] = {
    "1d": TargetSpec(
        horizon=get_horizon("1d"),
        kind="classification",
        buy_thresh=0.0025,
        sell_thresh=-0.0025,
        cost_pct_roundtrip=0.0,
        overlap_policy=OverlapPolicy.PURGE,
        description="legacy ml.multi_horizon 1d classification thresholds",
    ),
    "5d": TargetSpec(
        horizon=get_horizon("5d"),
        kind="classification",
        buy_thresh=0.0050,
        sell_thresh=-0.0050,
        cost_pct_roundtrip=0.0,
        overlap_policy=OverlapPolicy.PURGE,
        description="legacy ml.multi_horizon 5d classification thresholds",
    ),
    "10d": TargetSpec(
        horizon=get_horizon("10d"),
        kind="classification",
        buy_thresh=0.0100,
        sell_thresh=-0.0100,
        cost_pct_roundtrip=0.0,
        overlap_policy=OverlapPolicy.PURGE,
        description="legacy ml.multi_horizon 10d classification thresholds",
    ),
}


def get_legacy_mh_target(name: str) -> TargetSpec:
    key = str(name).strip().lower()
    if key not in LEGACY_MH_TARGETS:
        raise KeyError(f"no legacy MH target '{name}' — known: {sorted(LEGACY_MH_TARGETS)}")
    return LEGACY_MH_TARGETS[key]


def absolute_return_target(
    horizon_name: str,
    *,
    cost_pct_roundtrip: float = 0.0,
    purge_bars: int | None = None,
    embargo_bars: int | None = None,
) -> TargetSpec:
    """Convenience: continuous absolute forward return for a capability horizon."""
    return TargetSpec(
        horizon=get_horizon(horizon_name),
        kind="absolute_return",
        cost_pct_roundtrip=cost_pct_roundtrip,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        overlap_policy=OverlapPolicy.PURGE,
    )
