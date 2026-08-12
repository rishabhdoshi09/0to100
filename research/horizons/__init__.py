"""
research/horizons — generic, leakage-safe multi-horizon target framework (Phase A / A2).

This package defines *how* a research target is specified and labelled. It does not
train models, place trades, or alter live ``ml.multi_horizon`` behaviour.

Primary objective: prevent overlapping-label leakage and future contamination.
"""
from research.horizons.spec import HorizonSpec, TargetSpec, OverlapPolicy
from research.horizons.catalog import (
    CAPABILITY_HORIZONS,
    LEGACY_MH_TARGETS,
    capability_names,
    get_horizon,
    get_legacy_mh_target,
)
from research.horizons.labels import (
    LabelFrame,
    build_forward_return_labels,
    classification_from_returns,
)
from research.horizons.splits import purged_splits_for_target, assert_no_label_leakage

__all__ = [
    "HorizonSpec",
    "TargetSpec",
    "OverlapPolicy",
    "CAPABILITY_HORIZONS",
    "LEGACY_MH_TARGETS",
    "capability_names",
    "get_horizon",
    "get_legacy_mh_target",
    "LabelFrame",
    "build_forward_return_labels",
    "classification_from_returns",
    "purged_splits_for_target",
    "assert_no_label_leakage",
]
