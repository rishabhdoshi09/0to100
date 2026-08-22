"""SEPA-003 locked labels. Do not treat 2025–2026 as untouched OOS."""
from __future__ import annotations

from pathlib import Path

EXPERIMENT = "SEPA-003"
FEATURE_SET = "sepa-003.v1"
REGIME_VERSION = "regime_pit_v1"
SECTOR_VERSION = "sector_map_v1"
WIN_ERA_END = "2023-12-31"
WEAK_ERA_START = "2024-01-01"
CONF_START = "2025-01-01"
CONF_END = "2026-08-21"
HORIZON = 20
FILL_SEARCH_SESSIONS = 20
MIN_N = 30
FDR_Q = 0.10
RS_BUCKETS = ("50-69", "70-79", "80-89", "90-94", "95-99")
REGIME_STATES = (
    "STRONG_BULL", "BULL", "SIDEWAYS", "CORRECTION", "BEAR", "UNKNOWN",
)
FEATURE_CLASSES = (
    "ROBUST_POSITIVE", "CONTEXT_DEPENDENT", "UNSTABLE", "NO_SIGNAL", "INSUFFICIENT_DATA",
)
FORBIDDEN_LABELS = (
    "untouched confirmation", "final OOS", "production validated",
    "VALIDATED_EDGE", "DEPLOYMENT_ELIGIBLE",
)

R2_DIR = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "SEPA-001R2"
OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "SEPA-003"


def era_of(as_of: str) -> str:
    if as_of <= WIN_ERA_END:
        return "winning_era"
    if as_of >= WEAK_ERA_START:
        return "weak_era"
    return "other"


def rs_bucket(pct: float | None) -> str | None:
    if pct is None:
        return None
    p = float(pct)
    if 50 <= p < 70:
        return "50-69"
    if 70 <= p < 80:
        return "70-79"
    if 80 <= p < 90:
        return "80-89"
    if 90 <= p < 95:
        return "90-94"
    if p >= 95:
        return "95-99"
    return None
