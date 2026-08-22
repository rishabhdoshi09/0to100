"""FEATURE-001 locked labels. Consumed history is explanatory only."""
from __future__ import annotations

from pathlib import Path

from scan.unified_scanner import SIGNAL_META

EXPERIMENT = "FEATURE-001"
TREND_VERSION = "trend_features_v1"
RS_VERSION = "rs_features_v1"
RS_SOURCE = "rs_cs_v1"
CLAIM_CLASS = "EXPLANATORY"
CORE_SEPA_STATUS = "RETIRED_RESEARCH_BENCHMARK"

SAMPLE_STEP = 5
SCAN_LOOKBACK = 280
HORIZON = 20
MIN_SESSIONS = 260
MIN_PRICE = 20.0
MIN_TURNOVER = 5_000_000.0
MIN_N = 30
FDR_Q = 0.10
NEAR_STRUCTURE = 5
STRONG_RS = 70.0
RS_DELTA_SESSIONS = 21
YEARS = ("2020", "2021", "2022", "2023", "2024", "2025", "2026")

BREAKOUT_KEYS = ("BREAKOUT_52W", "BREAKOUT_RES", "GOLDEN_CROSS", "VOL_SQUEEZE")
MOMENTUM_KEYS = ("MOMENTUM",)
FAMILY_KEYS = tuple(SIGNAL_META.keys())
FAMILY_CATEGORY = {k: SIGNAL_META[k][1] for k in FAMILY_KEYS}

RS_BUCKETS = ("<50", "50-69", "70-79", "80-89", "90-94", "95-99")
TREND_BUCKETS = ("strict", "near", "non")
FEATURE_CLASSES = (
    "POSITIVE_RANK_FEATURE",
    "RISK_FILTER_VALUE",
    "REDUNDANT",
    "NEGATIVE",
    "UNSTABLE",
    "INSUFFICIENT_DATA",
)
FINAL_STATUS = (
    "FORWARD-VALIDATE AS RANK FEATURE",
    "FORWARD-VALIDATE AS RISK FILTER",
    "KEEP RESEARCH-ONLY",
    "RETIRE",
)
FORBIDDEN_LABELS = (
    "VALIDATED_EDGE",
    "DEPLOYMENT_ELIGIBLE",
    "untouched confirmation",
    "final OOS",
    "production validated",
)

OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "FEATURE-001"


def rs_bucket(pct: float | None) -> str | None:
    if pct is None:
        return None
    p = float(pct)
    if p < 50:
        return "<50"
    if p < 70:
        return "50-69"
    if p < 80:
        return "70-79"
    if p < 90:
        return "80-89"
    if p < 95:
        return "90-94"
    return "95-99"


def trend_bucket(structure_pass: bool | None, n_passed: int | None) -> str | None:
    if structure_pass is None or n_passed is None:
        return None
    if structure_pass:
        return "strict"
    if int(n_passed) >= NEAR_STRUCTURE:
        return "near"
    return "non"


def year_of(as_of: str) -> str:
    return str(as_of)[:4]
