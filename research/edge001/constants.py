"""EDGE-001 frozen spec. Do not retune after seeing confirmation."""
from __future__ import annotations

from datetime import date
from pathlib import Path

EXPERIMENT = "EDGE-001"
PRIMARY_RANKER = "M1_12_1"
PRIMARY_N = 20
PRIMARY_CADENCE = "monthly"
PRIMARY_REBALANCE = "monthly_last_session"
MIN_PRICE = 20.0
MIN_TURNOVER = 5_000_000.0
MIN_TURNOVER_INR = MIN_TURNOVER
MIN_SESSIONS = 260
SKIP = 21
SKIP_BARS = SKIP
M1_LOOKBACK = 252
M2_LOOKBACK = 126
M4_LOOKBACK = 189
ADV_LOOKBACK = 20
DEV_END = "2022-12-31"
VAL_END = "2024-12-31"
CONF_END = "2026-08-21"
DSR_N_TRIALS = 64
CAPITALS = (500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000)
COST_PRODUCT = "CNC"
# round_trip_cost_pct already includes 0.10% slippage; apply as percent/100.
PROTOCOL_ACTIVATED_IST = "2026-08-22T00:00:00+05:30"
HORIZON_LOOKBACK = {"M1": M1_LOOKBACK, "M2": M2_LOOKBACK, "M4": M4_LOOKBACK}
BLOCKS = {
    "development": (date(2019, 1, 1), date(2022, 12, 31)),
    "validation": (date(2023, 1, 1), date(2024, 12, 31)),
    "confirmation": (date(2025, 1, 1), date(2026, 8, 21)),
}
OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "EDGE-001"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "edge001"
PARTICIPATION_FLAG = 0.05  # 5% of ADV = unreasonable retail clip


def block_of(as_of: str) -> str:
    if as_of <= DEV_END:
        return "development"
    if as_of <= VAL_END:
        return "validation"
    return "confirmation"


def protocol_sha() -> str:
    import hashlib
    import json

    p = OUT_DIR / "edge_001_protocol.json"
    if not p.exists():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
