"""EDGE-003 frozen spec."""
from __future__ import annotations

from pathlib import Path

EXPERIMENT = "EDGE-003"
PRIMARY_SIGNAL = "T1_PRICE_GT_SMA200_AND_SMA200_RISING"
SMA_WINDOW = 200
SLOPE_LOOKBACK = 21
MIN_PRICE = 20.0
MIN_TURNOVER = 5_000_000.0
MIN_SESSIONS = 260
DEV_END = "2022-12-31"
VAL_END = "2024-12-31"
CONF_END = "2026-08-21"
DSR_N_TRIALS = 16
PROTOCOL_ACTIVATED_IST = "2026-08-22T00:00:00+05:30"
PRIMARY_BOOK = "all_qualifiers_equal_weight"
PRIMARY_CADENCE = "monthly"
CAPITALS = (500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000)
PARTICIPATION_FLAG = 0.05
OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "EDGE-003"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "edge003"


def protocol_sha() -> str:
    import hashlib
    p = OUT_DIR / "edge_003_protocol.json"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "missing"
