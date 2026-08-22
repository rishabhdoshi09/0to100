"""EDGE-002 frozen spec. Do not retune after seeing confirmation."""
from __future__ import annotations

from pathlib import Path

EXPERIMENT = "EDGE-002"
PRIMARY_RANKER = "V1_126_REALIZED_VOL"
PRIMARY_N = 20
PRIMARY_CADENCE = "monthly"
MIN_PRICE = 20.0
MIN_TURNOVER = 5_000_000.0
MIN_SESSIONS = 260
V1_LOOKBACK = 126
V2_LOOKBACK = 63
V3_LOOKBACK = 252
V0_LOOKBACK = 20  # diagnostic only — EXP-NEXT-02 overlap
DEV_END = "2022-12-31"
VAL_END = "2024-12-31"
CONF_END = "2026-08-21"
DSR_N_TRIALS = 48
CAPITALS = (500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000)
PROTOCOL_ACTIVATED_IST = "2026-08-22T00:00:00+05:30"
OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "EDGE-002"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "edge002"


def block_of(as_of: str) -> str:
    if as_of <= DEV_END:
        return "development"
    if as_of <= VAL_END:
        return "validation"
    return "confirmation"


def protocol_sha() -> str:
    import hashlib
    p = OUT_DIR / "edge_002_protocol.json"
    if not p.exists():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
