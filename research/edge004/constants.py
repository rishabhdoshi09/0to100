"""EDGE-004 frozen spec."""
from __future__ import annotations

from pathlib import Path

EXPERIMENT = "EDGE-004"
PRIMARY_RANKER = "R1_21_INCLUSIVE_LOSERS"
PRIMARY_N = 20
PRIMARY_CADENCE = "monthly"
R1_LOOKBACK = 21
R2_LOOKBACK = 10
R3_LOOKBACK = 42
R0_SKIP = 5
MIN_PRICE = 20.0
MIN_TURNOVER = 5_000_000.0
MIN_SESSIONS = 260
DEV_END = "2022-12-31"
VAL_END = "2024-12-31"
CONF_END = "2026-08-21"
DSR_N_TRIALS = 24
PROTOCOL_ACTIVATED_IST = "2026-08-22T00:00:00+05:30"
OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "EDGE-004"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "edge004"


def protocol_sha() -> str:
    import hashlib
    p = OUT_DIR / "edge_004_protocol.json"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "missing"
