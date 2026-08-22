"""EDGE-005 frozen spec."""
from __future__ import annotations

from pathlib import Path

EXPERIMENT = "EDGE-005"
PRIMARY_RANKER = "P1_252_NEAR_HIGH"
PRIMARY_N = 20
PRIMARY_CADENCE = "monthly"
P1_LOOKBACK = 252
P2_LOOKBACK = 126
P3_LOOKBACK = 63
MIN_PRICE = 20.0
MIN_TURNOVER = 5_000_000.0
MIN_SESSIONS = 260
DEV_END = "2022-12-31"
VAL_END = "2024-12-31"
CONF_END = "2026-08-21"
DSR_N_TRIALS = 20
PROTOCOL_ACTIVATED_IST = "2026-08-22T00:00:00+05:30"
OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "EDGE-005"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "edge005"


def protocol_sha() -> str:
    import hashlib
    p = OUT_DIR / "edge_005_protocol.json"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "missing"
