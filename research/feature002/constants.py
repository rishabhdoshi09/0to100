"""FEATURE-002 frozen protocol. Bump feature_set_version to start a new experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPERIMENT = "FEATURE-002"
FEATURE_SET_VERSION = "feature-002.v1"
SHADOW_RANK_VERSION = "feature-002.ranks.v1"
PRODUCTION_RANK_VERSION = "auto_scan.final_order.v1"
TREND_VERSION = "trend_features_v1"
RS_VERSION = "rs_features_v1"
RS_SOURCE = "rs_cs_v1"
EXCHANGE = "NSE"

FEATURE_001_FINAL_COMMIT = "aa2dc3b3ef5ff611b2cdd25faeabff93f80dae58"
FEATURE_001_LAST_SAMPLE = "2026-07-23"
FORWARD_START_DATE = "2026-07-24"
FORWARD_START_TS_IST = "2026-08-22T00:00:00+05:30"

R3_RS_WEIGHT = 0.67
R3_TREND_WEIGHT = 0.33
R3_FORMULA = (
    "0.67 * within_set_pctl(rs_percentile) + 0.33 * within_set_pctl(n_structure_passed)"
)

PRIMARY_SOURCE = "live_scan"
TEST_SOURCES = frozenset({"implementation_test", "synthetic", "replay"})

QUIET_MAX = 99
EARLY_MAX = 499
INTERIM_MAX = 1999
DECISION_RESOLVED = 2000
DECISION_MULTI_SETS = 250
DECISION_PER_FAMILY = 100
DECISION_MONTHS = 6

ALLOWED_LABELS = (
    "GRADUATE_RANK_FEATURE",
    "EXTEND_FORWARD_VALIDATION",
    "KEEP_RESEARCH_ONLY",
    "RETIRE",
)
UNTIL_MATURE = "FORWARD VALIDATION ACTIVE — INSUFFICIENT NEW DATA"

OUT_DIR = (
    Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "FEATURE-002"
)
LEDGER_DIR = Path(__file__).resolve().parents[2] / "logs" / "feature002"
DB_PATH = LEDGER_DIR / "shadow.db"

_PROTOCOL = {
    "experiment": EXPERIMENT,
    "feature_set_version": FEATURE_SET_VERSION,
    "shadow_rank_version": SHADOW_RANK_VERSION,
    "production_rank_version": PRODUCTION_RANK_VERSION,
    "trend_version": TREND_VERSION,
    "rs_version": RS_VERSION,
    "rs_source": RS_SOURCE,
    "feature_001_final_commit": FEATURE_001_FINAL_COMMIT,
    "feature_001_last_sample_date": FEATURE_001_LAST_SAMPLE,
    "forward_start_date": FORWARD_START_DATE,
    "forward_start_ts_ist": FORWARD_START_TS_IST,
    "r3_formula": R3_FORMULA,
    "r3_weights": [R3_RS_WEIGHT, R3_TREND_WEIGHT],
}


def protocol_hash() -> str:
    blob = json.dumps(_PROTOCOL, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def event_id(session_date: str, symbol: str) -> str:
    raw = f"{FEATURE_SET_VERSION}|{session_date}|{str(symbol).upper()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def candidate_set_id(scan_cycle_id: str) -> str:
    raw = f"{FEATURE_SET_VERSION}|{scan_cycle_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def eligible_primary(session_date: str, source: str, recorded_at: str,
                     feature_set_version: str) -> bool:
    if feature_set_version != FEATURE_SET_VERSION:
        return False
    if source != PRIMARY_SOURCE:
        return False
    if str(session_date) < FORWARD_START_DATE:
        return False
    rec = str(recorded_at or "")
    if rec and rec < FORWARD_START_TS_IST:
        return False
    return True
