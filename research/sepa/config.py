"""Versioned SEPA-001 research parameters. Not live trading knobs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


ELIGIBILITY_VERSION = "sepa-001.v1"
TREND_VERSION = "trend_template_v1"
RS_VERSION = "rs_cs_v1"
VCP_VERSION = "vcp_swing_v1"
BUY_ZONE_VERSION = "buy_zone_v1"


@dataclass(frozen=True)
class SepaConfig:
    """All thresholds that change eligibility. Hash these with the code version."""

    eligibility_version: str = ELIGIBILITY_VERSION
    trend_version: str = TREND_VERSION
    rs_version: str = RS_VERSION
    vcp_version: str = VCP_VERSION
    buy_zone_version: str = BUY_ZONE_VERSION

    sma50: int = 50
    sma150: int = 150
    sma200: int = 200
    sma200_slope_lookback: int = 21
    high_low_lookback: int = 252
    off_52w_low_pct: float = 30.0
    near_52w_high_pct: float = 25.0

    rs_threshold: float = 70.0
    rs_horizons: tuple[int, ...] = (63, 126, 189, 252)
    rs_weights: tuple[float, ...] = (0.40, 0.20, 0.20, 0.20)

    swing_left: int = 3
    swing_right: int = 3
    min_reversal_pct: float = 2.5
    vcp_lookback: int = 120
    min_contractions: int = 2
    max_contractions: int = 6
    depth_expand_tol: float = 1.15
    final_vs_first: float = 0.75
    max_final_depth_pct: float = 12.0
    max_base_depth_pct: float = 35.0
    min_recovery_bounce: float = 1.02
    near_pivot_frac: float = 0.92
    volume_dry_up_max: float = 0.90

    buy_zone_below_pct: float = 0.25
    buy_zone_above_pct: float = 1.5

    max_stop_pct: float = 8.0
    max_stop_atr: float = 3.0
    atr_period: int = 14

    def config_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


DEFAULT_CONFIG = SepaConfig()
