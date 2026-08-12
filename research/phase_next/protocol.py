"""Phase-next shared constants — frozen BEFORE outcome inspection."""
from __future__ import annotations

# Certified research surface (verified in repo)
SNAPSHOT_ID = "a7a9828ec37e09e4"
GLOBAL_TRUST = "OPERATIONAL_ONLY"

# Chronological partitions (deterministic; confirmation untouched until discovery done)
WARMUP_END = "2024-07-31"          # features may use ≤ this for τ fitting
DISCOVERY_START = "2024-08-01"
DISCOVERY_END = "2025-07-31"
CONFIRM_START = "2025-08-01"
# panel end = snapshot last date

# Costs
COST_PRODUCT = "CNC"
TURNOVER_ONE_WAY = 1.0  # conservative full turnover per rebalance

# EXP-NEXT-01
REVERSAL_FORMATIONS = (1, 3, 5)       # short lookbacks only — ≠ rejected 60d momentum
REVERSAL_HOLDS = (5, 10)
REVERSAL_Q = 0.2

# EXP-NEXT-02
LOWVOL_LOOKBACK = 20
LOWVOL_REBALANCE = 21
LOWVOL_HOLD = 21
LOWVOL_Q = 0.2

# EXP-NEXT-03
VOLCOMP_SHORT = 10
VOLCOMP_LONG = 60
# τ = train-only quantile of short/long vol ratio (frozen from warmup)
VOLCOMP_TAU_QUANTILE = 0.25  # bottom quartile = compressed
VOLCOMP_FWD = 10
