"""
⚙️ Versioned configuration for the Institutional Momentum Breakout framework.

Every result-determining knob lives here, in ONE frozen dataclass, so nothing is a
scattered magic number and every threshold is captured in the config hash. Changing
a primary threshold after seeing results is a NEW experiment (new hash, new EXP id),
never a silent edit — the anti-fishing discipline in docs/RESEARCH_LOG.md.

`primary_config()` is the pre-registered EXP-006 configuration. The research ranges
in the milestone (e.g. base 40–180 sessions, risk 2–8%) are exposed as configurable
fields; the primary values are ONE point in those ranges and must not be optimised
against the primary result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field

from research.momentum_breakout.pit import PIT_INDICATORS_VERSION

# Bumped when the DETECTOR/FEATURE/SCORING logic changes (independent of thresholds).
DETECTOR_VERSION = 1
FEATURES_VERSION = 1
SCORING_VERSION = 1


@dataclass(frozen=True)
class MomentumBreakoutConfig:
    """Result-determining configuration. Frozen: build a new one to change a knob.

    THRESHOLDS ARE PRE-REGISTERED. The primary momentum hypothesis is defined by
    these defaults; they must not be tuned to the primary result (sensitivity
    analysis uses explicitly-labelled alternate configs with their own hash)."""

    # ── identity / provenance ──
    strategy_id: str = "institutional_momentum_breakout"
    config_version: str = "v1"

    # ── prior upmove / leadership (group 1) ──
    prior_rs_lookback: int = 126          # ~6m relative-strength lookback (bars)
    min_prior_rs_pct: float = 10.0        # stock must beat benchmark by ≥ this (pp)
    min_rs_percentile: float = 70.0       # cross-sectional RS percentile floor
    require_above_200dma: bool = True     # base must sit above a rising long-term trend
    trend_ma: int = 200
    trend_slope_ago: int = 20             # 200-DMA slope measured vs this many bars back

    # ── base structure (group 2) ──
    base_min_len: int = 40                # min base duration (trading sessions)
    base_max_len: int = 180              # max base duration considered
    max_base_depth_pct: float = 35.0      # reject a base deeper than this (unstable)
    pivot_lookback: int = 10              # pivot = highest high of the base's late window
    min_range_contraction: float = 0.0    # base range must be tighter than prior (ratio<1); 0 disables
    require_contraction: bool = True      # ATR or range must contract across the base

    # ── breakout quality (group 3) ──
    breakout_buffer_pct: float = 0.25     # close must clear the pivot by ≥ this %
    require_confirmed_close: bool = True  # signal only after a CLOSE above pivot
    max_extension_atr: float = 4.0        # reject if close is > this many ATR above pivot (chase)
    min_breakout_rvol: float = 1.3        # breakout-day relative volume floor

    # ── structural stop / risk (group 4) ──
    max_initial_risk_pct: float = 8.0     # reject setups risking more than this
    min_initial_risk_pct: float = 1.5     # implausibly tight stop → likely bad structure
    atr_stop_mult: float = 1.0            # pivot − mult×ATR is one stop candidate
    swing_low_lookback: int = 10          # recent swing-low stop candidate window

    # ── sector strength (group 5) ──
    require_sector_strength: bool = True  # primary hypothesis needs sector participation
    sector_rs_lookback: int = 63          # ~3m sector RS lookback
    min_sector_rs_pct: float = 0.0        # sector must beat benchmark by ≥ this (pp)
    min_sector_breadth_pct: float = 50.0  # ≥ this % of sector members above 50-DMA

    # ── participation / liquidity (group 6) ──
    min_turnover_cr: float = 5.0          # median daily turnover floor (₹ crore)
    vol_ref_window: int = 50              # reference window for volume z / rvol
    base_dryup_ref: int = 50              # pre-base window for volume dry-up

    # ── valuation (group 7) — CONTEXT ONLY, never a primary reject ──
    extreme_pe: float = 80.0
    extreme_ps: float = 20.0

    # ── entry / exit convention (experiment) ──
    entry_next_bar: bool = True           # enter no earlier than the next tradable bar
    slippage_pct: float = 0.10            # per-side slippage assumption
    cost_pct_roundtrip: float = 0.22      # round-trip charges (labelled, modelled)
    max_hold_days: int = 60               # time-stop for the max-hold exit variant
    trail_ema: int = 20                   # trailing rule for the EMA-trail exit variant

    # ── dedup / re-eligibility ──
    reeligible_after_bars: int = 20       # a name can requalify only after a new base

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        """Deterministic hash of every result-determining knob PLUS the code
        versions of the primitives/detector/features/scoring. Any change flips it,
        binding an observation to the exact logic + thresholds that produced it."""
        payload = {
            "config": self.to_dict(),
            "pit_indicators_version": PIT_INDICATORS_VERSION,
            "detector_version": DETECTOR_VERSION,
            "features_version": FEATURES_VERSION,
            "scoring_version": SCORING_VERSION,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def primary_config() -> MomentumBreakoutConfig:
    """The ONE pre-registered EXP-006 primary configuration."""
    return MomentumBreakoutConfig()
