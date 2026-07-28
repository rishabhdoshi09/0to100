"""
🔬 MomentumBreakoutObservation — the canonical research object.

One immutable record per candidate breakout event. It carries the RAW component
features (never just an opaque score), the transparent component scores, the full
point-in-time provenance (which data was available when), the eligibility decision
with every rejection reason, and the reproducibility stamp (experiment id, config
hash, code commit, dataset snapshot). Two runs on the same data+config+code must
produce byte-identical observations and event ids.

This is a RESEARCH object. It is never handed to autopilot, the broker, Telegram,
GTT, or any execution path — candidate detection in this milestone is research-only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

# ── data-quality / limitation flags (transparent, never silent) ────────────────
FLAG_SURVIVORSHIP_INCOMPLETE = "SURVIVORSHIP_INCOMPLETE"
FLAG_SECTOR_MEMBERSHIP_NOT_PIT = "SECTOR_MEMBERSHIP_NOT_PIT"
FLAG_VALUATION_UNAVAILABLE = "VALUATION_DATA_UNAVAILABLE"
FLAG_VALUATION_STALE = "VALUATION_DATA_STALE"
FLAG_EXTREME_PE = "EXTREME_PE"
FLAG_EXTREME_PS = "EXTREME_PRICE_TO_SALES"
FLAG_HIGH_EXPECTATION_RISK = "HIGH_EXPECTATION_RISK"
FLAG_DELIVERY_UNAVAILABLE = "DELIVERY_DATA_UNAVAILABLE"
FLAG_INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"

# ── eligibility verdicts ───────────────────────────────────────────────────────
ELIGIBLE = "ELIGIBLE"
REJECTED = "REJECTED"


@dataclass(frozen=True)
class MomentumBreakoutObservation:
    # ── identity ──
    symbol: str
    exchange: str
    # ── timestamps (the six distinct clocks the framework must separate) ──
    observation_ts: str          # bar time the signal became known (breakout close)
    data_availability_ts: str    # when the data used was actually available
    candidate_date: str          # the trading day of the candidate
    # ── structure ──
    pivot: float
    base_start_date: str
    base_end_date: str
    base_duration: int
    entry_reference_price: float
    structural_stop: float
    initial_risk_pct: float
    initial_risk_atr: float
    # ── feature groups (raw values, transparent) ──
    prior_upmove: dict = field(default_factory=dict)
    base_quality: dict = field(default_factory=dict)
    breakout_quality: dict = field(default_factory=dict)
    sector_strength: dict = field(default_factory=dict)
    participation: dict = field(default_factory=dict)
    trend_extension: dict = field(default_factory=dict)
    valuation: dict = field(default_factory=dict)
    valuation_data_ts: str | None = None
    stop_candidates: dict = field(default_factory=dict)
    # ── transparent component scores ──
    component_scores: dict = field(default_factory=dict)
    combined_score: float | None = None
    # ── decision ──
    eligibility: str = REJECTED
    rejection_reasons: tuple = ()
    data_quality_flags: tuple = ()
    # ── reproducibility / provenance ──
    experiment_id: str = ""
    strategy_id: str = ""
    config_version: str = ""
    config_hash: str = ""
    dataset_snapshot_id: str = ""
    code_commit: str = ""
    detector_version: int = 0

    def event_id(self) -> str:
        """Canonical breakout-event identity. Deduplication key: the SAME breakout
        (symbol + base + pivot + breakout day + detector/config version) always
        hashes to the same id, so consecutive closes above one pivot cannot mint
        duplicate events and equivalent detectors cannot double-count it. A genuine
        NEW base (different base_start/pivot) yields a different id."""
        key = "|".join([
            self.symbol.upper(), self.exchange.upper(),
            self.base_start_date, self.base_end_date,
            f"{round(float(self.pivot), 4)}",
            self.candidate_date,
            f"d{self.detector_version}", self.config_hash,
        ])
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["event_id"] = self.event_id()
        # tuples → lists for stable JSON
        for k in ("rejection_reasons", "data_quality_flags"):
            d[k] = list(d[k])
        return d

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, default=str)
