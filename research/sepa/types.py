"""Canonical SEPA-001 eligibility object — deterministic and serializable."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonish(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(v) for v in value]
    return str(value)


@dataclass
class RuleResult:
    id: str
    passed: bool | None
    detail: str
    values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "passed": self.passed,
            "detail": self.detail,
            "values": _jsonish(self.values),
        }


@dataclass
class SepaEligibility:
    """Answer: is this name a SEPA-style *trade* at this as-of date?

    Distinguishes good stock / good setup / good entry / eligible.
    A great stock with a chased print is eligible=False.
    """

    symbol: str
    as_of_date: str
    data_timestamp: str
    eligibility_version: str
    config_hash: str
    universe_version: str = ""

    trend_rules: list[RuleResult] = field(default_factory=list)
    trend_template_pass: bool = False
    structure_pass: bool = False
    trend_passed: int = 0
    trend_total: int = 8
    levels: dict[str, Any] = field(default_factory=dict)

    rs_score: float | None = None
    rs_percentile: float | None = None
    rs_threshold: float = 70.0
    rs_pass: bool = False
    rs_components: dict[str, Any] = field(default_factory=dict)
    benchmark_rs: dict[str, Any] = field(default_factory=dict)

    setup_type: str = ""
    vcp_detected: bool = False
    contraction_count: int = 0
    contraction_depths: list[float] = field(default_factory=list)
    contraction_dates: list[str] = field(default_factory=list)
    contraction_durations: list[int] = field(default_factory=list)
    base_depth_pct: float | None = None
    final_contraction_pct: float | None = None
    tightness: float | None = None
    vol_first: float | None = None
    vol_final: float | None = None
    vol_recent_vs_base: float | None = None
    dry_up_ratio: float | None = None
    setup_quality: float | None = None
    setup_fail_reasons: list[str] = field(default_factory=list)

    pivot: float | None = None
    pivot_date: str | None = None
    pivot_type: str | None = None
    price: float | None = None
    distance_from_pivot_pct: float | None = None
    buy_zone_low: float | None = None
    buy_zone_high: float | None = None
    entry_valid: bool = False
    entry_rejection: str | None = None
    extended: bool = False
    proposed_entry: float | None = None

    structural_stop: float | None = None
    stop_basis: str | None = None
    stop_distance_pct: float | None = None
    atr: float | None = None
    stop_atr_multiple: float | None = None
    stop_ok: bool = False
    risk_r: float | None = None

    measured_move: float | None = None
    reward_price: float | None = None
    reward_risk: float | None = None
    reward_status: str = "UNKNOWN"
    resistance: dict[str, Any] = field(default_factory=dict)

    good_stock: bool = False
    good_setup: bool = False
    good_entry: bool = False
    eligible: bool = False
    rejection_codes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    headline: str = ""

    pit_safe: bool = False
    universe_complete: bool = False
    ca_complete: bool = False
    research_grade: bool = False
    pit_class: str = "PIT_UNVERIFIED"
    vcp_state: str = ""
    setup_id: str = ""
    pivot_knowable_date: str | None = None
    vcp_knowable_date: str | None = None
    base_start_date: str | None = None
    pivot_version: str = ""
    vcp_version: str = ""
    original_base_start: str | None = None
    left_censored: bool = False
    lifecycle_status: str = ""
    universe_date: str = ""
    candidate_count: int | None = None
    investable_count: int | None = None
    rs_denominator: int | None = None
    membership_hash: str = ""
    selection_reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["trend_rules"] = [r.to_dict() if isinstance(r, RuleResult) else r for r in self.trend_rules]
        return _jsonish(raw)

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
