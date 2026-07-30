"""
📐 Canonical schemas — the immutable, typed records the two brains speak through.

Every record is a frozen dataclass with a DETERMINISTIC record id derived from its content
(so the same facts always produce the same id — the basis of idempotent decoding), plus full
point-in-time provenance: event timestamp, optional knowledge-available timestamp, strategy id
and version, rules hash, data snapshot id, source, and schema version.

Nothing here trades, allocates, or scores — these are data. Brain 1 and Brain 2 only ever
exchange instances of these types; they never call each other directly.

Deliberately: `ResearchRationale` persists STRUCTURED reasoning only (observation, hypothesis,
supporting/conflicting evidence, decision, uncertainty, next test) — never raw chain-of-thought.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, fields

SCHEMA_VERSION = 1


def _rid(kind: str, payload: dict) -> str:
    """Deterministic record id from the semantic payload (provenance-invariant fields only).
    Same semantic content ⇒ same id ⇒ idempotent reprocessing."""
    blob = json.dumps({"k": kind, **payload}, sort_keys=True, default=str).encode()
    return f"{kind}-{hashlib.sha256(blob).hexdigest()[:16]}"


@dataclass(frozen=True)
class _Base:
    """Common provenance carried by every canonical record."""
    strategy_id: str = ""
    strategy_version: int = 0
    rules_hash: str = ""
    data_snapshot_id: str = ""
    source: str = ""
    event_ts: str = ""                 # when the event happened (point-in-time)
    knowledge_ts: str = ""             # when it became knowable (>= event_ts) if applicable
    schema_version: int = SCHEMA_VERSION
    record_id: str = ""                # deterministic; filled in __post_init__

    def _identity(self) -> dict:
        """Fields that DEFINE the record (everything except the derived id)."""
        d = asdict(self)
        d.pop("record_id", None)
        return d

    def __post_init__(self):
        if not self.record_id:
            object.__setattr__(self, "record_id",
                               _rid(type(self).__name__, self._identity()))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalSignal(_Base):
    symbol: str = ""
    direction: str = "LONG"
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    max_hold: int = 0
    rationale: str = ""                # short structured reason, not chain-of-thought


@dataclass(frozen=True)
class MarketContext(_Base):
    regime: str = "UNKNOWN"            # RISK_ON / RISK_OFF / NARROW …
    breadth: str = ""
    nifty_trend: str = ""
    vix: float = 0.0


@dataclass(frozen=True)
class StrategyDefinition(_Base):
    family: str = ""
    entry_rules: tuple = ()
    exit_rules: tuple = ()
    stop_rules: tuple = ()
    max_holding_days: int = 0
    frozen: bool = True               # rules are frozen before evaluation


@dataclass(frozen=True)
class ExecutionAssessment(_Base):
    symbol: str = ""
    intended_price: float = 0.0
    fill_price: float = 0.0
    slippage_bps: float = 0.0
    cost: float = 0.0
    exit_reason: str = ""             # STOP / TARGET / GAP_STOP / MAX_HOLD …


@dataclass(frozen=True)
class OutcomeObservation(_Base):
    symbol: str = ""
    split: str = "forward"           # in_sample / out_of_sample / forward
    realized_R: float = 0.0
    pnl: float = 0.0
    regime: str = ""
    sector: str = ""


@dataclass(frozen=True)
class ResearchRationale(_Base):
    """STRUCTURED research reasoning only — never raw chain-of-thought."""
    observation: str = ""
    hypothesis: str = ""
    supporting_evidence: tuple = ()
    conflicting_evidence: tuple = ()
    decision: str = ""
    uncertainty: str = ""
    next_test: str = ""


@dataclass(frozen=True)
class StrategyEvidenceCard(_Base):
    """Brain 1's verdict on a strategy. Immutable. Brain 2 may read but never modify it."""
    family: str = ""
    evidence_state: str = "INSUFFICIENT_EVIDENCE"
    confidence: float = 0.0
    in_sample_trades: int = 0
    out_of_sample_trades: int = 0
    forward_trades: int = 0
    expectancy_R: float = 0.0
    lower_bound_R: float = 0.0        # uncertainty-adjusted (mean − k·SE / CI lower)
    profit_factor: float | None = None
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    cost_sensitivity_R: float = 0.0
    forward_to_backtest: float = 0.0
    regime_results: dict = field(default_factory=dict)
    sector_concentration: float = 0.0
    correlation_cluster: str = ""
    evidence_freshness_days: float = 0.0
    decay_detected: bool = False
    overfit: bool = False
    lifecycle_recommendation: str = ""
    supporting_reasons: tuple = ()
    conflicting_reasons: tuple = ()
    data_quality_warnings: tuple = ()


@dataclass(frozen=True)
class PaperAllocationDecision(_Base):
    """Brain 2's allocation verdict. Immutable. References the card it acted on."""
    family: str = ""
    card_id: str = ""                 # the StrategyEvidenceCard.record_id it consumed
    action: str = "HOLD"             # DEPLOY / INCREASE / REDUCE / PAUSE / RETIRE / HOLD / SKIP
    risk_bucket: str = ""            # established / promising / exploratory
    target_risk_pct: float = 0.0
    prev_risk_pct: float = 0.0
    score: float = 0.0
    reasons: tuple = ()
    blocked_by: tuple = ()


@dataclass(frozen=True)
class LifecycleDecision(_Base):
    """A recommended lifecycle transition. USER_APPROVED is never emitted by a brain."""
    family: str = ""
    from_state: str = ""
    to_state: str = ""
    actor: str = ""                  # system / paper_autopilot — NEVER user here
    met_criteria: tuple = ()
    unmet_criteria: tuple = ()
    user_gate_required: bool = False


# registry of decodable/persistable types, keyed by class name (for reconstruction)
RECORD_TYPES = {c.__name__: c for c in (
    CanonicalSignal, MarketContext, StrategyDefinition, ExecutionAssessment,
    OutcomeObservation, ResearchRationale, StrategyEvidenceCard,
    PaperAllocationDecision, LifecycleDecision)}


def from_dict(kind: str, d: dict):
    """Rebuild a record from a persisted dict, tolerant of extra/missing keys."""
    cls = RECORD_TYPES[kind]
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in allowed})
