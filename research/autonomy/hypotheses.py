"""
🧪 Evidence-gap planner + constrained hypothesis generation.

Research begins from an OBSERVED weakness, never from random indicator combinations. A hypothesis may
change only approved dimensions of the existing frozen `StrategySpec` grammar and always produces a
NEW version (`bump_version`) — the parent is never mutated. Semantically equivalent hypotheses are
hashed and de-duplicated against `research.scientific_memory` so the system does not rediscover a
dead idea under slightly different wording. No LLM writes executable code here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# the ONLY spec dimensions a v1 hypothesis may touch (must already exist in the validated grammar)
ALLOWED_DIMENSIONS = frozenset({
    "family", "max_holding_days", "regime_conditions", "sector_conditions",
    "liquidity_conditions", "thresholds", "max_positions", "turnover_cap",
    "eligible_universe", "entry_rules", "exit_rules", "position_sizing",
})

# gap kinds → whether they call for a DATA task or a STRATEGY mutation
DATA_GAPS = frozenset({"data_insufficiency", "missing_universe_history", "missing_corporate_actions",
                       "missing_benchmark", "unsupported_runtime_family"})


@dataclass(frozen=True)
class EvidenceGap:
    kind: str
    strategy_id: str
    diagnosis: str
    economic_impact: float            # 0..1
    confidence: float                 # 0..1 confidence in the diagnosis
    data_available: bool
    data_mining_risk: float           # 0..1 (higher = worse)
    novelty: float = 0.5              # 0..1
    recommended_action: str = "strategy_mutation"   # or "data_task"

    @property
    def priority(self) -> float:
        # economic impact × diagnosis confidence × data availability × (1 - mining risk), + novelty nudge
        base = self.economic_impact * self.confidence * (1.0 if self.data_available else 0.2)
        return round(base * (1.0 - 0.5 * self.data_mining_risk) + 0.05 * self.novelty, 6)


def plan_gaps(diagnostics) -> list:
    """Turn per-strategy diagnostic dicts into ranked EvidenceGaps. A missing-DATA gap recommends a
    DATA task (never a strategy mutation). Deterministic: ties break on (kind, strategy_id)."""
    gaps: list[EvidenceGap] = []
    for d in diagnostics:
        kind = str(d.get("kind", ""))
        action = "data_task" if kind in DATA_GAPS else "strategy_mutation"
        gaps.append(EvidenceGap(
            kind=kind, strategy_id=str(d.get("strategy_id", "")),
            diagnosis=str(d.get("diagnosis", "")),
            economic_impact=float(d.get("economic_impact", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            data_available=bool(d.get("data_available", True)),
            data_mining_risk=float(d.get("data_mining_risk", 0.5)),
            novelty=float(d.get("novelty", 0.5)),
            recommended_action=action))
    gaps.sort(key=lambda g: (-g.priority, g.kind, g.strategy_id))
    return gaps


@dataclass(frozen=True)
class HypothesisProposal:
    hypothesis_id: str
    parent_strategy_id: str
    parent_version: int
    observed_problem: str
    causal_explanation: str
    changes: dict                     # dimension -> new value (allowed dimensions only)
    expected_improvement: str
    failure_condition: str
    target_regime: str
    dataset_requirements: tuple
    research_budget: int
    semantic_hash: str
    created_before_results: bool = True
    child_version: int = 0
    child_strategy_id: str = ""


def _normalise(value):
    if isinstance(value, (list, tuple)):
        return sorted(_normalise(v) for v in value)
    if isinstance(value, dict):
        return {k: _normalise(value[k]) for k in sorted(value)}
    return value


def hypothesis_hash(parent_spec, changes: dict) -> str:
    """Deterministic identity of a hypothesis = parent config identity + normalised changes.
    Two syntactically different but semantically identical changes hash the same."""
    parent_id = parent_spec.config_hash() if hasattr(parent_spec, "config_hash") else str(parent_spec)
    blob = json.dumps({"parent": parent_id, "changes": _normalise(changes)},
                      sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


_AUTO_BACKEND = object()


class ResearchMemory:
    """Thin wrapper over scientific memory plus an in-process seen set.

    Omitting ``backend`` enables the canonical persistent store. Passing ``backend=None`` is an
    explicit network/disk-free test mode.
    """

    def __init__(self, *, backend=_AUTO_BACKEND):
        self._seen: set[str] = set()
        if backend is _AUTO_BACKEND:
            try:
                import research.scientific_memory as backend  # type: ignore
            except Exception:
                backend = None
        self.backend = backend

    def _statement(self, h: str) -> str:
        return f"hypothesis:{h}"

    def is_known(self, semantic_hash: str) -> bool:
        if semantic_hash in self._seen:
            return True
        if self.backend is not None:
            try:
                if self.backend.is_known_dead(self._statement(semantic_hash)):
                    return True
            except Exception:
                pass
        return False

    def register(self, semantic_hash: str) -> None:
        self._seen.add(semantic_hash)

    def record_dead(self, semantic_hash: str, reason: str) -> None:
        self._seen.add(semantic_hash)
        if self.backend is not None:
            try:
                self.backend.record_negative(self._statement(semantic_hash), notes=reason)
            except Exception:
                pass


def propose_hypothesis(parent_spec, gap: EvidenceGap, changes: dict, *, memory: ResearchMemory,
                       expected_improvement: str = "", failure_condition: str = "",
                       target_regime: str = "any", dataset_requirements=(),
                       research_budget: int = 1) -> tuple:
    """Build a preregistered hypothesis (successor spec via bump_version) if it touches only allowed
    dimensions and is not a known-dead idea. Returns (proposal, child_spec) or (None, reason)."""
    raw_changes = dict(changes or {})
    causal = str(raw_changes.pop("_why", ""))
    bad = set(raw_changes) - ALLOWED_DIMENSIONS
    if bad:
        return None, f"changes touch non-grammar dimensions: {sorted(bad)}"
    h = hypothesis_hash(parent_spec, raw_changes)
    if memory.is_known(h):
        return None, "duplicate: an equivalent hypothesis was already tried"
    try:
        child = parent_spec.bump_version(**raw_changes)  # NEW version; parent untouched
    except (TypeError, ValueError) as exc:
        return None, f"invalid strategy-grammar change: {exc}"
    if child.config_hash() == parent_spec.config_hash():
        return None, "non-material hypothesis: result identity did not change"
    memory.register(h)
    proposal = HypothesisProposal(
        hypothesis_id=h, parent_strategy_id=parent_spec.strategy_id, parent_version=parent_spec.version,
        observed_problem=gap.diagnosis, causal_explanation=causal, changes=raw_changes,
        expected_improvement=expected_improvement, failure_condition=failure_condition,
        target_regime=target_regime, dataset_requirements=tuple(dataset_requirements),
        research_budget=int(research_budget), semantic_hash=h,
        child_version=child.version, child_strategy_id=child.strategy_id)
    return proposal, child
