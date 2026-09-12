"""The canonical trading decision.

Before this module a decision about one stock existed as at least a dozen
different dictionaries: a scanner row, a reco card, a compacted ledger entry, a
paper intent, a journal record, an outcome row. Each subsystem read the keys it
knew and quietly ignored the rest, so "why did the desk say BUY on this name"
had no single answer — it had one answer per subsystem, and they could disagree
without anything noticing.

:class:`Decision` is that single answer. It is deterministic, serialisable,
and carries its own provenance: which scan produced it, which evidence snapshot
it was reasoning over, which engine and schema versions were in force, and
which class of evidence its confidence rests on.

Three things it deliberately does NOT do:

* It does not compute anything. Scoring lives in the scanner and the ranker;
  this is the record of what they decided, not a second opinion.
* It does not fill gaps. Evidence the system could not obtain is listed in
  ``missing_evidence`` and stays missing. A decision that cannot see the
  fundamentals says so rather than scoring them as neutral.
* It does not explain itself in prose. The UI renders the structure. An
  LLM may narrate it, but the narration is never the authority.

The linkage fields are the spine of the learning loop:

    Decision.decision_id
      -> PaperIntent.decision_id
        -> PaperOrder.paper_intent_id
          -> PaperPosition.paper_order_id
            -> Outcome.position_id
              -> EvidenceUpdate.outcome_id

Every hop keeps the id of the hop before it, so an evidence number months later
can be walked all the way back to the bar that produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from product.evidence_class import (
    HISTORICAL_REPLAY,
    PAPER_FORWARD,
    TEST_FIXTURE,
    normalise as normalise_evidence_class,
)

SCHEMA_VERSION = 1

# ── decision states ────────────────────────────────────────────────────────
BUY = "BUY"
WATCH = "WATCH"
WAIT = "WAIT"
AVOID = "AVOID"
NO_TRADE = "NO_TRADE"

STATES = (BUY, WATCH, WAIT, AVOID, NO_TRADE)

#: Only these states may create a paper intent. WATCH is a reminder, not a
#: trade; NO_TRADE is the desk declining to act and must stay actionless.
ACTIONABLE_STATES = frozenset({BUY})

# ── evidence direction ─────────────────────────────────────────────────────
SUPPORTING = "SUPPORTING"
CONFLICTING = "CONFLICTING"
MISSING = "MISSING"

DIRECTIONS = (SUPPORTING, CONFLICTING, MISSING)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> Any:
    """JSON-safe, order-stable representation for hashing and persistence."""
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class EvidenceItem:
    """One named thing the decision knows, or knows it does not know.

    ``direction`` is the whole point. A card that lists ten facts tells the
    reader nothing about which of them argued against the trade. Evidence that
    conflicts with the decision is as much a part of the decision as evidence
    that supports it, and a decision taken with three conflicts is a different
    decision from the same score with none.
    """

    id: str
    label: str = ""
    direction: str = SUPPORTING
    detail: str = ""
    value: Any = None
    source: str = ""
    as_of: str = ""
    #: How this particular fact was produced. A setup whose win-rate comes from
    #: a backtest and one whose win-rate comes from settled paper trades are
    #: not making the same claim.
    evidence_class: str = ""

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"unknown evidence direction: {self.direction!r}")
        if not str(self.id or "").strip():
            raise ValueError("evidence item needs an id")

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass(frozen=True)
class Decision:
    """One stock, one moment, one authoritative verdict."""

    symbol: str
    state: str

    decision_id: str = ""
    generated_at: str = ""

    setup: str = ""

    # ── conviction ─────────────────────────────────────────────────────────
    score: float | None = None
    calibrated_confidence: float | None = None
    expected_value: float | None = None

    # ── context ────────────────────────────────────────────────────────────
    market_state: str = ""
    sector_state: str = ""

    # ── evidence, by family ────────────────────────────────────────────────
    technical_evidence: dict[str, Any] = field(default_factory=dict)
    fundamental_evidence: dict[str, Any] = field(default_factory=dict)
    news_evidence: dict[str, Any] = field(default_factory=dict)
    historical_evidence: dict[str, Any] = field(default_factory=dict)

    # ── evidence, by direction (what the UI actually renders) ──────────────
    supporting_evidence: tuple[EvidenceItem, ...] = ()
    conflicting_evidence: tuple[EvidenceItem, ...] = ()
    missing_evidence: tuple[EvidenceItem, ...] = ()

    # ── the trade ──────────────────────────────────────────────────────────
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    expected_R: float | None = None

    chase_risk: str = ""
    liquidity_state: str = ""

    portfolio_effect: dict[str, Any] = field(default_factory=dict)
    position_size: dict[str, Any] = field(default_factory=dict)

    invalidation_conditions: tuple[str, ...] = ()

    # ── provenance ─────────────────────────────────────────────────────────
    source_scan_id: str = ""
    evidence_snapshot_id: str = ""

    decision_engine_version: str = ""
    feature_schema_version: str = ""
    strategy_version: str = ""

    evidence_class: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(
                f"unknown decision state {self.state!r}; expected one of {STATES}"
            )
        if not str(self.symbol or "").strip():
            raise ValueError("a decision needs a symbol")
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        if not self.generated_at:
            object.__setattr__(self, "generated_at", _utc_now())
        if self.evidence_class:
            normalised = normalise_evidence_class(self.evidence_class)
            if not normalised:
                raise ValueError(f"unknown evidence class: {self.evidence_class!r}")
            object.__setattr__(self, "evidence_class", normalised)
        if not self.decision_id:
            object.__setattr__(self, "decision_id", self.derive_id())

    # ── identity ───────────────────────────────────────────────────────────
    def derive_id(self) -> str:
        """Deterministic id over what makes this decision *this* decision.

        Two runs of the same scan over the same data produce the same id, so a
        re-publish is recognisable as the same decision rather than a new one.
        Prices move and scores drift, so they are part of the identity; the
        evidence lists are not — enriching a decision does not make it another.
        """
        material = _clean({
            "symbol": self.symbol,
            "state": self.state,
            "setup": self.setup,
            "generated_at": self.generated_at,
            "source_scan_id": self.source_scan_id,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "score": self.score,
            "strategy_version": self.strategy_version,
        })
        digest = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"dec_{digest[:24]}"

    # ── derived truths ─────────────────────────────────────────────────────
    @property
    def is_actionable(self) -> bool:
        return self.state in ACTIONABLE_STATES

    @property
    def may_open_paper_position(self) -> bool:
        """A decision may reach the paper book only if it is actionable AND
        carries a real exit. A trade with no stop is not a trade, it is a hope.
        """
        return bool(
            self.is_actionable
            and self.entry is not None
            and self.stop is not None
            and float(self.stop) != float(self.entry)
        )

    @property
    def risk_per_share(self) -> float | None:
        if self.entry is None or self.stop is None:
            return None
        return abs(float(self.entry) - float(self.stop))

    @property
    def computed_expected_R(self) -> float | None:
        """R implied by the levels, independent of whatever was passed in."""
        risk = self.risk_per_share
        if not risk or self.target is None or self.entry is None:
            return None
        return (float(self.target) - float(self.entry)) / risk

    @property
    def is_market_evidence(self) -> bool:
        from product.evidence_class import is_market_evidence

        return is_market_evidence(self.evidence_class)

    def evidence_counts(self) -> dict[str, int]:
        return {
            SUPPORTING: len(self.supporting_evidence),
            CONFLICTING: len(self.conflicting_evidence),
            MISSING: len(self.missing_evidence),
        }

    # ── serialisation ──────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "generated_at": self.generated_at,
            "state": self.state,
            "setup": self.setup,
            "score": self.score,
            "calibrated_confidence": self.calibrated_confidence,
            "expected_value": self.expected_value,
            "market_state": self.market_state,
            "sector_state": self.sector_state,
            "technical_evidence": _clean(self.technical_evidence),
            "fundamental_evidence": _clean(self.fundamental_evidence),
            "news_evidence": _clean(self.news_evidence),
            "historical_evidence": _clean(self.historical_evidence),
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "conflicting_evidence": [e.to_dict() for e in self.conflicting_evidence],
            "missing_evidence": [e.to_dict() for e in self.missing_evidence],
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "expected_R": self.expected_R,
            "chase_risk": self.chase_risk,
            "liquidity_state": self.liquidity_state,
            "portfolio_effect": _clean(self.portfolio_effect),
            "position_size": _clean(self.position_size),
            "invalidation_conditions": list(self.invalidation_conditions),
            "source_scan_id": self.source_scan_id,
            "evidence_snapshot_id": self.evidence_snapshot_id,
            "decision_engine_version": self.decision_engine_version,
            "feature_schema_version": self.feature_schema_version,
            "strategy_version": self.strategy_version,
            "evidence_class": self.evidence_class,
            "provenance": _clean(self.provenance),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Decision":
        def items(key: str, direction: str) -> tuple[EvidenceItem, ...]:
            out = []
            for row in payload.get(key) or []:
                if not isinstance(row, Mapping):
                    continue
                data = dict(row)
                data.setdefault("direction", direction)
                data.setdefault("id", data.get("label") or key)
                out.append(EvidenceItem(**{
                    k: v for k, v in data.items()
                    if k in EvidenceItem.__dataclass_fields__
                }))
            return tuple(out)

        return cls(
            symbol=str(payload.get("symbol") or ""),
            state=str(payload.get("state") or ""),
            decision_id=str(payload.get("decision_id") or ""),
            generated_at=str(payload.get("generated_at") or ""),
            setup=str(payload.get("setup") or ""),
            score=payload.get("score"),
            calibrated_confidence=payload.get("calibrated_confidence"),
            expected_value=payload.get("expected_value"),
            market_state=str(payload.get("market_state") or ""),
            sector_state=str(payload.get("sector_state") or ""),
            technical_evidence=dict(payload.get("technical_evidence") or {}),
            fundamental_evidence=dict(payload.get("fundamental_evidence") or {}),
            news_evidence=dict(payload.get("news_evidence") or {}),
            historical_evidence=dict(payload.get("historical_evidence") or {}),
            supporting_evidence=items("supporting_evidence", SUPPORTING),
            conflicting_evidence=items("conflicting_evidence", CONFLICTING),
            missing_evidence=items("missing_evidence", MISSING),
            entry=payload.get("entry"),
            stop=payload.get("stop"),
            target=payload.get("target"),
            expected_R=payload.get("expected_R"),
            chase_risk=str(payload.get("chase_risk") or ""),
            liquidity_state=str(payload.get("liquidity_state") or ""),
            portfolio_effect=dict(payload.get("portfolio_effect") or {}),
            position_size=dict(payload.get("position_size") or {}),
            invalidation_conditions=tuple(payload.get("invalidation_conditions") or ()),
            source_scan_id=str(payload.get("source_scan_id") or ""),
            evidence_snapshot_id=str(payload.get("evidence_snapshot_id") or ""),
            decision_engine_version=str(payload.get("decision_engine_version") or ""),
            feature_schema_version=str(payload.get("feature_schema_version") or ""),
            strategy_version=str(payload.get("strategy_version") or ""),
            evidence_class=str(payload.get("evidence_class") or ""),
            provenance=dict(payload.get("provenance") or {}),
            schema_version=int(payload.get("schema_version") or SCHEMA_VERSION),
        )

    def with_evidence(
        self,
        *,
        supporting: Sequence[EvidenceItem] = (),
        conflicting: Sequence[EvidenceItem] = (),
        missing: Sequence[EvidenceItem] = (),
    ) -> "Decision":
        """Enrichment returns a new decision and keeps the same id."""
        return replace(
            self,
            supporting_evidence=self.supporting_evidence + tuple(supporting),
            conflicting_evidence=self.conflicting_evidence + tuple(conflicting),
            missing_evidence=self.missing_evidence + tuple(missing),
            decision_id=self.decision_id,
        )
