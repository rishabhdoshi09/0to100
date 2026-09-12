"""The spine that links a decision to what it eventually taught the system.

A statistic is only as trustworthy as its ability to name the trades that made
it. "This setup wins 61% of the time" is worth nothing if nobody can list the
trades, and worth a great deal if every one of them can be walked back to the
decision that opened it and the bar that closed it.

So each hop keeps the id of the hop before it:

    Decision         decision_id
    PaperIntent      intent_id          + decision_id
    PaperOrder       order_id           + paper_intent_id, decision_id
    PaperPositionRef position_id        + paper_order_id, paper_intent_id, decision_id
    Outcome          outcome_id         + position_id .. decision_id
    EvidenceUpdate   evidence_update_id + outcome_id .. decision_id

:func:`validate_chain` is the invariant: a chain with a broken link is a
statistic whose provenance cannot be reconstructed, and the system refuses it
rather than counting it.

Nothing here places an order. ``PaperOrder`` is a record of a simulated fill;
there is no code path from this module to a broker.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from product.evidence_class import normalise as normalise_evidence_class

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _derive(prefix: str, *parts: Any) -> str:
    material = json.dumps([str(p) for p in parts], separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


class BrokenChain(ValueError):
    """Raised when a record cannot name the decision it descends from."""


@dataclass(frozen=True)
class PaperIntent:
    """The desk deciding to take a decision. Still no fill, still no position."""

    decision_id: str
    symbol: str
    side: str = "BUY"
    qty: int = 0
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    created_at: str = ""
    evidence_class: str = ""
    reason: str = ""
    intent_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.decision_id or "").strip():
            raise BrokenChain("a paper intent must name the decision it came from")
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        if not self.created_at:
            object.__setattr__(self, "created_at", _now())
        if self.evidence_class:
            object.__setattr__(
                self, "evidence_class", normalise_evidence_class(self.evidence_class)
            )
        if not self.intent_id:
            object.__setattr__(self, "intent_id", _derive(
                "int", self.decision_id, self.symbol, self.side, self.created_at,
                self.entry, self.stop, self.target,
            ))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperOrder:
    """A simulated order and its simulated fill."""

    paper_intent_id: str
    decision_id: str
    symbol: str
    qty: int = 0
    status: str = "FILLED"
    placed_at: str = ""
    filled_at: str = ""
    fill_price: float | None = None
    order_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.paper_intent_id or "").strip():
            raise BrokenChain("a paper order must name its intent")
        if not str(self.decision_id or "").strip():
            raise BrokenChain("a paper order must name its decision")
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        if not self.placed_at:
            object.__setattr__(self, "placed_at", _now())
        if not self.order_id:
            object.__setattr__(self, "order_id", _derive(
                "ord", self.paper_intent_id, self.symbol, self.qty, self.placed_at,
            ))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperPositionRef:
    """The identity of an open paper position, carried alongside the book row.

    The book itself stays a pure simulator: this is the record that knows which
    decision the position came from.
    """

    paper_order_id: str
    paper_intent_id: str
    decision_id: str
    symbol: str
    strategy_id: str = ""
    opened_at: str = ""
    position_id: str = ""

    def __post_init__(self) -> None:
        for name in ("paper_order_id", "paper_intent_id", "decision_id"):
            if not str(getattr(self, name) or "").strip():
                raise BrokenChain(f"a paper position must name its {name}")
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        if not self.opened_at:
            object.__setattr__(self, "opened_at", _now())
        if not self.position_id:
            object.__setattr__(self, "position_id", _derive(
                "pos", self.paper_order_id, self.symbol, self.opened_at,
            ))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Outcome:
    """What the market did to the position, settled on completed sessions."""

    position_id: str
    paper_order_id: str
    paper_intent_id: str
    decision_id: str
    symbol: str
    exit_reason: str = ""
    realized_R: float | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    entry_session: str = ""
    exit_session: str = ""
    bars_held: int = 0
    mae_R: float | None = None
    mfe_R: float | None = None
    evidence_class: str = ""
    resolved_at: str = ""
    outcome_id: str = ""

    def __post_init__(self) -> None:
        for name in ("position_id", "paper_order_id", "paper_intent_id", "decision_id"):
            if not str(getattr(self, name) or "").strip():
                raise BrokenChain(f"an outcome must name its {name}")
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        if not self.resolved_at:
            object.__setattr__(self, "resolved_at", _now())
        if self.evidence_class:
            object.__setattr__(
                self, "evidence_class", normalise_evidence_class(self.evidence_class)
            )
        if not self.outcome_id:
            object.__setattr__(self, "outcome_id", _derive(
                "out", self.position_id, self.exit_session, self.exit_reason,
                self.realized_R,
            ))

    @property
    def is_win(self) -> bool | None:
        if self.realized_R is None:
            return None
        return float(self.realized_R) > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceUpdate:
    """The belief change one outcome caused, with the before and after kept.

    Storing both sides is what makes the loop auditable rather than asserted:
    a reader can see that this outcome, and not some later rewrite, is what
    moved the number.
    """

    outcome_id: str
    position_id: str
    decision_id: str
    context_key: str
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    evidence_class: str = ""
    updated_at: str = ""
    evidence_update_id: str = ""

    def __post_init__(self) -> None:
        for name in ("outcome_id", "position_id", "decision_id"):
            if not str(getattr(self, name) or "").strip():
                raise BrokenChain(f"an evidence update must name its {name}")
        if not str(self.context_key or "").strip():
            raise BrokenChain("an evidence update must name the context it changed")
        if not self.updated_at:
            object.__setattr__(self, "updated_at", _now())
        if self.evidence_class:
            object.__setattr__(
                self, "evidence_class", normalise_evidence_class(self.evidence_class)
            )
        if not self.evidence_update_id:
            object.__setattr__(self, "evidence_update_id", _derive(
                "evu", self.outcome_id, self.context_key, self.updated_at,
            ))

    @property
    def changed(self) -> bool:
        return self.before != self.after

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── conditional context ────────────────────────────────────────────────────
def volatility_bucket(atr_pct: float | None) -> str:
    """ATR as a percentage of price. Unknown stays unknown."""
    if atr_pct is None:
        return "UNKNOWN"
    value = float(atr_pct)
    if value < 2.0:
        return "LOW_VOL"
    if value < 4.0:
        return "MID_VOL"
    return "HIGH_VOL"


def extension_bucket(pct_from_pivot: float | None) -> str:
    """How far above the trigger the entry was taken. Chase risk, bucketed."""
    if pct_from_pivot is None:
        return "UNKNOWN"
    value = float(pct_from_pivot)
    if value <= 2.0:
        return "AT_PIVOT"
    if value <= 5.0:
        return "SLIGHTLY_EXTENDED"
    return "EXTENDED"


def confidence_bucket(confidence: float | None) -> str:
    """Calibration buckets: said 70%, did 70% happen?"""
    if confidence is None:
        return "UNKNOWN"
    value = float(confidence)
    if value < 0.4:
        return "CONF_LT_40"
    if value < 0.6:
        return "CONF_40_60"
    if value < 0.8:
        return "CONF_60_80"
    return "CONF_GTE_80"


def context_key(
    *,
    setup: str = "",
    market_regime: str = "",
    sector_state: str = "",
    volatility: str = "",
    extension: str = "",
    confidence: str = "",
) -> str:
    """The conditional cell an outcome updates.

    Deliberately explicit rather than a free-form dict: a statistic keyed by a
    dict nobody can reproduce is a statistic nobody can check.
    """
    parts = [
        f"setup={setup or 'UNKNOWN'}",
        f"regime={market_regime or 'UNKNOWN'}",
        f"sector={sector_state or 'UNKNOWN'}",
        f"vol={volatility or 'UNKNOWN'}",
        f"ext={extension or 'UNKNOWN'}",
        f"conf={confidence or 'UNKNOWN'}",
    ]
    return "|".join(parts)


# ── the invariant ──────────────────────────────────────────────────────────
def validate_chain(
    *,
    decision_id: str,
    intent: PaperIntent | None = None,
    order: PaperOrder | None = None,
    position: PaperPositionRef | None = None,
    outcome: Outcome | None = None,
    update: EvidenceUpdate | None = None,
) -> None:
    """Raise if any hop fails to name the hop before it.

    Called wherever an outcome is about to become a statistic. A chain that
    cannot be walked back to its decision is not counted.
    """
    if not str(decision_id or "").strip():
        raise BrokenChain("chain has no decision")
    if intent is not None and intent.decision_id != decision_id:
        raise BrokenChain(
            f"intent {intent.intent_id} names decision {intent.decision_id!r}, "
            f"not {decision_id!r}"
        )
    if order is not None:
        if intent is not None and order.paper_intent_id != intent.intent_id:
            raise BrokenChain(
                f"order {order.order_id} names intent {order.paper_intent_id!r}, "
                f"not {intent.intent_id!r}"
            )
        if order.decision_id != decision_id:
            raise BrokenChain(f"order {order.order_id} names a different decision")
    if position is not None:
        if order is not None and position.paper_order_id != order.order_id:
            raise BrokenChain(f"position {position.position_id} names a different order")
        if position.decision_id != decision_id:
            raise BrokenChain(f"position {position.position_id} names a different decision")
    if outcome is not None:
        if position is not None and outcome.position_id != position.position_id:
            raise BrokenChain(f"outcome {outcome.outcome_id} names a different position")
        if outcome.decision_id != decision_id:
            raise BrokenChain(f"outcome {outcome.outcome_id} names a different decision")
    if update is not None:
        if outcome is not None and update.outcome_id != outcome.outcome_id:
            raise BrokenChain(
                f"evidence update {update.evidence_update_id} names a different outcome"
            )
        if update.decision_id != decision_id:
            raise BrokenChain(
                f"evidence update {update.evidence_update_id} names a different decision"
            )


def chain_ids(
    *,
    decision_id: str,
    intent: PaperIntent | None = None,
    order: PaperOrder | None = None,
    position: PaperPositionRef | None = None,
    outcome: Outcome | None = None,
    update: EvidenceUpdate | None = None,
) -> dict[str, str]:
    """Every id in one flat record, for persisting next to a statistic."""
    return {
        "decision_id": decision_id,
        "paper_intent_id": intent.intent_id if intent else "",
        "paper_order_id": order.order_id if order else "",
        "position_id": position.position_id if position else "",
        "outcome_id": outcome.outcome_id if outcome else "",
        "evidence_update_id": update.evidence_update_id if update else "",
    }
