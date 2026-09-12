"""Ranking that reads what the system measured, not only what it scored.

The scanner's score says how good the setup looks. This says how that kind of
setup, in this kind of tape, has actually paid — and lets the second override
the first downward.

The asymmetry is deliberate and matches the rest of the desk: measured evidence
may demote a decision, never promote one. A setup that has lost money in this
context for thirty settled trades should fall behind one that has not, but a
setup that has merely survived its own history has earned nothing.

Every ranked row carries the evidence record that moved it, so a rank change is
explainable without re-running anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from product.conditional_evidence import MIN_SAMPLE, ranking_evidence
from product.decision import Decision
from product.decision_chain import (
    confidence_bucket,
    context_key,
    extension_bucket,
    volatility_bucket,
)
from product.evidence_class import PAPER_FORWARD


def decision_context_key(decision: Decision) -> str:
    """The conditional cell this decision belongs to.

    Derived from the decision alone so that the cell a decision is ranked
    against is the same cell its outcome will later update. If these two ever
    drift apart the loop silently stops learning, so they share one function.
    """
    technical = decision.technical_evidence or {}
    return context_key(
        setup=decision.setup,
        market_regime=decision.market_state,
        sector_state=decision.sector_state,
        volatility=volatility_bucket(technical.get("atr_pct")),
        extension=extension_bucket(technical.get("pct_from_pivot")),
        confidence=confidence_bucket(decision.calibrated_confidence),
    )


@dataclass(frozen=True)
class RankedDecision:
    decision: Decision
    base_score: float
    evidence_adjustment: float
    ranking_score: float
    evidence: dict[str, Any]

    @property
    def symbol(self) -> str:
        return self.decision.symbol

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision.decision_id,
            "symbol": self.symbol,
            "state": self.decision.state,
            "base_score": self.base_score,
            "evidence_adjustment": self.evidence_adjustment,
            "ranking_score": self.ranking_score,
            "evidence": dict(self.evidence),
        }


def rank(
    decisions: Iterable[Decision],
    *,
    evidence_class: str = PAPER_FORWARD,
    path: str | Path | None = None,
    min_sample: int = MIN_SAMPLE,
) -> list[RankedDecision]:
    """Rank decisions, reading measured evidence for each one's context.

    Ties break on symbol so the order is deterministic: an unstable sort would
    make "the ranking changed" impossible to attribute.
    """
    ranked: list[RankedDecision] = []
    for decision in decisions:
        key = decision_context_key(decision)
        evidence = ranking_evidence(
            key, evidence_class=evidence_class, path=path, min_sample=min_sample
        )
        base = float(decision.score if decision.score is not None else 0.0)
        adjustment = float(evidence.get("adjustment") or 0.0)
        ranked.append(RankedDecision(
            decision=decision,
            base_score=base,
            evidence_adjustment=adjustment,
            ranking_score=round(base + adjustment, 6),
            evidence=evidence,
        ))
    ranked.sort(key=lambda r: (-r.ranking_score, r.symbol))
    return ranked


def ranking_explanation(row: RankedDecision) -> str:
    """One line a human can check, rendered from the record, not narrated."""
    reason = str(row.evidence.get("reason") or "")
    if reason == "INSUFFICIENT_EVIDENCE":
        have = row.evidence.get("count", 0)
        need = row.evidence.get("min_sample", MIN_SAMPLE)
        return (
            f"{row.symbol}: scored {row.base_score:g}; no measured edge for this "
            f"context yet ({have}/{need} settled trades), so the score stands."
        )
    if reason == "MEASURED_NOT_NEGATIVE":
        return (
            f"{row.symbol}: scored {row.base_score:g}; measured over "
            f"{row.evidence.get('count')} settled trades and not losing, which "
            "earns no bonus."
        )
    if reason == "MEASURED_NEGATIVE_EXPECTANCY":
        return (
            f"{row.symbol}: scored {row.base_score:g}, demoted "
            f"{abs(row.evidence_adjustment):g} to {row.ranking_score:g} — this "
            f"setup has lost {abs(float(row.evidence.get('expectancy_R') or 0)):.2f}R "
            f"on average over {row.evidence.get('count')} settled trades in this context."
        )
    if reason == "NOT_MARKET_EVIDENCE":
        return (
            f"{row.symbol}: scored {row.base_score:g}; the only evidence for this "
            "context is not market evidence, so ranking ignores it."
        )
    return f"{row.symbol}: scored {row.base_score:g}."
