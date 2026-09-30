"""Evidence-aware ranking for F&O directional candidates.

The scan produces candidates in raw setup/contract-score order. This is the
single "rank" step both the directional scan (what the desk displays) and
the paper cycle (what actually gets a paper position opened first) consult,
so a rank change is explainable in one place instead of drifting between two
independent sort keys.

Ranking reads product.fno_evidence_fusion.fuse_fno_ranking_evidence, which
implements the desk's full evidence hierarchy for F&O: real settled
PAPER_FORWARD trades are the stronger vote (bounded demote AND, above a
stricter sample/confidence floor, bounded promote); historical
COUNTERFACTUAL walk-forward evidence is a much smaller, bounded prior that
can only ever act when forward evidence has nothing to say yet, or nudge a
forward-confirmed adjustment further when the two agree. Historical evidence
alone can never come close to the caps forward evidence can reach -- see
product.fno_evidence_fusion's module docstring for the exact policy table
and the reasoning behind every cap.

Contract-selection learning does NOT happen here. product.fo_options_pipeline
.evaluate_fo_opportunity is the one contract-selection seam: it re-ranks
EVERY eligible contract by learned score (raw score + contract evidence)
before an alternative is ever discarded, and ``selected_contract`` already
carries ``raw_contract_score`` / ``contract_evidence`` /
``learned_contract_score`` by the time a candidate reaches this function. By
the time ranking runs, only the one already-selected contract survives --
recomputing evidence here could not select among alternatives that no longer
exist, and would risk silently disagreeing with the real selection decision
if it used different inputs. This step only recomputes those fields as a
fallback for a candidate that never went through the real pipeline (e.g. a
hand-built fixture in a test), so a caller can always rely on them being
present.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.fno_contract_evidence import contract_ranking_adjustment
from product.fno_evidence_fusion import fuse_fno_ranking_evidence


def rank_fno_candidates(
    candidates: list[Mapping[str, Any]],
    *,
    path: str | None = None,
) -> list[dict[str, Any]]:
    """Sort candidates by evidence-adjusted score. Never mutates its input.

    Each returned row carries ``base_score`` (the raw setup score),
    ``historical_prior`` and ``forward_adjustment`` (the two bounded
    components fuse_fno_ranking_evidence computed), ``ranking_adjustment``
    (their fused, capped sum) and ``ranking_score`` (base + adjustment), plus
    ``ranking_evidence`` -- the full fusion record, so a UI or report can say
    exactly why a rank did or did not change -- alongside every original
    field.
    """
    rows: list[dict[str, Any]] = []
    for row in candidates:
        row = dict(row)
        setup = row.get("setup") if isinstance(row.get("setup"), Mapping) else {}
        base_score = float(setup.get("score") or 0.0)
        evidence = fuse_fno_ranking_evidence(setup, path=path)
        adjustment = float(evidence.get("adjustment") or 0.0)
        row["base_score"] = base_score
        row["historical_prior"] = float(evidence.get("historical_prior") or 0.0)
        row["forward_adjustment"] = float(evidence.get("forward_adjustment") or 0.0)
        row["ranking_adjustment"] = adjustment
        row["ranking_score"] = round(base_score + adjustment, 4)
        row["ranking_evidence"] = evidence

        contract = row.get("selected_contract")
        if isinstance(contract, Mapping):
            contract = dict(contract)
            if "learned_contract_score" not in contract:
                # Fallback only -- the real seam (product.fo_options_pipeline
                # .evaluate_fo_opportunity) already computes this using the
                # real setup context (holding horizon/regime/setup type) at
                # selection time, among every eligible contract, not just
                # this one. A candidate that reaches here without it never
                # went through that seam (e.g. a hand-built test fixture).
                contract_evidence = contract_ranking_adjustment(contract, path=path)
                raw_contract_score = float(contract.get("score") or 0.0)
                contract_adjustment = float(contract_evidence.get("adjustment") or 0.0)
                contract["contract_evidence"] = contract_evidence
                contract["raw_contract_score"] = raw_contract_score
                contract["learned_contract_score"] = round(
                    raw_contract_score + contract_adjustment, 4
                )
            row["selected_contract"] = contract
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row.get("ranking_score") or 0.0),
            float(
                (row.get("selected_contract") or {}).get("learned_contract_score")
                if (row.get("selected_contract") or {}).get("learned_contract_score") is not None
                else (row.get("selected_contract") or {}).get("score") or 0.0
            ),
        ),
        reverse=True,
    )
    return rows
