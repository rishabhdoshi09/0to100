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
"""
from __future__ import annotations

from typing import Any, Mapping

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
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row.get("ranking_score") or 0.0),
            float((row.get("selected_contract") or {}).get("score") or 0.0),
        ),
        reverse=True,
    )
    return rows
