"""Evidence-aware ranking for F&O directional candidates.

The scan produces candidates in raw setup/contract-score order. This is the
single "rank" step both the directional scan (what the desk displays) and
the paper cycle (what actually gets a paper position opened first) consult,
so a rank change is explainable in one place instead of drifting between two
independent sort keys.

Mirrors product.decision_ranking.rank() exactly: demote-only, and gated to
evidence_class=PAPER_FORWARD only. A real settled forward paper trade may
push a proven-losing context down the list; nothing here ever promotes one,
and COUNTERFACTUAL (historical walk-forward) evidence is never consulted --
see product.fno_historical_walkforward's module docstring for why historical
replay must never move production ranking. product.conditional_evidence
.ranking_evidence() already enforces this at the store level; this module
only supplies it with the right F&O context key.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.conditional_evidence import MIN_SAMPLE, ranking_evidence
from product.evidence_class import PAPER_FORWARD
from product.fno_evidence import fno_context_key


def rank_fno_candidates(
    candidates: list[Mapping[str, Any]],
    *,
    min_sample: int = MIN_SAMPLE,
    path: str | None = None,
) -> list[dict[str, Any]]:
    """Sort candidates by evidence-adjusted score. Never mutates its input.

    Each returned row carries ``base_score`` (the raw setup score),
    ``ranking_adjustment`` and ``ranking_score`` (their sum), and
    ``ranking_evidence`` (the full record from conditional_evidence so a UI
    or report can say exactly why a rank did or did not change) alongside
    every original field.
    """
    from product.conditional_evidence import load as load_evidence_store

    # Load the evidence store once for the whole batch rather than once per
    # candidate -- this is called for every symbol's LONG/SHORT pick, for the
    # full scan's candidate list, and again every paper-cycle tick, so a
    # per-candidate re-read/re-parse of the same file adds up fast.
    store = load_evidence_store(path)
    rows: list[dict[str, Any]] = []
    for row in candidates:
        row = dict(row)
        setup = row.get("setup") if isinstance(row.get("setup"), Mapping) else {}
        base_score = float(setup.get("score") or 0.0)
        key = fno_context_key(setup)
        if key:
            evidence = ranking_evidence(
                key, evidence_class=PAPER_FORWARD, path=path, min_sample=min_sample,
                store=store,
            )
        else:
            evidence = {
                "usable": False,
                "reason": "NO_CONTEXT_KEY",
                "adjustment": 0.0,
                "evidence_class": PAPER_FORWARD,
                "context_key": "",
                "count": 0,
            }
        adjustment = float(evidence.get("adjustment") or 0.0)
        row["base_score"] = base_score
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
