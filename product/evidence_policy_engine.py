"""Canonical EvidencePolicyEngine.

Consumes frozen learning policies plus the candidate's point-in-time evidence.
Outputs SUPPORT / NEUTRAL / PENALIZE / BLOCK. Never emits BUY.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.learning_policy_store import (
    ACTIVE,
    ELIGIBLE,
    EXPERIMENTAL,
    load_policies,
)

SUPPORT = "SUPPORT"
NEUTRAL = "NEUTRAL"
PENALIZE = "PENALIZE"
BLOCK = "BLOCK"

_EFFECT_RANK = {BLOCK: 3, PENALIZE: 2, SUPPORT: 1, NEUTRAL: 0}


def _bucket_of(candidate: Mapping[str, Any], dimension: str) -> str:
    if dimension == "reason_code":
        return str(candidate.get("reason_code") or candidate.get("last_reject_reason") or "")
    if dimension == "setup":
        return str(candidate.get("setup_label") or candidate.get("primary_thesis") or "")
    if dimension == "entry_state":
        return str(candidate.get("entry_state") or "")
    if dimension == "tier":
        return str(candidate.get("reco_tier") or "")
    if dimension == "sector":
        return str(candidate.get("sector") or "")
    if dimension == "extension":
        try:
            ext = float(candidate.get("extension_pct") or 0)
        except (TypeError, ValueError):
            return ""
        if ext > 8:
            return "extended_gt_8"
        if ext > 4:
            return "extended_4_8"
        return "extended_le_4"
    return str(candidate.get(dimension) or "")


def evaluate_policies(
    candidate: Mapping[str, Any],
    *,
    policies: Sequence[Mapping[str, Any]] | None = None,
    path=None,
) -> dict[str, Any]:
    """Return the empirical overlay for one candidate. Cannot create a BUY."""
    if policies is None:
        policies = list((load_policies(path).get("policies") or []))
    supportive: list[dict[str, Any]] = []
    cautionary: list[dict[str, Any]] = []
    blocking: list[dict[str, Any]] = []
    sample = 0
    coverage = 0
    final = NEUTRAL
    learned_edge = 0.0

    for raw in policies:
        policy = dict(raw)
        status = str(policy.get("production_status") or "")
        if status not in {ACTIVE, ELIGIBLE, EXPERIMENTAL}:
            continue
        dimension = str(policy.get("dimension") or "")
        bucket = str(policy.get("bucket") or "")
        if not dimension or not bucket:
            continue
        if _bucket_of(candidate, dimension) != bucket:
            continue
        coverage += 1
        n = int(policy.get("sample_size") or 0)
        sample = max(sample, n)
        edge = float(policy.get("expectancy_difference_R") or 0.0)
        learned_edge += edge
        confidence = str(policy.get("confidence") or "")
        if confidence == "INSUFFICIENT_EVIDENCE" and status != ACTIVE:
            cautionary.append({**policy, "effect": NEUTRAL, "note": "INSUFFICIENT_EVIDENCE"})
            continue
        if status == ACTIVE and edge <= -0.40:
            effect = BLOCK
            blocking.append({**policy, "effect": effect})
        elif edge <= -0.20 and status in {ACTIVE, ELIGIBLE}:
            effect = PENALIZE
            cautionary.append({**policy, "effect": effect})
        elif edge >= 0.25 and status in {ACTIVE, ELIGIBLE}:
            effect = SUPPORT
            supportive.append({**policy, "effect": effect})
        else:
            effect = NEUTRAL
            cautionary.append({**policy, "effect": effect})
        if _EFFECT_RANK[effect] > _EFFECT_RANK[final]:
            final = effect

    return {
        "supportive": supportive,
        "cautionary": cautionary,
        "blocking": blocking,
        "learned_edge_score": round(learned_edge, 4),
        "evidence_coverage": coverage,
        "sample_size": sample,
        "confidence": (
            "INSUFFICIENT_EVIDENCE" if coverage == 0 or sample < 8
            else "MEASURED"
        ),
        "final_effect": final,
        "invents_buy": False,
    }
