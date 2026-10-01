"""Phase 8: one-click scientific explainability for a promotion candidate.

Answers "why does QuantTerm believe this policy deserves PAPER authority"
(or why it doesn't, yet) without reading source code. Every field here is
reused from an existing computation (promotion.py, scorecard.py,
policy_registry.py, auto_promotion_readiness.py) -- this module invents no
new statistic, it only assembles and serializes what those modules already
compute internally into one read-only response.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from product.evolution import policy_registry, promotion, scorecard
from product.evolution.auto_promotion_readiness import evaluate_auto_promotion_readiness


def _manifest(policy: dict[str, Any] | None) -> dict[str, Any]:
    if not policy:
        return {}
    policy = dict(policy)
    return {
        "policy_id": policy.get("policy_id"),
        "domain": policy.get("domain"),
        "version": policy.get("version"),
        "parent_policy_id": policy.get("parent_policy_id"),
        "hypothesis": policy.get("hypothesis"),
        "weights": dict(policy.get("weights") or {}),
        "status": policy.get("status"),
        "created_at": policy.get("created_at"),
        "fingerprint": policy_registry.policy_manifest_fingerprint(policy),
    }


def _paired_snapshot_detail(
    champion_policy_id: str, challenger_policy_id: str,
    *, domain: str, ledger_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    champ = {
        r["market_snapshot_id"]: r
        for r in scorecard.graded_rows(champion_policy_id, domain=domain, path=ledger_path)
    }
    chal = {
        r["market_snapshot_id"]: r
        for r in scorecard.graded_rows(challenger_policy_id, domain=domain, path=ledger_path)
    }
    out = []
    for sid in sorted(set(champ) & set(chal)):
        c, h = champ[sid], chal[sid]
        out.append({
            "market_snapshot_id": sid,
            "symbol": h.get("symbol") or c.get("symbol"),
            "regime": h.get("regime") or c.get("regime"),
            "sector": h.get("sector") or c.get("sector"),
            "champion_decision": c.get("decision"),
            "champion_classification": c.get("classification"),
            "champion_evidence_class": c.get("evidence_class"),
            "challenger_decision": h.get("decision"),
            "challenger_classification": h.get("classification"),
            "challenger_evidence_class": h.get("evidence_class"),
        })
    return out


def explain_promotion_candidate(
    domain: str, challenger_policy_id: str,
    *, registry_path: str | Path | None = None,
    ledger_path: str | Path | None = None, proof_path: str | Path | None = None,
) -> dict[str, Any]:
    """Everything an operator (or auditor) needs to answer "why does this
    policy deserve -- or not yet deserve -- PAPER authority" in one place."""
    champion = policy_registry.current_champion(domain, path=registry_path)
    policy = policy_registry.get_policy(challenger_policy_id, path=registry_path)
    champion_id = champion["policy_id"] if champion else ""

    eligibility = promotion.evaluate_promotion(
        domain, challenger_policy_id, registry_path=registry_path, ledger_path=ledger_path,
    )
    pairs = (
        _paired_snapshot_detail(champion_id, challenger_policy_id, domain=domain, ledger_path=ledger_path)
        if champion_id else []
    )
    regime_distribution = dict(Counter(str(p.get("regime") or "UNKNOWN") for p in pairs))
    sector_distribution = dict(Counter(str(p.get("sector") or "UNKNOWN") for p in pairs))
    non_forward_evidence = [
        p for p in pairs
        if str(p.get("champion_evidence_class") or "") not in ("EVOLUTION_SHADOW", "PAPER_FORWARD")
        or str(p.get("challenger_evidence_class") or "") not in ("EVOLUTION_SHADOW", "PAPER_FORWARD")
    ]

    lifecycle = list((policy or {}).get("lifecycle_history") or [])
    probation = dict((policy or {}).get("probation") or {})
    promotion_history = list((policy or {}).get("promotion_history") or [])

    readiness = (
        evaluate_auto_promotion_readiness(
            domain, challenger_policy_id, registry_path=registry_path, ledger_path=ledger_path,
        )
        if policy else None
    )

    return {
        "domain": domain,
        "policy_manifest": _manifest(policy),
        "champion_manifest": _manifest(champion),
        "scientific_eligibility": eligibility,
        "paired_snapshot_ids": [p["market_snapshot_id"] for p in pairs],
        "paired_snapshot_detail": pairs,
        "paired_sample_size": len(pairs),
        "regime_distribution": regime_distribution,
        "sector_distribution": sector_distribution,
        "evidence_integrity": {
            "all_forward_evidence": not non_forward_evidence,
            "non_forward_evidence_count": len(non_forward_evidence),
            "note": (
                "Every paired observation is EVOLUTION_SHADOW/PAPER_FORWARD -- "
                "promotion never reads product.evolution.historical_priors, "
                "so historical/counterfactual evidence cannot appear here."
                if not non_forward_evidence
                else "WARNING: non-forward evidence_class found in paired evidence."
            ),
        },
        "lifecycle_history": lifecycle,
        "probation": probation or None,
        "promotion_history": promotion_history,
        "latest_promotion_proof": promotion.latest_promotion_proof(
            challenger_policy_id, path=proof_path,
        ),
        "auto_promotion_readiness": readiness,
    }
