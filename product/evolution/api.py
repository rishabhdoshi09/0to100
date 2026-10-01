"""Read-only aggregation for the Evolution Lab UI surface.

One function, one job: assemble everything the Evolution Lab page needs from
already-persisted stores (policy registry, shadow-decision ledger, consensus
board). Never scans the market, never mutates anything -- a page view must
stay cheap, exactly like every other board endpoint in terminal_product_api.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from product.evolution import policy_registry as PR
from product.evolution import scorecard as SC


def _policy_summary(
    policy: dict[str, Any], *, champion_id: str | None,
    promotion_by_policy: dict[str, dict[str, Any]] | None = None,
    exposure_by_policy: dict[str, dict[str, Any]] | None = None,
    retirement_by_policy: dict[str, dict[str, Any]] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    card = SC.scorecard(policy["policy_id"], domain=policy["domain"], path=path)
    paired = None
    if champion_id and policy["policy_id"] != champion_id:
        paired = SC.paired_comparison(champion_id, policy["policy_id"], domain=policy["domain"], path=path)
    try:
        from product.evolution.promotion import latest_promotion_proof
        proof = latest_promotion_proof(policy["policy_id"])
    except Exception:
        proof = None
    since_probation = None
    if policy.get("status") == PR.PROBATION:
        try:
            from product.evolution.promotion import evaluate_probation_evidence
            since_probation = evaluate_probation_evidence(
                policy["domain"], policy["policy_id"], registry_path=path, ledger_path=path,
            )
        except Exception:
            since_probation = None
    try:
        from product.evolution.historical_priors import historical_scorecard
        historical = historical_scorecard(
            policy["policy_id"], domain=policy["domain"],
        )
    except Exception:
        historical = {
            "policy_id": policy["policy_id"],
            "domain": policy["domain"],
            "observations": 0,
            "selected": 0,
            "selected_expectancy_R": None,
            "not_promotion_evidence": True,
        }
    return {
        "policy_id": policy["policy_id"],
        "version": policy.get("version"),
        "parent_policy_id": policy.get("parent_policy_id"),
        "status": policy.get("status"),
        "hypothesis": policy.get("hypothesis"),
        "created_at": policy.get("created_at"),
        "scorecard": card,
        "paired_vs_champion": paired,
        "controls_paper_decisions": bool(policy.get("status") == PR.CHAMPION),
        "manifest_fingerprint": PR.policy_manifest_fingerprint(policy),
        "promotion_evaluation": dict(
            (promotion_by_policy or {}).get(policy["policy_id"]) or {}
        ),
        "latest_promotion_proof": proof,
        "historical_prior": historical,
        "probation": dict(policy.get("probation") or {}) or None,
        "since_probation_checkpoint": since_probation,
        "exposure": dict((exposure_by_policy or {}).get(policy["policy_id"]) or {}) or None,
        "retirement_evaluation": dict((retirement_by_policy or {}).get(policy["policy_id"]) or {}) or None,
    }


def evolution_lab_board(domain: str = PR.EQUITY, *, path: str | Path | None = None) -> dict[str, Any]:
    """Current Champion, its full challenger leaderboard, and recent
    promotion/rollback history -- everything the Evolution Lab page shows."""
    champion = PR.current_champion(domain, path=path)
    champion_id = champion["policy_id"] if champion else None
    try:
        from product.evolution.promotion import evaluate_promotion_batch
        promotion_batch = evaluate_promotion_batch(
            domain, registry_path=path, ledger_path=path, persist_proofs=False,
        )
    except Exception:
        promotion_batch = []
    promotion_by_policy = {
        str(row.get("policy_id") or ""): row for row in promotion_batch
        if row.get("policy_id")
    }
    try:
        from product.evolution.exposure import exposure_report
        exposure_by_policy = {
            row["policy_id"]: row
            for row in exposure_report(domain, registry_path=path, ledger_path=path)["rows"]
        }
    except Exception:
        exposure_by_policy = {}
    try:
        from product.evolution.promotion import evaluate_retirement_batch
        retirement_by_policy = {
            row["policy_id"]: row
            for row in evaluate_retirement_batch(domain, registry_path=path, ledger_path=path)
        }
    except Exception:
        retirement_by_policy = {}

    challengers = [
        p for p in PR.list_policies(domain=domain, path=path)
        if p["status"] in (PR.CHALLENGER, PR.SHADOW, PR.PROBATION)
    ]
    leaderboard = [
        _policy_summary(
            p, champion_id=champion_id,
            promotion_by_policy=promotion_by_policy,
            exposure_by_policy=exposure_by_policy,
            retirement_by_policy=retirement_by_policy, path=path,
        )
        for p in challengers
    ]
    leaderboard.sort(
        key=lambda row: (row["paired_vs_champion"] or {}).get("incremental_expectancy_R") or float("-inf"),
        reverse=True,
    )

    retired = [
        p for p in PR.list_policies(domain=domain, path=path)
        if p["status"] in (PR.RETIRED, PR.REJECTED)
    ]
    recent_events: list[dict[str, Any]] = []
    for p in PR.list_policies(domain=domain, path=path):
        for event in p.get("lifecycle_history") or []:
            recent_events.append({
                "policy_id": p["policy_id"], "at": event.get("at"),
                "status": event.get("status"), "reason": event.get("reason"),
            })
    recent_events.sort(key=lambda e: str(e.get("at") or ""), reverse=True)

    champion_tenure: dict[str, Any] | None = None
    if champion is not None:
        promotion_history = list(champion.get("promotion_history") or [])
        last_promotion = promotion_history[-1] if promotion_history else None
        rollback_candidates = [
            p for p in PR.list_policies(domain=domain, status=PR.PROBATION, path=path)
            if any(h.get("status") == PR.CHAMPION for h in p.get("lifecycle_history") or [])
        ]
        champion_tenure = {
            "promoted_at": (last_promotion or {}).get("at") or champion.get("created_at"),
            "promoted_by_explicit_action": bool(last_promotion),
            "rollback_target_available": bool(rollback_candidates),
            "rollback_target_policy_id": (
                rollback_candidates[0]["policy_id"] if rollback_candidates else None
            ),
        }

    return {
        "domain": domain,
        "champion": (
            {
                **_policy_summary(
                    champion, champion_id=champion_id,
                    promotion_by_policy=promotion_by_policy, path=path,
                ),
                "tenure": champion_tenure,
            } if champion else None
        ),
        "challenger_leaderboard": leaderboard,
        "retired_count": len(retired),
        "recent_events": recent_events[:20],
        "auto_promotion_enabled": _auto_promotion_enabled(),
        "live_locked": True,
        "live_execution_authorized": False,
    }


def _auto_promotion_enabled() -> bool:
    from product.evolution.promotion import AUTO_PROMOTION_ENABLED
    return AUTO_PROMOTION_ENABLED
