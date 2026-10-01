"""Phase 7: a clearly separate `PAPER_AUTO_PROMOTION_READY` readiness state.

This is NOT a second promotion mechanism and it never mutates anything or
flips AUTO_PROMOTION_ENABLED. It exists because "is this challenger
PROMOTION_ELIGIBLE right now" (promotion.evaluate_promotion) answers a
narrower, instantaneous question than "has the whole mechanism earned enough
trust that letting it replace the Champion unattended would be reasonable."
The second question needs more evidence than the first, checked together
in one auditable place rather than inferred from scattered signals an
operator would otherwise have to assemble by hand.

Passing every check here changes nothing by itself. The only way a policy
ever actually becomes Champion remains promotion.promote_to_champion(),
called explicitly, which re-validates the full scientific gate itself
regardless of what this module reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from product.evolution import policy_registry, promotion, scorecard
from product.evolution.health import SECTOR_DOMINANCE_THRESHOLD, _dominance
from product.evolution.shadow_decisions import SCHEMA_VERSION as SHADOW_SCHEMA_VERSION

PAPER_AUTO_PROMOTION_READY = "PAPER_AUTO_PROMOTION_READY"
NOT_READY = "NOT_READY"


def _check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _rollback_target_available(domain: str, *, registry_path: str | Path | None = None) -> bool:
    """Mirrors promotion.rollback()'s own candidate search: is there a
    PROBATION policy that was itself once CHAMPION, so an automatic
    promotion could actually be rolled back if it went wrong."""
    candidates = [
        p for p in policy_registry.list_policies(
            domain=domain, status=policy_registry.PROBATION, path=registry_path,
        )
        if any(h.get("status") == policy_registry.CHAMPION for h in p.get("lifecycle_history") or [])
    ]
    return bool(candidates)


def _schema_consistency(
    champion_policy_id: str, challenger_policy_id: str,
    *, domain: str, ledger_path: str | Path | None = None,
) -> bool:
    """All graded rows feeding the paired comparison must share the current
    shadow-ledger schema version -- a schema change mid-history is exactly
    the kind of "recent evaluator/schema change invalidating comparability"
    the brief warns about."""
    rows = scorecard.graded_rows(champion_policy_id, domain=domain, path=ledger_path)
    rows += scorecard.graded_rows(challenger_policy_id, domain=domain, path=ledger_path)
    return all(int(r.get("schema_version") or 0) == SHADOW_SCHEMA_VERSION for r in rows) if rows else True


def _tenure_would_not_block(
    domain: str, *, registry_path: str | Path | None = None,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    champion = policy_registry.current_champion(domain, path=registry_path)
    if champion is None or not list(champion.get("promotion_history") or []):
        return {"blocked": False, "tenure_days": None}
    last = promotion._dt((champion.get("promotion_history") or [])[-1].get("at"))
    if last is None:
        return {"blocked": False, "tenure_days": None}
    tenure_days = (datetime.now(timezone.utc) - last).total_seconds() / 86400.0
    return {
        "blocked": tenure_days < promotion.MIN_CHAMPION_TENURE_DAYS,
        "tenure_days": round(tenure_days, 2),
    }


def evaluate_auto_promotion_readiness(
    domain: str, challenger_policy_id: str,
    *, registry_path: str | Path | None = None, ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Every prerequisite the brief lists, checked together. Returns
    PAPER_AUTO_PROMOTION_READY only when ALL of them pass; otherwise
    NOT_READY with the full breakdown so an operator can see exactly what
    is still missing. AUTO_PROMOTION_ENABLED is reported but never touched."""
    champion = policy_registry.current_champion(domain, path=registry_path)
    checks: list[dict[str, Any]] = []

    eligibility = promotion.evaluate_promotion(
        domain, challenger_policy_id, registry_path=registry_path, ledger_path=ledger_path,
    )
    checks.append(_check(
        "scientific_promotion_eligible",
        eligibility.get("status") == promotion.PROMOTION_ELIGIBLE,
        eligibility.get("reason"),
    ))

    policy = policy_registry.get_policy(challenger_policy_id, path=registry_path)
    in_probation = bool(policy and policy.get("status") == policy_registry.PROBATION)
    checks.append(_check("currently_in_probation", in_probation))

    probation_evidence = (
        promotion.evaluate_probation_evidence(
            domain, challenger_policy_id, registry_path=registry_path, ledger_path=ledger_path,
        )
        if in_probation
        else {"passed": False, "reason": "not in probation"}
    )
    checks.append(_check(
        "probation_evidence_genuinely_new_and_not_regressed",
        probation_evidence.get("passed", False),
        probation_evidence.get("reason"),
    ))

    champion_id = champion["policy_id"] if champion else ""
    sector_dom = None
    schema_ok = True
    if champion_id:
        shared_rows = scorecard.graded_rows(champion_id, domain=domain, path=ledger_path)
        shared_rows += scorecard.graded_rows(challenger_policy_id, domain=domain, path=ledger_path)
        sector_dom = _dominance(shared_rows, "sector", SECTOR_DOMINANCE_THRESHOLD)
        schema_ok = _schema_consistency(
            champion_id, challenger_policy_id, domain=domain, ledger_path=ledger_path,
        )
    checks.append(_check("evidence_not_sector_dominated", sector_dom is None, sector_dom))
    checks.append(_check("no_schema_version_drift_in_evidence", schema_ok))

    rollback_ok = _rollback_target_available(domain, registry_path=registry_path)
    checks.append(_check("rollback_target_available", rollback_ok))

    tenure = _tenure_would_not_block(domain, registry_path=registry_path)
    checks.append(_check("champion_tenure_hysteresis_clear", not tenure["blocked"], tenure))

    manifest_fp = policy_registry.policy_manifest_fingerprint(policy) if policy else ""
    checks.append(_check("policy_manifest_content_addressed", bool(manifest_fp), manifest_fp))

    # Live-money lock is a structural invariant of this package (verified
    # independently by the AST-based static-analysis tests), not something
    # recomputed here -- reported for a complete, one-place audit view.
    checks.append(_check("live_money_locked", True, {"live_locked": True, "live_execution_authorized": False}))

    all_passed = all(c["passed"] for c in checks)
    return {
        "policy_id": challenger_policy_id,
        "domain": domain,
        "champion_policy_id": champion_id or None,
        "status": PAPER_AUTO_PROMOTION_READY if all_passed else NOT_READY,
        "checks": checks,
        "auto_promotion_enabled": promotion.AUTO_PROMOTION_ENABLED,
        "note": (
            "This status is informational only -- it never enables automatic "
            "promotion by itself; AUTO_PROMOTION_ENABLED stays False until a "
            "separate, deliberate code change turns it on."
        ),
    }
