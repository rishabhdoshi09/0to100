"""Selection-bias / exploration-starvation diagnostic (section 5 of the
Evolution brief).

The Champion gets executed PAPER observations; Challengers mostly get
shadow-only observations, and a per-cycle budget (tournament.py's
max_challengers / max_seconds) means a Challenger is not guaranteed to be
evaluated on every opportunity the Champion saw. Promotion itself is already
immune to this -- scorecard.paired_comparison() only ever compares the
SHARED snapshots, never the full unpaired history -- but a narrow paired
slice can still be an unrepresentative sample of the real opportunity space,
and nothing before this module made that visible to an operator.

This module answers one question, directly, without new statistics: of
everything the Champion was evaluated on, how much of it did this Challenger
ALSO get evaluated on? A low fraction is not disqualifying by itself (the
paired comparison it does have remains fair on what it covers), but it is
exactly the caution signal the brief asks for: "Champion looks better
because it received more observable decisions" is a coverage question, not
a comparison-math question.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from product.evolution import policy_registry, scorecard

# Below this fraction of the Champion's own total opportunity set, a
# Challenger's paired evidence is flagged as a narrow/unrepresentative slice
# -- an observability caution, never a promotion gate (promotion already
# only reads the paired intersection, so it cannot be fooled by this; this
# flag exists so an operator can see WHY a Challenger's sample is small).
LOW_COVERAGE_THRESHOLD = 0.5


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def exposure_report(
    domain: str, *, registry_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Per-Challenger exposure/coverage vs the domain's current Champion.

    Returns an empty report (not an error) when there is no Champion or no
    active Challengers yet -- a fresh domain has nothing to flag."""
    champion = policy_registry.current_champion(domain, path=registry_path)
    if champion is None:
        return {"domain": domain, "champion_policy_id": None, "rows": []}
    champion_id = champion["policy_id"]
    champion_card = scorecard.scorecard(champion_id, domain=domain, path=ledger_path)
    champion_total = int(champion_card.get("decision_snapshots") or 0)

    rows: list[dict[str, Any]] = []
    for policy in policy_registry.active_challengers(domain, path=registry_path):
        policy_id = policy["policy_id"]
        card = scorecard.scorecard(policy_id, domain=domain, path=ledger_path)
        challenger_total = int(card.get("decision_snapshots") or 0)
        paired = scorecard.paired_comparison(
            champion_id, policy_id, domain=domain, path=ledger_path,
        )
        paired_n = int(paired.get("paired_snapshots") or 0)
        coverage_vs_champion = _ratio(paired_n, champion_total)
        coverage_vs_self = _ratio(paired_n, challenger_total)
        rows.append({
            "policy_id": policy_id,
            "status": policy.get("status"),
            "challenger_total_snapshots": challenger_total,
            "champion_total_snapshots": champion_total,
            "paired_snapshots": paired_n,
            "coverage_of_champion_opportunities": coverage_vs_champion,
            "coverage_of_own_opportunities": coverage_vs_self,
            "narrow_sample_risk": bool(
                coverage_vs_champion is not None
                and coverage_vs_champion < LOW_COVERAGE_THRESHOLD
            ),
        })

    return {
        "domain": domain,
        "champion_policy_id": champion_id,
        "champion_total_snapshots": champion_total,
        "low_coverage_threshold": LOW_COVERAGE_THRESHOLD,
        "rows": rows,
    }
