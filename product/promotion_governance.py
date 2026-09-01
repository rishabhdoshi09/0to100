"""Formal promotion governance for research components.

A module existing is not a reason to promote it. Promotion must survive OOS,
forward, adversarial, and execution-adjusted evidence checks. This authority
never executes trades and never enables live money.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

MIN_EXECUTION_COVERAGE = 0.80
MIN_EXECUTION_ADJUSTED_N = 30


def challenger_promotion_reasons(
    comparison: Mapping[str, Any],
    *,
    adversarial_status: str = "SURVIVED",
    min_execution_coverage: float = MIN_EXECUTION_COVERAGE,
    min_execution_adjusted_n: int = MIN_EXECUTION_ADJUSTED_N,
) -> list[str]:
    """Return extra fail-closed reasons beyond ordinary OOS/sample checks."""
    cmp = dict(comparison or {})
    reasons: list[str] = []
    adj_n = int(cmp.get("execution_adjusted_n") or 0)
    oos_n = int(cmp.get("oos_n") or cmp.get("sample_size") or 0)
    coverage = float(cmp.get("execution_adjusted_coverage") or 0.0)
    adj_exp = cmp.get("execution_adjusted_expectancy")
    gross_exp = cmp.get("expectancy")

    if oos_n >= min_execution_adjusted_n:
        if adj_n < min_execution_adjusted_n or coverage < float(min_execution_coverage):
            reasons.append("EXECUTION_EVIDENCE_INCOMPLETE")
        elif adj_exp is None:
            reasons.append("EXECUTION_ADJUSTED_EXPECTANCY_MISSING")
        elif float(adj_exp) <= 0.0:
            reasons.append(
                "GROSS_EDGE_DID_NOT_SURVIVE_EXECUTION"
                if gross_exp is not None and float(gross_exp) > 0
                else "EXECUTION_ADJUSTED_EDGE_NON_POSITIVE"
            )

    if str(adversarial_status or "").upper() in {"FAILED", "FRAGILE"}:
        code = "ADVERSARIAL_FAILED" if str(adversarial_status).upper() == "FAILED" else "ADVERSARIAL_FRAGILE"
        if code not in reasons:
            reasons.append(code)
    return reasons


def assess_component(
    *,
    component: str,
    status: str,
    forward_n: int = 0,
    gross_expectancy: float | None = None,
    execution_adjusted_expectancy: float | None = None,
    execution_adjusted_coverage: float | None = None,
    adversarial_status: str = "",
    explicit_promotion_required: bool = True,
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if forward_n < 30:
        blockers.append("FORWARD_SAMPLE_TOO_SMALL")
    if gross_expectancy is None:
        blockers.append("GROSS_EXPECTANCY_MISSING")
    if execution_adjusted_expectancy is None:
        blockers.append("EXECUTION_ADJUSTED_EXPECTANCY_MISSING")
    elif execution_adjusted_expectancy <= 0:
        blockers.append(
            "GROSS_EDGE_DID_NOT_SURVIVE_EXECUTION"
            if gross_expectancy is not None and gross_expectancy > 0
            else "EXECUTION_ADJUSTED_EDGE_NON_POSITIVE"
        )
    if execution_adjusted_coverage is None or execution_adjusted_coverage < MIN_EXECUTION_COVERAGE:
        blockers.append("EXECUTION_EVIDENCE_INCOMPLETE")
    if str(adversarial_status or "").upper() in {"FAILED", "FRAGILE"}:
        blockers.append(f"ADVERSARIAL_{str(adversarial_status).upper()}")
    decision = "ELIGIBLE" if not blockers else "KEEP_SHADOW"
    return {
        "component": component,
        "current_status": status,
        "forward_n": int(forward_n),
        "gross_expectancy": gross_expectancy,
        "execution_adjusted_expectancy": execution_adjusted_expectancy,
        "execution_adjusted_coverage": execution_adjusted_coverage,
        "adversarial_status": adversarial_status or "UNKNOWN",
        "decision": decision,
        "blockers": list(dict.fromkeys(blockers)),
        "explicit_promotion_required": bool(explicit_promotion_required),
        "live_locked": True,
        "notes": list(notes or []),
    }


def promotion_board(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [assess_component(**dict(component)) for component in components]
    return {
        "schema_version": 1,
        "live_locked": True,
        "components": rows,
        "eligible": [r["component"] for r in rows if r["decision"] == "ELIGIBLE"],
        "shadow": [r["component"] for r in rows if r["decision"] != "ELIGIBLE"],
        "note": "No component promotes itself; eligibility still requires an explicit owner promotion action.",
    }
