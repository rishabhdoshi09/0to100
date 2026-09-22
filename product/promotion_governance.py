"""Formal promotion governance for research components.

A module existing is not a reason to promote it. Promotion must survive OOS,
forward, adversarial, and execution-adjusted evidence checks. This authority
never executes trades and never enables live money.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.live_safety import live_safety_projection

MIN_EXECUTION_COVERAGE = 0.80
MIN_EXECUTION_ADJUSTED_N = 30




MIN_STABILITY_BUCKET_N = 10
MAX_DRAWDOWN_RATIO = 1.50
MIN_CALIBRATION_IMPROVEMENT = 0.01


def _f(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _bucket_blockers(
    breakdown: Mapping[str, Any] | None,
    *,
    prefix: str,
    min_n: int = MIN_STABILITY_BUCKET_N,
) -> list[str]:
    blockers: list[str] = []
    for name, raw in sorted(dict(breakdown or {}).items()):
        if isinstance(raw, Mapping):
            n = int(raw.get("n") or raw.get("sample_size") or 0)
            expectancy = _f(raw.get("expectancy"))
        else:
            # Legacy scalar summaries are descriptive only because they carry no
            # sample size. Promotion governance must not invent confidence.
            n = 0
            expectancy = _f(raw)
        if n >= int(min_n) and expectancy is not None and expectancy <= 0.0:
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(name).upper()).strip("_")
            blockers.append(f"{prefix}_INSTABILITY:{safe or 'UNKNOWN'}")
    return blockers


def promotion_dossier(
    comparison: Mapping[str, Any],
    *,
    component: str,
    adversarial_status: str = "SURVIVED",
    require_calibration_edge: bool = False,
    require_positive_expectancy: bool = True,
    require_execution_adjusted_edge: bool = False,
    min_oos_n: int = 30,
    min_forward_n: int = 30,
    min_execution_coverage: float = MIN_EXECUTION_COVERAGE,
    min_execution_adjusted_n: int = MIN_EXECUTION_ADJUSTED_N,
    min_calibration_improvement: float = MIN_CALIBRATION_IMPROVEMENT,
    min_stability_bucket_n: int = MIN_STABILITY_BUCKET_N,
    champion_drawdown: float | None = None,
) -> dict[str, Any]:
    """Canonical promotion dossier shared by rule and learned challengers.

    It is deliberately conservative:
    - exact evidence counts are required instead of inferred confidence;
    - negative sufficiently-sampled regime/sector buckets block promotion;
    - calibration is required only for probabilistic challengers;
    - execution-adjusted edge is required only for challengers that claim to
      collect execution-aware evidence;
    - no result here enables live money.
    """
    cmp = dict(comparison or {})
    blockers: list[str] = []
    oos_n = int(cmp.get("oos_n") or 0)
    forward_n = int(
        cmp.get("forward_n")
        or cmp.get("n")
        or cmp.get("sample_size")
        or 0
    )
    gross = _f(cmp.get("expectancy"))
    adjusted = _f(cmp.get("execution_adjusted_expectancy"))
    adjusted_n = int(cmp.get("execution_adjusted_n") or 0)
    adjusted_cov = _f(cmp.get("execution_adjusted_coverage"))
    improvement = _f(
        cmp.get("improvement")
        if cmp.get("improvement") is not None
        else cmp.get("calibration_improvement")
    )
    lower = _f(
        cmp.get("improvement_lower_95")
        if cmp.get("improvement_lower_95") is not None
        else cmp.get("calibration_improvement_lower_95")
    )
    drawdown = _f(cmp.get("drawdown"))
    champ_dd = _f(champion_drawdown)
    exact_version = cmp.get("exact_version_evidence")

    if oos_n < int(min_oos_n):
        blockers.append("OOS_SAMPLE_TOO_SMALL")
    if forward_n < int(min_forward_n):
        blockers.append("FORWARD_SAMPLE_TOO_SMALL")

    if require_positive_expectancy:
        if gross is None:
            blockers.append("GROSS_EXPECTANCY_MISSING")
        elif gross <= 0.0:
            blockers.append("GROSS_EXPECTANCY_NON_POSITIVE")

    if require_calibration_edge:
        if improvement is None:
            blockers.append("CALIBRATION_IMPROVEMENT_MISSING")
        elif improvement < float(min_calibration_improvement):
            blockers.append("CALIBRATION_EDGE_TOO_SMALL")
        if lower is None:
            blockers.append("CALIBRATION_CONFIDENCE_MISSING")
        elif lower <= 0.0:
            blockers.append("CALIBRATION_LOWER_BOUND_NON_POSITIVE")
        if exact_version is False:
            blockers.append("EXACT_VERSION_FORWARD_EVIDENCE_MISSING")

    if require_execution_adjusted_edge:
        if adjusted_n < int(min_execution_adjusted_n):
            blockers.append("EXECUTION_EVIDENCE_INCOMPLETE")
        if adjusted_cov is None or adjusted_cov < float(min_execution_coverage):
            blockers.append("EXECUTION_EVIDENCE_INCOMPLETE")
        if adjusted is None:
            blockers.append("EXECUTION_ADJUSTED_EXPECTANCY_MISSING")
        elif adjusted <= 0.0:
            blockers.append(
                "GROSS_EDGE_DID_NOT_SURVIVE_EXECUTION"
                if gross is not None and gross > 0.0
                else "EXECUTION_ADJUSTED_EDGE_NON_POSITIVE"
            )

    adv = str(adversarial_status or "UNKNOWN").upper()
    if adv in {"FAILED", "FRAGILE"}:
        blockers.append(f"ADVERSARIAL_{adv}")

    blockers.extend(
        _bucket_blockers(
            cmp.get("regime_breakdown"),
            prefix="REGIME",
            min_n=min_stability_bucket_n,
        )
    )
    blockers.extend(
        _bucket_blockers(
            cmp.get("sector_breakdown"),
            prefix="SECTOR",
            min_n=min_stability_bucket_n,
        )
    )

    drawdown_ratio = None
    if drawdown is not None and champ_dd is not None and champ_dd > 0.0:
        drawdown_ratio = drawdown / champ_dd
        if drawdown_ratio > MAX_DRAWDOWN_RATIO:
            blockers.append("DRAWDOWN_REGRESSION")

    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": 1,
        "component": str(component),
        "decision": "ELIGIBLE" if not blockers else "KEEP_SHADOW",
        "blockers": blockers,
        "oos_n": oos_n,
        "forward_n": forward_n,
        "gross_expectancy": gross,
        "execution_adjusted_expectancy": adjusted,
        "execution_adjusted_n": adjusted_n,
        "execution_adjusted_coverage": adjusted_cov,
        "calibration_improvement": improvement,
        "calibration_improvement_lower_95": lower,
        "exact_version_evidence": exact_version,
        "drawdown": drawdown,
        "champion_drawdown": champ_dd,
        "drawdown_ratio": None if drawdown_ratio is None else round(drawdown_ratio, 6),
        "regime_breakdown": dict(cmp.get("regime_breakdown") or {}),
        "sector_breakdown": dict(cmp.get("sector_breakdown") or {}),
        "adversarial_status": adv,
        "explicit_promotion_required": True,
        **live_safety_projection(),
        "note": (
            "Promotion eligibility is research/paper governance only. "
            "It never grants live execution authority."
        ),
    }


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
        **live_safety_projection(),
        "notes": list(notes or []),
    }


def promotion_board(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [assess_component(**dict(component)) for component in components]
    safety = live_safety_projection()
    for row in rows:
        row.update(safety)
    return {
        "schema_version": 1,
        **safety,
        "components": rows,
        "eligible": [r["component"] for r in rows if r["decision"] == "ELIGIBLE"],
        "shadow": [r["component"] for r in rows if r["decision"] != "ELIGIBLE"],
        "note": "No component promotes itself; eligibility still requires an explicit owner promotion action.",
    }
