"""Transparent investor-style fundamental intelligence built from measured report evidence.

This module does not acquire data and does not create a second stock-research engine.
It projects the evidence already produced by ``StockResearchEngine`` into investor
questions that are easier to reason about: business quality, financial quality,
cash flow, governance, capital allocation, growth and valuation.

Important invariants:
- missing evidence is never scored as zero;
- not-applicable dimensions are removed from the coverage denominator;
- current valuation multiples alone are context, not an attractiveness score;
- management commentary is not treated as execution quality unless measured outcomes exist;
- this score is descriptive fundamental evidence, never an independent BUY signal.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


COMPONENT_WEIGHTS: dict[str, float] = {
    "business_quality": 20.0,
    "financial_quality": 20.0,
    "cash_flow": 15.0,
    "management_governance": 15.0,
    "capital_allocation": 10.0,
    "growth_quality": 10.0,
    "valuation": 10.0,
}

MIN_DISPLAY_COVERAGE = 0.50
MIN_SCORED_COMPONENTS = 3

_LABEL_POINTS: dict[str, float] = {
    "strong": 86.0,
    "improving": 82.0,
    "adequate": 68.0,
    "stable": 64.0,
    "mixed": 50.0,
    "watch": 45.0,
    "weakening": 38.0,
    "weak": 28.0,
    "deteriorating": 24.0,
    "elevated": 25.0,
}

_BUSINESS_PILLARS = {
    "sector", "asset_quality", "funding", "capital", "consistency",
}
_GROWTH_PILLARS = {"growth"}
_GOVERNANCE_PILLARS = {"governance"}
_GOVERNANCE_FLAG_WORDS = {
    "governance", "promoter", "pledge", "auditor", "related_party",
    "related party", "dilution", "warrant", "fraud", "regulatory_action",
}


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _pct(value: Any) -> float:
    number = _f(value)
    if number is None:
        return 0.0
    return max(0.0, min(1.0, number / 100.0 if number > 1 else number))


def _label_points(label: Any) -> float | None:
    text = str(label or "").strip().lower()
    if not text or text in {"unmeasured", "data unavailable", "not applicable", "see sector kpis"}:
        return None
    return _LABEL_POINTS.get(text)


def _weighted_findings(
    findings: Sequence[Mapping[str, Any]],
    *,
    pillars: set[str] | None = None,
    critical_fallback: bool = False,
) -> tuple[float | None, float, list[str], str]:
    selected = [
        f for f in findings
        if pillars is None or str(f.get("pillar") or "") in pillars
    ]
    if not selected and critical_fallback:
        selected = [
            f for f in findings
            if str(f.get("importance") or "").lower() == "critical"
        ]
    if not selected:
        return None, 0.0, [], "No applicable measured KPI bucket exists in this framework."

    total_weight = sum(max(_f(f.get("weight")) or 0.0, 0.0) for f in selected)
    usable = [f for f in selected if _f(f.get("points")) is not None]
    used_weight = sum(max(_f(f.get("weight")) or 0.0, 0.0) for f in usable)
    if total_weight <= 0:
        total_weight = float(len(selected))
        used_weight = float(len(usable))
        weights = [1.0 for _ in usable]
    else:
        weights = [max(_f(f.get("weight")) or 0.0, 0.0) for f in usable]

    coverage = used_weight / total_weight if total_weight > 0 else 0.0
    if not usable or sum(weights) <= 0:
        return None, coverage, [], "Relevant KPI definitions exist, but none has a measured score."

    score = sum(float(f["points"]) * w for f, w in zip(usable, weights)) / sum(weights)
    evidence = [
        str(f.get("fact") or f.get("interpretation") or f.get("label") or "")
        for f in usable
        if f.get("fact") or f.get("interpretation") or f.get("label")
    ][:5]
    explain = (
        f"{len(usable)}/{len(selected)} applicable KPI(s) measured; weighted evidence coverage "
        f"{coverage * 100:.0f}%."
    )
    return round(score, 1), coverage, evidence, explain


def _component(
    component_id: str,
    label: str,
    *,
    score: float | None,
    coverage: float,
    explain: str,
    evidence: Sequence[str] = (),
    applicable: bool = True,
    formula: str = "",
) -> dict[str, Any]:
    weight = COMPONENT_WEIGHTS[component_id]
    coverage = max(0.0, min(1.0, float(coverage)))
    awarded = round(score / 100.0 * weight, 1) if score is not None else None
    if not applicable:
        display = "Not applicable"
    elif score is None:
        display = f"Data unavailable / {weight:.0f}"
    else:
        display = f"{awarded:.1f}/{weight:.0f} · {coverage * 100:.0f}% evidence"
    return {
        "id": f"intelligence_{component_id}",
        "label": f"Intelligence · {label}",
        "score": round(score, 1) if score is not None else None,
        "coverage_pct": round(coverage * 100.0, 1),
        "awarded": awarded,
        "max": weight,
        "display": display,
        "explain": explain,
        "formula": formula,
        "evidence": [str(x) for x in evidence if str(x).strip()][:6],
        "applicable": applicable,
        "missing_is_zero": False,
    }


def _business_component(report: Mapping[str, Any]) -> dict[str, Any]:
    findings = list(report.get("kpis") or [])
    score, coverage, evidence, explain = _weighted_findings(
        findings, pillars=_BUSINESS_PILLARS, critical_fallback=True,
    )
    framework = str((report.get("framework") or {}).get("label") or "sector framework")
    return _component(
        "business_quality", "Business Quality",
        score=score, coverage=coverage,
        explain=f"Business-specific evidence from {framework}. {explain}",
        evidence=evidence,
        formula="Weighted average of measured business/sector KPI points; missing KPIs are skipped.",
    )


def _financial_component(report: Mapping[str, Any]) -> dict[str, Any]:
    quality = dict(report.get("fundamental_quality") or {})
    score = _f(quality.get("score"))
    coverage = _pct(
        quality.get("score_coverage_pct")
        if quality.get("score_coverage_pct") is not None
        else quality.get("coverage_pct")
    )
    evidence = [str(quality.get("explain") or "")]
    return _component(
        "financial_quality", "Financial Quality",
        score=score, coverage=coverage,
        explain=(
            "Uses the existing sector-weighted QuantTerm fundamental-quality score. "
            "It is not recomputed from a second model."
            if score is not None else
            "Existing sector-weighted fundamental quality is Unmeasured because evidence coverage is too thin."
        ),
        evidence=evidence,
        formula="Existing StockResearchEngine fundamental_quality.score + its measured evidence coverage.",
    )


def _cash_component(report: Mapping[str, Any]) -> dict[str, Any]:
    cash = dict(report.get("cash_flow_quality") or {})
    applicable = cash.get("applicable") is not False
    if not applicable:
        return _component(
            "cash_flow", "Cash Flow Quality",
            score=None, coverage=0.0, applicable=False,
            explain=str(cash.get("detail") or "Cash-conversion analysis is not applicable to this business model."),
            formula="Excluded from the overall evidence denominator when not applicable.",
        )
    metrics = list(cash.get("metrics") or [])
    available = [m for m in metrics if m.get("available")]
    coverage = len(available) / len(metrics) if metrics else 0.0
    score = _label_points(cash.get("label"))
    evidence = [str(m.get("fact") or "") for m in available if m.get("fact")][:5]
    return _component(
        "cash_flow", "Cash Flow Quality",
        score=score, coverage=coverage,
        explain=str(cash.get("detail") or "Rule-based CFO/PAT, FCF and working-capital evidence."),
        evidence=evidence,
        formula="Rule label from measured cash-conversion checks; no estimate is created when inputs are absent.",
    )


def _governance_component(report: Mapping[str, Any]) -> dict[str, Any]:
    findings = list(report.get("kpis") or [])
    score, coverage, evidence, explain = _weighted_findings(findings, pillars=_GOVERNANCE_PILLARS)
    flags = list(report.get("red_flags") or [])
    governance_flags: list[Mapping[str, Any]] = []
    for flag in flags:
        blob = " ".join(
            str(flag.get(key) or "")
            for key in ("kind", "id", "title", "rule")
        ).lower()
        if any(word in blob for word in _GOVERNANCE_FLAG_WORDS):
            governance_flags.append(flag)

    if governance_flags:
        severities = {str(f.get("severity") or "monitor").lower() for f in governance_flags}
        flag_score = 20.0 if "critical" in severities else 38.0 if "warning" in severities else 55.0
        score = min(score, flag_score) if score is not None else flag_score
        coverage = max(coverage, 0.50)
        evidence = [
            str(f.get("evidence") or f.get("fact") or f.get("title") or "")
            for f in governance_flags
        ][:5] + evidence
        explain = (
            f"{len(governance_flags)} measured governance/promoter flag(s) cap the component score. "
            + explain
        )

    guidance_n = len(list(report.get("extracted_guidance") or []))
    if guidance_n:
        evidence.append(
            f"{guidance_n} management guidance/commentary item(s) are on file, but promise-vs-result execution is not yet scored."
        )
    if score is None and not governance_flags:
        explain = (
            "No validated governance KPI or adverse governance flag is measurable. "
            "Management commentary alone is not treated as management quality."
        )
    return _component(
        "management_governance", "Management & Governance",
        score=score, coverage=coverage,
        explain=explain,
        evidence=evidence,
        formula="Governance KPI evidence, capped by measured governance/promoter red flags; commentary alone earns no points.",
    )


def _capital_allocation_component(report: Mapping[str, Any]) -> dict[str, Any]:
    balance = dict(report.get("balance_sheet_rules") or {})
    cash = dict(report.get("cash_flow_quality") or {})
    sources: list[tuple[str, float]] = []
    evidence: list[str] = []

    balance_score = _label_points(balance.get("label"))
    balance_metrics = [m for m in list(balance.get("metrics") or []) if m.get("available")]
    if balance_score is not None:
        sources.append(("balance-sheet funding", balance_score))
        evidence.extend(str(m.get("fact") or "") for m in balance_metrics[:3] if m.get("fact"))

    if cash.get("applicable") is not False:
        cash_score = _label_points(cash.get("label"))
        cash_metrics = [m for m in list(cash.get("metrics") or []) if m.get("available")]
        if cash_score is not None:
            sources.append(("cash deployment", cash_score))
            evidence.extend(str(m.get("fact") or "") for m in cash_metrics[:3] if m.get("fact"))

    score = round(sum(v for _, v in sources) / len(sources), 1) if sources else None
    coverage = min(1.0, len(sources) / 2.0)
    explain = (
        "Capital-allocation proxy from measured funding/leverage and cash-generation evidence. "
        "Acquisitions, buybacks, dividends and project-level returns are not inferred."
        if sources else
        "Capital allocation is Unmeasured: the desk does not yet have enough cash/debt deployment evidence to score it."
    )
    return _component(
        "capital_allocation", "Capital Allocation",
        score=score, coverage=coverage,
        explain=explain,
        evidence=evidence,
        formula="Average of measured balance-sheet funding and cash-generation rule scores; strategic uses of capital remain explicit gaps.",
    )


def _growth_component(report: Mapping[str, Any]) -> dict[str, Any]:
    findings = list(report.get("kpis") or [])
    score, coverage, evidence, explain = _weighted_findings(findings, pillars=_GROWTH_PILLARS)
    growth = dict(report.get("growth_quality") or {})
    if score is None:
        qualitative = _label_points(growth.get("label"))
        if qualitative is not None:
            score = qualitative
            measured = int(growth.get("n_growth") or 0)
            coverage = min(1.0, measured / 2.0) if measured else 0.5
            evidence = [str(x) for x in list(growth.get("notes") or [])[:5]]
            explain = f"Growth-quality rule label: {growth.get('label')}."
    return _component(
        "growth_quality", "Growth Quality",
        score=score, coverage=coverage,
        explain=explain,
        evidence=evidence,
        formula="Measured growth KPI trend points; falls back only to the rule-based growth-quality label when available.",
    )


def _valuation_component(report: Mapping[str, Any]) -> dict[str, Any]:
    valuation = [v for v in list(report.get("valuation") or []) if v.get("available", True)]
    scored = [
        v for v in valuation
        if _f(v.get("score")) is not None or _f(v.get("points")) is not None
    ]
    evidence = [str(v.get("fact") or v.get("interpretation") or v.get("label") or "") for v in valuation][:6]
    if scored:
        values = [(_f(v.get("score")) if _f(v.get("score")) is not None else _f(v.get("points"))) for v in scored]
        numbers = [v for v in values if v is not None]
        score = round(sum(numbers) / len(numbers), 1) if numbers else None
        coverage = len(scored) / max(len(valuation), 1)
        explain = "Valuation attractiveness is scored only from valuation evidence that already carries an explicit validated score."
    else:
        score = None
        coverage = 0.0
        explain = (
            f"{len(valuation)} current valuation snapshot(s) may be on file, but no validated historical/peer-implied "
            "attractiveness score exists. Current PE/PB/EV-EBITDA alone is context, not 'cheap' or 'expensive'."
        )
    return _component(
        "valuation", "Valuation",
        score=score, coverage=coverage,
        explain=explain,
        evidence=evidence,
        formula="No valuation points without an explicit contextual scoring method; current multiples alone earn zero guessed points.",
    )


def build_fundamental_intelligence(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build the investor-facing scorecard from one completed due-diligence report."""
    components = [
        _business_component(report),
        _financial_component(report),
        _cash_component(report),
        _governance_component(report),
        _capital_allocation_component(report),
        _growth_component(report),
        _valuation_component(report),
    ]

    applicable = [c for c in components if c.get("applicable", True)]
    applicable_weight = sum(float(c.get("max") or 0.0) for c in applicable) or 1.0
    evidence_weight = sum(
        float(c.get("max") or 0.0) * _pct(c.get("coverage_pct"))
        for c in applicable
    )
    coverage = evidence_weight / applicable_weight
    scored = [c for c in applicable if c.get("score") is not None]
    numerator = sum(
        float(c["score"]) * float(c.get("max") or 0.0) * _pct(c.get("coverage_pct"))
        for c in scored
    )
    denominator = sum(
        float(c.get("max") or 0.0) * _pct(c.get("coverage_pct"))
        for c in scored
    )
    raw_score = round(numerator / denominator) if denominator > 0 else None
    display_score = (
        raw_score
        if raw_score is not None and coverage >= MIN_DISPLAY_COVERAGE and len(scored) >= MIN_SCORED_COMPONENTS
        else None
    )
    if display_score is None:
        label = "INSUFFICIENT EVIDENCE"
    elif display_score >= 80:
        label = "STRONG"
    elif display_score >= 65:
        label = "GOOD"
    elif display_score >= 50:
        label = "MIXED"
    else:
        label = "WEAK"

    missing = [c["label"].replace("Intelligence · ", "") for c in applicable if c.get("score") is None]
    summary = {
        "id": "fundamental_intelligence_total",
        "label": "Fundamental Intelligence",
        "score": display_score,
        "coverage_pct": round(coverage * 100.0, 1),
        "awarded": float(display_score) if display_score is not None else None,
        "max": 100.0,
        "display": (
            f"{display_score}/100 · {coverage * 100:.0f}% evidence"
            if display_score is not None else
            f"INSUFFICIENT EVIDENCE · {coverage * 100:.0f}% coverage"
        ),
        "explain": (
            "Investor-style synthesis of the same due-diligence evidence. Missing dimensions are skipped, not scored as zero. "
            f"Unmeasured: {', '.join(missing) if missing else 'none'}."
        ),
        "formula": (
            "coverage-weighted component score; displayed only when ≥50% applicable evidence coverage and ≥3 components are scored"
        ),
        "evidence": [],
        "applicable": True,
        "missing_is_zero": False,
    }
    return {
        "schema_version": 1,
        "score": display_score,
        "raw_score_if_coverage_improves": raw_score,
        "label": label,
        "coverage_pct": round(coverage * 100.0, 1),
        "components": components,
        "summary_pillar": summary,
        "missing_components": missing,
        "score_is_buy_signal": False,
        "valuation_without_context_is_scored": False,
        "management_commentary_is_execution_score": False,
        "explain": summary["explain"],
    }
