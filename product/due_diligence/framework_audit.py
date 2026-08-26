"""Framework Coverage Audit — system capability, not a company score.

Answers: of the metrics this business requires, how many have a validated
acquisition path? Distinct from Data Coverage (datasets on disk) and
Decision Coverage (this company's populated evidence).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.due_diligence.frameworks import get_framework
from product.due_diligence.metric_impl import (
    PRIORITY_FRAMEWORKS,
    get_impl,
    reliability_label,
)
from product.due_diligence.sector_frameworks import list_frameworks


DECISION = frozenset({"critical", "important"})


def _importance(spec: Any) -> str:
    return str(getattr(spec, "importance", "") or "supporting")


def audit_metric(spec: Any, *, finding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    impl = get_impl(getattr(spec, "id", ""))
    populated = bool(finding and finding.get("available") and finding.get("points") is not None)
    company_state = "not evaluated"
    if finding is not None:
        if populated:
            company_state = "populated"
        elif impl.implemented:
            company_state = "missing from sources"
        else:
            company_state = "no acquisition path"
    return {
        "id": getattr(spec, "id", ""),
        "label": getattr(spec, "label", ""),
        "importance": _importance(spec),
        "implemented": impl.implemented,
        "reliability": impl.reliability,
        "reliability_label": reliability_label(impl.reliability),
        "paths": list(impl.paths),
        "definition": impl.definition,
        "period_policy": impl.period_policy,
        "false_positive_guard": impl.false_positive_guard,
        "tests": list(impl.tests),
        "populated": populated,
        "company_state": company_state,
    }


def audit_framework(
    framework_id: str,
    *,
    findings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    fw = get_framework(framework_id)
    by_id = {str(f.get("id")): f for f in (findings or []) if isinstance(f, Mapping)}
    metrics = [
        audit_metric(spec, finding=by_id.get(spec.id))
        for spec in fw["kpis"]
    ]
    decision = [row for row in metrics if row["importance"] in DECISION]
    critical = [row for row in metrics if row["importance"] == "critical"]
    implemented = [row for row in decision if row["implemented"]]
    obtainable = [row for row in decision if row["reliability"] == "obtainable"]
    decision_n = len(decision)
    implemented_n = len(implemented)
    coverage = round(100.0 * implemented_n / decision_n, 1) if decision_n else 0.0
    populated_n = sum(1 for row in decision if row["populated"])
    return {
        "id": fw["id"],
        "label": fw["label"],
        "blurb": fw.get("blurb") or "",
        "priority": fw["id"] in PRIORITY_FRAMEWORKS,
        "decision_n": decision_n,
        "critical_n": len(critical),
        "implemented_n": implemented_n,
        "obtainable_n": len(obtainable),
        "implementation_coverage_pct": coverage,
        "populated_n": populated_n,
        "company_decision_coverage_pct": (
            round(100.0 * populated_n / decision_n, 1) if decision_n and findings is not None else None
        ),
        "metrics": metrics,
        "decision_metrics": decision,
        "summary": (
            f"{implemented_n}/{decision_n} decision metrics have a validated acquisition path "
            f"({coverage}%)."
        ),
        "not_a_quality_score": True,
        "not_decision_coverage": True,
    }


def audit_all_frameworks() -> dict[str, Any]:
    rows = [audit_framework(fid) for fid in list_frameworks()]
    priority = [row for row in rows if row["priority"]]
    return {
        "schema_version": 1,
        "engine": "StockResearchEngine",
        "not_an_llm": True,
        "places_orders": False,
        "not_a_quality_score": True,
        "explain": (
            "Implementation coverage is system capability: whether QuantTerm can "
            "acquire, validate, period-stamp and test a metric. It is not this "
            "company's Decision Coverage and not Fundamental Quality."
        ),
        "priority_frameworks": list(PRIORITY_FRAMEWORKS),
        "rows": [
            {
                "id": row["id"],
                "label": row["label"],
                "priority": row["priority"],
                "decision_n": row["decision_n"],
                "implemented_n": row["implemented_n"],
                "implementation_coverage_pct": row["implementation_coverage_pct"],
            }
            for row in rows
        ],
        "frameworks": rows,
        "priority_summary": [
            {
                "id": row["id"],
                "label": row["label"],
                "decision_n": row["decision_n"],
                "implemented_n": row["implemented_n"],
                "implementation_coverage_pct": row["implementation_coverage_pct"],
            }
            for row in priority
        ],
        "point_in_time": True,
        "point_in_time_note": (
            "A metric print keeps the dated period and reporting basis from the source. "
            "Freshness uses fetched_at / as_of on file, never a live 'latest' restatement."
        ),
    }


def coverage_table(payload: Mapping[str, Any] | None = None) -> str:
    data = dict(payload or audit_all_frameworks())
    lines = [
        "| Framework | Decision metrics | Reliable acquisition | Coverage |",
        "| --------- | ---------------: | -------------------: | -------: |",
    ]
    for row in list(data.get("rows") or data.get("priority_summary") or []):
        lines.append(
            f"| {row.get('label') or row.get('id')} | {row.get('decision_n')} | "
            f"{row.get('implemented_n')} | {row.get('implementation_coverage_pct')}% |"
        )
    return "\n".join(lines)
