"""Readable fundamental-intelligence projection over the existing StockResearchEngine.

No scraping, no new scoring engine and no recommendation path lives here.  The
module converts the cache-only due-diligence report into a compact company dossier
for Stock Intelligence.  Missing evidence stays missing and the existing
StockResearchEngine score/coverage remain authoritative.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in list(value or []) if isinstance(item, Mapping)]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _compact_kpi(item: Mapping[str, Any]) -> dict[str, Any]:
    snap = _mapping(item.get("snapshot"))
    prov = _mapping(item.get("provenance"))
    return {
        "id": item.get("id"),
        "label": item.get("label") or item.get("id"),
        "pillar": item.get("pillar"),
        "available": bool(item.get("available")),
        "trend": item.get("trend") or "unknown",
        "current": snap.get("current"),
        "period": snap.get("current_period") or item.get("source_date"),
        "fact": item.get("fact") or "Data unavailable",
        "interpretation": item.get("interpretation") or "",
        "implication": item.get("implication") or "",
        "source": prov.get("source") or item.get("source") or "",
        "source_url": prov.get("source_url") or item.get("source_url") or "",
        "confidence": prov.get("confidence") or item.get("confidence") or "",
        "source_consensus": item.get("source_consensus") or "",
    }


def _red_flag(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id") or item.get("kind"),
        "severity": item.get("severity") or "watch",
        "kind": item.get("kind") or "fundamental",
        "title": item.get("title") or item.get("label") or item.get("flag") or item.get("id"),
        "detail": item.get("detail") or item.get("reason") or item.get("explain") or "",
        "source": item.get("source") or "",
        "source_url": item.get("source_url") or "",
        "source_date": item.get("source_date") or item.get("date") or "",
    }


def _capital_allocation_rows(kpis: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tokens = (
        "capital", "capex", "debt", "borrow", "dividend", "buyback", "acquisition",
        "working_capital", "working capital", "cash", "free_cash", "free cash",
        "receivable", "inventory", "dilution", "equity",
    )
    out: list[dict[str, Any]] = []
    for item in kpis:
        blob = " ".join(
            _text(item.get(key)).lower()
            for key in ("id", "label", "pillar", "fact", "interpretation")
        )
        if any(token in blob for token in tokens):
            out.append(_compact_kpi(item))
    return out[:12]


def _management_rows(guidance: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in guidance[:10]:
        actual = item.get("actual") or item.get("result") or item.get("outcome")
        status = item.get("status") or item.get("execution_status")
        out.append({
            "metric": item.get("metric") or item.get("guidance_metric") or item.get("topic") or "Management guidance",
            "guidance": item.get("value") or item.get("guidance_value") or item.get("excerpt") or "",
            "tone": item.get("tone") or "",
            "source": item.get("source") or "",
            "source_url": item.get("source_url") or "",
            "source_date": item.get("source_date") or item.get("event_date") or "",
            "actual": actual,
            "execution_status": status or ("UNMEASURED" if actual in (None, "") else "RECORDED"),
        })
    return out


def _valuation_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _rows(report.get("valuation")):
        out.append({
            "id": item.get("id"),
            "label": item.get("label") or item.get("id"),
            "value": item.get("value") if item.get("value") is not None else _mapping(item.get("snapshot")).get("current"),
            "unit": item.get("unit") or "",
            "period": item.get("period") or _mapping(item.get("snapshot")).get("current_period") or item.get("source_date"),
            "source": item.get("source") or _mapping(item.get("provenance")).get("source") or "",
            "source_url": item.get("source_url") or _mapping(item.get("provenance")).get("source_url") or "",
        })
    return out[:10]


def build_fundamental_intelligence(
    stock_workspace: Mapping[str, Any] | None,
    due_diligence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    base = _mapping(stock_workspace)
    report = _mapping(due_diligence)
    symbol = _text(report.get("symbol") or base.get("symbol")).upper()
    if not report:
        return {
            "schema_version": 1,
            "symbol": symbol,
            "available": False,
            "state": "NO_RESEARCH_DOSSIER",
            "score": None,
            "evidence_coverage_pct": 0.0,
            "message": "Deep fundamental research is not on file. Run Investigate/Acquire; QuantTerm will not infer missing company evidence from price data.",
            "thesis_breakers": [],
            "missing_evidence": [],
            "places_orders": False,
        }

    framework = _mapping(report.get("framework"))
    profile = _mapping(report.get("profile"))
    quality = _mapping(report.get("fundamental_quality"))
    coverage = _mapping(report.get("research_coverage"))
    decision_coverage = _mapping(report.get("decision_coverage"))
    kpis = _rows(report.get("kpis"))
    measured = [_compact_kpi(item) for item in kpis if item.get("available")]
    missing_kpis = [_compact_kpi(item) for item in kpis if not item.get("available")]
    flags = [_red_flag(item) for item in _rows(report.get("red_flags"))]
    concerns = [_text(x) for x in list(report.get("concerns") or []) if _text(x)]
    critical_missing = [
        _text(item.get("label") or item.get("id") or item)
        if isinstance(item, Mapping) else _text(item)
        for item in list(report.get("critical_metrics_missing") or [])
    ]
    missing_evidence = _rows(report.get("missing_evidence"))

    breakers: list[dict[str, Any]] = []
    for flag in flags[:8]:
        breakers.append({
            "type": "RED_FLAG",
            "severity": flag.get("severity"),
            "title": flag.get("title") or flag.get("id"),
            "detail": flag.get("detail"),
            "source": flag.get("source"),
            "source_url": flag.get("source_url"),
            "source_date": flag.get("source_date"),
        })
    for concern in concerns:
        breakers.append({"type": "DETERIORATING_KPI", "severity": "watch", "title": concern, "detail": ""})
    for label in critical_missing:
        if label:
            breakers.append({
                "type": "MISSING_CRITICAL_EVIDENCE",
                "severity": "unknown",
                "title": label,
                "detail": "A missing critical check reduces confidence; it is not scored as a pass or a fail.",
            })
    # Never claim that an empty breaker list means a clean bill of health.
    breaker_state = "MEASURED" if breakers else "NO_MEASURED_BREAKER_ON_FILE"

    score = quality.get("score")
    score_coverage = quality.get("score_coverage_pct")
    research_cov = coverage.get("coverage_pct")
    decision_cov = report.get("decision_coverage_pct") or decision_coverage.get("coverage_pct")
    evidence_cov = research_cov if research_cov is not None else decision_cov if decision_cov is not None else score_coverage

    named = _mapping(report.get("named_quality_scores"))
    sections = {
        "business_quality": named.get("business_quality") or named.get("business") or named.get("quality"),
        "financial_quality": named.get("financial_quality") or named.get("financial") or quality.get("score"),
        "cash_flow_quality": report.get("cash_flow_quality"),
        "balance_sheet_quality": report.get("balance_sheet_quality"),
        "growth_quality": report.get("growth_quality"),
        "governance_risk": report.get("governance_risk"),
    }

    return {
        "schema_version": 1,
        "symbol": symbol,
        "company": report.get("company") or base.get("company") or symbol,
        "available": True,
        "state": "RESEARCHED" if measured else "EVIDENCE_THIN",
        "business": {
            "framework_id": framework.get("id"),
            "framework_label": framework.get("label"),
            "sub_sector": framework.get("sub_sector") or profile.get("sub_sector"),
            "business_model": framework.get("business_model") or profile.get("business_model"),
            "description": profile.get("about") or profile.get("description") or framework.get("blurb") or _mapping(base.get("fundamentals")).get("company_about") or "",
            "peer_note": framework.get("peer_note") or "",
        },
        "fundamental_score": {
            "score": score,
            "label": quality.get("label") or "Unmeasured",
            "score_coverage_pct": score_coverage,
            "evidence_coverage_pct": evidence_cov,
            "explain": quality.get("explain") or "",
            "missing_is_zero": False,
        },
        "quality": sections,
        "business_specific_kpis": {
            "measured_count": len(measured),
            "total_count": len(kpis),
            "measured": measured[:20],
            "missing": missing_kpis[:20],
            "sector_kpi_label": report.get("sector_kpi_label"),
            "sector_kpi_detail": report.get("sector_kpi_detail"),
        },
        "financial_quality": {
            "financial_strength": report.get("financial_strength"),
            "earnings_quality": report.get("earnings_quality"),
            "growth_quality": report.get("growth_quality"),
            "balance_sheet_quality": report.get("balance_sheet_quality"),
            "cash_flow_quality": report.get("cash_flow_quality"),
            "strengths": [_text(x) for x in list(report.get("strengths") or []) if _text(x)][:8],
            "concerns": concerns[:8],
        },
        "accounting_governance": {
            "governance_risk": report.get("governance_risk") or "Unmeasured",
            "red_flags": flags[:20],
            "source_conflicts": _rows(report.get("source_conflicts"))[:12],
        },
        "management_execution": {
            "guidance": _management_rows(_rows(report.get("extracted_guidance"))),
            "note": (
                "Promise-versus-result is shown only when both guidance and an attributable actual outcome are on file. "
                "Otherwise execution_status remains UNMEASURED."
            ),
        },
        "capital_allocation": {
            "evidence": _capital_allocation_rows(kpis),
            "note": "Capital-allocation observations are projections of measured KPIs only; absence is not interpreted as good allocation.",
        },
        "valuation": {
            "metrics": _valuation_rows(report),
            "peers": _rows(report.get("peers"))[:12],
            "note": "Valuation is contextual evidence, not an automatic cheap/expensive verdict and not a target price.",
        },
        "thesis": _mapping(report.get("thesis")),
        "thesis_breakers": breakers[:20],
        "thesis_breaker_state": breaker_state,
        "what_changed": [_text(x) for x in list(report.get("what_changed") or []) if _text(x)][:8],
        "watch_next": [_text(x) for x in list(report.get("watch_next") or []) if _text(x)][:8],
        "missing_evidence": missing_evidence[:20],
        "critical_metrics_missing": critical_missing[:20],
        "coverage": {
            "research": coverage,
            "decision": decision_coverage,
            "implementation_coverage_pct": report.get("implementation_coverage_pct"),
        },
        "as_of": _mapping(report.get("as_of")),
        "sources": _rows(report.get("sources"))[:30],
        "filings": _rows(report.get("filings"))[:20],
        "fundamental_confirmation": report.get("fundamental_confirmation"),
        "confirmation_reason": report.get("confirmation_reason"),
        "vs_technical_setup": report.get("vs_technical_setup"),
        "places_orders": False,
        "uses_llm": False,
        "invariant": "Deep fundamentals explain business evidence; they do not independently generate or override a trading BUY.",
    }
