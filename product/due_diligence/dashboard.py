"""First-screen research dashboard. Assembled from already-measured fields."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.due_diligence.fundamental_intelligence import build_fundamental_intelligence

CONFIRM_LABEL = {
    "STRONGLY SUPPORTS": "STRONG SUPPORT",
    "SUPPORTS": "SUPPORT",
    "NEUTRAL": "NEUTRAL",
    "CAUTION": "CAUTION",
    "CONTRADICTS": "CONTRADICTS",
    "STRONGLY CONTRADICTS": "CONTRADICTS",
    "UNMEASURED": "NEUTRAL",
}


def confirmation_label(vs_setup: str) -> str:
    return CONFIRM_LABEL.get(str(vs_setup or "").upper(), str(vs_setup or "NEUTRAL"))


def _selected_by(technical: Mapping[str, Any]) -> str:
    if not technical.get("available"):
        return "Manual investigator — not on the current scanner shortlist."
    bits: list[str] = []
    status = str(technical.get("scanner_status") or "").strip()
    if status and status != "Data unavailable":
        bits.append(status)
    sepa = technical.get("sepa_score")
    if sepa is not None:
        bits.append(f"SEPA {sepa:.0f}" if isinstance(sepa, (int, float)) else f"SEPA {sepa}")
    grade = technical.get("breakout_grade")
    if grade:
        bits.append(f"Breakout {grade}")
    signals = [str(s) for s in (technical.get("signals") or []) if s]
    for signal in signals[:3]:
        if signal not in " ".join(bits):
            bits.append(signal.replace("_", " "))
    return " · ".join(bits) if bits else "Scanner shortlist"


def company_snapshot(
    *,
    symbol: str,
    company: str,
    profile: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
    technical: Mapping[str, Any],
    as_of: Mapping[str, Any],
    scan_row: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {str(f.get("id")): f for f in findings}

    def holding(kpi_id: str) -> Any:
        snap = dict((by_id.get(kpi_id) or {}).get("snapshot") or {})
        return snap.get("current")

    def holding_period(kpi_id: str) -> str:
        snap = dict((by_id.get(kpi_id) or {}).get("snapshot") or {})
        return str(snap.get("current_period") or "")

    price = scan_row.get("close") or scan_row.get("ltp") or scan_row.get("price")
    high = scan_row.get("high_52w") or scan_row.get("high52") or scan_row.get("week_52_high")
    low = scan_row.get("low_52w") or scan_row.get("low52") or scan_row.get("week_52_low")
    mcap = scan_row.get("market_cap") or scan_row.get("mcap") or raw.get("market_cap")
    about = str(profile.get("about") or profile.get("business_model") or "").strip()
    return {
        "company": company,
        "nse_ticker": symbol,
        "sector": str(profile.get("sector") or "Unclassified"),
        "industry": str(profile.get("industry") or profile.get("sector") or "Unclassified"),
        "market_cap": mcap if mcap not in (None, "") else None,
        "market_cap_display": str(mcap) if mcap not in (None, "") else "Data unavailable",
        "current_price": price if price not in (None, "") else None,
        "current_price_display": str(price) if price not in (None, "") else "Data unavailable",
        "high_52w": high if high not in (None, "") else None,
        "high_52w_display": str(high) if high not in (None, "") else "Data unavailable",
        "low_52w": low if low not in (None, "") else None,
        "low_52w_display": str(low) if low not in (None, "") else "Data unavailable",
        "promoter_holding": holding("promoter"),
        "fii_holding": holding("fii"),
        "dii_holding": holding("dii"),
        "public_holding": holding("public"),
        "promoter_pledge": holding("pledge"),
        "shareholding_period": holding_period("promoter") or holding_period("fii") or "Data unavailable",
        "latest_reported_quarter": as_of.get("latest_financial_period") or "Data unavailable",
        "latest_annual_report": as_of.get("latest_annual_report") or "Data unavailable",
        "data_freshness": as_of.get("fundamentals_freshness") or "MISSING",
        "about": about or "Data unavailable",
        "selected_by": _selected_by(technical),
        "sub_sector": str(profile.get("sub_sector") or ""),
        "business_model": str(profile.get("business_model") or "Data unavailable"),
        "framework_id": str(profile.get("framework_id") or ""),
        "classification_note": str(profile.get("classification_note") or ""),
    }


def attach_fundamental_intelligence(report: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the investor scorecard to a completed report without creating new evidence.

    ``StockResearchEngine`` already owns acquisition and measurement. This is a pure
    projection over that completed evidence. The mutation is intentional because the
    existing Fundamentals panel reads ``fundamental_quality.breakdown.pillars``.
    """
    intelligence = build_fundamental_intelligence(report)
    if not isinstance(report, dict):
        return intelligence

    report["fundamental_intelligence"] = intelligence
    quality = report.get("fundamental_quality")
    if not isinstance(quality, dict):
        return intelligence
    breakdown = quality.get("breakdown")
    if not isinstance(breakdown, dict):
        return intelligence

    existing = list(breakdown.get("sector_pillars") or breakdown.get("pillars") or [])
    # Idempotence: if an already-enriched report is projected again, preserve the
    # original sector pillars instead of nesting the intelligence rows repeatedly.
    sector_pillars = [
        p for p in existing
        if not str((p or {}).get("id") or "").startswith("intelligence_")
        and str((p or {}).get("id") or "") != "fundamental_intelligence_total"
    ]
    breakdown["sector_pillars"] = sector_pillars
    breakdown["intelligence_score"] = intelligence.get("score")
    breakdown["intelligence_label"] = intelligence.get("label")
    breakdown["intelligence_coverage_pct"] = intelligence.get("coverage_pct")
    breakdown["intelligence_missing_components"] = list(intelligence.get("missing_components") or [])
    breakdown["pillars"] = [
        dict(intelligence.get("summary_pillar") or {}),
        *[dict(x) for x in list(intelligence.get("components") or [])],
        *sector_pillars,
    ]
    return intelligence


def first_screen(report: Mapping[str, Any]) -> dict[str, Any]:
    intelligence = attach_fundamental_intelligence(report)
    quality = dict(report.get("fundamental_quality") or {})
    technical = dict(report.get("technical_context") or {})
    flags = dict(report.get("flag_groups") or {})
    vs = str(report.get("vs_technical_setup") or "UNMEASURED")
    confirmation = str(report.get("fundamental_confirmation") or confirmation_label(vs))
    snapshot = dict(report.get("company_snapshot") or {})
    coverage = dict(report.get("research_coverage") or {})
    decision = dict(report.get("decision_coverage") or {})
    audit = dict(report.get("framework_audit") or {})
    as_of = dict(report.get("as_of") or {})
    missing = list(report.get("missing_evidence") or [])
    confirmation_reason = str(report.get("confirmation_reason") or "")
    qualifier = str(report.get("confirmation_qualifier") or "")
    return {
        "company": snapshot.get("company") or report.get("company"),
        "ticker": report.get("symbol"),
        "selected_by": snapshot.get("selected_by") or _selected_by(technical),
        "technical_score": technical.get("scanner_score"),
        "sepa_score": technical.get("sepa_score"),
        "breakout_quality": technical.get("breakout_quality"),
        "fundamental_quality": quality.get("score"),
        "fundamental_quality_label": quality.get("label"),
        "fundamental_intelligence_score": intelligence.get("score"),
        "fundamental_intelligence_label": intelligence.get("label"),
        "fundamental_intelligence_coverage_pct": intelligence.get("coverage_pct"),
        "fundamental_intelligence_missing": list(intelligence.get("missing_components") or []),
        "score_coverage_pct": quality.get("score_coverage_pct") if quality.get("score_coverage_pct") is not None else quality.get("coverage_pct"),
        "research_coverage_pct": coverage.get("coverage_pct"),
        "research_coverage_summary": coverage.get("summary"),
        "research_coverage_needs_acquire": bool(coverage.get("needs_acquire")),
        "data_coverage_pct": coverage.get("coverage_pct"),
        "implementation_coverage_pct": (
            audit.get("implementation_coverage_pct")
            if audit.get("implementation_coverage_pct") is not None
            else report.get("implementation_coverage_pct")
        ),
        "implementation_coverage_summary": audit.get("summary") or "",
        "framework_audit_metrics": list(audit.get("decision_metrics") or []),
        "decision_coverage_pct": decision.get("coverage_pct") if decision.get("coverage_pct") is not None else report.get("decision_coverage_pct"),
        "fundamental_confirmation": confirmation,
        "confirmation_reason": confirmation_reason,
        "confirmation_qualifier": qualifier,
        "vs_detail": report.get("vs_detail"),
        "business_trend": report.get("business_trend"),
        "earnings_trend": report.get("earnings_quality"),
        "sector_kpis": report.get("sector_kpi_label") or (
            report.get("framework", {}).get("label") if report.get("kpis") else "Unmeasured"
        ),
        "sector_kpi_framework": report.get("framework", {}).get("label"),
        "sector_kpi_detail": report.get("sector_kpi_detail"),
        "sub_sector": snapshot.get("sub_sector") or report.get("profile", {}).get("sub_sector") or "",
        "business_model": snapshot.get("business_model") or report.get("profile", {}).get("business_model") or "",
        "critical_metrics_missing": list(report.get("critical_metrics_missing") or []),
        "missing_evidence": missing,
        "deeper_acquire_available": bool(report.get("deeper_acquire_available")),
        "balance_sheet": report.get("balance_sheet_quality"),
        "critical_red_flags": flags.get("n_critical", 0),
        "warnings": flags.get("n_warnings", 0),
        "latest_financial_quarter": snapshot.get("latest_reported_quarter") or "Data unavailable",
        "latest_data_refresh": as_of.get("latest_data_refresh") or "Data unavailable",
        "data_freshness": snapshot.get("data_freshness") or "MISSING",
        "improving": list(report.get("strengths") or []),
        "deteriorating": list(report.get("concerns") or []),
        "recent_material_events": [
            {
                "date": e.get("published_at") or "date unavailable",
                "headline": e.get("headline"),
                "category": e.get("category") or e.get("event_type"),
                "materiality": e.get("materiality") or "Unmeasured",
                "source": e.get("source"),
                "url": e.get("url"),
            }
            for e in list(report.get("events") or [])[:5]
        ],
        "technical_reason": list(technical.get("reasons") or technical.get("signals") or [])[:4],
        "fundamental_evidence": [
            *(list(report.get("strengths") or [])[:3]),
            *(list(report.get("concerns") or [])[:2]),
        ],
        "sections": [
            "Overview", "Thesis Breakers", "Fundamentals", "Sector KPIs", "Quarterly", "Annual",
            "Cash Flow", "Peers", "Shareholding", "Valuation", "News",
            "Filings", "Red Flags", "Sources",
        ],
    }


def cache_schedule(as_of: Mapping[str, Any]) -> list[dict[str, Any]]:
    fetched = str(as_of.get("fundamentals_fetched_at") or "")
    news = str(as_of.get("latest_material_news") or "")
    acquired = str(as_of.get("autonomy_acquired_at") or "")
    return [
        {"class": "Live price", "refresh": "Very frequent / current", "last_checked_at": fetched or "Data unavailable", "next_refresh_at": "On next price poll"},
        {"class": "News", "refresh": "Frequent", "last_checked_at": news or "Data unavailable", "next_refresh_at": "Hourly when curator runs"},
        {"class": "Exchange filings", "refresh": "Frequent", "last_checked_at": acquired or "Data unavailable", "next_refresh_at": "On Acquire / desk pipeline"},
        {"class": "Quarterly results", "refresh": "Around result announcements", "last_checked_at": fetched or "Data unavailable", "next_refresh_at": "After next result filing"},
        {"class": "Shareholding", "refresh": "Quarterly", "last_checked_at": fetched or "Data unavailable", "next_refresh_at": "Next shareholding filing"},
        {"class": "Annual reports", "refresh": "Yearly", "last_checked_at": str(as_of.get("latest_annual_report") or "Data unavailable"), "next_refresh_at": "Next annual report"},
        {"class": "Peer fundamentals", "refresh": "Periodic", "last_checked_at": fetched or "Data unavailable", "next_refresh_at": "On fundamentals refresh"},
    ]
