"""Rule-based desk synthesis. Not a language model and not a number source."""
from __future__ import annotations

from typing import Any, Mapping


def compose_thesis(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Turn already-measured fields into a readable paragraph.

    Every clause is copied from the report. Empty inputs stay empty.
    """
    symbol = str(payload.get("symbol") or "")
    company = str(payload.get("company") or symbol)
    framework = dict(payload.get("framework") or {})
    technical = dict(payload.get("technical_context") or {})
    quality = dict(payload.get("fundamental_quality") or {})
    vs_setup = str(payload.get("vs_technical_setup") or "UNMEASURED")
    vs_detail = str(payload.get("vs_detail") or "")
    news = str(payload.get("news_event_impact") or "Neutral")
    trend = str(payload.get("business_trend") or "Unmeasured")
    strengths = [str(x) for x in (payload.get("strengths") or []) if x][:3]
    concerns = [str(x) for x in (payload.get("concerns") or []) if x][:3]
    unavailable = [str(x) for x in (payload.get("unavailable") or []) if x][:6]
    guidance = list(payload.get("extracted_guidance") or [])
    basis = ["framework", "technical_context", "fundamental_quality", "vs_technical_setup"]

    parts: list[str] = []
    parts.append(
        f"{company} ({symbol}) is researched on the {framework.get('label') or 'generic'} framework."
    )
    if technical.get("available"):
        sepa = technical.get("sepa_score")
        sepa_bit = f" SEPA {sepa}/100." if sepa is not None else ""
        parts.append(
            f"The saved scanner setup is {technical.get('scanner_status') or 'on file'}."
            f"{sepa_bit}"
        )
        basis.append("scan_row")
    else:
        parts.append("No current scanner setup is on file, so vs-setup stays UNMEASURED.")

    score = quality.get("score")
    coverage = quality.get("coverage_pct")
    decision_pct = (payload.get("decision_coverage") or {}).get("coverage_pct")
    if decision_pct is None:
        decision_pct = payload.get("decision_coverage_pct")
    if score is None:
        miss = ", ".join(unavailable) if unavailable else "sector KPIs"
        parts.append(
            f"Fundamental quality is Unmeasured"
            f"{f' ({coverage}% score coverage)' if coverage is not None else ''}. "
            f"Still missing: {miss}."
        )
    else:
        extra = f"; decision coverage {decision_pct}%" if decision_pct is not None else ""
        parts.append(
            f"Fundamental quality is {score}/100 ({quality.get('label')}); "
            f"score coverage {coverage}%; business trend is {trend}{extra}."
        )
        basis.append("sector_kpis")

    missing_ev = [str(x.get("label") or x.get("id") or "") for x in (payload.get("missing_evidence") or []) if x][:6]
    if missing_ev:
        parts.append("Important missing evidence: " + ", ".join(missing_ev) + ".")
        basis.append("missing_evidence")
    sector_label = str(payload.get("sector_kpi_label") or "")
    if sector_label:
        parts.append(f"{framework.get('label') or 'Sector'} KPIs: {sector_label}.")

    if strengths:
        parts.append("Measured strengths: " + "; ".join(strengths) + ".")
        basis.append("strengths")
    if concerns:
        parts.append("Measured concerns: " + "; ".join(concerns) + ".")
        basis.append("concerns")

    parts.append(f"Material news impact is {news}.")
    if guidance:
        first = dict(guidance[0])
        tone = first.get("tone") or "Unmeasured"
        excerpt = str(first.get("excerpt") or "").strip()
        if tone != "Unmeasured":
            bit = f"Filing/commentary tone is {tone}"
            if excerpt:
                bit += f' (“{excerpt[:160]}”)'
            parts.append(bit + ".")
            basis.append("extracted_guidance")

    parts.append(f"Therefore vs the technical setup: {vs_setup} — {vs_detail}".rstrip(" —"))
    parts.append("This paragraph is a rule-based synthesis of files on disk. It is not a language model.")
    return {
        "kind": "rule_based_synthesis",
        "not_an_llm": True,
        "text": " ".join(parts),
        "basis": basis,
    }
