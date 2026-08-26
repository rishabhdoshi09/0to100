"""Deterministic red flags with severity, threshold and evidence. No LLM."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _flag(
    *,
    flag_id: str,
    title: str,
    kind: str,
    severity: str,
    rule: str,
    triggered_value: Any,
    threshold: Any,
    evidence: str,
    source: str = "",
    source_date: str = "",
    url: str = "",
) -> dict[str, Any]:
    return {
        "id": flag_id,
        "title": title,
        "kind": kind,
        "severity": severity,  # critical | warning | monitor
        "rule": rule,
        "triggered_value": triggered_value,
        "threshold": threshold,
        "evidence": evidence,
        "fact": evidence,
        "source": source or "Source unavailable",
        "source_date": source_date or "date unavailable",
        "url": url,
    }


def collect_red_flags(
    findings: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    extra: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for finding in findings:
        snap = dict(finding.get("snapshot") or {})
        current = snap.get("current")
        if finding.get("id") in {"gnpa", "nnpa"} and finding.get("trend") == "deteriorating":
            yoy = snap.get("yoy_change")
            severity = "critical" if yoy is not None and abs(float(yoy)) >= 0.5 else "warning"
            flags.append(_flag(
                flag_id=f"flag-{finding['id']}",
                title=f"{finding['label']} is deteriorating",
                kind="asset_quality",
                severity=severity,
                rule=f"{finding['label']} YoY change vs 0.25 pt worsening threshold",
                triggered_value=yoy,
                threshold={"yoy_pt": 0.25, "critical_pt": 0.5},
                evidence=str(finding.get("fact") or ""),
                source=str(finding.get("source") or ""),
                source_date=str(finding.get("source_date") or ""),
            ))
        if finding.get("id") == "pledge" and finding.get("available"):
            if finding.get("trend") == "deteriorating" or (current is not None and current > 20):
                severity = "critical" if current is not None and current >= 40 else "warning"
                flags.append(_flag(
                    flag_id="flag-pledge",
                    title="Promoter pledge is elevated or rising",
                    kind="governance",
                    severity=severity,
                    rule="Pledge > 20% or pledge rising (lower is better)",
                    triggered_value=current,
                    threshold=20,
                    evidence=str(finding.get("fact") or ""),
                    source=str(finding.get("source") or ""),
                    source_date=str(finding.get("source_date") or ""),
                ))
        if finding.get("id") == "promoter" and finding.get("trend") == "deteriorating":
            qoq = snap.get("qoq_change")
            if qoq is not None and qoq <= -3:
                flags.append(_flag(
                    flag_id="flag-promoter-drop",
                    title="Promoter holding fell by more than 3 percentage points last quarter",
                    kind="governance",
                    severity="warning",
                    rule="Promoter QoQ change ≤ −3 pp",
                    triggered_value=qoq,
                    threshold=-3,
                    evidence=str(finding.get("fact") or ""),
                    source=str(finding.get("source") or ""),
                    source_date=str(finding.get("source_date") or ""),
                ))
        if finding.get("id") == "opm" and finding.get("trend") == "deteriorating":
            yoy = snap.get("yoy_change")
            if yoy is not None and yoy <= -4:
                flags.append(_flag(
                    flag_id="flag-margin-collapse",
                    title="Operating margin deteriorated sharply",
                    kind="financial",
                    severity="warning",
                    rule="OPM YoY ≤ −4 pp",
                    triggered_value=yoy,
                    threshold=-4,
                    evidence=str(finding.get("fact") or ""),
                    source=str(finding.get("source") or ""),
                    source_date=str(finding.get("source_date") or ""),
                ))
    for event in events:
        category = str(event.get("category") or event.get("event_type") or "")
        material = str(event.get("materiality") or "")
        impact = str(event.get("impact") or "")
        severe_cat = category in {
            "Regulatory Action", "regulatory_action", "Auditor Change", "Plant Shutdown",
            "governance", "pledge",
        }
        if severe_cat or (impact == "negative" and event.get("thesis_change")):
            severity = "critical" if material in {"Very High", "High"} or category in {
                "Regulatory Action", "regulatory_action", "Auditor Change",
            } else "warning"
            flags.append(_flag(
                flag_id=f"flag-news-{event.get('event_type')}",
                title=str(event.get("headline") or "Material event"),
                kind=str(event.get("event_type") or "news"),
                severity=severity,
                rule=f"Material {category or 'event'} with signed negative / regulatory taxonomy",
                triggered_value=material or impact,
                threshold="High materiality or regulatory taxonomy",
                evidence=str(event.get("headline") or ""),
                source=str(event.get("source") or ""),
                source_date=str(event.get("published_at") or ""),
                url=str(event.get("url") or ""),
            ))
    for item in extra or []:
        if not isinstance(item, Mapping):
            continue
        if item.get("id") in {f.get("id") for f in flags}:
            continue
        flags.append(_flag(
            flag_id=str(item.get("id") or f"flag-extra-{len(flags)}"),
            title=str(item.get("title") or item.get("rule") or "Flag"),
            kind=str(item.get("kind") or "financial"),
            severity=str(item.get("severity") or "monitor"),
            rule=str(item.get("rule") or item.get("title") or ""),
            triggered_value=item.get("triggered_value"),
            threshold=item.get("threshold"),
            evidence=str(item.get("evidence") or item.get("fact") or ""),
            source=str(item.get("source") or ""),
            source_date=str(item.get("source_date") or ""),
            url=str(item.get("url") or ""),
        ))
    return flags


def partition_flags(flags: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    critical = [dict(f) for f in flags if f.get("severity") == "critical"]
    warnings = [dict(f) for f in flags if f.get("severity") == "warning"]
    monitor = [dict(f) for f in flags if f.get("severity") not in {"critical", "warning"}]
    return {
        "critical": critical,
        "warnings": warnings,
        "monitor": monitor,
        "n_critical": len(critical),
        "n_warnings": len(warnings),
        "n_monitor": len(monitor),
    }
