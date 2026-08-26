"""Deterministic evidence-sufficiency rules. Missing is unknown, never zero."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

DECISION_IMPORTANCE = frozenset({"critical", "important"})
SCORE_IMPORTANCE = frozenset({"critical", "important", "supporting"})
OPTIONAL_IMPORTANCE = "optional"

DEFAULT_MIN_CRITICAL = 2
DEFAULT_MIN_DECISION_COVERAGE = 0.45
DEFAULT_MIN_SCORE_COVERAGE = 0.40
STRONG_SUPPORT_DECISION = 0.70
LOW_EVIDENCE_DECISION = 0.60
NEUTRAL_DECISION = 0.40

SECTOR_VERDICTS = ("Strong", "Healthy", "Mixed", "Weak", "Insufficient Evidence")

MISSING_REASON = {
    "not_yet_acquired": "Dataset not yet acquired",
    "acquisition_failed": "Acquisition failed",
    "source_unavailable": "Source unavailable",
    "metric_not_reported": "Metric not reported in acquired structured source",
    "not_implemented": "No validated acquisition path yet",
    "not_applicable": "Not applicable for this issuer",
    "reported": "Reported",
}


def importance_of(finding: Mapping[str, Any]) -> str:
    value = str(finding.get("importance") or "").strip().lower()
    if value in {"critical", "important", "supporting", "optional"}:
        return value
    if finding.get("missing_ok"):
        return "optional"
    return "supporting"


def _weight(finding: Mapping[str, Any]) -> float:
    try:
        return float(finding.get("weight") or 0)
    except (TypeError, ValueError):
        return 0.0


def _available(finding: Mapping[str, Any]) -> bool:
    return bool(finding.get("available") and finding.get("points") is not None)


def decision_coverage(findings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Importance-weighted share of critical+important evidence that is actually measured."""
    pool = [f for f in findings if importance_of(f) in DECISION_IMPORTANCE]
    total = sum(_weight(f) for f in pool)
    used = sum(_weight(f) for f in pool if _available(f))
    coverage = (used / total) if total else 0.0
    critical = [f for f in pool if importance_of(f) == "critical"]
    critical_n = len(critical)
    critical_available = sum(1 for f in critical if _available(f))
    missing = [
        {
            "id": f.get("id"),
            "label": f.get("label"),
            "importance": importance_of(f),
        }
        for f in pool
        if not _available(f)
    ]
    return {
        "coverage": coverage,
        "coverage_pct": round(coverage * 100.0, 1),
        "used_weight": used,
        "total_weight": total,
        "critical_n": critical_n,
        "critical_available": critical_available,
        "missing": missing,
    }


def score_evidence(
    findings: Sequence[Mapping[str, Any]],
    *,
    min_score_coverage: float = DEFAULT_MIN_SCORE_COVERAGE,
) -> dict[str, Any]:
    """Raw awarded points / evaluated weight. Missing KPIs are skipped, not zero-filled.

    Display a /100 score only when evaluated weight clears min_score_coverage.
    """
    pool = [
        f for f in findings
        if importance_of(f) in SCORE_IMPORTANCE and f.get("implemented", True)
    ]
    usable = [f for f in pool if _available(f)]
    total_w = sum(_weight(f) for f in pool)
    used_w = sum(_weight(f) for f in usable)
    coverage = (used_w / total_w) if total_w else 0.0
    raw_awarded = sum(float(f["points"]) / 100.0 * _weight(f) for f in usable)
    normalized = (raw_awarded / used_w * 100.0) if used_w else None
    display = None
    if usable and coverage >= min_score_coverage and normalized is not None:
        display = int(round(normalized))
    return {
        "score": display,
        "normalized": None if normalized is None else round(normalized, 1),
        "raw_awarded": round(raw_awarded, 2),
        "evaluated_weight": round(used_w, 2),
        "scoring_weight": round(total_w, 2),
        "coverage": coverage,
        "coverage_pct": round(coverage * 100.0, 1),
        "n": len(usable),
        "used_weight": used_w,
        "total_weight": total_w,
        "unmeasured_because": (
            None if display is not None else
            "No measurable sector KPIs" if not usable else
            f"Score coverage {round(coverage * 100.0, 1)}% is below {int(min_score_coverage * 100)}%"
        ),
    }


def sector_kpi_verdict(
    findings: Sequence[Mapping[str, Any]],
    *,
    min_critical: int = DEFAULT_MIN_CRITICAL,
    min_decision_coverage: float = DEFAULT_MIN_DECISION_COVERAGE,
    cycle_aware: bool = False,
) -> dict[str, Any]:
    """Emit Strong/Healthy/Mixed/Weak only when enough important evidence exists."""
    decision = decision_coverage(findings)
    if decision["critical_available"] < int(min_critical) or decision["coverage"] < float(min_decision_coverage):
        return {
            "label": "Insufficient Evidence",
            "reason": "Insufficient Evidence",
            "detail": (
                f"{decision['critical_available']}/{decision['critical_n']} critical KPIs measured; "
                f"decision coverage {decision['coverage_pct']}%."
            ),
            "decision": decision,
        }
    useful = [
        str(f.get("trend") or "unknown")
        for f in findings
        if _available(f) and importance_of(f) in DECISION_IMPORTANCE
        and str(f.get("trend") or "unknown") != "unknown"
    ]
    improving = useful.count("improving")
    deteriorating = useful.count("deteriorating")
    stable = useful.count("stable")
    critical_trends = [
        str(f.get("trend") or "unknown")
        for f in findings
        if _available(f) and importance_of(f) == "critical"
        and str(f.get("trend") or "unknown") != "unknown"
    ]
    critical_det = critical_trends.count("deteriorating")
    critical_imp = critical_trends.count("improving")
    if not useful:
        label = "Healthy"
        reason = "Fundamentally Neutral"
    elif deteriorating and not improving:
        label = "Weak"
        reason = "Fundamental Caution"
    elif critical_det > critical_imp:
        label = "Weak"
        reason = "Fundamental Caution"
    elif deteriorating and improving:
        if improving >= deteriorating + 2 and critical_det == 0:
            label = "Healthy"
            reason = "Positive Confirmation"
        else:
            label = "Mixed"
            reason = "Mixed Fundamentals"
    elif improving and not deteriorating:
        label = "Strong" if improving >= max(2, stable) else "Healthy"
        reason = "Positive Confirmation"
    else:
        label = "Healthy"
        reason = "Fundamentally Neutral"
    if cycle_aware and label == "Strong":
        label = "Healthy"
        reason = "Positive Confirmation"
    return {
        "label": label,
        "reason": reason,
        "detail": (
            f"{improving} improving / {stable} stable / {deteriorating} deteriorating "
            f"among measured critical+important KPIs."
        ),
        "decision": decision,
    }


def missing_evidence(findings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Critical and important KPIs that cannot be evaluated."""
    out: list[dict[str, Any]] = []
    for finding in findings:
        if _available(finding) or importance_of(finding) not in DECISION_IMPORTANCE:
            continue
        state = str(finding.get("availability_state") or "")
        reason = MISSING_REASON.get(state, "Metric not reliably extracted")
        if state in {"", "reported"}:
            reason = "Metric not reliably extracted"
        out.append({
            "id": finding.get("id"),
            "label": finding.get("label"),
            "importance": importance_of(finding),
            "availability_state": state or "metric_not_reported",
            "reason": reason,
        })
    return out


def critical_metrics_missing(findings: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(item.get("label") or item.get("id") or "")
        for item in missing_evidence(findings)
        if item.get("importance") in DECISION_IMPORTANCE and item.get("label")
    ]


def confirmation_from_evidence(
    *,
    vs_setup: str,
    vs_detail: str,
    score: int | None,
    decision_coverage_pct: float,
) -> dict[str, Any]:
    """Map SUPPORT/NEUTRAL/CAUTION onto an evidence-sufficiency reason.

    Distinguishes NEUTRAL because mixed from NEUTRAL because we do not know enough.
    """
    label = str(vs_setup or "NEUTRAL").upper()
    detail = str(vs_detail or "")
    coverage = float(decision_coverage_pct or 0.0) / 100.0
    qualifier = ""
    reason = "Fundamentally Neutral"

    if label == "UNMEASURED":
        return {
            "vs_setup": "UNMEASURED",
            "vs_detail": detail or "No current scanner setup to compare against.",
            "reason": "Insufficient Evidence" if score is None else "Fundamentally Neutral",
            "qualifier": "",
            "display": "NEUTRAL",
        }

    if label == "NEUTRAL" and (score is None or coverage < NEUTRAL_DECISION):
        return {
            "vs_setup": "NEUTRAL",
            "vs_detail": detail or "Insufficient fundamental evidence to raise or cut conviction.",
            "reason": "Insufficient Evidence",
            "qualifier": "",
            "display": "NEUTRAL — insufficient fundamental evidence",
        }

    if coverage < NEUTRAL_DECISION and label not in {"CONTRADICTS", "STRONGLY CONTRADICTS", "CAUTION"}:
        return {
            "vs_setup": "NEUTRAL",
            "vs_detail": "Insufficient fundamental evidence to raise or cut conviction.",
            "reason": "Insufficient Evidence",
            "qualifier": "",
            "display": "NEUTRAL — insufficient fundamental evidence",
        }

    if label == "STRONGLY SUPPORTS" and coverage < STRONG_SUPPORT_DECISION:
        label = "SUPPORTS"
        detail = detail or "Fundamentals do not contradict the setup."
    if coverage < LOW_EVIDENCE_DECISION and label in {"STRONGLY SUPPORTS", "SUPPORTS"}:
        label = "SUPPORTS"
        qualifier = "Low Evidence"
        if "incomplete" not in detail.lower():
            detail = (
                f"{detail} Decision coverage {round(coverage * 100.0, 1)}% — "
                "important sector evidence is still missing."
            ).strip()
    if coverage < LOW_EVIDENCE_DECISION and label == "STRONGLY CONTRADICTS":
        label = "CONTRADICTS"
        qualifier = "Low Evidence"

    if label in {"STRONGLY SUPPORTS", "SUPPORTS"}:
        reason = "Positive Confirmation"
    elif label == "CAUTION":
        reason = "Fundamental Caution"
    elif label in {"CONTRADICTS", "STRONGLY CONTRADICTS"}:
        reason = "Fundamental Contradiction"
    elif score is not None and 40 <= score < 60:
        reason = "Mixed Fundamentals"
    elif score is None or coverage < NEUTRAL_DECISION:
        reason = "Insufficient Evidence"
    else:
        reason = "Fundamentally Neutral"

    if qualifier:
        display = "SUPPORT (Low Evidence)" if label == "SUPPORTS" else f"{label.replace('SUPPORTS', 'SUPPORT').title()} ({qualifier})"
        if label == "CONTRADICTS":
            display = "CONTRADICTS (Low Evidence)"
    else:
        display = {
            "STRONGLY SUPPORTS": "STRONG SUPPORT",
            "SUPPORTS": "SUPPORT",
            "NEUTRAL": "NEUTRAL",
            "CAUTION": "CAUTION",
            "CONTRADICTS": "CONTRADICTS",
            "STRONGLY CONTRADICTS": "CONTRADICTS",
        }.get(label, label)

    if reason == "Insufficient Evidence" and label == "NEUTRAL":
        display = "NEUTRAL — insufficient fundamental evidence"
    elif reason == "Mixed Fundamentals" and label == "NEUTRAL":
        display = "NEUTRAL — mixed fundamentals"

    return {
        "vs_setup": label,
        "vs_detail": detail,
        "reason": reason,
        "qualifier": qualifier,
        "display": display,
    }
