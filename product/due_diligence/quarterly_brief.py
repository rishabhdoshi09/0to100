"""Read-only, traceable quarterly research brief for the canonical Stock Investigator.

This is a projection of EXISTING measured due-diligence evidence, not a
new data source, scanner, trade signal, or autonomous portfolio decision.
Missing provenance and time-ineligible claims are exposed, never promoted.
"""
from __future__ import annotations

from datetime import date
from math import isfinite
from typing import Any, Mapping


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def check_price_track(
    *, buy: Any = None, cmp: Any = None, target: Any = None,
    reported_vs_buy_pct: Any = None, reported_to_target_pct: Any = None,
    price_as_of: str = "", tolerance_pp: float = 0.2,
) -> dict[str, Any]:
    """Audit third-party price tables without trusting their published percentages.

    An upside is from CMP to target, NOT from the buy price; no future return
    is implied. Price timestamps, splits, target dates and adjustment methodology
    still require independent verification.
    """
    prices = {key: _number(value) for key, value in (
        ("buy", buy), ("cmp", cmp), ("target", target),
    )}
    issues: list[dict[str, Any]] = []
    for key, value in prices.items():
        if value is not None and value <= 0:
            issues.append({"code": "INVALID_PRICE", "field": key})
            prices[key] = None
    versus_buy = (
        round((prices["cmp"] / prices["buy"] - 1) * 100, 2)
        if prices["cmp"] is not None and prices["buy"] is not None else None
    )
    upside = (
        round((prices["target"] / prices["cmp"] - 1) * 100, 2)
        if prices["cmp"] is not None and prices["target"] is not None else None
    )
    for field, reported, calculated in (
        ("vs_buy_pct", reported_vs_buy_pct, versus_buy),
        ("to_target_pct", reported_to_target_pct, upside),
    ):
        claimed = _number(reported)
        if reported not in (None, "") and claimed is None:
            issues.append({"code": "INVALID_REPORTED_RETURN", "field": field})
        elif claimed is not None and calculated is not None:
            if abs(claimed - calculated) > tolerance_pp:
                issues.append({
                    "code": "RETURN_MISMATCH", "field": field,
                    "reported_pct": claimed, "recomputed_pct": calculated,
                    "difference_pp": round(claimed - calculated, 2),
                })
    if not _iso_date(price_as_of):
        issues.append({"code": "PRICE_DATE_UNVERIFIED", "field": "price_as_of"})
    if versus_buy is None or upside is None:
        issues.append({"code": "PRICE_INPUT_INCOMPLETE", "field": "price_track"})
    return {
        "price_as_of": price_as_of or None,
        "prices": prices,
        "computed_vs_buy_pct": versus_buy,
        "computed_to_target_pct": upside,
        "issues": issues,
        "arithmetic_valid": not any(
            i["code"] in {"INVALID_PRICE", "INVALID_REPORTED_RETURN", "RETURN_MISMATCH", "PRICE_INPUT_INCOMPLETE"}
            for i in issues
        ),
        "source_verified": False,
        "not_a_trade_signal": True,
    }


def compose_quarterly_brief(
    report: Mapping[str, Any], *, price_track: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Synthesize only the report's existing findings, flags, dates and gaps.

    All claims reference their source KPI/flag. An ISO-formatted source date
    later than as_of is excluded even if upstream supplied it. Dates that
    cannot be verified are labelled, not silently treated as trustworthy.
    """
    pit = bool(report.get("point_in_time"))
    as_of = (
        str(report.get("pit_as_of") or "")[:10] if pit else
        str((report.get("as_of") or {}).get("generated_at") or "")[:10]
    )
    cutoff = _iso_date(as_of)
    thesis: list[dict[str, Any]] = []
    antithesis: list[dict[str, Any]] = []
    quarterly: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for finding in list(report.get("kpis") or []):
        if not isinstance(finding, Mapping) or finding.get("available") is not True:
            continue
        snap = dict(finding.get("snapshot") or {})
        value = _number(snap.get("current"))
        if value is None:
            continue
        source_date = str(finding.get("source_date") or "")
        parsed_date = _iso_date(source_date)
        if cutoff and parsed_date and parsed_date > cutoff:
            warnings.append({
                "code": "FUTURE_CLAIM_WITHHELD",
                "id": str(finding.get("id") or ""),
                "source_date": source_date,
            })
            continue
        source = str(finding.get("source") or "")
        source_url = str(finding.get("source_url") or "")
        source_status = (
            "SOURCE_MISSING" if not source or source.lower() == "source unavailable"
            else "SOURCE_DATE_UNVERIFIED" if parsed_date is None
            else "REPORTED_IN_RESEARCH"
        )
        item = {
            "id": str(finding.get("id") or ""),
            "label": str(finding.get("label") or finding.get("id") or "Metric"),
            "value": value,
            "period": str(snap.get("current_period") or ""),
            "trend": str(finding.get("trend") or ""),
            "evidence": str(finding.get("fact") or finding.get("interpretation") or ""),
            "source": source or "Source unavailable",
            "source_url": source_url,
            "source_date": source_date,
            "source_status": source_status,
        }
        if len(quarterly) < 8:
            quarterly.append(item)
        trend = item["trend"].lower()
        if trend == "improving" and len(thesis) < 3:
            thesis.append(item)
        elif trend == "deteriorating" and len(antithesis) < 3:
            antithesis.append(item)

    risk_flags = []
    for flag in list(report.get("red_flags") or []):
        if not isinstance(flag, Mapping):
            continue
        severity = str(flag.get("severity") or "monitor").lower()
        if severity not in {"critical", "warning", "monitor"}:
            severity = "monitor"
        source_date = str(flag.get("source_date") or "")
        parsed_date = _iso_date(source_date)
        if cutoff and parsed_date and parsed_date > cutoff:
            warnings.append({
                "code": "FUTURE_FLAG_WITHHELD",
                "id": str(flag.get("id") or ""),
                "source_date": source_date,
            })
            continue
        risk_flags.append({
            "id": str(flag.get("id") or ""),
            "title": str(flag.get("title") or flag.get("rule") or "Risk"),
            "severity": severity,
            "evidence": str(flag.get("evidence") or flag.get("fact") or ""),
            "source": str(flag.get("source") or "Source unavailable"),
            "source_date": source_date,
            "source_url": str(flag.get("url") or ""),
        })
    risk_flags.sort(key=lambda r: {"critical": 0, "warning": 1, "monitor": 2}[r["severity"]])
    for flag in risk_flags[:3]:
        if not any(row.get("id") == flag["id"] for row in antithesis):
            antithesis.append(flag)
    coverage = report.get("decision_coverage") or {}
    coverage_pct = _number(coverage.get("coverage_pct")) if isinstance(coverage, Mapping) else None
    if coverage_pct is None or coverage_pct < 50 or len(quarterly) < 2:
        status = "EVIDENCE_INCOMPLETE"
    elif any(flag["severity"] == "critical" for flag in risk_flags):
        status = "RISK_REVIEW"
    elif antithesis:
        status = "MIXED_EVIDENCE"
    elif thesis:
        status = "MEASURED_IMPROVEMENT"
    else:
        status = "NO_MEASURED_TREND"
    checked_prices = check_price_track(**price_track) if price_track is not None else None
    return {
        "schema_version": 1,
        "symbol": str(report.get("symbol") or "").upper(),
        "as_of": as_of or None,
        "evidence_mode": "POINT_IN_TIME_INPUT" if pit else "CURRENT_RESEARCH_CACHE",
        "review_status": status,
        "thesis": thesis,
        "anti_thesis": antithesis,
        "quarterly_evidence": quarterly,
        "risk_flags": risk_flags,
        "missing_evidence": [
            str(item.get("label") or item.get("id") or "")
            for item in list(report.get("missing_evidence") or [])
            if isinstance(item, Mapping)
        ][:8],
        "watch_next": [str(s) for s in list(report.get("watch_next") or []) if s][:6],
        "quality_warnings": warnings,
        "price_arithmetic": checked_prices,
        "source_verified_independently": False,
        "execution_authority": False,
        "not_a_buy_sell_signal": True,
        "disclaimer": "Evidence summary only; secondary data may be stale. Not a recommendation or live execution approval.",
    }
