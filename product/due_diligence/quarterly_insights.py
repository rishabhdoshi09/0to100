"""Read-only, point-in-time quarterly thesis/anti-thesis review.

This is a presentation over the canonical PIT financial warehouse, not another
financial dataset, trading model, or analyst recommendation. Only evidence
published by the requested as-of date may contribute measured findings.
"""
from __future__ import annotations

import math
import re
from calendar import monthrange
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Any, Mapping, Sequence

from product.pit_financials import parsed_results

SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._&-]{0,31}$")
FIELDS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue",),
    "pat": ("pat",),
    "total_debt": ("bs_total_debt", "total_debt"),
    "operating_cash_flow": ("cfo",),
    "receivables": ("bs_trade_receivables", "trade_receivables"),
}
LABELS = {
    "revenue": "Revenue",
    "pat": "Profit after tax",
    "total_debt": "Total debt",
    "operating_cash_flow": "Cash from operations",
    "receivables": "Trade receivables",
}
VERSION = "quarterly_insights.v1"

# Avoid treating a lender's liability growth, receivables, or cash flow as
# ordinary manufacturing-company risk. The public read API currently has no
# independently verified as-of sector classification, so it must not assume
# the company is non-financial.
_FINANCIAL_FRAMEWORKS = frozenset({
    "bank", "nbfc", "nbfc_gold", "nbfc_housing", "life_insurance",
    "general_insurance", "broker", "amc", "exchange",
})


def _operating_risk_rules_applicable(framework_id: str) -> bool:
    name = str(framework_id or "").strip().lower()
    if not name or name == "generic" or name in _FINANCIAL_FRAMEWORKS:
        return False
    # Use the existing source of truth for supported sector frameworks, not
    # an independent allowlist that can drift over time.
    from product.due_diligence.sector_frameworks import list_frameworks

    return name in list_frameworks()



def _date(value: Any) -> date | None:
    try:
        text = str(value or "")
        if len(text) != 10:
            return None
        return date.fromisoformat(text)
    except (ValueError, TypeError):
        return None


def _number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fact(row: Mapping[str, Any], metric: str) -> float | None:
    data = row.get("facts")
    if not isinstance(data, Mapping):
        return None
    for key in FIELDS[metric]:
        value = _number(data.get(key))
        if value is not None:
            return value
    return None


def _source(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": str(row.get("evidence_id") or ""),
        "publication_date": str(row.get("publication_date") or row.get("filing_date") or "")[:10],
        "available_from": str(row.get("available_from") or "")[:10],
        "period_start": str(row.get("period_start") or "")[:10],
        "period_end": str(row.get("period_end") or "")[:10],
        "source": str(row.get("source") or ""),
        "source_url": str(row.get("source_url") or ""),
        "parser_version": str(row.get("parser_version") or ""),
    }


def _pct(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline <= 0:
        return None
    result = (current - baseline) / abs(baseline) * 100.0
    return round(result, 2) if math.isfinite(result) else None


def _year_ago(period: date) -> date | None:
    try:
        return period.replace(year=period.year - 1)
    except ValueError:
        return None  # Feb 29 requires an explicit comparable-quarter policy.


def _is_calendar_quarter(row: Mapping[str, Any]) -> bool:
    """Only a proven three-month calendar quarter may be called quarterly.

    NSE result metadata can describe H1 cumulative P&L while XBRL reports
    quarter and year-to-date values. Reject ambiguous or conflicting periods
    until their actual source contexts have been independently reconciled.
    """
    start = _date(row.get("period_start"))
    end = _date(row.get("period_end"))
    parsed = row.get("extracted")
    parsed_start = _date(parsed.get("period_start")) if isinstance(parsed, Mapping) and parsed.get("period_start") else None
    if start is not None and parsed_start is not None and start != parsed_start:
        return False
    start = parsed_start or start
    if start is None or end is None or start.year != end.year:
        return False
    if start.day != 1 or start.month not in (1, 4, 7, 10):
        return False
    if end.month != start.month + 2 or end.day != monthrange(end.year, end.month)[1]:
        return False
    return True


def _evidence_item(
    metric: str, latest: Mapping[str, Any], earlier: Mapping[str, Any] | None
) -> dict[str, Any]:
    current = _fact(latest, metric)
    previous = _fact(earlier, metric) if earlier else None
    change = _pct(current, previous)
    return {
        "metric": metric,
        "label": LABELS[metric],
        "current": current,
        "prior_year_same_quarter": previous,
        "yoy_pct": change,
        "comparison": "SAME_FISCAL_QUARTER_YEAR_AGO" if earlier else "UNAVAILABLE",
        "unit": "INR_CRORE",
        "sources": [_source(r) for r in (latest, earlier) if r is not None],
    }


def compose_quarterly_insights(
    symbol: str,
    *,
    as_of: str,
    rows: Sequence[Mapping[str, Any]],
    framework_id: str = "",
) -> dict[str, Any]:
    """Pure, deterministic reporting seam. Input rows are rechecked for PIT safety.

    The trusted production caller passes parsed_results(), which already
    enforces warehouse temporal eligibility. This second filter intentionally
    protects injected test rows and future integrations as well.
    """
    ticker = str(symbol or "").upper().strip()
    day = _date(as_of)
    if not SYMBOL_RE.fullmatch(ticker):
        raise ValueError("Invalid NSE symbol")
    if day is None:
        raise ValueError("as_of must be an ISO date YYYY-MM-DD")
    valid: list[dict[str, Any]] = []
    unverified_periods = 0
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        if str(row.get("symbol") or "").upper() != ticker:
            continue
        published = _date(row.get("publication_date") or row.get("filing_date"))
        available = _date(row.get("available_from"))
        period = _date(row.get("period_end"))
        if (
            published is None or available is None or period is None
            or published > day or available > day or period > day
            or not str(row.get("evidence_id") or "")
            or not isinstance(row.get("facts"), Mapping)
        ):
            continue
        if not _is_calendar_quarter(row):
            unverified_periods += 1
            continue
        valid.append(row)
    # Prefer the same accounting scope for every comparison. Do not conflate
    # consolidated and standalone statements in the numerator/denominator.
    consolidated = [r for r in valid if r.get("consolidated") is True]
    selected = consolidated if consolidated else [
        r for r in valid if r.get("consolidated") is not True
    ]
    selected.sort(
        key=lambda r: (str(r.get("period_end")), str(r.get("available_from"))),
        reverse=True,
    )
    by_period: dict[str, dict[str, Any]] = {}
    for row in selected:
        by_period.setdefault(str(row["period_end"]), row)
    periods = list(by_period.values())[:12]
    latest = periods[0] if periods else None
    year_ago = _year_ago(_date(latest["period_end"])) if latest else None
    prior = by_period.get(year_ago.isoformat()) if year_ago else None

    metrics = {
        metric: _evidence_item(metric, latest, prior)
        for metric in FIELDS
    } if latest else {}
    supporting: list[dict[str, Any]] = []
    challenging: list[dict[str, Any]] = []
    missing: list[str] = []
    risk_framework = str(framework_id or "").strip().lower()
    operating_risk_applicable = _operating_risk_rules_applicable(risk_framework)
    if not operating_risk_applicable:
        missing.append("SECTOR_SPECIFIC_RISK_FRAMEWORK_NOT_VERIFIED")

    if unverified_periods:
        missing.append("NON_QUARTER_OR_UNVERIFIED_REPORTING_PERIOD")
    if not latest:
        missing.extend(["PUBLISHED_PARSED_QUARTER", "THESIS_EVIDENCE", "ANTI_THESIS_EVIDENCE"])
    else:
        if prior is None:
            missing.append("SAME_QUARTER_PRIOR_YEAR")
        for metric in ("revenue", "pat"):
            fact = metrics[metric]
            change = fact["yoy_pct"]
            if change is None:
                # A negative/zero PAT base makes conventional percentage growth
                # misleading. State a verified turnaround or continuing loss
                # rather than printing a fictitious +200% YoY number.
                if metric == "pat" and fact["current"] is not None and fact["prior_year_same_quarter"] is not None:
                    current_pat = fact["current"]
                    prior_pat = fact["prior_year_same_quarter"]
                    if prior_pat <= 0 < current_pat:
                        supporting.append({
                            "id": "pat_loss_to_profit",
                            "metric": "pat",
                            "observation": "PAT moved from a non-positive prior-year base to profit; percentage growth is not meaningful.",
                            "value": current_pat,
                            "unit": "INR_CRORE",
                            "sources": fact["sources"],
                        })
                    elif prior_pat < 0 and current_pat < 0:
                        challenging.append({
                            "id": "pat_still_negative",
                            "metric": "pat",
                            "observation": "PAT remains negative in both comparable quarters; conventional percentage growth is not meaningful.",
                            "value": current_pat,
                            "unit": "INR_CRORE",
                            "sources": fact["sources"],
                        })
                missing.append(f"{metric.upper()}_YOY")
                continue
            note = {
                "id": f"{metric}_yoy",
                "metric": metric,
                "observation": f"{LABELS[metric]} changed {change:+.2f}% against the same quarter last year.",
                "value": change,
                "unit": "PCT",
                "sources": fact["sources"],
            }
            if change > 0:
                supporting.append(note)
            elif change < 0:
                challenging.append(note)
            # A flat comparison is neutral, never supportive.

        # Debt stock comparisons only use reported same-quarter balances.
        debt = metrics["total_debt"]
        if (
            operating_risk_applicable and debt["yoy_pct"] is not None
            and debt["yoy_pct"] >= 25
        ):
            challenging.append({
                "id": "debt_growth",
                "metric": "total_debt",
                "observation": f"Reported debt grew {debt['yoy_pct']:+.2f}% year over year.",
                "value": debt["yoy_pct"],
                "unit": "PCT",
                "sources": debt["sources"],
            })
        current_cfo = metrics["operating_cash_flow"]["current"]
        if operating_risk_applicable and current_cfo is not None and current_cfo < 0:
            challenging.append({
                "id": "negative_operating_cash_flow",
                "metric": "operating_cash_flow",
                "observation": "Reported operating cash flow is negative.",
                "value": current_cfo,
                "unit": "INR_CRORE",
                "sources": metrics["operating_cash_flow"]["sources"][:1],
            })
        if current_cfo is None:
            missing.append("CASH_FLOW_QUALITY")
        recv = metrics["receivables"]
        rev = metrics["revenue"]
        if (
            recv["yoy_pct"] is not None and rev["yoy_pct"] is not None
            and recv["yoy_pct"] > max(20, rev["yoy_pct"] + 15)
            and operating_risk_applicable
        ):
            challenging.append({
                "id": "receivables_outpace_revenue",
                "metric": "receivables",
                "observation": "Receivables growth materially exceeds revenue growth.",
                "value": recv["yoy_pct"],
                "unit": "PCT",
                "sources": recv["sources"] + rev["sources"],
            })
        if _fact(latest, "total_debt") is None:
            missing.append("DEBT_EVIDENCE")
        missing.extend(["MANAGEMENT_GUIDANCE", "VALUATION", "COMPETITIVE_POSITION"])

    if not latest:
        status = "NO_VERIFIABLE_QUARTER"
    elif all(metrics[m]["yoy_pct"] is not None for m in ("revenue", "pat")):
        status = "QUARTER_COMPARABLE"
    else:
        status = "PARTIAL_QUARTER_EVIDENCE"
    published_on = _date(latest.get("publication_date") or latest.get("filing_date")) if latest else None
    days_old = (day - published_on).days if published_on else None
    freshness = (
        "UNAVAILABLE" if days_old is None else
        "STALE" if days_old > 135 else "RECENT"
    )
    return {
        "schema_version": 1,
        "engine": VERSION,
        "symbol": ticker,
        "as_of": day.isoformat(),
        "status": status,
        "publication_freshness": freshness,
        "days_since_publication": days_old,
        "latest_period_end": str(latest.get("period_end")) if latest else None,
        "accounting_scope": "CONSOLIDATED" if latest and latest.get("consolidated") else (
            "STANDALONE_OR_UNSPECIFIED" if latest else "UNKNOWN"
        ),
        "comparison_period_end": year_ago.isoformat() if prior and year_ago else None,
        "metrics": metrics,
        "thesis_evidence": supporting,
        "anti_thesis_evidence": challenging,
        "unknowns": sorted(set(missing)),
        "source_documents": [_source(r) for r in [latest, prior] if r is not None],
        "price_track": {"status": "UNAVAILABLE", "reason": "No independently verified quote, buy price or valuation target is supplied by the PIT financial warehouse."},
        "conviction": {"status": "NOT_CALIBRATED", "win_probability": None},
        "decision": "RESEARCH_ONLY",
        "places_orders": False,
        "may_override_risk": False,
        "note": (
            "Rule-based dated financial observations, not an AI prediction or buy/sell signal. "
            "Absence of an anti-thesis trigger does not establish absence of risk. "
            "All missing research remains explicitly unknown."
        ),
    }


def quarterly_insights(
    symbol: str,
    *,
    as_of: str | None = None,
    path=None,
    framework_id: str = "",
) -> dict[str, Any]:
    """Query canonical PIT storage only; never fetch network data or mutate state."""
    ticker = str(symbol or "").strip().upper()
    cutoff = as_of or datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    # Validate prior to touching the warehouse.
    if _date(cutoff) is None or not SYMBOL_RE.fullmatch(ticker):
        raise ValueError("Invalid symbol or as_of date")
    return compose_quarterly_insights(
        ticker,
        as_of=cutoff,
        rows=parsed_results(ticker, as_of=cutoff, path=path),
        framework_id=framework_id,
    )
