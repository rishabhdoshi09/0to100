"""Per-category PIT coverage, replay grade, data debt, and courtroom notes.

Categories stay separate. One percentage is not allowed to hide a hole.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from product.pit_availability import (
    PIT_MARKET_ONLY,
    PIT_PARTIAL,
    PIT_STRONG,
    PIT_UNAVAILABLE,
    PIT_UNVERIFIED,
)
from product.pit_query import (
    get_financial_snapshot,
    get_research_snapshot,
    get_sector_context,
    replay_grade_for_symbol,
)
from product.pit_warehouse import (
    DOC_ANNUAL_REPORT,
    DOC_CORPORATE_ANNOUNCEMENT,
    DOC_CREDIT_RATING,
    DOC_EXCHANGE_FILING,
    DOC_INVESTOR_PRESENTATION,
    DOC_QUARTERLY_RESULT,
    DOC_SHAREHOLDING_PATTERN,
    get_evidence,
    get_evidence_raw,
)

STRONG = "STRONG"
PARTIAL = "PARTIAL"
UNAVAILABLE = "UNAVAILABLE"
UNVERIFIED = "UNVERIFIED"

CATEGORIES = (
    "MARKET_DATA",
    "FINANCIALS",
    "BUSINESS",
    "SECTOR",
    "ANNOUNCEMENTS",
    "SHAREHOLDING",
    "OTHER_FRAMEWORK_EVIDENCE",
)


def _count_types(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get("evidence_type") or "")
        out[key] = out.get(key, 0) + 1
    return out


def _grade_bucket(*, dated: int, unverified: int, strong_n: int = 3) -> str:
    if dated >= strong_n:
        return STRONG
    if dated > 0:
        return PARTIAL
    if unverified > 0:
        return UNVERIFIED
    return UNAVAILABLE


def category_coverage(
    symbol: str,
    *,
    as_of: str,
    market_bars_ok: bool = True,
    path=None,
) -> dict[str, Any]:
    eligible = get_evidence(symbol, as_of=as_of, path=path)
    raw = get_evidence_raw(symbol, path=path)
    unverified = [
        r for r in raw
        if str(r.get("pit_status") or "") == PIT_UNVERIFIED
        or not str(r.get("available_from") or "")
    ]
    by_ok = _count_types(eligible)
    by_bad = _count_types(unverified)
    fin = get_financial_snapshot(symbol, as_of=as_of, path=path)
    research = get_research_snapshot(symbol, as_of=as_of, path=path)
    sector = get_sector_context(symbol, as_of=as_of)
    financials = (
        STRONG if fin.get("numbers_parsed") else
        PARTIAL if fin.get("available") else
        UNVERIFIED if by_bad.get(DOC_QUARTERLY_RESULT) else
        UNAVAILABLE
    )
    business = (
        STRONG if research.get("quality_label") not in {"", "Unmeasured", None} and fin.get("numbers_parsed") else
        PARTIAL if by_ok.get(DOC_ANNUAL_REPORT) else
        UNVERIFIED if by_bad.get(DOC_ANNUAL_REPORT) else
        UNAVAILABLE
    )
    announcements = _grade_bucket(
        dated=by_ok.get(DOC_CORPORATE_ANNOUNCEMENT, 0) + by_ok.get(DOC_EXCHANGE_FILING, 0),
        unverified=by_bad.get(DOC_CORPORATE_ANNOUNCEMENT, 0) + by_bad.get(DOC_EXCHANGE_FILING, 0),
    )
    shareholding = _grade_bucket(
        dated=by_ok.get(DOC_SHAREHOLDING_PATTERN, 0),
        unverified=by_bad.get(DOC_SHAREHOLDING_PATTERN, 0),
        strong_n=1,
    )
    other = _grade_bucket(
        dated=by_ok.get(DOC_INVESTOR_PRESENTATION, 0) + by_ok.get(DOC_CREDIT_RATING, 0),
        unverified=by_bad.get(DOC_INVESTOR_PRESENTATION, 0) + by_bad.get(DOC_CREDIT_RATING, 0),
        strong_n=1,
    )
    categories = {
        "MARKET_DATA": STRONG if market_bars_ok else UNAVAILABLE,
        "FINANCIALS": financials,
        "BUSINESS": business,
        "SECTOR": UNVERIFIED if sector.get("sector") else UNAVAILABLE,
        "ANNOUNCEMENTS": announcements,
        "SHAREHOLDING": shareholding,
        "OTHER_FRAMEWORK_EVIDENCE": other,
    }
    missing = [name for name, grade in categories.items() if grade in {UNAVAILABLE, UNVERIFIED}]
    available = [name for name, grade in categories.items() if grade in {STRONG, PARTIAL}]
    return {
        "symbol": str(symbol).upper(),
        "as_of": str(as_of)[:10],
        "categories": categories,
        "available": available,
        "missing": missing,
        "n_eligible": len(eligible),
        "n_unverified": len(unverified),
        "sector_limitation": sector.get("limitation"),
        "financial_numbers_parsed": bool(fin.get("numbers_parsed")),
    }


def overall_replay_grade(
    symbol: str,
    *,
    as_of: str,
    market_bars_ok: bool = True,
    path=None,
) -> dict[str, Any]:
    """PIT_STRONG is committee-comparable, not 'two PDFs were found'."""
    coverage = category_coverage(symbol, as_of=as_of, market_bars_ok=market_bars_ok, path=path)
    grade = replay_grade_for_symbol(symbol, as_of=as_of, market_bars_ok=market_bars_ok, path=path)
    cats = coverage["categories"]
    if not market_bars_ok:
        overall = PIT_UNAVAILABLE
        reason = "PIT market bars unavailable"
    elif all(cats[k] in {UNAVAILABLE, UNVERIFIED} for k in CATEGORIES if k != "MARKET_DATA") and cats["MARKET_DATA"] == STRONG:
        if coverage["n_unverified"] and coverage["n_eligible"] <= 0:
            overall = PIT_UNVERIFIED
            reason = "company artifacts exist but publication dates are unproven"
        else:
            overall = PIT_MARKET_ONLY
            reason = "OHLCV/regime only — no dated company evidence at T"
    elif grade.get("production_comparable"):
        overall = PIT_STRONG
        reason = (
            "required decision inputs are historically supportable: "
            "official bars, multi-period parsed financials, and derived "
            "quality measures the live committee also consumes"
        )
    else:
        overall = PIT_PARTIAL
        reason = grade.get("reason") or "dated company evidence exists; production-comparable judgment is not"
    out = dict(grade)
    out["grade"] = overall
    out["reason"] = reason
    out["coverage"] = coverage
    out["comparable_to_forward"] = overall == PIT_STRONG
    return out


def explain_downgrade(coverage: Mapping[str, Any], *, decision: str = "", reason_code: str = "") -> dict[str, Any]:
    cats = dict((coverage or {}).get("categories") or {})
    numbers = bool((coverage or {}).get("financial_numbers_parsed"))
    available = []
    missing_for_judgment = []
    for name, grade in cats.items():
        if name == "FINANCIALS" and grade in {STRONG, PARTIAL} and not numbers:
            available.append("FINANCIALS_METADATA")
            missing_for_judgment.append("PARSED_FINANCIAL_FACTS")
        elif grade in {STRONG, PARTIAL}:
            available.append(name)
        elif grade == UNAVAILABLE:
            missing_for_judgment.append(name)
        elif grade == UNVERIFIED:
            missing_for_judgment.append(f"{name}_UNVERIFIED")
    unavailable = [k for k, v in cats.items() if v == UNAVAILABLE]
    unverified = [k for k, v in cats.items() if v == UNVERIFIED]
    return {
        "available": available,
        "unavailable": unavailable,
        "unverified": unverified,
        "missing_for_judgment": missing_for_judgment,
        "decision": decision,
        "reason_code": reason_code or "INSUFFICIENT_INDEPENDENT_EVIDENCE",
        "note": (
            "WAIT/AVOID from missing historical evidence is a data gap, not a silent pass. "
            "A dated result filing is not FINANCIAL_QUALITY. Unknown stays unknown."
        ),
    }


def explain_historical_buy(row: Mapping[str, Any]) -> dict[str, Any]:
    """Courtroom packet. Call only when a historical BUY actually printed."""
    return {
        "date": row.get("as_of"),
        "symbol": row.get("symbol"),
        "market_evidence_cutoff": (row.get("pit") or {}).get("max_bar_date") or row.get("as_of"),
        "financial_report_versions": (row.get("pit_financial") or {}).get("latest_publication"),
        "publication_dates": (row.get("pit_financial") or {}).get("latest_publication"),
        "business_evidence": (row.get("pit_research") or {}).get("answered"),
        "sector_evidence": (row.get("pit_sector") or {}).get("limitation"),
        "independent_families": row.get("evidence_family_votes") or row.get("families"),
        "committee_decision": row.get("decision"),
        "entry": row.get("entry"),
        "stop": row.get("stop"),
        "portfolio_decision": row.get("execution_state") or row.get("raw_decision"),
        "pit_grade": row.get("pit_grade"),
        "future_outcome": {
            "forward_return_pct": row.get("forward_return_pct"),
            "classification": row.get("classification"),
        },
        "versions": row.get("versions"),
    }


def coverage_map(
    symbols: Sequence[str],
    *,
    as_of: str,
    path=None,
) -> dict[str, Any]:
    """Symbol × fiscal-year-ish × evidence class. Years come from available_from."""
    table: dict[str, dict[str, dict[str, str]]] = {}
    for symbol in symbols:
        rows = get_evidence(symbol, as_of=as_of, path=path)
        raw = get_evidence_raw(symbol, path=path)
        years: dict[str, dict[str, str]] = defaultdict(dict)
        for row in raw:
            day = str(row.get("available_from") or row.get("period_end") or "")[:10]
            year = day[:4] if len(day) >= 4 else "unknown"
            kind = str(row.get("evidence_type") or "OTHER")
            eligible = bool(row.get("available_from")) and str(row.get("pit_status") or "") != PIT_UNVERIFIED
            mark = "yes" if eligible else "?"
            years[year][kind] = mark if years[year].get(kind) != "yes" else "yes"
        table[str(symbol).upper()] = dict(years)
        table[str(symbol).upper()]["_eligible_at_as_of"] = str(len(rows))
    return {"as_of": str(as_of)[:10], "symbols": table}


def data_debt(
    decisions: Sequence[Mapping[str, Any]],
    *,
    default_as_of: str = "",
    path=None,
) -> dict[str, Any]:
    """Persistent concept: what historical information is worth acquiring next."""
    missing_fin = 0
    missing_biz = 0
    missing_sector = 0
    unverified_dates = 0
    market_only = 0
    rows = []
    for row in decisions:
        symbol = str(row.get("symbol") or "").upper()
        as_of = str(row.get("as_of") or default_as_of)[:10]
        if not symbol or not as_of:
            continue
        cov = row.get("pit_coverage") or category_coverage(symbol, as_of=as_of, path=path)
        cats = dict(cov.get("categories") or {})
        if str(row.get("pit_grade") or "") == PIT_MARKET_ONLY:
            market_only += 1
        if cats.get("FINANCIALS") in {UNAVAILABLE, UNVERIFIED, PARTIAL} and not cov.get("financial_numbers_parsed"):
            missing_fin += 1
        if cats.get("BUSINESS") in {UNAVAILABLE, UNVERIFIED}:
            missing_biz += 1
        if cats.get("SECTOR") in {UNAVAILABLE, UNVERIFIED}:
            missing_sector += 1
        if int(cov.get("n_unverified") or 0) > 0:
            unverified_dates += 1
        rows.append({
            "symbol": symbol,
            "as_of": as_of,
            "pit_grade": row.get("pit_grade"),
            "missing": cov.get("missing"),
            "decision": row.get("decision"),
        })
    return {
        "n_decisions": len(list(decisions)),
        "could_not_fully_evaluate": market_only + missing_fin,
        "missing_financial_publication_history": missing_fin,
        "missing_business_evidence": missing_biz,
        "missing_sector_context": missing_sector,
        "unverified_source_dates": unverified_dates,
        "market_only": market_only,
        "priority": (
            "financial result publication history with parseable numbers"
            if missing_fin >= missing_biz and missing_fin > 0 else
            "versioned sector membership and dated annual-report metadata"
            if missing_sector or unverified_dates else
            "dated business / annual-report evidence"
        ),
        "rows": rows[:200],
    }
