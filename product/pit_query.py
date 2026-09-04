"""Temporal query API. Callers do not filter future data themselves.

get_evidence / get_financial_snapshot / get_research_snapshot / get_sector_context
all enforce available_from <= T. There is no path here that returns today's
Screener cache for a historical as_of.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from product.pit_availability import (
    PIT_MARKET_ONLY,
    PIT_PARTIAL,
    PIT_STRONG,
    PIT_UNAVAILABLE,
    PIT_UNVERIFIED,
    grade_replay,
)
from product.pit_events import get_events
from product.pit_financials import get_business_snapshot, get_fact
from product.pit_warehouse import (
    DOC_ANNUAL_REPORT,
    DOC_CORPORATE_ANNOUNCEMENT,
    DOC_CREDIT_RATING,
    DOC_EXCHANGE_FILING,
    DOC_INVESTOR_PRESENTATION,
    DOC_QUARTERLY_RESULT,
    DOC_SHAREHOLDING_PATTERN,
    get_evidence,
)

# Sector membership is a current static map. Do not pretend it is versioned.
SECTOR_CLASSIFICATION_LIMITATION = (
    "Sector labels use the current static NIFTY-comment map. "
    "Historical classification revisions are not versioned. "
    "Replay must not treat this as a production-grade SECTOR_CONTEXT confirm."
)


def get_financial_snapshot(symbol: str, *, as_of: str, path=None) -> dict[str, Any]:
    """Only reports whose publication/filing date is on or before T."""
    from product.pit_financials import get_financial_snapshot_v2

    parsed = get_financial_snapshot_v2(symbol, as_of=as_of, path=path)
    if parsed.get("numbers_parsed"):
        return parsed
    results = get_evidence(symbol, as_of=as_of, evidence_types=(DOC_QUARTERLY_RESULT,), path=path)
    latest = results[0] if results else None
    return {
        "symbol": str(symbol).upper(),
        "as_of": str(as_of)[:10],
        "available": bool(results),
        "n_results": len(results),
        "latest_publication": (latest or {}).get("available_from"),
        "latest_period_end": (latest or {}).get("period_end"),
        "latest_source_url": (latest or {}).get("source_url"),
        "facts": {},
        "derived": {},
        "tables": {},
        "numbers_parsed": False,
        "quality_status": "UNKNOWN",
        "note": (
            "Result metadata is dated. Numbers are not inferred from period labels."
            if results else
            "No financial result with a proven publication date on or before T."
        ),
    }


def get_research_snapshot(symbol: str, *, as_of: str, path=None) -> dict[str, Any]:
    """PIT-compatible research view. Unknown stays unknown."""
    items = get_evidence(symbol, as_of=as_of, path=path)
    by_type: dict[str, int] = {}
    for item in items:
        by_type[str(item.get("evidence_type") or "")] = by_type.get(str(item.get("evidence_type") or ""), 0) + 1
    answered = []
    if by_type.get(DOC_QUARTERLY_RESULT):
        answered.append("financial_result_filed")
    if by_type.get(DOC_SHAREHOLDING_PATTERN):
        answered.append("shareholding_filed")
    if by_type.get(DOC_ANNUAL_REPORT):
        answered.append("annual_report_present")
    biz = get_business_snapshot(symbol, as_of=as_of, path=path)
    for key in (biz.get("answered") or {}):
        if key not in answered:
            answered.append(key)
    unknown = [
        name for name in (
            "business_quality_score", "framework_kpis", "margins",
            "cash_flow_quality", "valuation",
            *(biz.get("unknown") or []),
        )
        if name not in answered
    ]
    # Metadata-only filings stay Unmeasured. Parsed facts can answer Partial.
    quality = "Unmeasured"
    if biz.get("answered"):
        quality = str(biz.get("quality_label") or "Partial")
    return {
        "symbol": str(symbol).upper(),
        "as_of": str(as_of)[:10],
        "available": bool(items),
        "answered": answered,
        "unknown": unknown,
        "pit_unavailable": [] if items else ["company_evidence"],
        "by_type": by_type,
        "coverage": {
            "filed_result": bool(by_type.get(DOC_QUARTERLY_RESULT)),
            "shareholding": bool(by_type.get(DOC_SHAREHOLDING_PATTERN)),
            "annual_report": bool(by_type.get(DOC_ANNUAL_REPORT)),
            "announcements": bool(by_type.get(DOC_CORPORATE_ANNOUNCEMENT) or by_type.get(DOC_EXCHANGE_FILING)),
        },
        "quality_label": quality,
        "vs_technical": "",
        "acquired_at": "",
        "business": biz,
        "note": "PIT research does not load today's autonomy_facts or Screener cache.",
    }


def get_sector_context(symbol: str, *, as_of: str, scan_row: dict[str, Any] | None = None) -> dict[str, Any]:
    sector = ""
    if scan_row:
        sector = str(scan_row.get("sector") or "")
    if not sector:
        try:
            from scan.sector_heat import sector_of

            sector = str(sector_of(symbol) or "")
        except Exception:
            sector = ""
    return {
        "symbol": str(symbol).upper(),
        "as_of": str(as_of)[:10],
        "sector": sector,
        "classification_versioned": False,
        "status": "SECTOR_MEMBERSHIP_APPROXIMATE" if sector else "UNAVAILABLE",
        "limitation": SECTOR_CLASSIFICATION_LIMITATION,
        "usable_as_family_confirm": False,
    }


def company_items_for_grade(symbol: str, *, as_of: str, path=None) -> list[dict[str, Any]]:
    """Shape warehouse rows for pit_availability.grade_replay."""
    out = []
    for row in get_evidence(symbol, as_of=as_of, path=path):
        out.append({
            "id": row.get("evidence_id"),
            "evidence_type": row.get("evidence_type"),
            "period_end": row.get("period_end"),
            "publication_date": row.get("publication_date") or row.get("available_from"),
            "filing_date": row.get("filing_date"),
            "acquired_at": row.get("acquired_at"),
        })
    return out


def attach_pit_to_card(card: dict[str, Any], *, as_of: str, path=None) -> dict[str, Any]:
    """Add inspectable PIT notes. Does not mint fake method BUY chips."""
    out = dict(card)
    symbol = str(out.get("symbol") or "").upper()
    fin = get_financial_snapshot(symbol, as_of=as_of, path=path)
    research = get_research_snapshot(symbol, as_of=as_of, path=path)
    sector = get_sector_context(symbol, as_of=as_of, scan_row=out)
    out["pit_financial"] = fin
    out["pit_research"] = research
    out["pit_sector"] = sector
    out["pit_as_of"] = str(as_of)[:10]
    # Do not add Funds/Sector method passes from metadata-only filings.
    return out


def production_comparable(*, fin: dict[str, Any], research: dict[str, Any]) -> bool:
    """Committee can run the same financial quality path as live Investigate.

    One dated XBRL print is not enough. Trend/margin derivation needs at
    least two parsed periods, and the tables Investigate consumes must
    actually contain those periods. Sector labels remain approximate and
    are not part of this gate.
    """
    derived = dict(fin.get("derived") or {})
    tables = dict(fin.get("tables") or {})
    quarterly = list(tables.get("quarterly_results") or [])
    periods = 0
    for row in quarterly:
        if str(row.get("row_label") or "").startswith("Sales"):
            periods = sum(1 for key, value in row.items() if key != "row_label" and value not in (None, ""))
            break
    answered = [str(x) for x in (research.get("answered") or []) if x not in {"financial_result_filed"}]
    return bool(
        fin.get("numbers_parsed")
        and not fin.get("stale_for_production")
        and int(fin.get("n_parsed_results") or 0) >= 2
        and periods >= 2
        and (
            derived.get("pat_margin_pct") is not None
            or derived.get("revenue_yoy_pct") is not None
            or derived.get("pbt_margin_pct") is not None
        )
        and bool(answered)
    )


def replay_grade_for_symbol(symbol: str, *, as_of: str, market_bars_ok: bool, path=None) -> dict[str, Any]:
    items = company_items_for_grade(symbol, as_of=as_of, path=path)
    grade = grade_replay(
        as_of=as_of,
        market_bars_ok=market_bars_ok,
        company_items=items,
        used_today_fundamentals=False,
        used_today_research=False,
        used_future_bar=not market_bars_ok,
    )
    # PIT_STRONG requires more than "two filings exist".
    fin = get_financial_snapshot(symbol, as_of=as_of, path=path)
    research = get_research_snapshot(symbol, as_of=as_of, path=path)
    comparable = production_comparable(fin=fin, research=research)
    if grade.get("grade") == PIT_STRONG and not comparable:
        grade["grade"] = PIT_PARTIAL
        grade["reason"] = (
            "Dated filings exist but the production committee cannot make a "
            "comparable judgment: parsed multi-period financial facts or "
            "framework answers are missing."
        )
        grade["comparable_to_forward"] = False
    if not items and market_bars_ok:
        grade["grade"] = PIT_MARKET_ONLY
    grade["financial_numbers_parsed"] = bool(fin.get("numbers_parsed"))
    grade["n_parsed_results"] = int(fin.get("n_parsed_results") or 0)
    grade["research_answered"] = list(research.get("answered") or [])
    grade["production_comparable"] = comparable
    return grade


def pit_research_inputs(symbol: str, *, as_of: str, path=None) -> dict[str, Any]:
    """Inputs for StockResearchEngine that cannot leak current-world caches.

    News matches the live curator window: items published in the 90 days
    ending at T, newest 40. Older warehouse rows remain queryable but are
    not dumped into the news layer.
    """
    items = get_evidence(symbol, as_of=as_of, path=path)
    cutoff = ""
    try:
        cutoff = (date.fromisoformat(str(as_of)[:10]) - timedelta(days=90)).isoformat()
    except ValueError:
        cutoff = ""
    news = []
    for row in items:
        if row.get("evidence_type") not in {
            DOC_CORPORATE_ANNOUNCEMENT, DOC_EXCHANGE_FILING, DOC_QUARTERLY_RESULT,
            DOC_CREDIT_RATING, DOC_INVESTOR_PRESENTATION,
        }:
            continue
        published = str(row.get("available_from") or "")[:10]
        if cutoff and published and published < cutoff:
            continue
        news.append({
            "symbol": symbol.upper(),
            "headline": (row.get("extracted") or {}).get("headline") or row.get("evidence_type"),
            "published_at": row.get("available_from"),
            "source": row.get("source"),
            "url": row.get("source_url"),
        })
        if len(news) >= 40:
            break
    from product.pit_financials import get_financial_snapshot_v2

    fin = get_financial_snapshot_v2(symbol, as_of=as_of, path=path)
    tables = dict(fin.get("tables") or {})
    raw = {
        "data": tables,
        "fetched_at": str(fin.get("latest_publication") or ""),
        "point_in_time": True,
        "as_of": str(as_of)[:10],
        "source": "pit_warehouse_xbrl",
        "freshness": "PIT" if fin.get("numbers_parsed") else "MISSING",
    }
    return {
        "scan_payload": {"records": [], "point_in_time": True, "as_of_session": str(as_of)[:10]},
        "long_term_payload": {"records": [], "point_in_time": True},
        "raw_fundamentals": raw,
        "news": news,
    }
