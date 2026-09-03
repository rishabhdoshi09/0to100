"""PIT financial facts and derived ratios.

Ratios are recomputed from warehouse facts whose available_from <= T.
Current Screener TTM is never read.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

from product.pit_warehouse import DOC_QUARTERLY_RESULT, get_evidence

PARSER_VERSION = "pit_financials.v1"


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return round(num / den, 4)


def _growth(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev in (None, 0):
        return None
    return round((cur - prev) / abs(prev) * 100.0, 2)


def parsed_results(symbol: str, *, as_of: str, path=None) -> list[dict[str, Any]]:
    rows = []
    for item in get_evidence(symbol, as_of=as_of, evidence_types=(DOC_QUARTERLY_RESULT,), path=path):
        extracted = dict(item.get("extracted") or {})
        if not extracted.get("numbers_parsed"):
            continue
        facts = dict(extracted.get("facts") or {})
        if not facts:
            continue
        nature = str(extracted.get("nature") or "").lower()
        rows.append({
            **item,
            "facts": facts,
            "source_tags": extracted.get("source_tags") or {},
            "nature": extracted.get("nature") or "",
            "consolidated": nature.startswith("consol"),
            "confidence": extracted.get("confidence"),
            "parser_version": extracted.get("parser_version") or item.get("parser_version"),
        })
    preferred: list[dict[str, Any]] = []
    seen_period = set()
    consol = [r for r in rows if r.get("consolidated")]
    pool = consol or rows
    pool.sort(key=lambda r: str(r.get("available_from") or ""), reverse=True)
    for row in pool:
        period = str(row.get("period_end") or "")
        if period in seen_period:
            continue
        seen_period.add(period)
        preferred.append(row)
    return preferred


def get_fact(symbol: str, field: str, *, as_of: str, path=None) -> dict[str, Any]:
    """Latest eligible value for one field. Future artifacts are invisible."""
    for row in parsed_results(symbol, as_of=as_of, path=path):
        facts = row.get("facts") or {}
        if field in facts:
            return {
                "symbol": str(symbol).upper(),
                "field": field,
                "as_of": str(as_of)[:10],
                "value": facts[field],
                "available": True,
                "available_from": row.get("available_from"),
                "period_end": row.get("period_end"),
                "evidence_id": row.get("evidence_id"),
                "source_tag": (row.get("source_tags") or {}).get(field),
                "source": row.get("source"),
                "pit_status": "AVAILABLE_AT_T",
            }
    return {
        "symbol": str(symbol).upper(),
        "field": field,
        "as_of": str(as_of)[:10],
        "value": None,
        "available": False,
        "pit_status": "UNAVAILABLE_AT_T",
    }


def _derive(latest: Mapping[str, Any], previous: Mapping[str, Any] | None) -> dict[str, Any]:
    facts = dict(latest.get("facts") or {})
    prev = dict((previous or {}).get("facts") or {})
    revenue = _f(facts.get("revenue") or facts.get("total_income"))
    pat = _f(facts.get("pat"))
    pbt = _f(facts.get("pbt"))
    finance = _f(facts.get("finance_costs"))
    dep = _f(facts.get("depreciation"))
    equity = _f(facts.get("bs_equity") or facts.get("paid_up_equity"))
    debt = _f(facts.get("bs_total_debt") or facts.get("total_debt"))
    assets = _f(facts.get("bs_total_assets"))
    cash = _f(facts.get("bs_cash"))
    cfo = _f(facts.get("cfo"))
    paid = _f(facts.get("paid_up_equity"))
    face = _f(facts.get("face_value"))
    ebit = None if pbt is None or finance is None else pbt + finance
    ebitda = None if ebit is None or dep is None else ebit + dep
    shares = _ratio(paid, face)
    derived = {
        "pat_margin_pct": None if revenue is None or pat is None else round(pat / revenue * 100.0, 2),
        "pbt_margin_pct": None if revenue is None or pbt is None else round(pbt / revenue * 100.0, 2),
        "ebit_cr": ebit,
        "ebitda_cr": ebitda,
        "ebit_margin_pct": None if revenue is None or ebit is None else round(ebit / revenue * 100.0, 2),
        "revenue_yoy_pct": _growth(revenue, _f(prev.get("revenue") or prev.get("total_income"))),
        "pat_yoy_pct": _growth(pat, _f(prev.get("pat"))),
        "debt_to_equity": _ratio(debt, equity),
        "roe_approx_pct": None if equity in (None, 0) or pat is None else round(pat / equity * 100.0, 2),
        "roce_approx_pct": None if ebit is None or assets in (None, 0) else round(ebit / assets * 100.0, 2),
        "cash_conversion": _ratio(cfo, pat),
        "shares_outstanding_cr": shares,
        "share_count_change_pct": _growth(shares, _ratio(_f(prev.get("paid_up_equity")), _f(prev.get("face_value")))),
        "cash_cr": cash,
        "source_fact_ids": {
            "latest": latest.get("evidence_id"),
            "previous": (previous or {}).get("evidence_id"),
        },
    }
    return {k: v for k, v in derived.items() if v is not None or k == "source_fact_ids"}


def _period_label(period_end: str) -> str:
    d = _day(period_end)
    if not d:
        return period_end or "period"
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    return f"{months[d.month - 1]} {d.year}"


def screener_tables(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Shape PIT facts like Investigate tables so the same quality rules run."""
    chronological = list(reversed(list(rows)[:8]))
    labels = [_period_label(str(r.get("period_end") or "")) for r in chronological]

    def series(field: str, label: str) -> dict[str, Any]:
        out: dict[str, Any] = {"row_label": label}
        for lab, row in zip(labels, chronological):
            val = (row.get("facts") or {}).get(field)
            if val is not None:
                out[lab] = val
        return out

    quarterly = [
        series("revenue", "Sales+"),
        series("pat", "Net Profit+"),
        series("pbt", "Profit before tax"),
        series("finance_costs", "Interest"),
        series("depreciation", "Depreciation"),
        series("employee_expense", "Employee cost"),
    ]
    opm: dict[str, Any] = {"row_label": "OPM %"}
    for lab, row in zip(labels, chronological):
        facts = row.get("facts") or {}
        rev = _f(facts.get("revenue"))
        pbt = _f(facts.get("pbt"))
        fin = _f(facts.get("finance_costs"))
        if rev and pbt is not None:
            ebit = pbt + (fin or 0)
            opm[lab] = round(ebit / rev * 100.0, 2)
    quarterly.append(opm)
    return {
        "quarterly_results": quarterly,
        "profit_loss": [
            series("revenue", "Sales+"),
            series("pat", "Net Profit+"),
            series("pbt", "Profit before tax"),
        ],
        "balance_sheet": [
            series("bs_total_assets", "Total Assets"),
            series("bs_equity", "Equity"),
            series("total_debt", "Borrowings"),
            series("bs_cash", "Cash Equivalents"),
            series("paid_up_equity", "Equity Share Capital"),
        ],
        "cash_flow": [
            series("cfo", "Cash from Operating Activity+"),
            series("cfi", "Cash from Investing Activity"),
            series("cff", "Cash from Financing Activity"),
        ],
    }


def get_financial_snapshot_v2(symbol: str, *, as_of: str, path=None) -> dict[str, Any]:
    rows = parsed_results(symbol, as_of=as_of, path=path)
    latest = rows[0] if rows else None
    previous = None
    if latest:
        for row in rows[1:]:
            if str(row.get("period_end") or "") != str(latest.get("period_end") or ""):
                previous = row
                break
    derived = _derive(latest, previous) if latest else {}
    tables = screener_tables(rows) if rows else {}
    stale = False
    if latest:
        pub = _day(latest.get("available_from"))
        t = _day(as_of)
        if pub and t and (t - pub).days > 400:
            stale = True
    return {
        "symbol": str(symbol).upper(),
        "as_of": str(as_of)[:10],
        "available": bool(latest),
        "numbers_parsed": bool(latest),
        "n_parsed_results": len(rows),
        "latest_publication": (latest or {}).get("available_from"),
        "latest_period_end": (latest or {}).get("period_end"),
        "latest_source_url": (latest or {}).get("source_url"),
        "latest_evidence_id": (latest or {}).get("evidence_id"),
        "nature": (latest or {}).get("nature"),
        "facts": (latest or {}).get("facts") or {},
        "derived": derived,
        "tables": tables,
        "stale_for_production": stale,
        "quality_status": "UNKNOWN" if not latest or stale else "MEASURED",
        "parser_version": PARSER_VERSION,
        "note": (
            "No parsed official XBRL/HTML result on or before T."
            if not latest else
            "Derived only from filings published on or before T."
        ),
    }


def get_business_snapshot(symbol: str, *, as_of: str, path=None) -> dict[str, Any]:
    """Answer only questions the available facts can support. Unknown stays unknown."""
    fin = get_financial_snapshot_v2(symbol, as_of=as_of, path=path)
    facts = dict(fin.get("facts") or {})
    derived = dict(fin.get("derived") or {})
    answered: dict[str, Any] = {}
    unknown = []
    if facts.get("revenue") is not None:
        answered["revenue"] = facts["revenue"]
    else:
        unknown.append("revenue")
    if derived.get("pat_margin_pct") is not None:
        answered["pat_margin_pct"] = derived["pat_margin_pct"]
    else:
        unknown.append("margins")
    if derived.get("revenue_yoy_pct") is not None:
        answered["growth"] = derived["revenue_yoy_pct"]
    else:
        unknown.append("growth")
    if derived.get("debt_to_equity") is not None:
        answered["leverage"] = derived["debt_to_equity"]
    else:
        unknown.append("leverage")
    if facts.get("employee_expense") is not None and facts.get("revenue"):
        answered["employee_intensity_pct"] = round(facts["employee_expense"] / facts["revenue"] * 100.0, 2)
    else:
        unknown.append("cost_structure")
    for name in ("segment_mix", "order_book", "geographic_mix", "customer_dependence", "capacity"):
        unknown.append(name)
    return {
        "symbol": str(symbol).upper(),
        "as_of": str(as_of)[:10],
        "answered": answered,
        "unknown": unknown,
        "pit_unavailable": [] if fin.get("available") else ["company_financials"],
        "quality_label": "Unmeasured" if not answered else "Partial",
        "note": "Framework questions without a dated fact remain UNKNOWN.",
    }
