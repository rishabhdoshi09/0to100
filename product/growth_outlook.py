"""Growth & Financial Outlook — evidence-only analysis for Stock Intelligence.

Builds a structured research brief from:
  • Screener deep fundamentals (P&L / quarterly / key ratios / extracted CAGRs)
  • Uploaded financial_history, management_commentary, order_book_guidance
  • Official price technicals when provided

Never invents sales targets, margin forecasts, concall quotes, or prices.
Missing evidence stays MISSING and is listed under open gaps.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        if isinstance(value, str):
            text = value.replace(",", "").replace("%", "").replace("₹", "").strip()
            if text in {"", "-", "--", "N/A", "—"}:
                return None
            value = text
        result = float(value)
        if not math.isfinite(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _label(row: Mapping[str, Any]) -> str:
    for key in ("row_label", "label", "", "Particulars", "particulars"):
        if key in row and str(row.get(key) or "").strip():
            return str(row.get(key)).strip().lower()
    # First string cell often is the row label.
    for value in row.values():
        if isinstance(value, str) and value.strip() and not re.fullmatch(r"[-+]?\d[\d,.]*", value.strip()):
            return value.strip().lower()
    return ""


def _numeric_values(row: Mapping[str, Any]) -> list[float]:
    out: list[float] = []
    for key, value in row.items():
        if str(key).lower() in {"row_label", "label", "particulars", ""}:
            continue
        num = _f(value)
        if num is not None:
            out.append(num)
    return out


def _series(table: Sequence[Mapping[str, Any]] | None, *needles: str) -> list[float]:
    for row in table or []:
        if not isinstance(row, Mapping):
            continue
        label = _label(row)
        if any(n in label for n in needles):
            return _numeric_values(row)
    return []


def _yoy_pct(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    prev, latest = values[-2], values[-1]
    if prev == 0:
        return None
    return round((latest / prev - 1.0) * 100.0, 2)


def _cagr_pct(values: list[float], years: int = 3) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return None
    intervals = min(years, len(clean) - 1)
    start, end = clean[-(intervals + 1)], clean[-1]
    if start <= 0 or end < 0:
        return None
    try:
        return round(((end / start) ** (1.0 / intervals) - 1.0) * 100.0, 2)
    except Exception:
        return None


def _margin_series(sales: list[float], profit: list[float]) -> list[float | None]:
    out: list[float | None] = []
    n = min(len(sales), len(profit))
    for i in range(n):
        if sales[-(n - i)] and sales[-(n - i)] != 0:
            out.append(round(profit[-(n - i)] / sales[-(n - i)] * 100.0, 2))
        else:
            out.append(None)
    return out


def _claim(
    key: str,
    label: str,
    value: Any,
    *,
    unit: str = "",
    source: str,
    as_of: str = "",
    note: str = "",
) -> dict[str, Any]:
    missing = value is None or value == ""
    return {
        "key": key,
        "label": label,
        "value": None if missing else value,
        "unit": unit,
        "source": source,
        "as_of": as_of or "",
        "status": "MISSING" if missing else "AVAILABLE",
        "note": note,
    }


def _metric_from_workspace(fundamentals: Mapping[str, Any], key: str) -> float | None:
    for row in fundamentals.get("metrics") or []:
        if str(row.get("key")) == key:
            return _f(row.get("value"))
    raw = fundamentals.get("raw_values") or {}
    return _f(raw.get(key))


def _key_ratio(fundamentals: Mapping[str, Any], *needles: str) -> float | None:
    for row in fundamentals.get("key_ratios") or []:
        name = str(row.get("name") or "").lower()
        if any(n in name for n in needles):
            return _f(row.get("value"))
    return None


def _guidance_rows(symbol: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from reporting.evidence_intake import structured_rows

        return (
            list(structured_rows(symbol, "management_commentary") or []),
            list(structured_rows(symbol, "order_book_guidance") or []),
            list(structured_rows(symbol, "financial_history") or []),
        )
    except Exception:
        return [], [], []


def _source_packs(symbol: str) -> list[dict[str, Any]]:
    """Clickable source packs so users can open filings and upload evidence."""
    try:
        from reporting.evidence_intake import RESOURCE_SPECS, resource_links

        links = resource_links(symbol)
    except Exception:
        return []
    # Outlook prioritises packs that unlock the growth/guidance brief.
    priority = (
        "management_commentary",
        "order_book_guidance",
        "financial_history",
        "business_profile",
        "annual_report",
        "business_segments",
        "shareholding_history",
    )
    packs: list[dict[str, Any]] = []
    for key in priority:
        spec = RESOURCE_SPECS.get(key)
        if spec is None:
            continue
        packs.append(
            {
                "key": key,
                "label": spec.label,
                "why": spec.why,
                "instructions": spec.instructions,
                "accepted_extensions": list(spec.accepted_extensions),
                "template_url": f"/evidence/templates/{key}.csv" if spec.template_columns else "",
                "links": list(links.get(key) or []),
                "upload_hint": f"Research Data → {spec.label} → paste source URL + upload file",
            }
        )
    return packs


def _uploaded_margin_trend(financial_history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(r) for r in financial_history if isinstance(r, Mapping) and _f(r.get("ebitda_margin_pct")) is not None]
    if len(rows) < 2:
        return {"available": False, "from": None, "to": None, "delta_pp": None}
    rows.sort(key=lambda r: str(r.get("period_end") or r.get("as_of_date") or ""))
    first = _f(rows[0].get("ebitda_margin_pct"))
    last = _f(rows[-1].get("ebitda_margin_pct"))
    if first is None or last is None:
        return {"available": False, "from": None, "to": None, "delta_pp": None}
    return {
        "available": True,
        "from": first,
        "to": last,
        "delta_pp": round(last - first, 2),
        "periods": len(rows),
        "source": "uploaded financial_history",
        "as_of": str(rows[-1].get("period_end") or rows[-1].get("as_of_date") or ""),
    }


def _thesis(
    *,
    sales_cagr: float | None,
    profit_cagr: float | None,
    sales_yoy: float | None,
    margin_expanding: bool | None,
    guidance_present: bool,
) -> dict[str, Any]:
    engines: list[str] = []
    if sales_cagr is not None and sales_cagr >= 15:
        engines.append("multi-year sales CAGR")
    elif sales_yoy is not None and sales_yoy >= 15:
        engines.append("recent sales growth")
    if profit_cagr is not None and profit_cagr >= 15:
        engines.append("multi-year profit CAGR")
    if margin_expanding is True:
        engines.append("margin expansion in available periods")

    if len(engines) >= 2:
        label = "DOUBLE ENGINE (evidence-backed)"
        text = (
            "Available financial evidence shows more than one growth driver: "
            + " and ".join(engines)
            + ". This is a research label from disclosed numbers — not a buy ticket."
        )
    elif engines:
        label = "SINGLE ENGINE (evidence-backed)"
        text = f"Available financial evidence shows one clear driver: {engines[0]}."
    else:
        label = "INCOMPLETE EVIDENCE"
        text = (
            "Not enough verified growth/margin evidence to call a multi-driver thesis. "
            "Refresh Screener fundamentals or upload financial history / concall guidance."
        )
    if guidance_present:
        text += " Dated management/guidance uploads are present and cited separately."
    else:
        text += " Concall / management guidance is MISSING until you upload it under Research Data."
    return {"label": label, "engines": engines, "text": text}


def build_growth_outlook(
    symbol: str,
    *,
    fundamentals: Mapping[str, Any] | None = None,
    technical: Mapping[str, Any] | None = None,
    company: str = "",
    sector: str = "",
    raw_fundamentals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a structured Growth & Financial Outlook pack for one symbol."""
    sym = re.sub(r"[^A-Z0-9&.-]", "", str(symbol or "").strip().upper())
    fundamentals = dict(fundamentals or {})
    technical = dict(technical or {})
    raw_record = dict(raw_fundamentals or {})
    raw = dict(raw_record.get("data") or raw_record or {})

    # Prefer workspace fundamentals; fill series from raw Screener tables.
    sales = _series(raw.get("profit_loss"), "sales", "revenue")
    profit = _series(raw.get("profit_loss"), "net profit", "profit after tax", "pat")
    op_profit = _series(raw.get("profit_loss"), "operating profit", "opm", "ebit")
    q_sales = _series(raw.get("quarterly_results"), "sales", "revenue")
    q_profit = _series(raw.get("quarterly_results"), "net profit", "profit after tax", "pat")

    sales_cagr = _metric_from_workspace(fundamentals, "sales_growth_3y")
    if sales_cagr is None:
        sales_cagr = _cagr_pct(sales, 3)
    profit_cagr = _metric_from_workspace(fundamentals, "profit_growth_3y")
    if profit_cagr is None:
        profit_cagr = _cagr_pct(profit, 3)

    sales_yoy = _yoy_pct(sales) if sales else _yoy_pct(q_sales)
    profit_yoy = _yoy_pct(profit) if profit else _yoy_pct(q_profit)

    # Margin: prefer OPM / EBITDA from key ratios or uploaded history; else PAT/Sales.
    opm = _key_ratio(fundamentals, "opm", "operating profit margin", "ebitda margin")
    npm = _key_ratio(fundamentals, "npm", "net profit margin", "profit margin")
    if opm is None and op_profit and sales and len(op_profit) == len(sales) and sales[-1]:
        opm = round(op_profit[-1] / sales[-1] * 100.0, 2)
    if npm is None and profit and sales and len(profit) == len(sales) and sales[-1]:
        npm = round(profit[-1] / sales[-1] * 100.0, 2)

    mgmt, order_book, fin_hist = _guidance_rows(sym)
    uploaded_margin = _uploaded_margin_trend(fin_hist)

    margin_expanding: bool | None = None
    if uploaded_margin.get("available"):
        margin_expanding = float(uploaded_margin["delta_pp"]) > 0.5
    else:
        margins = _margin_series(sales, profit)
        clean = [m for m in margins if m is not None]
        if len(clean) >= 2:
            margin_expanding = clean[-1] - clean[0] > 0.5

    pe = _metric_from_workspace(fundamentals, "pe")
    roe = _metric_from_workspace(fundamentals, "roe")
    roce = _metric_from_workspace(fundamentals, "roce")
    de = _metric_from_workspace(fundamentals, "debt_to_equity")
    mcap = _metric_from_workspace(fundamentals, "market_cap")
    as_of = str(
        (fundamentals.get("section_as_of") or {}).get("financial_history")
        or fundamentals.get("fetched_at")
        or raw_record.get("fetched_at")
        or ""
    )

    claims = [
        _claim("sales_cagr_3y", "Sales CAGR (≈3Y)", sales_cagr, unit="%", source="Screener P&L / extracted", as_of=as_of),
        _claim("profit_cagr_3y", "Profit CAGR (≈3Y)", profit_cagr, unit="%", source="Screener P&L / extracted", as_of=as_of),
        _claim("sales_yoy", "Latest period sales growth", sales_yoy, unit="%", source="Screener P&L or quarterly", as_of=as_of),
        _claim("profit_yoy", "Latest period profit growth", profit_yoy, unit="%", source="Screener P&L or quarterly", as_of=as_of),
        _claim("opm", "Operating / EBITDA margin (latest)", opm, unit="%", source="Screener key ratios or P&L", as_of=as_of),
        _claim("npm", "Net profit margin (latest)", npm, unit="%", source="Screener key ratios or P&L", as_of=as_of),
        _claim("roe", "ROE", roe, unit="%", source="Fundamentals snapshot", as_of=as_of),
        _claim("roce", "ROCE", roce, unit="%", source="Fundamentals snapshot", as_of=as_of),
        _claim("debt_to_equity", "Debt / equity", de, unit="x", source="Fundamentals snapshot", as_of=as_of),
        _claim("pe", "P/E", pe, unit="x", source="Fundamentals snapshot", as_of=as_of),
        _claim("market_cap", "Market cap", mcap, unit="INR Cr", source="Fundamentals snapshot", as_of=as_of),
        _claim(
            "ebitda_margin_trend",
            "EBITDA margin trend (uploaded)",
            (
                f"{uploaded_margin['from']}% → {uploaded_margin['to']}%"
                if uploaded_margin.get("available")
                else None
            ),
            source="uploaded financial_history",
            as_of=str(uploaded_margin.get("as_of") or ""),
            note="Upload financial_history with ebitda_margin_pct to populate.",
        ),
    ]

    guidance_present = bool(mgmt or order_book)
    thesis = _thesis(
        sales_cagr=sales_cagr,
        profit_cagr=profit_cagr,
        sales_yoy=sales_yoy,
        margin_expanding=margin_expanding,
        guidance_present=guidance_present,
    )

    price = _f(technical.get("close"))
    trend = str(technical.get("trend") or "UNAVAILABLE")
    trend_text = str(technical.get("trend_explanation") or "Official price history unavailable.")

    guidance_items: list[dict[str, Any]] = []
    for row in mgmt[:8]:
        guidance_items.append(
            {
                "kind": "management_commentary",
                "event_date": str(row.get("event_date") or row.get("as_of_date") or ""),
                "speaker": str(row.get("speaker") or ""),
                "topic": str(row.get("topic") or ""),
                "commentary": str(row.get("commentary") or "")[:500],
                "guidance_metric": str(row.get("guidance_metric") or ""),
                "guidance_value": str(row.get("guidance_value") or ""),
                "guidance_period": str(row.get("guidance_period") or ""),
                "source_url": str(row.get("source_url") or ""),
            }
        )
    for row in order_book[:6]:
        guidance_items.append(
            {
                "kind": "order_book_guidance",
                "event_date": str(row.get("as_of_date") or ""),
                "speaker": "management disclosure",
                "topic": str(row.get("metric") or ""),
                "commentary": str(row.get("management_wording") or "")[:500],
                "guidance_metric": str(row.get("metric") or ""),
                "guidance_value": f"{row.get('value') or ''} {row.get('unit') or ''}".strip(),
                "guidance_period": str(row.get("period") or ""),
                "source_url": str(row.get("source_url") or ""),
            }
        )

    gaps: list[str] = []
    if sales_cagr is None and sales_yoy is None:
        gaps.append("Sales growth series missing — refresh Screener fundamentals.")
    if profit_cagr is None and profit_yoy is None:
        gaps.append("Profit growth series missing — refresh Screener fundamentals.")
    if opm is None and not uploaded_margin.get("available"):
        gaps.append("EBITDA / operating margin not in cache — upload financial_history or refresh Screener.")
    if not guidance_present:
        gaps.append(
            "Concall / management guidance missing — upload management_commentary or order_book_guidance under Research Data."
        )
    if not technical.get("available"):
        gaps.append("Official price history missing — technical outlook incomplete.")
    if pe is None:
        gaps.append("P/E unavailable in current fundamentals pack.")

    sections = [
        {
            "id": "overview",
            "title": "Company overview and core thesis",
            "body": (
                f"{company or sym} ({sym})"
                + (f" · {sector}" if sector else "")
                + f". Thesis label: {thesis['label']}. {thesis['text']}"
            ),
        },
        {
            "id": "financials",
            "title": "Financial performance highlights",
            "body": (
                "Figures below are only those present in Screener cache or your uploads. "
                "Blank / MISSING cells are intentional — QuantTerm does not invent FY targets."
            ),
        },
        {
            "id": "technicals",
            "title": "Technical and market structure",
            "body": (
                f"Official close {('₹' + format(price, ',.1f')) if price is not None else '—'}. "
                f"Trend: {trend}. {trend_text} "
                "Technicals are EOD structure, not a forecast."
            ),
        },
        {
            "id": "guidance",
            "title": "Management guidance and investor communication",
            "body": (
                f"{len(guidance_items)} dated guidance/commentary row(s) from your uploads."
                if guidance_items
                else (
                    "No concall transcript, investor-presentation guidance, or order-book disclosure "
                    "has been uploaded for this symbol. Upload under Research Data → management_commentary "
                    "or order_book_guidance. QuantTerm will not invent management quotes."
                )
            ),
        },
        {
            "id": "risks",
            "title": "Investment risks and open gaps",
            "body": (
                "This is research context only — not a buy/sell ticket and never places orders. "
                + ("Open gaps: " + " ".join(gaps) if gaps else "No critical data gaps flagged in this pack.")
            ),
        },
    ]

    summary_bits: list[str] = []
    if sales_cagr is not None:
        summary_bits.append(f"≈3Y sales CAGR {sales_cagr:.1f}%")
    elif sales_yoy is not None:
        summary_bits.append(f"latest sales growth {sales_yoy:.1f}%")
    if profit_cagr is not None:
        summary_bits.append(f"≈3Y profit CAGR {profit_cagr:.1f}%")
    if opm is not None:
        summary_bits.append(f"latest OPM/EBITDA margin {opm:.1f}%")
    elif uploaded_margin.get("available"):
        summary_bits.append(
            f"uploaded EBITDA margin {uploaded_margin['from']}% → {uploaded_margin['to']}%"
        )
    if margin_expanding is True:
        summary_bits.append("margin expanding on available periods")
    elif margin_expanding is False:
        summary_bits.append("margin not expanding on available periods")

    summary = (
        f"{company or sym}: " + (", ".join(summary_bits) if summary_bits else "insufficient financial evidence")
        + f". Thesis: {thesis['label']}. "
        + (
            "Management guidance cited from uploads."
            if guidance_present
            else "Management/concall guidance not uploaded."
        )
        + " Research only — verify filings before any decision."
    )

    available = any(c["status"] == "AVAILABLE" for c in claims) or bool(guidance_items)
    source_packs = _source_packs(sym)
    return {
        "available": available,
        "symbol": sym,
        "company": company or sym,
        "sector": sector or "",
        "title": f"Growth & Financial Outlook — {company or sym}",
        "thesis": thesis,
        "claims": claims,
        "sections": sections,
        "guidance": guidance_items,
        "source_packs": source_packs,
        "technical": {
            "available": bool(technical.get("available")),
            "price": price,
            "trend": trend,
            "trend_explanation": trend_text,
            "as_of": str(technical.get("latest_date") or ""),
        },
        "gaps": gaps,
        "summary": summary,
        "places_orders": False,
        "honesty": (
            "Evidence-only outlook. Sales/profit/margins come from Screener cache or your uploads. "
            "Concall guidance appears only from uploaded management_commentary / order_book_guidance. "
            "Open the source links below, download filings, then upload under Research Data. "
            "No invented FY targets, quotes, or prices. Not a buy/sell ticket."
        ),
    }
