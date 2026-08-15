"""Reco-style recommendation detail — Performance + Thesis + KPI tabs.

High-level model (matches Reco Wealth layout language, QuantTerm evidence):

  Performance tab
    · Price grid: Entry / CMP / Target (from scan or LT card — never invented)
    · Stop / downside protection when stop exists on the setup
    · KPI segmented control: Profitability | Valuation | Margins
        Profitability ← ROE, ROCE, CFO/PAT, sales/profit CAGR (stock workspace)
        Valuation     ← PE, market cap, dividend yield (+ peer PE when present)
        Margins       ← operating/net margin from ratios engine when inputs exist;
                        interest coverage + debt/equity as balance-sheet context

  Thesis tab
    · Narrative from qualify_reason + quality_factors + technical thesis
    · Risk flags as "what can go wrong"
    · No fabricated research PDF — link to Stock Intelligence instead

Honest empty: missing fund cache → KPI values null + coverage note.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric(
    *,
    key: str,
    label: str,
    value: float | None,
    unit: str,
    hint: str = "",
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "value": round(value, 2) if value is not None else None,
        "unit": unit,
        "available": value is not None,
        "hint": hint,
        "display": (
            f"{value:.1f}{unit}" if value is not None and unit == "%"
            else f"{value:.2f}{unit}" if value is not None and unit == "x"
            else f"{value:,.0f}" if value is not None and unit == "INR Cr"
            else f"{value}" if value is not None else "—"
        ),
    }


def _metrics_by_key(fundamentals: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in fundamentals.get("metrics") or []:
        if isinstance(row, Mapping) and row.get("key"):
            out[str(row["key"])] = row
    raw_vals = fundamentals.get("raw_values") or {}
    if isinstance(raw_vals, Mapping):
        for k, v in raw_vals.items():
            out.setdefault(str(k), {"key": k, "value": v})
    return out


def _ratio_map(symbol: str, raw: Mapping[str, Any]) -> dict[str, float | None]:
    if not raw:
        return {}
    try:
        from data_platform.ratios import ratios_from_fundamentals
        rows = ratios_from_fundamentals(symbol, raw) or []
    except Exception:
        return {}
    out: dict[str, float | None] = {}
    for r in rows:
        if hasattr(r, "key"):
            out[str(r.key)] = _f(getattr(r, "value", None))
        elif isinstance(r, Mapping):
            out[str(r.get("key"))] = _f(r.get("value"))
    return out


def _find_card(
    symbol: str,
    *,
    category_id: str = "",
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from product.recommendations_workspace import build_recommendations_workspace

    desk = build_recommendations_workspace(
        scan_payload=scan_payload,
        long_term_payload=long_term_payload,
        refresh_technicals=False,
    )
    sym = symbol.upper()
    for cat in desk.get("categories") or []:
        if category_id and cat.get("id") != category_id:
            continue
        for card in cat.get("cards") or []:
            if str(card.get("symbol") or "").upper() == sym:
                return dict(card)
    # Fallback: any lifecycle / category match.
    for cat in desk.get("categories") or []:
        for card in cat.get("cards") or []:
            if str(card.get("symbol") or "").upper() == sym:
                return dict(card)
    for card in (desk.get("lifecycle") or {}).get("active") or []:
        if str(card.get("symbol") or "").upper() == sym:
            return dict(card)
    return {"symbol": sym, "company": sym, "category_id": category_id or "", "category_label": ""}


def build_recommendation_detail(
    symbol: str,
    *,
    category_id: str = "",
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one pick into Reco-style Performance / Thesis detail."""
    sym = str(symbol or "").strip().upper()
    card = _find_card(
        sym,
        category_id=category_id,
        scan_payload=scan_payload,
        long_term_payload=long_term_payload,
    )

    workspace: dict[str, Any] = {}
    try:
        from product.stock_workspace import build_stock_workspace
        workspace = build_stock_workspace(sym) or {}
    except Exception:
        workspace = {}

    fundamentals = dict(workspace.get("fundamentals") or {})
    by_key = _metrics_by_key(fundamentals)
    raw = {}
    try:
        from fundamentals.cache import FundamentalsCache
        raw = FundamentalsCache().get(sym) or FundamentalsCache().get_any(sym) or {}
    except Exception:
        raw = {}
    ratios = _ratio_map(sym, raw if isinstance(raw, Mapping) else {})

    def val(key: str) -> float | None:
        row = by_key.get(key) or {}
        if isinstance(row, Mapping) and row.get("value") is not None:
            return _f(row.get("value"))
        return _f((fundamentals.get("raw_values") or {}).get(key))

    profitability = [
        _metric(key="roe", label="ROE", value=val("roe"), unit="%", hint="Return on equity"),
        _metric(key="roce", label="ROCE", value=val("roce"), unit="%", hint="Return on capital employed"),
        _metric(key="cfo_to_pat", label="CFO / PAT", value=val("cfo_to_pat"), unit="x", hint="Cash conversion"),
        _metric(key="sales_growth_3y", label="Sales CAGR (3Y)", value=val("sales_growth_3y"), unit="%", hint="Revenue growth"),
        _metric(key="profit_growth_3y", label="Profit CAGR (3Y)", value=val("profit_growth_3y"), unit="%", hint="Profit growth"),
    ]
    valuation = [
        _metric(key="pe", label="P/E", value=val("pe"), unit="x", hint="Price / earnings"),
        _metric(key="market_cap", label="Market cap", value=val("market_cap") or val("market_cap_cr"), unit="INR Cr", hint="Listed equity value"),
        _metric(
            key="dividend_yield",
            label="Dividend yield",
            value=val("dividend_yield"),
            unit="%",
            hint="Trailing dividend yield when disclosed",
        ),
    ]
    # Peer PE from workspace metrics if attached.
    for row in fundamentals.get("metrics") or []:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("key") or "") in {"peer_avg_pe", "pe_vs_peer_avg"}:
            valuation.append(_metric(
                key=str(row["key"]),
                label=str(row.get("label") or row["key"]),
                value=_f(row.get("value")),
                unit=str(row.get("unit") or "x"),
                hint=str(row.get("interpretation") or ""),
            ))

    margins = [
        _metric(
            key="operating_margin",
            label="Operating margin",
            value=ratios.get("operating_margin"),
            unit="%",
            hint="From latest P&L when revenue/OP available",
        ),
        _metric(
            key="net_margin",
            label="Net margin",
            value=ratios.get("net_margin"),
            unit="%",
            hint="Net profit / revenue when available",
        ),
        _metric(key="interest_coverage", label="Interest coverage", value=val("interest_coverage"), unit="x", hint="Debt service cushion"),
        _metric(key="debt_to_equity", label="Debt / equity", value=val("debt_to_equity"), unit="x", hint="Balance-sheet leverage"),
    ]

    entry = _f(card.get("entry"))
    cmp_ = _f(card.get("cmp") or card.get("price"))
    target = _f(card.get("target"))
    stop = _f(card.get("stop"))
    downside = None
    if stop is not None and cmp_ and cmp_ > 0:
        downside = round((stop / cmp_ - 1.0) * 100.0, 1)

    coverage = fundamentals.get("coverage_pct")
    fund_ready = bool(fundamentals.get("available")) and _f(coverage, 0) is not None and float(coverage or 0) >= 40
    any_kpi = any(m["available"] for m in profitability + valuation + margins)

    lt_row: dict[str, Any] = {}
    try:
        from product.long_term_store import load_long_term_scan
        lt = dict(long_term_payload or load_long_term_scan() or {})
        for r in lt.get("records") or []:
            if str(r.get("symbol") or "").upper() == sym:
                lt_row = dict(r)
                break
    except Exception:
        lt_row = {}

    quality = [str(x) for x in (lt_row.get("quality_factors") or fundamentals.get("quality_factors") or []) if x]
    risks = [str(x) for x in (lt_row.get("risk_flags") or fundamentals.get("risk_flags") or []) if x]
    thesis_line = str(lt_row.get("thesis") or card.get("qualify_reason") or card.get("reason") or "").strip()
    our_take_parts = [p for p in [thesis_line, " · ".join(quality[:4])] if p]
    our_take = " ".join(our_take_parts) if our_take_parts else (
        "No narrative thesis on file yet — open Stock Intelligence after fundamentals refresh."
    )

    company = str(
        card.get("company")
        or workspace.get("company")
        or (workspace.get("profile") or {}).get("name")
        or sym
    )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": sym,
        "company": company,
        "category_id": str(card.get("category_id") or category_id or ""),
        "category_label": str(card.get("category_label") or ""),
        "action_badge": str(card.get("action_badge") or "Watch"),
        "risk_tier": str(card.get("risk_tier") or "Medium"),
        "setup_label": str(card.get("setup_label") or ""),
        "sector": str(card.get("sector") or workspace.get("sector") or "—"),
        "performance": {
            "entry": entry,
            "cmp": cmp_,
            "target": target,
            "stop": stop,
            "upside_from_entry_pct": card.get("upside_from_entry_pct"),
            "upside_to_target_pct": card.get("upside_to_target_pct"),
            "downside_from_cmp_pct": downside,
            "price_tag": str(card.get("price_tag") or ""),
        },
        "kpis": {
            "profitability": profitability,
            "valuation": valuation,
            "margins": margins,
        },
        "fundamentals_ready": bool(fund_ready and any_kpi),
        "fundamentals_note": (
            f"Coverage {float(coverage or 0):.0f}% from verified fundamentals pack."
            if any_kpi else
            "Profitability / valuation / margins need a fundamentals refresh — "
            "numbers are never invented."
        ),
        "thesis": {
            "our_take": our_take[:600],
            "quality_factors": quality[:8],
            "risk_flags": risks[:8],
            "qualify_reason": str(card.get("qualify_reason") or card.get("reason") or ""),
            "classification": str(lt_row.get("classification") or fundamentals.get("classification") or ""),
        },
        "stock_intelligence_path": f"/stock/{sym}",
        "disclaimer": (
            "Research detail from QuantTerm evidence. Not a broker recommendation. "
            "CMP may be delayed."
        ),
    }
