"""Central financial ratio calculations from canonical fundamentals observations."""
from __future__ import annotations

from typing import Any, Mapping

from data_platform.contracts import FinancialRatioSnapshot, ObservationMeta, QualityStatus, utc_now_iso


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _ratio(
    symbol: str,
    key: str,
    label: str,
    value: float | None,
    formula: str,
    numerator: str,
    denominator: str,
    period: str,
    scope: str,
    missing: str,
) -> FinancialRatioSnapshot:
    meta = ObservationMeta(
        symbol=symbol,
        source="data_platform.ratios",
        retrieved_at=utc_now_iso(),
        period_end=period,
        scope=scope,
        quality_status=QualityStatus.FRESH if value is not None else QualityStatus.MISSING,
        missing_reason=missing if value is None else "",
    )
    return FinancialRatioSnapshot(
        symbol=symbol,
        key=key,
        label=label,
        value=value,
        formula=formula,
        numerator=numerator,
        denominator=denominator,
        period=period,
        scope=scope,
        missing_reason=missing if value is None else "",
        meta=meta,
    )


def ratios_from_fundamentals(symbol: str, raw: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    sym = str(symbol or "").upper()
    data = dict(raw or {})
    period = str(data.get("period") or data.get("as_of") or "")
    scope = str(data.get("scope") or "consolidated")
    revenue = _f(data.get("revenue") or data.get("sales"))
    op_profit = _f(data.get("operating_profit") or data.get("ebit"))
    net_profit = _f(data.get("net_profit") or data.get("profit"))
    equity = _f(data.get("equity") or data.get("shareholders_equity"))
    debt = _f(data.get("total_debt") or data.get("borrowings"))
    eps = _f(data.get("eps"))
    price = _f(data.get("current_price") or data.get("price"))
    book = _f(data.get("book_value") or data.get("bvps"))
    ocf = _f(data.get("operating_cash_flow") or data.get("cash_from_operations"))
    interest = _f(data.get("finance_cost") or data.get("interest"))

    specs: list[FinancialRatioSnapshot] = []
    op_margin = (op_profit / revenue * 100) if revenue and op_profit is not None and revenue > 0 else None
    specs.append(_ratio(sym, "operating_margin", "Operating margin", round(op_margin, 2) if op_margin is not None else None,
                        "operating_profit / revenue", "operating_profit", "revenue", period, scope,
                        "operating profit or revenue missing"))
    net_margin = (net_profit / revenue * 100) if revenue and net_profit is not None and revenue > 0 else None
    specs.append(_ratio(sym, "net_margin", "Net margin", round(net_margin, 2) if net_margin is not None else None,
                        "net_profit / revenue", "net_profit", "revenue", period, scope,
                        "net profit or revenue missing"))
    roe = (net_profit / equity * 100) if equity and net_profit is not None and equity > 0 else None
    specs.append(_ratio(sym, "roe", "ROE", round(roe, 2) if roe is not None else None,
                        "net_profit / equity", "net_profit", "equity", period, scope, "net profit or equity missing"))
    dte = (debt / equity) if equity and debt is not None and equity > 0 else None
    specs.append(_ratio(sym, "debt_equity", "Debt / equity", round(dte, 2) if dte is not None else None,
                        "total_debt / equity", "total_debt", "equity", period, scope, "debt or equity missing"))
    ic = (op_profit / interest) if interest and op_profit is not None and interest > 0 else None
    specs.append(_ratio(sym, "interest_coverage", "Interest coverage", round(ic, 2) if ic is not None else None,
                        "operating_profit / finance_cost", "operating_profit", "finance_cost", period, scope,
                        "operating profit or finance cost missing"))
    pe = (price / eps) if eps and price is not None and eps > 0 else None
    specs.append(_ratio(sym, "pe", "P/E", round(pe, 2) if pe is not None else None,
                        "price / eps", "price", "eps", period, scope, "price or eps missing"))
    pb = (price / book) if book and price is not None and book > 0 else None
    specs.append(_ratio(sym, "pb", "P/B", round(pb, 2) if pb is not None else None,
                        "price / book_value", "price", "book_value", period, scope, "price or book value missing"))
    cfc = (ocf / net_profit) if net_profit and ocf is not None and net_profit != 0 else None
    specs.append(_ratio(sym, "cash_flow_conversion", "Cash flow conversion", round(cfc, 2) if cfc is not None else None,
                        "operating_cash_flow / net_profit", "operating_cash_flow", "net_profit", period, scope,
                        "OCF or net profit missing"))

    out: list[dict[str, Any]] = []
    for s in specs:
        row = {
            "key": s.key,
            "label": s.label,
            "value": s.value,
            "formula": s.formula,
            "numerator": s.numerator,
            "denominator": s.denominator,
            "period": s.period,
            "scope": s.scope,
            "missing_reason": s.missing_reason,
        }
        if s.meta:
            row["quality_status"] = s.meta.quality_status.value
        out.append(row)
    return out
