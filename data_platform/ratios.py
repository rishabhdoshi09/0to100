"""Central financial ratio calculations from canonical fundamentals observations."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

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


def _row_latest(data: Mapping[str, Any], section: str, *needles: str) -> float | None:
    """Latest numeric cell from a Screener.in table section (chronological columns)."""
    for row in data.get(section, []) or []:
        if not isinstance(row, Mapping):
            continue
        label = str(row.get("", row.get("row_label", "")) or "").lower().strip()
        if not any(needle in label for needle in needles):
            continue
        vals: list[float] = []
        for key, value in row.items():
            if key in ("", "row_label"):
                continue
            parsed = _f(value)
            if parsed is not None:
                vals.append(parsed)
        if vals:
            return vals[-1]
    return None


def _key_ratio_lookup(data: Mapping[str, Any], *needles: str) -> float | None:
    for item in data.get("key_ratios", []) or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).lower().strip()
        if any(needle in name for needle in needles):
            return _f(item.get("value"))
    return None


def flatten_screener_snapshot(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Map Screener.in deep snapshot → flat fields for ratio math + direct overrides."""
    data = dict(raw or {})
    if not data:
        return {}

    extracted: dict[str, Any] = {}
    try:
        from screener.engine import _extract_fundamentals

        extracted = dict(_extract_fundamentals(data) or {})
    except Exception:
        extracted = {}

    equity: float | None = None
    debt: float | None = None
    for row in data.get("balance_sheet", []) or []:
        if not isinstance(row, Mapping):
            continue
        label = str(row.get("", row.get("row_label", "")) or "").lower().strip()
        vals: list[float] = []
        for key, value in row.items():
            if key in ("", "row_label"):
                continue
            parsed = _f(value)
            if parsed is not None:
                vals.append(parsed)
        latest = vals[-1] if vals else None
        if latest is None:
            continue
        if "borrowing" in label:
            debt = (debt or 0.0) + latest
        if (
            "equity capital" in label
            or "total equity" in label
            or ("reserves" in label and "share" in label)
        ):
            equity = (equity or 0.0) + latest

    flat: dict[str, Any] = {
        "period": str(data.get("period") or data.get("as_of") or ""),
        "scope": str(data.get("scope") or "consolidated"),
        "revenue": _row_latest(data, "profit_loss", "sales", "revenue"),
        "operating_profit": _row_latest(
            data, "profit_loss", "operating profit", "ebit", "op profit"
        ),
        "net_profit": _row_latest(
            data, "profit_loss", "net profit", "profit after tax", "pat"
        ),
        "operating_cash_flow": _row_latest(
            data, "cash_flow", "cash from operating", "cash flow from operating"
        ),
        "finance_cost": _row_latest(data, "profit_loss", "finance cost", "interest"),
        "equity": equity,
        "total_debt": debt,
        "eps": _key_ratio_lookup(data, "eps", "earning per share"),
        "current_price": _key_ratio_lookup(data, "current price", "cmp", "price"),
        "book_value": _key_ratio_lookup(data, "book value", "bvps"),
    }
    if flat.get("_direct_pe") is None:
        pe = _key_ratio_lookup(
            data,
            "stock p/e",
            "p/e",
            "pe ratio",
            "price to earning",
            "price/earning",
            "price earning",
        )
        if pe is not None:
            flat["_direct_pe"] = pe

    for src, dst in (
        ("pe", "_direct_pe"),
        ("roe", "_direct_roe"),
        ("debt_to_equity", "_direct_debt_to_equity"),
        ("interest_coverage", "_direct_interest_coverage"),
        ("cfo_to_pat", "_direct_cash_flow_conversion"),
    ):
        if extracted.get(src) is not None:
            flat[dst] = extracted[src]

    return flat


def _coerce_fundamentals_payload(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(raw or {})
    if "key_ratios" in data or "profit_loss" in data or "balance_sheet" in data:
        return flatten_screener_snapshot(data)
    return data


def stock_pe_from_payload(raw: Mapping[str, Any] | None) -> float | None:
    """Best-effort stock P/E from Screener snapshot or price/eps."""
    data = _coerce_fundamentals_payload(raw)
    direct = _f(data.get("_direct_pe"))
    if direct is not None and direct > 0:
        return direct
    eps = _f(data.get("eps"))
    price = _f(data.get("current_price") or data.get("price"))
    if eps and price and eps > 0:
        return round(price / eps, 2)
    return None


def _pe_from_peer_comparison_row(row: Mapping[str, Any]) -> float | None:
    for key, value in row.items():
        label = str(key or "").strip().lower()
        if label in {"p/e", "pe", "stock p/e", "price to earning", "price/earning", "price earning"}:
            parsed = _f(value)
            if parsed is not None and parsed > 0:
                return parsed
    return None


def compute_peer_average_pe(
    symbol: str,
    screener_data: Mapping[str, Any] | None,
    extra_peer_symbols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Mean peer P/E from (1) Screener peer_comparison rows and (2) cached fundamentals
    for same-sector peer symbols. No network — cache-only for peer symbols.
    """
    sym = str(symbol or "").upper().strip()
    data = dict(screener_data or {})
    values: list[float] = []
    sources: list[str] = []

    for row in data.get("peer_comparison", []) or []:
        if not isinstance(row, Mapping):
            continue
        pe = _pe_from_peer_comparison_row(row)
        if pe is not None:
            values.append(pe)

    if values:
        sources.append("screener_peer_table")

    try:
        from fundamentals.cache import FundamentalsCache

        cache = FundamentalsCache()
        cache_hits = 0
        for peer_sym in extra_peer_symbols or []:
            peer = str(peer_sym or "").upper().strip()
            if not peer or peer == sym:
                continue
            cached = cache.get(peer) or cache.get_any(peer)
            if not cached:
                continue
            pe = stock_pe_from_payload(cached)
            if pe is not None and pe > 0:
                values.append(pe)
                cache_hits += 1
        if cache_hits:
            sources.append("peer_fundamentals_cache")
    except Exception:
        pass

    stock_pe = stock_pe_from_payload(data)
    average_pe = round(sum(values) / len(values), 2) if values else None
    pe_vs_peer = (
        round(stock_pe / average_pe, 2)
        if stock_pe is not None and average_pe is not None and average_pe > 0
        else None
    )

    note = ""
    if not values:
        note = (
            "Peer P/E needs Screener peer_comparison on this symbol or cached fundamentals "
            "for same-sector peers (open peers / retry fundamentals)."
        )
    else:
        note = (
            f"Mean P/E from {len(values)} peer sample(s): "
            + ", ".join(sources)
            + ". Screener peer table + fundamentals cache only — not point-in-time history."
        )

    return {
        "average_pe": average_pe,
        "sample_count": len(values),
        "stock_pe": stock_pe,
        "pe_vs_peer_avg": pe_vs_peer,
        "sources": sources,
        "note": note,
    }


def peer_pe_ratio_rows(symbol: str, peer_stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Ratio-tab rows for peer average P/E and optional premium/discount vs peers."""
    sym = str(symbol or "").upper()
    period = ""
    scope = "peer_set"
    rows: list[dict[str, Any]] = []
    avg = peer_stats.get("average_pe")
    n = int(peer_stats.get("sample_count") or 0)
    missing = str(peer_stats.get("note") or "peer P/E samples missing")

    avg_row = _ratio(
        sym,
        "peer_avg_pe",
        "Average peer P/E",
        avg if isinstance(avg, (int, float)) else None,
        "mean(peer P/E) from Screener peer table + cached peer fundamentals",
        "sum(peer P/E)",
        f"count={n}",
        period,
        scope,
        missing if avg is None else f"{n} peer sample(s)",
    )
    rows.append({
        "key": avg_row.key,
        "label": avg_row.label,
        "value": avg_row.value,
        "formula": avg_row.formula,
        "numerator": avg_row.numerator,
        "denominator": avg_row.denominator,
        "period": avg_row.period,
        "scope": avg_row.scope,
        "missing_reason": avg_row.missing_reason,
        "quality_status": avg_row.meta.quality_status.value if avg_row.meta else QualityStatus.MISSING.value,
    })

    vs = peer_stats.get("pe_vs_peer_avg")
    vs_row = _ratio(
        sym,
        "pe_vs_peer_avg",
        "P/E vs peer average",
        vs if isinstance(vs, (int, float)) else None,
        "stock P/E ÷ average peer P/E",
        "stock_pe",
        "peer_avg_pe",
        period,
        scope,
        "stock or peer average P/E missing" if vs is None else "",
    )
    rows.append({
        "key": vs_row.key,
        "label": vs_row.label,
        "value": vs_row.value,
        "formula": vs_row.formula,
        "numerator": vs_row.numerator,
        "denominator": vs_row.denominator,
        "period": vs_row.period,
        "scope": vs_row.scope,
        "missing_reason": vs_row.missing_reason,
        "quality_status": vs_row.meta.quality_status.value if vs_row.meta else QualityStatus.MISSING.value,
    })
    return rows


def peer_pe_fundamental_metrics(peer_stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Metric cards for Stock Intelligence fundamentals snapshot."""
    avg = peer_stats.get("average_pe")
    vs = peer_stats.get("pe_vs_peer_avg")
    n = int(peer_stats.get("sample_count") or 0)
    note = str(peer_stats.get("note") or "")
    metrics: list[dict[str, Any]] = []
    if avg is not None:
        metrics.append({
            "key": "peer_avg_pe",
            "label": "Average peer P/E",
            "value": avg,
            "unit": "x",
            "meaning": f"Mean P/E across {n} peers (Screener peer table + cached peer fundamentals).",
            "interpretation": (
                "Compare the stock's own P/E to this peer average — context only, not a buy signal."
            ),
        })
    if vs is not None:
        label = "Premium to peers" if vs > 1.05 else "Discount to peers" if vs < 0.95 else "Near peer average"
        metrics.append({
            "key": "pe_vs_peer_avg",
            "label": "P/E vs peer average",
            "value": vs,
            "unit": "x",
            "meaning": "Stock P/E divided by average peer P/E (>1 = richer than peers on P/E).",
            "interpretation": label,
        })
    if not metrics and note:
        metrics.append({
            "key": "peer_avg_pe",
            "label": "Average peer P/E",
            "value": None,
            "unit": "x",
            "meaning": "Peer valuation context from Screener peer table and cached peer fundamentals.",
            "interpretation": note,
        })
    return metrics


def ratios_from_fundamentals(
    symbol: str,
    raw: Mapping[str, Any] | None,
    *,
    peer_stats: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sym = str(symbol or "").upper()
    data = _coerce_fundamentals_payload(raw)
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

    direct_overrides = {
        "pe": "_direct_pe",
        "roe": "_direct_roe",
        "debt_equity": "_direct_debt_to_equity",
        "interest_coverage": "_direct_interest_coverage",
        "cash_flow_conversion": "_direct_cash_flow_conversion",
    }
    for row in out:
        direct_key = direct_overrides.get(row["key"])
        if direct_key and data.get(direct_key) is not None:
            row["value"] = data[direct_key]
            row["missing_reason"] = ""
            row["quality_status"] = QualityStatus.FRESH.value
            if direct_key.startswith("_direct_"):
                row["formula"] = f"Screener.in key ratio ({row['key']})"

    if peer_stats:
        out.extend(peer_pe_ratio_rows(sym, peer_stats))

    return out
