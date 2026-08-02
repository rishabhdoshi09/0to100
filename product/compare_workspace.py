"""Side-by-side stock comparison from persisted workspaces — no invented winners."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from product.stock_workspace import build_stock_workspace, clean_symbol


COMPARE_SECTIONS = (
    ("market", "Market"),
    ("growth", "Growth"),
    ("quality", "Quality"),
    ("balance_sheet", "Balance sheet"),
    ("valuation", "Valuation"),
    ("technical", "Technical state"),
)


def _metric(label: str, value: Any, unit: str = "", source: str = "") -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "unit": unit,
        "source": source,
        "available": value is not None and value != "" and value != "—",
    }


def _pick_metric(metrics: Sequence[Mapping[str, Any]], key: str) -> Any:
    for item in metrics:
        if str(item.get("key", "")).lower() == key.lower():
            return item.get("value")
    return None


def _workspace_slice(symbol: str) -> dict[str, Any]:
    try:
        return build_stock_workspace(symbol)
    except Exception as exc:
        return {
            "symbol": symbol,
            "available": False,
            "error": str(exc),
            "company": symbol,
            "sector": "—",
        }


def build_compare_workspace(symbols: Sequence[str], *, max_symbols: int = 5) -> dict[str, Any]:
    clean = []
    seen: set[str] = set()
    for raw in symbols:
        try:
            sym = clean_symbol(str(raw))
        except ValueError:
            continue
        if sym in seen:
            continue
        seen.add(sym)
        clean.append(sym)
        if len(clean) >= max_symbols:
            break

    workspaces = [_workspace_slice(sym) for sym in clean]
    rows: list[dict[str, Any]] = []

    for ws in workspaces:
        tech = dict(ws.get("technical", {}) or {})
        fund = dict(ws.get("fundamentals", {}) or {})
        if ws.get("error") and not tech.get("available") and not fund.get("available"):
            rows.append({
                "symbol": ws.get("symbol", ""),
                "company": ws.get("symbol", ""),
                "available": False,
                "error": ws.get("error"),
                "sections": {},
            })
            continue

        scanner = dict(ws.get("scanner", {}) or {})
        tech_metrics = list(tech.get("metrics", []) or [])
        fund_metrics = list(fund.get("metrics", []) or [])

        sections = {
            "market": [
                _metric("Price", tech.get("close"), "₹", tech.get("latest_date", "")),
                _metric("52W from high", _pick_metric(tech_metrics, "from_high_pct"), "%"),
                _metric("5D return", _pick_metric(tech_metrics, "return_5d"), "%"),
                _metric("Volume ratio", tech.get("volume_ratio") or scanner.get("volume_ratio")),
                _metric("Relative strength", scanner.get("score")),
            ],
            "growth": [
                _metric("Sales growth", _pick_metric(fund_metrics, "revenue_growth"), "%", fund.get("fetched_at", "")),
                _metric("Profit growth", _pick_metric(fund_metrics, "profit_growth"), "%"),
                _metric("EPS growth", _pick_metric(fund_metrics, "eps_growth"), "%"),
            ],
            "quality": [
                _metric("Operating margin", _pick_metric(fund_metrics, "operating_margin"), "%"),
                _metric("Net margin", _pick_metric(fund_metrics, "net_margin"), "%"),
                _metric("ROE", _pick_metric(fund_metrics, "roe"), "%"),
                _metric("ROCE", _pick_metric(fund_metrics, "roce"), "%"),
            ],
            "balance_sheet": [
                _metric("Debt / equity", _pick_metric(fund_metrics, "debt_to_equity")),
                _metric("Interest coverage", _pick_metric(fund_metrics, "interest_coverage")),
            ],
            "valuation": [
                _metric("P/E", _pick_metric(fund_metrics, "pe"), "x"),
                _metric("P/B", _pick_metric(fund_metrics, "pb"), "x"),
                _metric("Dividend yield", _pick_metric(fund_metrics, "dividend_yield"), "%"),
            ],
            "technical": [
                _metric("Trend", tech.get("trend")),
                _metric("Momentum 5D", scanner.get("momentum_5d"), "%"),
                _metric("Setup", scanner.get("status") or scanner.get("verdict")),
                _metric("Extension", "Chase risk" if scanner.get("chase_risk") else "Normal"),
            ],
        }

        rows.append({
            "symbol": ws.get("symbol", ""),
            "company": ws.get("company", ws.get("symbol", "")),
            "sector": ws.get("sector", "—"),
            "available": True,
            "confidence_pct": ws.get("confidence_pct"),
            "sections": sections,
        })

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": clean,
        "rows": rows,
        "section_labels": {key: label for key, label in COMPARE_SECTIONS},
        "disclaimer": (
            "Comparison highlights dimension-specific readings only. "
            "No universal winner is declared."
        ),
    }
