"""On-file fundamentals for Top Stocks — calculated packs, never a live scrape.

Tape for this desk is official NSE bhavcopy plus Kite/NSE quotes.
Valuation metrics come from the last long-term research pack (ratios already
calculated and stored). Missing fields stay missing. Google/Screener are not
called to fill this list.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

QUALITY_CLASSES = frozenset({
    "QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE",
})

_COMMON = (
    ("pe", "P/E", "x"),
    ("roe", "ROE", "%"),
    ("roce", "ROCE", "%"),
)
_GROWTH = (
    ("sales_growth_3y", "Sales 3Y", "%"),
    ("profit_growth_3y", "Profit 3Y", "%"),
)
_LEVERAGE = (
    ("debt_to_equity", "D/E", "x"),
    ("interest_coverage", "Int. cover", "x"),
)
_BANKISH = ("bank", "nbfc", "finance", "financial", "insurance", "capital market")


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        if out != out:
            return None
        return out
    except (TypeError, ValueError):
        return None


def is_financial_sector(sector: str) -> bool:
    text = str(sector or "").lower()
    return any(token in text for token in _BANKISH)


def sector_metric_specs(sector: str) -> tuple[tuple[str, str, str], ...]:
    """Relevant ratios for the sector — banks skip generic industrial leverage."""
    if is_financial_sector(sector):
        return _COMMON + (("sales_growth_3y", "Book/NI 3Y", "%"),)
    return _COMMON + _LEVERAGE[:1] + _GROWTH[:1]


def pack_fundamentals(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project stored ratios. Empty pack if nothing is on file — never invented."""
    nested = dict(row.get("fundamentals") or {}) if isinstance(row.get("fundamentals"), Mapping) else {}
    sector = str(row.get("sector") or "")
    coverage = _f(row.get("fundamental_coverage"))
    classification = str(row.get("classification") or "")
    metrics: list[dict[str, Any]] = []
    for key, label, unit in sector_metric_specs(sector):
        raw = nested.get(key) if key in nested else row.get(key)
        value = _f(raw)
        if value is None:
            continue
        metrics.append({
            "key": key,
            "label": label,
            "value": round(value, 2),
            "unit": unit,
        })
    available = bool(metrics) or (coverage is not None and coverage > 0)
    return {
        "available": available,
        "coverage_pct": round(coverage * 100.0, 1) if coverage is not None else None,
        "classification": classification or None,
        "sector": sector or None,
        "metrics": metrics,
        "source": "long_term_pack" if available else "",
        "note": (
            "Ratios from the on-file long-term pack — calculated, not live-scraped."
            if available
            else "No fundamental pack on file for this name. Missing stays missing."
        ),
    }


def fund_rank(row: Mapping[str, Any]) -> int:
    """Tie-break only. Technical SEPA still ranks first."""
    cls = str(row.get("classification") or "")
    cov = _f(row.get("fundamental_coverage")) or 0.0
    if cls in QUALITY_CLASSES and cov >= 0.50:
        return 2 if cls == "QUALITY_COMPOUNDER" else 1
    if cov >= 0.50:
        return 1
    return 0


def tape_policy() -> dict[str, str]:
    return {
        "price": (
            "Last print: Kite when logged in, else NSE snapshot. "
            "Otherwise official NSE bhavcopy EOD. Google is not used on this desk."
        ),
        "technical": (
            "Minervini SEPA / Trend Template on official NSE daily OHLCV "
            "(simple 50/150/200 averages and 52-week range)."
        ),
        "fundamental": (
            "Valuation metrics are calculated from the on-file long-term pack. "
            "Sector chooses which ratios show (banks skip generic D/E). "
            "This desk does not scrape Screener or Google to fill Top Stocks."
        ),
    }


def attach_to_card(card: dict[str, Any], row: Mapping[str, Any], sepa: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Stamp tape + on-file funds onto a Best Setups card."""
    quote = dict((sepa or {}).get("quote") or {})
    if quote.get("close") and not card.get("cmp"):
        card["cmp"] = quote.get("close")
        card["price_tag"] = "EOD"
        card["tech_source"] = "nse_bhavcopy"
    if quote.get("change_pct") is not None and card.get("change_pct") is None:
        card["change_pct"] = quote.get("change_pct")
    fund = pack_fundamentals(row)
    card["fundamentals"] = fund
    card["fund_available"] = bool(fund.get("available"))
    card["rank_fund"] = fund_rank(row)
    return card
