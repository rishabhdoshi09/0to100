"""Evidence-first research dossier assembly for QuantTerm.

This module builds a deterministic report payload from QuantTerm's persisted stores.
It does not invent business descriptions, management quotes, institutional holdings,
or financial history. Missing sections become explicit open items in the final report.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "logs" / "reports"


def _as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def _clean_symbol(value: str) -> str:
    symbol = re.sub(r"[^A-Z0-9&.-]", "", str(value or "").strip().upper())
    if not symbol or len(symbol) > 32:
        raise ValueError("invalid NSE symbol")
    return symbol


def _record(payload: Mapping[str, Any] | None, symbol: str) -> dict[str, Any]:
    for item in list((payload or {}).get("records", []) or []):
        if isinstance(item, Mapping) and str(item.get("symbol", "")).upper() == symbol:
            return dict(item)
    return {}


def _return_pct(frame: Any, periods: int) -> float | None:
    try:
        closes = frame["close"].dropna()
        if len(closes) <= periods:
            return None
        return round((float(closes.iloc[-1]) / float(closes.iloc[-periods - 1]) - 1.0) * 100.0, 2)
    except Exception:
        return None


def _price_metrics(frame: Any) -> dict[str, Any]:
    if frame is None or len(frame) == 0:
        return {
            "available": False,
            "latest_price": None,
            "latest_date": "",
            "return_1m_pct": None,
            "return_3m_pct": None,
            "return_6m_pct": None,
            "return_12m_pct": None,
            "high_52w": None,
            "from_high_pct": None,
            "avg_volume_20d": None,
        }
    try:
        closes = frame["close"].dropna()
        latest = float(closes.iloc[-1])
        high = float(frame["high"].tail(min(252, len(frame))).max())
        volume = frame["volume"].tail(20).mean() if "volume" in frame.columns else None
        index = frame.index[-1]
        latest_date = getattr(index, "date", lambda: index)()
        return {
            "available": True,
            "latest_price": round(latest, 2),
            "latest_date": str(latest_date),
            "return_1m_pct": _return_pct(frame, 21),
            "return_3m_pct": _return_pct(frame, 63),
            "return_6m_pct": _return_pct(frame, 126),
            "return_12m_pct": _return_pct(frame, 252),
            "high_52w": round(high, 2),
            "from_high_pct": round((latest / high - 1.0) * 100.0, 2) if high > 0 else None,
            "avg_volume_20d": round(float(volume), 0) if volume is not None else None,
        }
    except Exception:
        return {
            "available": False,
            "latest_price": None,
            "latest_date": "",
            "return_1m_pct": None,
            "return_3m_pct": None,
            "return_6m_pct": None,
            "return_12m_pct": None,
            "high_52w": None,
            "from_high_pct": None,
            "avg_volume_20d": None,
        }


def _default_inputs(symbol: str) -> dict[str, Any]:
    from product.long_term_store import load_long_term_scan
    from product.scan_store import load_scan

    scan = load_scan() or {}
    long_term = load_long_term_scan() or {}
    try:
        from data.bhavcopy_runtime import get_ohlcv
        frame = get_ohlcv(symbol)
    except Exception:
        frame = None
    try:
        from product.market_view import current_market_view
        market_obj = current_market_view()
        market = {
            "health": market_obj.health,
            "summary": market_obj.summary,
            "trade_stance": market_obj.trade_stance,
            "breadth": market_obj.breadth,
            "leaders": list(market_obj.leaders),
            "laggards": list(market_obj.laggards),
        }
    except Exception as exc:
        market = {"health": "Unavailable", "summary": "", "trade_stance": "", "error": str(exc)}
    try:
        from news.curator_store import NewsCuratorStore
        store = NewsCuratorStore(ROOT / "logs" / "news_curator.sqlite3")
        try:
            news = [item.as_dict() for item in store.recent(hours=24 * 30, limit=20, symbol=symbol)]
        finally:
            store.close()
    except Exception:
        news = []
    try:
        fno_path = ROOT / "logs" / "product" / "fno_universe.json"
        if fno_path.exists():
            fno = json.loads(fno_path.read_text(encoding="utf-8"))
        else:
            from data.fno_universe import current_fno_universe
            report = current_fno_universe()
            fno = {
                "generated_at": "",
                "source": report.source,
                "underlyings": [item.__dict__ for item in report.underlyings],
            }
    except Exception:
        fno = {}
    return {
        "scan_payload": scan,
        "long_term_payload": long_term,
        "frame": frame,
        "market": market,
        "news": news,
        "fno_payload": fno,
    }


def build_equity_dossier(
    symbol: str,
    *,
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    frame: Any = None,
    market: Mapping[str, Any] | None = None,
    news: Sequence[Mapping[str, Any]] | None = None,
    fno_payload: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build one auditable single-stock research dossier payload."""
    symbol = _clean_symbol(symbol)
    if all(value is None for value in (scan_payload, long_term_payload, frame, market, news, fno_payload)):
        defaults = _default_inputs(symbol)
        scan_payload = defaults["scan_payload"]
        long_term_payload = defaults["long_term_payload"]
        frame = defaults["frame"]
        market = defaults["market"]
        news = defaults["news"]
        fno_payload = defaults["fno_payload"]

    scan_row = _record(scan_payload, symbol)
    long_row = _record(long_term_payload, symbol)
    price = _price_metrics(frame)
    fundamentals = dict(long_row.get("fundamentals", {}) or {})
    company = str(scan_row.get("company") or long_row.get("company") or symbol)
    sector = str(long_row.get("sector") or scan_row.get("sector") or "Unclassified")
    quality = list(dict.fromkeys([str(x) for x in (long_row.get("quality_factors", []) or []) if str(x).strip()]))
    technical = list(dict.fromkeys([str(x) for x in (scan_row.get("reasons", []) or []) if str(x).strip()]))
    risks = list(dict.fromkeys([str(x) for x in (long_row.get("risk_flags", []) or []) if str(x).strip()]))
    if scan_row.get("chase_risk"):
        risks.append("Current price structure is flagged as extended; do not chase without a fresh base or pullback.")

    news_rows = [dict(item) for item in (news or []) if isinstance(item, Mapping)]
    news_rows.sort(key=lambda item: (int(item.get("impact_score", 0) or 0), str(item.get("published_at", ""))), reverse=True)
    management_evidence = [
        item for item in news_rows
        if str(item.get("event_type", "")) in {"results", "order_or_contract", "fund_raising", "promoter_or_insider"}
        or bool(item.get("official"))
    ][:10]

    fno_match = None
    for item in list((fno_payload or {}).get("underlyings", []) or []):
        if isinstance(item, Mapping) and str(item.get("symbol", "")).upper() == symbol:
            fno_match = dict(item)
            break

    open_items: list[str] = []
    if not long_row:
        open_items.append("No completed Long-Term record is available for this symbol.")
    if not fundamentals:
        open_items.append("Current fundamental metrics are unavailable or not yet refreshed.")
    if not price.get("available"):
        open_items.append("Official bhavcopy price history is unavailable for this symbol.")
    if not news_rows:
        open_items.append("No curated company-linked news or filing evidence is available in the current news store.")
    if fundamentals.get("fii_holding") is None or fundamentals.get("dii_holding") is None:
        open_items.append("Quarterly FII and DII ownership history is not present in the current canonical data pack.")
    if not any(key in fundamentals for key in ("sales_growth_3y", "profit_growth_3y", "roce", "roe")):
        open_items.append("A verified multi-year financial series has not been attached to this dossier.")
    open_items.append("Business-segment mix and management quotations require traced filing/transcript sources before publication.")

    sources = [
        {
            "name": "Whole-market scanner",
            "status": "available" if scan_row else "missing",
            "timestamp": str((scan_payload or {}).get("scanned_at", "")),
            "point_in_time": False,
            "note": "Current saved technical scan projection.",
        },
        {
            "name": "Long-Term research store",
            "status": "available" if long_row else "missing",
            "timestamp": str((long_term_payload or {}).get("scanned_at", "")),
            "point_in_time": False,
            "note": "Current technical + current-fundamental decision aid; not historical PIT evidence.",
        },
        {
            "name": "Official NSE bhavcopy",
            "status": "available" if price.get("available") else "missing",
            "timestamp": str(price.get("latest_date", "")),
            "point_in_time": True,
            "note": "Daily OHLCV history from the canonical persisted store.",
        },
        {
            "name": "Curated news and filings",
            "status": "available" if news_rows else "missing",
            "timestamp": str(news_rows[0].get("fetched_at", "")) if news_rows else "",
            "point_in_time": False,
            "note": f"{len(news_rows)} linked article(s) in the current 30-day window.",
        },
        {
            "name": "Current F&O instrument master",
            "status": "available" if fno_match else "not_applicable_or_missing",
            "timestamp": str((fno_payload or {}).get("generated_at", "")),
            "point_in_time": False,
            "note": "Current nearest futures contract metadata; not a historical derivatives series.",
        },
    ]
    available_sections = sum(
        bool(value)
        for value in (scan_row, long_row, price.get("available"), fundamentals, news_rows)
    )
    coverage_pct = round(available_sections / 5 * 100)

    thesis = list(dict.fromkeys(quality + technical + [
        str(item.get("why_it_matters", "")) for item in news_rows[:5] if str(item.get("why_it_matters", "")).strip()
    ]))[:12]
    if not thesis:
        thesis = ["QuantTerm does not yet have enough traced evidence to publish a positive investment thesis."]

    return {
        "schema_version": 1,
        "report_type": "EQUITY_RESEARCH_DOSSIER",
        "symbol": symbol,
        "company": company,
        "sector": sector,
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "classification": str(long_row.get("classification") or scan_row.get("status") or "UNCLASSIFIED"),
        "coverage_pct": coverage_pct,
        "price": price,
        "scan": scan_row,
        "long_term": long_row,
        "fundamentals": fundamentals,
        "market": dict(market or {}),
        "news": news_rows[:15],
        "management_evidence": management_evidence,
        "fno": fno_match or {},
        "thesis": thesis,
        "quality_factors": quality,
        "technical_evidence": technical,
        "risks": list(dict.fromkeys(risks)) or ["No explicit risk list has been recorded; treat the evidence pack as incomplete."],
        "sources": sources,
        "open_items": list(dict.fromkeys(open_items)),
        "disclaimer": (
            "QuantTerm Research is an evidence-organising decision aid, not a buy or sell recommendation. "
            "Current fundamentals and news are not point-in-time historical evidence unless explicitly labelled."
        ),
        "_frame": frame,
    }


def build_long_term_basket(
    *,
    symbols: Iterable[str] | None = None,
    limit: int = 3,
    long_term_payload: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a multi-company research basket using the current Long-Term shortlist."""
    if long_term_payload is None:
        from product.long_term_store import load_long_term_scan
        long_term_payload = load_long_term_scan() or {}
    records = [dict(item) for item in (long_term_payload.get("records", []) or []) if isinstance(item, Mapping)]
    requested = [_clean_symbol(item) for item in symbols] if symbols else []
    if requested:
        chosen = [item for symbol in requested for item in records if str(item.get("symbol", "")).upper() == symbol]
    else:
        chosen = records[: max(1, min(int(limit), 10))]
    dossiers = [build_equity_dossier(str(item.get("symbol", "")), long_term_payload=long_term_payload) for item in chosen]
    common_quality: dict[str, int] = {}
    common_risks: dict[str, int] = {}
    for dossier in dossiers:
        for item in dossier["quality_factors"]:
            common_quality[item] = common_quality.get(item, 0) + 1
        for item in dossier["risks"]:
            common_risks[item] = common_risks.get(item, 0) + 1
    return {
        "schema_version": 1,
        "report_type": "LONG_TERM_BASKET",
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "title": "QuantTerm Long-Term Research Basket",
        "subtitle": "Evidence-backed quality, valuation and technical-timing brief",
        "companies": dossiers,
        "common_quality": [key for key, count in sorted(common_quality.items(), key=lambda item: (-item[1], item[0])) if count >= 2][:8],
        "common_risks": [key for key, count in sorted(common_risks.items(), key=lambda item: (-item[1], item[0])) if count >= 2][:8],
        "open_items": list(dict.fromkeys(item for dossier in dossiers for item in dossier["open_items"])),
        "disclaimer": (
            "This basket is a current research publication generated from persisted QuantTerm evidence. "
            "It is not a portfolio recommendation and does not replace independent due diligence."
        ),
    }


def _report_path(prefix: str, name: str, report_dir: str | Path = DEFAULT_REPORT_DIR) -> Path:
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "report"
    return target / f"{prefix}_{safe}_{stamp}.pdf"


def generate_equity_report(symbol: str, *, report_dir: str | Path = DEFAULT_REPORT_DIR) -> Path:
    from reporting.pdf_renderer import render_equity_pdf

    dossier = build_equity_dossier(symbol)
    path = _report_path("equity_research", dossier["symbol"], report_dir)
    render_equity_pdf(dossier, path)
    return path


def generate_basket_report(*, limit: int = 3, report_dir: str | Path = DEFAULT_REPORT_DIR) -> Path:
    from reporting.pdf_renderer import render_basket_pdf

    basket = build_long_term_basket(limit=limit)
    path = _report_path("long_term_basket", f"top_{limit}", report_dir)
    render_basket_pdf(basket, path)
    return path
