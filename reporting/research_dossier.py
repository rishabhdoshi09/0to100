"""Evidence-first research dossier assembly for QuantTerm.

Every section is derived from persisted stores or validated user evidence. Runtime
sources inherit their actual as-of date and freshness state; attached but unparsed
source documents do not become analytical claims.
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
        number = float(str(value).replace(",", "").replace("%", "").strip())
        return number if number == number else None
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
    empty = {
        "available": False, "latest_price": None, "latest_date": "",
        "return_1m_pct": None, "return_3m_pct": None, "return_6m_pct": None,
        "return_12m_pct": None, "high_52w": None, "from_high_pct": None,
        "avg_volume_20d": None,
    }
    if frame is None or len(frame) == 0:
        return empty
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
        return empty


def _default_inputs(symbol: str) -> dict[str, Any]:
    from product.long_term_store import load_long_term_scan
    from product.scan_store import load_scan
    from reporting.evidence_intake import load_raw_fundamentals

    scan = load_scan() or {}
    long_term = load_long_term_scan() or {}
    raw_fundamentals = load_raw_fundamentals(symbol)
    try:
        from data.bhavcopy_runtime import get_ohlcv
        frame = get_ohlcv(symbol)
    except Exception:
        frame = None
    try:
        from product.market_view import current_market_view
        obj = current_market_view()
        market = {
            "health": obj.health, "summary": obj.summary,
            "trade_stance": obj.trade_stance, "breadth": obj.breadth,
            "leaders": list(obj.leaders), "laggards": list(obj.laggards),
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
        path = ROOT / "logs" / "product" / "fno_universe.json"
        if path.exists():
            fno = json.loads(path.read_text(encoding="utf-8"))
        else:
            from data.fno_universe import current_fno_universe
            report = current_fno_universe()
            fno = {"generated_at": "", "source": report.source, "underlyings": [row.__dict__ for row in report.underlyings]}
    except Exception:
        fno = {}
    return {
        "scan_payload": scan, "long_term_payload": long_term,
        "raw_fundamentals": raw_fundamentals, "frame": frame,
        "market": market, "news": news, "fno_payload": fno,
    }


def _row_label(row: Mapping[str, Any]) -> str:
    for key in ("", "row_label", "Particulars", "PARTICULARS", "Particular"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return str(next(iter(row.values()), "") or "")


def _normalise_table(rows: Sequence[Mapping[str, Any]] | None, limit: int = 24) -> list[dict[str, Any]]:
    output = []
    for raw in list(rows or [])[:limit]:
        row = dict(raw)
        label = _row_label(row)
        values = {str(key or "period"): value for key, value in row.items() if str(value or "").strip() and str(value) != label}
        output.append({"label": label, "values": values})
    return output


def _latest_series_value(rows: Sequence[Mapping[str, Any]], needles: Sequence[str]) -> float | None:
    for row in rows:
        label = _row_label(row).lower()
        if any(needle in label for needle in needles):
            values = [_as_float(value) for key, value in row.items() if key not in ("", "row_label")]
            clean = [value for value in values if value is not None]
            if clean:
                return clean[-1]
    return None


def _uploaded_management(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        commentary = str(row.get("commentary") or row.get("management_wording") or "").strip()
        if commentary:
            output.append({
                "published_at": str(row.get("event_date") or row.get("as_of_date") or ""),
                "fetched_at": "", "source": "User-uploaded traced evidence",
                "headline": str(row.get("topic") or row.get("guidance_metric") or "Management commentary"),
                "why_it_matters": commentary, "speaker": str(row.get("speaker") or ""),
                "source_url": str(row.get("source_url") or ""), "official": False,
                "event_type": "management_commentary", "impact_score": 0,
            })
    return output


def _strict_coverage(requirements: Mapping[str, Any], runtime_presence: Mapping[str, bool]) -> tuple[int, list[dict[str, Any]]]:
    weights = {
        "price_history": 10, "scanner": 10, "long_term": 10,
        "business_profile": 10, "financial_history": 20,
        "shareholding_history": 15, "business_segments": 10,
        "management_commentary": 10, "news": 5,
    }
    requirement_map = {item.get("key"): item for item in requirements.get("requirements", [])}
    runtime_map = {item.get("key"): item for item in requirements.get("runtime_sources", [])}
    score = 0.0
    rows: list[dict[str, Any]] = []
    for key, weight in weights.items():
        if key in runtime_map:
            source = dict(runtime_map[key])
            # Symbol-specific availability can be narrower than the global store timestamp.
            available = bool(source.get("available")) and bool(runtime_presence.get(key))
            status = str(source.get("status") or "MISSING") if available else "MISSING"
        else:
            source = dict(requirement_map.get(key, {}))
            available = bool(source.get("available"))
            status = str(source.get("status") or "MISSING")
        factor = 1.0 if available and status == "FRESH" else 0.5 if available and status in {"STALE", "UNKNOWN_DATE"} else 0.0
        score += weight * factor
        rows.append({
            "key": key, "weight": weight, "status": status,
            "available": available, "as_of": str(source.get("as_of") or ""),
            "age_days": source.get("age_days"),
        })
    return round(score), rows


def build_equity_dossier(
    symbol: str, *,
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    raw_fundamentals: Mapping[str, Any] | None = None,
    frame: Any = None, market: Mapping[str, Any] | None = None,
    news: Sequence[Mapping[str, Any]] | None = None,
    fno_payload: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    symbol = _clean_symbol(symbol)
    if any(value is None for value in (scan_payload, long_term_payload, raw_fundamentals, frame, market, news, fno_payload)):
        defaults = _default_inputs(symbol)
        scan_payload = defaults["scan_payload"] if scan_payload is None else scan_payload
        long_term_payload = defaults["long_term_payload"] if long_term_payload is None else long_term_payload
        raw_fundamentals = defaults["raw_fundamentals"] if raw_fundamentals is None else raw_fundamentals
        frame = defaults["frame"] if frame is None else frame
        market = defaults["market"] if market is None else market
        news = defaults["news"] if news is None else news
        fno_payload = defaults["fno_payload"] if fno_payload is None else fno_payload

    from reporting.evidence_intake import evidence_requirements, structured_rows

    scan_row = _record(scan_payload, symbol)
    long_row = _record(long_term_payload, symbol)
    price = _price_metrics(frame)
    raw_record = dict(raw_fundamentals or {})
    raw = dict(raw_record.get("data", {}) or {})
    fundamentals = dict(long_row.get("fundamentals", {}) or {})
    uploaded_shareholding = structured_rows(symbol, "shareholding_history")
    shareholding_rows = uploaded_shareholding or list(raw.get("shareholding", []) or [])
    if fundamentals.get("fii_holding") is None:
        fundamentals["fii_holding"] = _latest_series_value(shareholding_rows, ("fii", "foreign institutional", "foreign portfolio"))
    if fundamentals.get("dii_holding") is None:
        fundamentals["dii_holding"] = _latest_series_value(shareholding_rows, ("dii", "domestic institutional"))

    company = str(scan_row.get("company") or long_row.get("company") or symbol)
    sector = str(long_row.get("sector") or scan_row.get("sector") or "Unclassified")
    quality = list(dict.fromkeys(str(value) for value in (long_row.get("quality_factors", []) or []) if str(value).strip()))
    technical = list(dict.fromkeys(str(value) for value in (scan_row.get("reasons", []) or []) if str(value).strip()))
    risks = list(dict.fromkeys(str(value) for value in (long_row.get("risk_flags", []) or []) if str(value).strip()))
    if scan_row.get("chase_risk"):
        risks.append("Current price structure is flagged as extended; do not chase without a fresh base or pullback.")

    news_rows = [dict(item) for item in (news or []) if isinstance(item, Mapping)]
    news_rows.sort(key=lambda item: (int(item.get("impact_score", 0) or 0), str(item.get("published_at", ""))), reverse=True)
    management_evidence = _uploaded_management(structured_rows(symbol, "management_commentary"))
    management_evidence += [
        item for item in news_rows
        if str(item.get("event_type", "")) in {"results", "order_or_contract", "fund_raising", "promoter_or_insider"} or bool(item.get("official"))
    ][:10]

    institutional_context: dict[str, Any] = {}
    symbol_bulk_deals: list[dict[str, Any]] = []
    try:
        from data.fii_dii_store import workspace_payload

        inst = workspace_payload(days=30)
        institutional_context = {
            "market_bias": str((inst.get("cash") or {}).get("bias") or ""),
            "market_note": str(inst.get("insight") or (inst.get("cash") or {}).get("note") or ""),
            "fii_net_30d_cr": (inst.get("cash") or {}).get("totals", {}).get("fii_net_cr"),
            "dii_net_30d_cr": (inst.get("cash") or {}).get("totals", {}).get("dii_net_cr"),
            "bulk_buy_symbols": list(inst.get("bulk_buy_symbols") or [])[:20],
            "derivatives": dict(inst.get("derivatives") or {}),
        }
        symbol_bulk_deals = [
            dict(d) for d in (inst.get("bulk_deals") or [])
            if str(d.get("symbol", "")).upper() == symbol
        ]
        if symbol in institutional_context.get("bulk_buy_symbols", []):
            institutional_context["symbol_bulk_buy_flag"] = True
    except Exception:
        institutional_context = {}

    options_context: dict[str, Any] = {}
    try:
        from options.chain_fetch import chain_workspace

        spot = price.get("latest_price")
        options_context = chain_workspace(symbol, spot=float(spot) if spot else None)
    except Exception:
        options_context = {"available": False}

    fno_match = next((dict(item) for item in list((fno_payload or {}).get("underlyings", []) or []) if isinstance(item, Mapping) and str(item.get("symbol", "")).upper() == symbol), None)
    news_as_of = str(news_rows[0].get("published_at") or news_rows[0].get("fetched_at") or "") if news_rows else ""
    requirements = evidence_requirements(
        symbol,
        price_as_of=str(price.get("latest_date") or ""),
        scan_as_of=str((scan_payload or {}).get("scanned_at", "")),
        long_term_as_of=str((long_term_payload or {}).get("scanned_at", "")),
        news_as_of=news_as_of,
        fno_as_of=str((fno_payload or {}).get("generated_at", "")),
    )
    coverage_pct, section_coverage = _strict_coverage(requirements, {
        "price_history": bool(price.get("available")),
        "scanner": bool(scan_row), "long_term": bool(long_row), "news": bool(news_rows),
    })

    open_items: list[str] = []
    for item in requirements.get("requirements", []):
        if not item.get("available"):
            links = item.get("links", [])
            first = str(links[0].get("url")) if links else ""
            state = " A source is attached but still needs structured extraction." if item.get("source_attached") else ""
            open_items.append(f"{item.get('label')} is not usable.{state} {item.get('instructions')}" + (f" Primary source: {first}." if first else ""))
        elif item.get("status") == "STALE":
            open_items.append(f"{item.get('label')} is stale: as of {item.get('as_of')} ({item.get('age_days')} days old). Refresh it before circulation.")
    if not price.get("available"):
        open_items.append("Official bhavcopy price history is unavailable for this symbol; no synthetic price series is allowed.")
    if not news_rows:
        open_items.append("No curated company-linked news or filing evidence is available in the current 30-day store.")

    company_about = str(raw.get("about") or "").strip()
    profile_rows = structured_rows(symbol, "business_profile")
    if profile_rows:
        company_about = str(profile_rows[0].get("business_summary") or company_about)
    uploaded_financials = structured_rows(symbol, "financial_history")
    financial_tables = {
        "uploaded": uploaded_financials,
        "quarterly_results": _normalise_table(raw.get("quarterly_results", []), 20),
        "profit_loss": _normalise_table(raw.get("profit_loss", []), 24),
        "balance_sheet": _normalise_table(raw.get("balance_sheet", []), 24),
        "cash_flow": _normalise_table(raw.get("cash_flow", []), 20),
        "peer_comparison": _normalise_table(raw.get("peer_comparison", []), 15),
        "as_of": str((uploaded_financials or [{}])[-1].get("as_of_date") or raw_record.get("fetched_at") or ""),
    }

    sources = [
        {"name": "Whole-market scanner", "status": "available" if scan_row else "missing", "as_of": str((scan_payload or {}).get("scanned_at", "")), "point_in_time": False, "note": "Current saved technical scan projection."},
        {"name": "Long-Term research store", "status": "available" if long_row else "missing", "as_of": str((long_term_payload or {}).get("scanned_at", "")), "point_in_time": False, "note": "Current technical plus current-fundamental decision aid."},
        {"name": "Deep fundamentals cache", "status": str(raw_record.get("freshness") or "MISSING"), "as_of": str(raw_record.get("fetched_at") or ""), "point_in_time": False, "note": "Raw company description, financial tables and shareholding snapshot."},
        {"name": "Official NSE bhavcopy", "status": "available" if price.get("available") else "missing", "as_of": str(price.get("latest_date", "")), "point_in_time": True, "note": "Daily OHLCV history from the canonical store."},
        {"name": "Curated news and filings", "status": "available" if news_rows else "missing", "as_of": news_as_of, "point_in_time": False, "note": f"{len(news_rows)} linked article(s) in the current 30-day window."},
        {"name": "Current F&O instrument master", "status": "available" if fno_match else "not_applicable_or_missing", "as_of": str((fno_payload or {}).get("generated_at", "")), "point_in_time": False, "note": "Current nearest futures metadata; not a historical series."},
    ]
    for upload in requirements.get("uploads", []):
        sources.append({
            "name": f"Uploaded evidence: {upload.get('kind')}",
            "status": str(upload.get("extraction_status") or "SOURCE_ATTACHED_UNPARSED"),
            "as_of": str(upload.get("as_of") or ""), "point_in_time": True,
            "note": f"{upload.get('filename')} · SHA256 {str(upload.get('sha256', ''))[:12]} · {upload.get('source_url') or 'source URL missing'}",
        })

    thesis = list(dict.fromkeys(quality + technical + [str(item.get("why_it_matters", "")) for item in news_rows[:5] if str(item.get("why_it_matters", "")).strip()]))[:12]
    if not thesis:
        thesis = ["QuantTerm does not yet have enough traced evidence to publish a positive investment thesis."]

    return {
        "schema_version": 3, "report_type": "EQUITY_RESEARCH_DOSSIER",
        "symbol": symbol, "company": company, "sector": sector,
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "classification": str(long_row.get("classification") or scan_row.get("status") or "UNCLASSIFIED"),
        "coverage_pct": coverage_pct, "section_coverage": section_coverage,
        "price": price, "scan": scan_row, "long_term": long_row,
        "fundamentals": fundamentals, "deep_fundamentals": raw,
        "deep_fundamentals_fetched_at": str(raw_record.get("fetched_at") or ""),
        "company_about": company_about, "financial_tables": financial_tables,
        "shareholding_history": _normalise_table(shareholding_rows, 30),
        "business_segments": structured_rows(symbol, "business_segments"),
        "order_book_guidance": structured_rows(symbol, "order_book_guidance"),
        "market": dict(market or {}), "news": news_rows[:15],
        "management_evidence": management_evidence[:15], "fno": fno_match or {},
        "institutional_context": institutional_context,
        "symbol_bulk_deals": symbol_bulk_deals[:10],
        "options_context": options_context,
        "thesis": thesis, "quality_factors": quality, "technical_evidence": technical,
        "risks": list(dict.fromkeys(risks)) or ["No explicit risk list has been recorded; treat the evidence pack as incomplete."],
        "sources": sources, "evidence_requirements": requirements,
        "open_items": list(dict.fromkeys(open_items)),
        "disclaimer": "QuantTerm Research is an evidence-organising decision aid, not a buy or sell recommendation. Current fundamentals and news are not point-in-time historical evidence unless explicitly labelled.",
        "_frame": frame,
    }


def build_long_term_basket(
    *, symbols: Iterable[str] | None = None, limit: int = 3,
    long_term_payload: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    if long_term_payload is None:
        from product.long_term_store import load_long_term_scan
        long_term_payload = load_long_term_scan() or {}
    records = [dict(item) for item in (long_term_payload.get("records", []) or []) if isinstance(item, Mapping)]
    requested = [_clean_symbol(item) for item in symbols] if symbols else []
    chosen = [item for symbol in requested for item in records if str(item.get("symbol", "")).upper() == symbol] if requested else records[:max(1, min(int(limit), 10))]
    dossiers = [build_equity_dossier(str(item.get("symbol", "")), long_term_payload=long_term_payload) for item in chosen]
    quality_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for dossier in dossiers:
        for item in dossier["quality_factors"]:
            quality_counts[item] = quality_counts.get(item, 0) + 1
        for item in dossier["risks"]:
            risk_counts[item] = risk_counts.get(item, 0) + 1
    return {
        "schema_version": 3, "report_type": "LONG_TERM_BASKET",
        "generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "title": "QuantTerm Long-Term Research Basket",
        "subtitle": "Evidence-backed quality, valuation and technical-timing brief",
        "companies": dossiers,
        "common_quality": [key for key, count in sorted(quality_counts.items(), key=lambda item: (-item[1], item[0])) if count >= 2][:8],
        "common_risks": [key for key, count in sorted(risk_counts.items(), key=lambda item: (-item[1], item[0])) if count >= 2][:8],
        "open_items": list(dict.fromkeys(item for dossier in dossiers for item in dossier["open_items"])),
        "disclaimer": "This basket is a current research publication generated from persisted QuantTerm evidence. It is not a portfolio recommendation and does not replace independent due diligence.",
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
    path = _report_path("equity_evidence_brief", dossier["symbol"], report_dir)
    render_equity_pdf(dossier, path)
    return path


def generate_basket_report(*, limit: int = 3, report_dir: str | Path = DEFAULT_REPORT_DIR) -> Path:
    from reporting.pdf_renderer import render_basket_pdf
    basket = build_long_term_basket(limit=limit)
    path = _report_path("long_term_basket", f"top_{limit}", report_dir)
    render_basket_pdf(basket, path)
    return path
