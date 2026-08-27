"""Single-stock intelligence projection for QuantTerm.

The module builds one explainable workspace from persisted scanner, long-term,
official-history, fundamentals, news and F&O stores. It performs no trading and
never fills missing metrics with estimates.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]


def clean_symbol(value: str) -> str:
    symbol = re.sub(r"[^A-Z0-9&.-]", "", str(value or "").strip().upper())
    if not symbol or len(symbol) > 32:
        raise ValueError("invalid NSE symbol")
    return symbol


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        result = float(str(value).replace(",", "").replace("%", "").strip())
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def _find(payload: Mapping[str, Any] | None, symbol: str) -> dict[str, Any]:
    for row in list((payload or {}).get("records", []) or []):
        if isinstance(row, Mapping) and str(row.get("symbol", "")).upper() == symbol:
            return dict(row)
    return {}


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(text[:11], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _source(name: str, available: bool, as_of: Any, max_age_days: int, meaning: str,
            *, now: datetime) -> dict[str, Any]:
    stamp = _parse_time(as_of)
    age_days = None if stamp is None else max(0, int((now - stamp.astimezone(timezone.utc)).total_seconds() // 86400))
    if not available:
        status = "MISSING"
    elif age_days is None:
        status = "UNKNOWN_DATE"
    elif age_days > max_age_days:
        status = "STALE"
    else:
        status = "FRESH"
    return {
        "name": name,
        "available": bool(available),
        "status": status,
        "as_of": str(as_of or ""),
        "age_days": age_days,
        "max_age_days": max_age_days,
        "meaning": meaning,
    }


def _return_pct(series: Any, periods: int) -> float | None:
    try:
        clean = series.dropna()
        if len(clean) <= periods:
            return None
        return round((float(clean.iloc[-1]) / float(clean.iloc[-periods - 1]) - 1.0) * 100.0, 2)
    except Exception:
        return None


def _rsi(close: Any, periods: int = 14) -> float | None:
    try:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean()
        last_loss = float(loss.iloc[-1])
        if last_loss == 0:
            return 100.0
        rs = float(gain.iloc[-1]) / last_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 2)
    except Exception:
        return None


def _atr(frame: Any, periods: int = 14) -> float | None:
    try:
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        previous = close.shift(1)
        true_range = (high - low).to_frame("a")
        true_range["b"] = (high - previous).abs()
        true_range["c"] = (low - previous).abs()
        value = true_range.max(axis=1).ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean().iloc[-1]
        return round(float(value), 2)
    except Exception:
        return None


def _interpret_rsi(value: float | None) -> str:
    if value is None:
        return "RSI is unavailable."
    if value >= 75:
        return "Momentum is strong but stretched; fresh entries need price discipline."
    if value >= 55:
        return "Momentum is positive without being deeply overbought."
    if value <= 30:
        return "Price is oversold; this is not automatically a reversal signal."
    if value < 45:
        return "Momentum is weak or cooling."
    return "Momentum is balanced."


def _trend(close: float, ema20: float | None, ema50: float | None, ema200: float | None) -> tuple[str, str]:
    if ema20 is None or ema50 is None:
        return "UNCONFIRMED", "Not enough history for a reliable moving-average trend."
    if ema200 is not None and close > ema20 > ema50 > ema200:
        return "PRIMARY UPTREND", "Price and short/medium/long averages are positively stacked."
    if close > ema20 > ema50:
        return "UPTREND", "Price is above the 20- and 50-day averages; the long-term stack is incomplete or mixed."
    if ema200 is not None and close < ema20 < ema50 < ema200:
        return "PRIMARY DOWNTREND", "Price and moving averages are negatively stacked."
    if close < ema20 and close < ema50:
        return "DOWNTREND", "Price is below both short- and medium-term trend references."
    return "MIXED", "Trend references disagree; wait for cleaner structure or use a wider time horizon."


def _technical(frame: Any) -> dict[str, Any]:
    if frame is None or len(frame) == 0:
        return {"available": False, "metrics": [], "trend": "UNAVAILABLE", "trend_explanation": "Official history is unavailable."}
    try:
        data = frame.sort_index().copy()
        close_series = data["close"].astype(float).dropna()
        close = float(close_series.iloc[-1])
        ema20_series = close_series.ewm(span=20, adjust=False).mean()
        ema50_series = close_series.ewm(span=50, adjust=False).mean()
        ema200_series = close_series.ewm(span=200, adjust=False).mean()
        ema20 = round(float(ema20_series.iloc[-1]), 2) if len(close_series) >= 20 else None
        ema50 = round(float(ema50_series.iloc[-1]), 2) if len(close_series) >= 50 else None
        ema200 = round(float(ema200_series.iloc[-1]), 2) if len(close_series) >= 200 else None
        rsi14 = _rsi(close_series)
        atr14 = _atr(data)
        atr_pct = round(atr14 / close * 100.0, 2) if atr14 is not None and close else None
        high_52w = round(float(data["high"].tail(min(252, len(data))).max()), 2)
        low_52w = round(float(data["low"].tail(min(252, len(data))).min()), 2)
        from_high = round((close / high_52w - 1.0) * 100.0, 2) if high_52w else None
        from_low = round((close / low_52w - 1.0) * 100.0, 2) if low_52w else None
        avg_volume = float(data["volume"].tail(20).mean()) if "volume" in data.columns else None
        volume = float(data["volume"].iloc[-1]) if "volume" in data.columns else None
        volume_ratio = round(volume / avg_volume, 2) if volume is not None and avg_volume not in (None, 0) else None
        trend, trend_explanation = _trend(close, ema20, ema50, ema200)
        latest_index = data.index[-1]
        latest_date = str(getattr(latest_index, "date", lambda: latest_index)())
        metrics = [
            {"key": "price", "label": "Current close", "value": round(close, 2), "unit": "INR", "meaning": "Latest official daily closing price in the local history store.", "interpretation": trend_explanation},
            {"key": "return_1m", "label": "1-month return", "value": _return_pct(close_series, 21), "unit": "%", "meaning": "Approximate 21-session price change.", "interpretation": "Useful for recent momentum; not a forecast."},
            {"key": "return_3m", "label": "3-month return", "value": _return_pct(close_series, 63), "unit": "%", "meaning": "Approximate 63-session price change.", "interpretation": "Shows medium-term direction and strength."},
            {"key": "return_12m", "label": "12-month return", "value": _return_pct(close_series, 252), "unit": "%", "meaning": "Approximate 252-session price change.", "interpretation": "Longer-horizon momentum, unavailable when history is shallow."},
            {"key": "rsi14", "label": "RSI (14)", "value": rsi14, "unit": "", "meaning": "A 0-100 momentum oscillator; high values can mean strength and extension at the same time.", "interpretation": _interpret_rsi(rsi14)},
            {"key": "volume_ratio", "label": "Volume / 20-day average", "value": volume_ratio, "unit": "x", "meaning": "Today’s volume divided by the recent average.", "interpretation": "Above 1.5x usually means unusually strong participation." if volume_ratio is not None else "Volume data unavailable."},
            {"key": "atr_pct", "label": "ATR (14) as % of price", "value": atr_pct, "unit": "%", "meaning": "Typical daily movement relative to price; a practical volatility and stop-distance reference.", "interpretation": "Higher ATR means wider day-to-day movement and larger position-sizing risk."},
            {"key": "from_high", "label": "Distance from 52-week high", "value": from_high, "unit": "%", "meaning": "How far the current close is below or above the trailing 52-week high.", "interpretation": "Near zero means price is close to its yearly high; deeply negative values indicate a larger drawdown."},
        ]
        return {
            "available": True,
            "latest_date": latest_date,
            "close": round(close, 2),
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "rsi14": rsi14,
            "atr14": atr14,
            "atr_pct": atr_pct,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "from_high_pct": from_high,
            "from_low_pct": from_low,
            "volume_ratio": volume_ratio,
            "trend": trend,
            "trend_explanation": trend_explanation,
            "metrics": metrics,
        }
    except Exception as exc:
        return {"available": False, "metrics": [], "trend": "ERROR", "trend_explanation": str(exc), "error": str(exc)}


FUNDAMENTAL_META: dict[str, tuple[str, str, str]] = {
    "market_cap": ("Market capitalisation", "INR Cr", "Approximate market value of the listed equity."),
    "pe": ("Price / earnings", "x", "Price paid for each rupee of current earnings; compare with growth, quality and peers."),
    "roe": ("Return on equity", "%", "Profit generated relative to shareholder equity."),
    "roce": ("Return on capital employed", "%", "Operating return generated on debt plus equity capital."),
    "sales_growth_3y": ("Sales CAGR (3Y)", "%", "Compounded revenue growth over roughly three years."),
    "profit_growth_3y": ("Profit CAGR (3Y)", "%", "Compounded profit growth over roughly three years."),
    "debt_to_equity": ("Debt / equity", "x", "Balance-sheet leverage; not directly comparable for banks and financial companies."),
    "interest_coverage": ("Interest coverage", "x", "Operating profit available to service interest expense."),
    "cfo_to_pat": ("Cash flow / PAT", "x", "Operating cash flow relative to accounting profit."),
    "promoter_holding": ("Promoter holding", "%", "Promoter ownership in the company."),
    "promoter_pledge": ("Promoter pledge", "%", "Promoter shares pledged as collateral; higher values can add governance and financing risk."),
    "fii_holding": ("FII holding", "%", "Foreign institutional ownership in the latest available disclosure."),
    "dii_holding": ("DII holding", "%", "Domestic institutional ownership in the latest available disclosure."),
}


def _fund_interpretation(key: str, value: float | None, *, financial_sector: bool) -> str:
    if value is None:
        return "Not available in the current verified data pack."
    if key in {"roe", "roce"}:
        return "Strong" if value >= 15 else "Weak" if value < 8 else "Moderate"
    if key in {"sales_growth_3y", "profit_growth_3y"}:
        return "Healthy growth" if value >= 10 else "Contracting" if value < 0 else "Modest growth"
    if key == "debt_to_equity":
        if financial_sector:
            return "Use bank/NBFC-specific capital metrics instead of this generic leverage ratio."
        return "Low leverage" if value <= 0.7 else "High leverage" if value > 2 else "Manageable but requires context"
    if key == "interest_coverage":
        return "Comfortable" if value >= 3 else "Weak debt service" if value < 1.5 else "Watch"
    if key == "cfo_to_pat":
        return "Healthy cash conversion" if value >= 0.8 else "Weak cash conversion" if value < 0.5 else "Mixed cash conversion"
    if key == "promoter_pledge":
        return "No reported pledge" if value == 0 else "Elevated pledge risk" if value > 10 else "Limited reported pledge"
    if key == "pe":
        return "Valuation needs peer and growth context; low is not automatically cheap and high is not automatically bad."
    return "Use the metric together with trend, history, peers and source date."


def _fundamentals(long_row: Mapping[str, Any], raw_record: Mapping[str, Any], sector: str) -> dict[str, Any]:
    values = dict(long_row.get("fundamentals", {}) or {})
    raw = dict(raw_record.get("data", {}) or {})
    if raw:
        try:
            from screener.engine import _extract_fundamentals
            extracted = dict(_extract_fundamentals(raw) or {})
            if extracted.get("market_cap_cr") is not None and values.get("market_cap") in (None, ""):
                values["market_cap"] = extracted.get("market_cap_cr")
            for key, value in extracted.items():
                if values.get(key) in (None, ""):
                    values[key] = value
        except Exception:
            pass
    financial_sector = any(token in sector.lower() for token in ("bank", "finance", "financial", "insurance", "nbfc"))
    metrics = []
    available_count = 0
    for key, (label, unit, meaning) in FUNDAMENTAL_META.items():
        value = _f(values.get(key))
        if value is not None:
            available_count += 1
        metrics.append({
            "key": key,
            "label": label,
            "value": value,
            "unit": unit,
            "meaning": meaning,
            "interpretation": _fund_interpretation(key, value, financial_sector=financial_sector),
        })
    coverage = round(available_count / len(FUNDAMENTAL_META) * 100) if FUNDAMENTAL_META else 0
    return {
        "available": bool(available_count),
        "coverage_pct": coverage,
        "score": _f(long_row.get("fundamental_score")),
        "classification": str(long_row.get("classification") or ""),
        "quality_factors": list(long_row.get("quality_factors", []) or []),
        "risk_flags": list(long_row.get("risk_flags", []) or []),
        "metrics": metrics,
        "raw_values": values,
        "company_about": str(raw.get("about") or "").strip(),
        "fetched_at": str(raw_record.get("fetched_at") or ""),
        "section_as_of": dict(raw_record.get("section_as_of", {}) or {}),
    }


def _default_inputs(symbol: str) -> dict[str, Any]:
    from product.scan_store import load_scan
    from product.long_term_store import load_long_term_scan
    from reporting.evidence_intake import load_raw_fundamentals

    scan = load_scan() or {}
    long_term = load_long_term_scan() or {}
    raw = load_raw_fundamentals(symbol)
    try:
        from data.bhavcopy_runtime import get_ohlcv
        frame = get_ohlcv(symbol)
    except Exception:
        frame = None
    try:
        from news.curator_store import NewsCuratorStore
        store = NewsCuratorStore(ROOT / "logs" / "news_curator.sqlite3")
        try:
            news = [item.as_dict() for item in store.recent(hours=24 * 30, limit=20, symbol=symbol)]
        finally:
            store.close()
    except Exception:
        news = []
    fno = {}
    try:
        path = ROOT / "logs" / "product" / "fno_universe.json"
        fno = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        fno = {}
    return {"scan": scan, "long_term": long_term, "raw": raw, "frame": frame, "news": news, "fno": fno}


def build_stock_workspace(
    symbol: str,
    *,
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    raw_fundamentals: Mapping[str, Any] | None = None,
    frame: Any = None,
    news: Sequence[Mapping[str, Any]] | None = None,
    fno_payload: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a data-packed, source-dated single-stock workspace."""
    symbol = clean_symbol(symbol)
    now = now or datetime.now(timezone.utc)
    if any(value is None for value in (scan_payload, long_term_payload, raw_fundamentals, frame, news, fno_payload)):
        defaults = _default_inputs(symbol)
        scan_payload = defaults["scan"] if scan_payload is None else scan_payload
        long_term_payload = defaults["long_term"] if long_term_payload is None else long_term_payload
        raw_fundamentals = defaults["raw"] if raw_fundamentals is None else raw_fundamentals
        frame = defaults["frame"] if frame is None else frame
        news = defaults["news"] if news is None else news
        fno_payload = defaults["fno"] if fno_payload is None else fno_payload

    scan_row = _find(scan_payload, symbol)
    long_row = _find(long_term_payload, symbol)
    raw_record = dict(raw_fundamentals or {})
    technical = _technical(frame)
    sector = str(long_row.get("sector") or scan_row.get("sector") or "Unclassified")
    company = str(scan_row.get("company") or long_row.get("company") or symbol)
    fundamentals = _fundamentals(long_row, raw_record, sector)
    news_rows = [dict(item) for item in (news or []) if isinstance(item, Mapping)]
    news_rows.sort(key=lambda item: (str(item.get("published_at") or item.get("fetched_at") or ""), int(item.get("impact_score", 0) or 0)), reverse=True)
    instrument = next((dict(item) for item in list((fno_payload or {}).get("underlyings", []) or []) if isinstance(item, Mapping) and str(item.get("symbol", "")).upper() == symbol), {})
    option_chain: dict[str, Any] = {}
    autonomy_acquired_at = ""
    try:
        from product.due_diligence.acquire import load_autonomy_facts
        facts = load_autonomy_facts(symbol)
        autonomy_acquired_at = str(facts.get("acquired_at") or "")
        chain = facts.get("option_chain")
        if isinstance(chain, Mapping) and chain:
            option_chain = dict(chain)
    except Exception:
        option_chain = {}
    fno_out = dict(instrument)
    if option_chain:
        fno_out["option_chain"] = option_chain

    sources = [
        _source("Official price history", technical.get("available", False), technical.get("latest_date"), 4, "Charts and technical calculations use saved NSE daily OHLCV.", now=now),
        _source("Whole-market scanner", bool(scan_row), (scan_payload or {}).get("scanned_at"), 1, "Current technical setup, entry framework and scanner reasons.", now=now),
        _source("Long-term research", bool(long_row), (long_term_payload or {}).get("scanned_at"), 4, "Current business-quality snapshot combined with technical timing.", now=now),
        _source(
            "Deep fundamentals",
            fundamentals.get("available", False),
            (fundamentals.get("section_as_of", {}) or {}).get("financial_history") or raw_record.get("fetched_at"),
            120,
            "Current cached company description and financial tables; freshness follows the latest disclosed financial period when available.",
            now=now,
        ),
        _source("Company-linked news", bool(news_rows), (news_rows[0].get("published_at") or news_rows[0].get("fetched_at")) if news_rows else "", 7, "Dated context from configured sources; never a standalone order signal.", now=now),
        _source("F&O instrument master", bool(instrument), (fno_payload or {}).get("generated_at"), 2, "Current derivatives eligibility, nearest future, expiry and lot size.", now=now),
    ]
    if option_chain:
        sources.append(_source(
            "Option-chain snapshot",
            bool(option_chain.get("available")),
            autonomy_acquired_at,
            1,
            "Nearest-expiry OI / IV / PCR from last Investigate acquire. Not a trade signal.",
            now=now,
        ))
    weights = {"Official price history": 30, "Whole-market scanner": 15, "Long-term research": 20, "Deep fundamentals": 25, "Company-linked news": 5, "F&O instrument master": 5}
    confidence = 0.0
    for source in sources:
        factor = 1.0 if source["status"] == "FRESH" else 0.5 if source["available"] else 0.0
        confidence += weights.get(source["name"], 0) * factor
    confidence_pct = round(confidence)

    technical_ready = bool(technical.get("available"))
    fundamental_ready = bool(fundamentals.get("available")) and fundamentals.get("coverage_pct", 0) >= 40
    if technical_ready and fundamental_ready and confidence_pct >= 70:
        state = "RESEARCH_READY"
        summary = "Technicals and a usable fundamental snapshot are both present. Check source dates and invalidation before acting."
    elif technical_ready:
        state = "TECHNICAL_ONLY"
        summary = "Price structure is available, but fundamental coverage is incomplete or stale."
    elif fundamental_ready:
        state = "FUNDAMENTAL_ONLY"
        summary = "Fundamental data is available, but official price history or technical context is missing."
    else:
        state = "DATA_INCOMPLETE"
        summary = "QuantTerm does not yet have enough verified stock-level evidence for a practical research view."

    gaps = [source["name"] for source in sources if source["status"] in {"MISSING", "STALE", "UNKNOWN_DATE"}]
    next_actions = []
    if not technical_ready:
        next_actions.append({"control": "REFRESH_DATA_NOW", "label": "Prepare official price history"})
    if not scan_row or sources[1]["status"] != "FRESH":
        next_actions.append({"control": "RUN_SCAN_NOW", "label": "Scan market (all setups)"})
    elif not long_row or sources[2]["status"] != "FRESH":
        next_actions.append({"control": "REFRESH_LONG_TERM_NOW", "label": "Refresh long-term funds"})
    if not fundamentals.get("available") or sources[3]["status"] != "FRESH":
        next_actions.append({"control": "REFRESH_STOCK_FUNDAMENTALS", "label": f"Refresh {symbol} fundamentals"})
    if not news_rows or sources[4]["status"] != "FRESH":
        next_actions.append({"control": "REFRESH_NEWS_NOW", "label": "Refresh news and filings"})

    case: dict[str, Any] = {}
    try:
        from product.case_memory import remember_case
        seed = dict(scan_row or long_row or {})
        seed.setdefault("symbol", symbol)
        seed.setdefault("company", company)
        seed.setdefault("sector", sector)
        cat = ""
        if scan_row:
            try:
                from product.recommendations_workspace import primary_scan_category
                assigned = primary_scan_category(scan_row)
                cat = assigned[0] if assigned else ""
            except Exception:
                cat = ""
        elif long_row:
            cat = "wealth_builders"
        if cat:
            seed.setdefault("category_id", cat)
        inv = [
            flag for flag in (fundamentals.get("risk_flags") or []) if flag
        ][:4]
        if not inv:
            try:
                from product.decision_card import what_changes_mind
                inv = what_changes_mind(seed, category_id=cat or str(seed.get("category_id") or ""))
            except Exception:
                inv = []
        case = remember_case(
            {
                "symbol": symbol,
                "company": company,
                "category_id": seed.get("category_id") or cat,
                "setup_label": str(seed.get("status") or seed.get("classification") or ""),
                "why_now": list(seed.get("reasons") or [])[:4],
                "what_changes_mind": inv,
            },
            row=seed,
            persist=bool(scan_row or long_row),
        )
    except Exception:
        case = {
            "n_similar": 0,
            "proven": False,
            "verdict": "unmeasured",
            "memory_line": "Case memory is unavailable on this snapshot.",
            "places_orders": False,
        }

    decision_mem: dict[str, Any] = {}
    try:
        from product.decision_memory import for_symbol
        decision_mem = for_symbol(symbol, row=scan_row or long_row, frame=frame)
    except Exception:
        decision_mem = {"stance": "WAIT", "places_orders": False}

    return {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "symbol": symbol,
        "company": company,
        "sector": sector,
        "state": state,
        "summary": summary,
        "confidence_pct": confidence_pct,
        "gaps": gaps,
        "technical": technical,
        "fundamentals": fundamentals,
        "scanner": scan_row,
        "long_term": long_row,
        "news": news_rows[:10],
        "fno": fno_out,
        "sources": sources,
        "next_actions": next_actions,
        "case": case,
        "decision_memory": decision_mem,
    }
