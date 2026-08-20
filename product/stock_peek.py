"""Fast stock snapshot for the Ideas popup — on-file numbers, timed tape.

The full stock-intelligence GET hydrates filings (can scrape) and overlays
today's NSE bar on the request thread. Opening a snapshot then sits on
"Failed to fetch" while Buy/Stop/Target were already on the card.

This builder:
  - reads scan + long-term + fundamentals cache (no scrape)
  - scores official OHLCV with a hard timeout
  - stamps CMP/change from Kite/NSE with a hard timeout
  - never invents a missing ratio
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Callable, Mapping

from product.stock_workspace import (
    _f,
    _find,
    _fundamentals,
    _sanitize_json,
    _technical,
    clean_symbol,
)

OHLCV_SECONDS = 2.5
QUOTE_SECONDS = 2.0


def _call_with_timeout(fn: Callable[[], Any], seconds: float, default: Any = None) -> Any:
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=max(0.1, float(seconds)))
    except (FuturesTimeout, Exception):
        return default
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _scan_and_long_term() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from product.scan_store import load_scan
        scan = load_scan() or {}
    except Exception:
        scan = {}
    try:
        from product.long_term_store import load_long_term_scan
        lt = load_long_term_scan() or {}
    except Exception:
        lt = {}
    return dict(scan), dict(lt)


def _cached_raw(symbol: str) -> dict[str, Any]:
    """On-file pack only. Never ensure_deep_fundamentals / Screener."""
    try:
        from reporting.evidence_intake import load_raw_fundamentals
        raw = dict(load_raw_fundamentals(symbol, auto_fetch=False) or {})
    except Exception:
        raw = {}
    if raw.get("data"):
        return raw
    try:
        from fundamentals.cache import FundamentalsCache
        cache = FundamentalsCache()
        fresh = cache.get(symbol)
        status = "TODAY"
        if fresh is None:
            fresh = cache.get_any(symbol)
            status = "STALE" if fresh else ""
        if fresh:
            return {
                "available": True,
                "data": dict(fresh),
                "fetched_at": str(fresh.get("_qt_fetched_at") or ""),
                "section_as_of": {},
                "cache_status": status or str(fresh.get("_qt_cache_status") or ""),
            }
    except Exception:
        pass
    return raw


def _load_frame(symbol: str) -> Any:
    from data.bhavcopy_runtime import get_ohlcv
    return get_ohlcv(symbol)


def _load_quote(symbol: str) -> dict[str, Any]:
    from data.live_quotes import get_live_quotes
    quotes = get_live_quotes([symbol], ttl=8.0, allow_google=False) or {}
    raw = quotes.get(symbol) or {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _sector_for(symbol: str, scan_row: Mapping[str, Any], lt_row: Mapping[str, Any]) -> str:
    sector = str(lt_row.get("sector") or scan_row.get("sector") or "").strip()
    if sector and sector.lower() not in {"unclassified", "—", "-"}:
        return sector
    try:
        from scan.sector_heat import sector_of
        found = str(sector_of(symbol) or "").strip()
        if found:
            return found
    except Exception:
        pass
    return sector or "Sector not classified"


def _upside(buy: float | None, target: float | None) -> float | None:
    if buy is None or target is None or buy <= 0:
        return None
    return round((target / buy - 1.0) * 100.0, 1)


def _merge_row(scan_row: Mapping[str, Any], lt_row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(lt_row or {})
    out.update({k: v for k, v in dict(scan_row or {}).items() if v not in (None, "", [], {})})
    funds = dict(lt_row.get("fundamentals") or {}) if isinstance(lt_row.get("fundamentals"), Mapping) else {}
    scan_funds = dict(scan_row.get("fundamentals") or {}) if isinstance(scan_row.get("fundamentals"), Mapping) else {}
    if scan_funds:
        funds.update({k: v for k, v in scan_funds.items() if v not in (None, "")})
    if funds:
        out["fundamentals"] = funds
    return out


def _tech_from_row(row: Mapping[str, Any], quote: Mapping[str, Any]) -> dict[str, Any]:
    close = _f(quote.get("price")) or _f(row.get("price")) or _f(row.get("cmp"))
    change = _f(quote.get("chg_pct"))
    if change is None:
        change = _f(row.get("change_pct"))
    rsi = _f(row.get("rsi"))
    vol = _f(row.get("volume_ratio"))
    metrics = []
    if close is not None:
        metrics.append({"key": "close", "label": "Close", "value": round(close, 2), "unit": "INR"})
    if change is not None:
        metrics.append({"key": "change_pct", "label": "Change", "value": round(change, 2), "unit": "%"})
    if rsi is not None:
        metrics.append({"key": "rsi14", "label": "RSI (14)", "value": round(rsi, 1), "unit": ""})
    if vol is not None:
        metrics.append({"key": "volume_ratio", "label": "Vol vs 20d", "value": round(vol, 2), "unit": "x"})
    return {
        "available": bool(metrics),
        "close": round(close, 2) if close is not None else None,
        "change_pct": round(change, 2) if change is not None else None,
        "rsi14": round(rsi, 1) if rsi is not None else None,
        "volume_ratio": round(vol, 2) if vol is not None else None,
        "metrics": metrics,
        "trend": str(row.get("status") or ""),
        "trend_explanation": "From the last scan row — official history did not arrive in time.",
        "source": "scan_row",
    }


def _promote_metrics(technical: Mapping[str, Any]) -> dict[str, Any]:
    """Expose close/change/RSI on the metric grid even when the quote stamp misses."""
    out = dict(technical)
    metrics = [dict(m) for m in (out.get("metrics") or []) if isinstance(m, Mapping)]
    by_key = {str(m.get("key")): m for m in metrics}
    if "close" not in by_key and "price" in by_key:
        by_key["close"] = {**by_key["price"], "key": "close", "label": "Close"}
    if "from_high_pct" not in by_key and "from_high" in by_key:
        by_key["from_high_pct"] = {
            **by_key["from_high"],
            "key": "from_high_pct",
            "label": "From 52w high",
            "unit": "%",
        }
    fields = (
        ("close", "Close", out.get("close"), "INR"),
        ("change_pct", "Change", out.get("change_pct"), "%"),
        ("rsi14", "RSI (14)", out.get("rsi14"), ""),
        ("ema20", "EMA 20", out.get("ema20"), "INR"),
        ("ema50", "EMA 50", out.get("ema50"), "INR"),
        ("ema200", "EMA 200", out.get("ema200"), "INR"),
        ("atr_pct", "ATR %", out.get("atr_pct"), "%"),
        ("volume_ratio", "Vol vs 20d", out.get("volume_ratio"), "x"),
        ("high_52w", "52w high", out.get("high_52w"), "INR"),
        ("low_52w", "52w low", out.get("low_52w"), "INR"),
        ("from_high_pct", "From 52w high", out.get("from_high_pct"), "%"),
    )
    for key, label, value, unit in fields:
        if value is None:
            continue
        prev = by_key.get(key) or {}
        by_key[key] = {
            **prev,
            "key": key,
            "label": prev.get("label") or label,
            "value": value,
            "unit": prev.get("unit") or unit,
        }
    preferred = [row[0] for row in fields]
    ordered = [by_key[k] for k in preferred if k in by_key]
    ordered.extend(m for k, m in by_key.items() if k not in preferred)
    out["metrics"] = ordered
    return out


def _apply_quote(technical: dict[str, Any], quote: Mapping[str, Any]) -> dict[str, Any]:
    px = _f(quote.get("price"))
    chg = _f(quote.get("chg_pct"))
    out = dict(technical)
    if px is not None:
        out["close"] = round(px, 2)
    if chg is not None:
        out["change_pct"] = round(chg, 2)
    src = str(quote.get("source") or "")
    if src:
        out["quote_source"] = src
        out["price_tag"] = "LIVE" if src in {"kite", "nse"} else (src.upper() or out.get("price_tag") or "EOD")
    elif not out.get("price_tag"):
        out["price_tag"] = "EOD"
    return _promote_metrics(out)


def _ratios_for(symbol: str, raw: Mapping[str, Any], price: float | None) -> list[dict[str, Any]]:
    try:
        from data_platform.ratios import ratios_from_fundamentals
        blob = dict(raw.get("data") or raw or {})
        if price is not None:
            blob = {**blob, "current_price": price, "price": price}
        rows = ratios_from_fundamentals(symbol, blob)
        return [row for row in rows if isinstance(row, Mapping) and row.get("value") is not None]
    except Exception:
        return []


def build_stock_peek(
    symbol: str,
    *,
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    raw_fundamentals: Mapping[str, Any] | None = None,
    frame: Any = None,
    quote: Mapping[str, Any] | None = None,
    load_history: bool = True,
    load_live: bool = True,
) -> dict[str, Any]:
    """Numbers for the snapshot popup. Missing stays missing. No scrape."""
    symbol = clean_symbol(symbol)
    if scan_payload is None or long_term_payload is None:
        scan_file, lt_file = _scan_and_long_term()
        scan_payload = scan_file if scan_payload is None else scan_payload
        long_term_payload = lt_file if long_term_payload is None else long_term_payload
    scan_row = _find(scan_payload, symbol)
    lt_row = _find(long_term_payload, symbol)
    merged = _merge_row(scan_row, lt_row)
    if raw_fundamentals is None:
        raw_fundamentals = _cached_raw(symbol)

    history_note = ""
    need_history = frame is None and load_history
    need_quote = quote is None and load_live
    if need_history and need_quote:
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            f_fut = pool.submit(_load_frame, symbol)
            q_fut = pool.submit(_load_quote, symbol)
            try:
                frame = f_fut.result(timeout=OHLCV_SECONDS)
            except Exception:
                frame = None
            try:
                quote = q_fut.result(timeout=QUOTE_SECONDS)
            except Exception:
                quote = {}
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if frame is None:
            history_note = "Official history did not arrive in time — using last scan numbers."
    elif need_history:
        frame = _call_with_timeout(lambda: _load_frame(symbol), OHLCV_SECONDS, None)
        if frame is None:
            history_note = "Official history did not arrive in time — using last scan numbers."
    elif need_quote:
        quote = _call_with_timeout(lambda: _load_quote(symbol), QUOTE_SECONDS, {}) or {}

    technical = _technical(frame) if frame is not None else _tech_from_row(merged, quote or {})
    if quote is None and load_live:
        quote = _call_with_timeout(lambda: _load_quote(symbol), QUOTE_SECONDS, {}) or {}
    quote = dict(quote or {})
    technical = _apply_quote(technical, quote)

    sector = _sector_for(symbol, scan_row, lt_row)
    company = str(scan_row.get("company") or lt_row.get("company") or symbol)
    fundamentals = _fundamentals(merged, dict(raw_fundamentals or {}), sector)
    filled_fund = [
        m for m in (fundamentals.get("metrics") or [])
        if isinstance(m, Mapping) and _f(m.get("value")) is not None
    ]
    fundamentals = {**fundamentals, "metrics": filled_fund, "available": bool(filled_fund)}

    cmp = _f(technical.get("close")) or _f(merged.get("price")) or _f(merged.get("cmp"))
    entry = _f(merged.get("entry") or merged.get("entry_price"))
    stop = _f(merged.get("stop"))
    target = _f(merged.get("target") or merged.get("target_price"))
    buy = entry if entry and entry > 0 else cmp
    upside = _upside(buy, target)
    ratios = _ratios_for(symbol, dict(raw_fundamentals or {}), cmp)

    sepa: dict[str, Any] = {}
    if frame is not None:
        try:
            from product.sepa_setup import score_sepa
            sepa = score_sepa(frame)
        except Exception:
            sepa = {}

    from product.top_stocks import pack_fundamentals
    packed = pack_fundamentals(merged)
    if packed.get("metrics") and not filled_fund:
        fundamentals = {
            **fundamentals,
            "available": True,
            "metrics": packed["metrics"],
            "classification": packed.get("classification") or fundamentals.get("classification"),
        }
        filled_fund = packed["metrics"]

    src = str(quote.get("source") or technical.get("quote_source") or "")
    payload = {
        "schema_version": 1,
        "symbol": symbol,
        "company": company,
        "sector": sector,
        "cmp": cmp,
        "change_pct": _f(technical.get("change_pct")),
        "price_tag": technical.get("price_tag") or ("LIVE" if src in {"kite", "nse"} else "EOD"),
        "quote_source": src,
        "entry": entry,
        "stop": stop,
        "target": target,
        "upside_from_buy_pct": upside,
        "rsi": _f(technical.get("rsi14")) or _f(merged.get("rsi")),
        "volume_ratio": _f(technical.get("volume_ratio")) or _f(merged.get("volume_ratio")),
        "technical": technical,
        "fundamentals": fundamentals,
        "ratios": ratios,
        "sepa": sepa or None,
        "history_note": history_note,
        "fundamentals_cache": str((raw_fundamentals or {}).get("cache_status") or ""),
        "disclaimer": (
            "Snapshot numbers from the last scan, official history, and on-file packs. "
            "Missing ratios stay missing — this popup does not scrape."
        ),
    }
    return _sanitize_json(payload)
