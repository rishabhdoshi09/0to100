"""Bounded technical refresh helpers for worker-side desk projections.

Important boundary:
- GET/page-open projections never call this module unless an explicit worker path
  asks for refresh_technicals=True.
- No per-symbol scraper loop. Quotes are fetched in one bulk call.
- RSI/volume come from QuantTerm official OHLCV cache.
- apply_current_trade_levels(row, None) is pure and network-free.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _f(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rsi(close: np.ndarray, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    delta = np.diff(close[-(period + 1):].astype(float))
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss <= 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _cached_technicals(symbol: str) -> dict[str, Any]:
    try:
        from scan.bulk_fetcher import get_cached
        df = get_cached(str(symbol or "").upper())
    except Exception:
        df = None
    if df is None or len(df) < 20:
        return {}
    try:
        close = df["close"].to_numpy(dtype=float)
    except Exception:
        return {}
    out: dict[str, Any] = {
        "price": round(float(close[-1]), 2),
        "cmp": round(float(close[-1]), 2),
        "tech_source": "official_ohlcv_cache",
    }
    rsi = _rsi(close)
    if rsi is not None:
        out["rsi"] = round(float(rsi), 1)
    try:
        volume = df["volume"].to_numpy(dtype=float)
        if len(volume) >= 21:
            baseline = float(np.nanmean(volume[-21:-1]))
            if baseline > 0:
                out["volume_ratio"] = round(float(volume[-1] / baseline), 3)
    except Exception:
        pass
    try:
        idx = df.index[-1]
        day = idx.date() if hasattr(idx, "date") else idx
        out["eod_as_of"] = str(day)[:10]
    except Exception:
        pass
    return out


def apply_current_trade_levels(
    row: dict[str, Any],
    quote: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Overlay a supplied quote and recompute display percentages.

    Passing quote=None is intentionally pure/network-free. Entry/stop/target
    are never invented or shifted here.
    """
    q = dict(quote or {})
    px = _f(q.get("price"))
    if px > 0:
        row["price"] = round(px, 2)
        row["cmp"] = round(px, 2)
        source = str(q.get("source") or "").strip().lower()
        row["quote_source"] = source
        row["price_tag"] = f"{source.upper()} quote" if source else "Quote"
    else:
        px = _f(row.get("cmp") or row.get("price"))
        if px > 0:
            row.setdefault("cmp", round(px, 2))
            row.setdefault("price", round(px, 2))
            if not row.get("price_tag"):
                eod = str(row.get("eod_as_of") or row.get("freshness") or "").strip()
                row["price_tag"] = f"EOD {eod}".strip() if eod else "Saved scan price"

    entry = _f(row.get("entry") or row.get("entry_price"))
    target = _f(row.get("target") or row.get("target_price"))
    stop = _f(row.get("stop") or row.get("stop_price"))
    cmp_px = _f(row.get("cmp") or row.get("price"))
    if cmp_px > 0 and entry > 0:
        row["upside_from_entry_pct"] = round((cmp_px / entry - 1.0) * 100.0, 2)
    if cmp_px > 0 and target > 0:
        row["upside_to_target_pct"] = round((target / cmp_px - 1.0) * 100.0, 2)
    if cmp_px > 0 and stop > 0:
        row["downside_to_stop_pct"] = round((stop / cmp_px - 1.0) * 100.0, 2)
    return row


def refresh_rows_technicals(
    rows: Sequence[Mapping[str, Any]],
    *,
    bulk_overlay: bool = False,
) -> list[dict[str, Any]]:
    """Refresh a bounded row set from existing caches plus one bulk quote call.

    bulk_overlay is retained as the worker-call contract. Both modes remain one
    bulk quote request; there is never a per-symbol fallback scrape here.
    """
    out = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not out:
        return []
    symbols = [
        str(row.get("symbol") or "").strip().upper()
        for row in out
        if str(row.get("symbol") or "").strip()
    ]
    try:
        from data.live_quotes import get_live_quotes
        quotes = dict(get_live_quotes(symbols) or {})
    except Exception:
        quotes = {}

    for row in out:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        cached = _cached_technicals(symbol)
        for key, value in cached.items():
            if value not in (None, ""):
                row[key] = value
        quote = quotes.get(symbol)
        apply_current_trade_levels(row, quote)
        if quote:
            base = str(row.get("tech_source") or "official_ohlcv_cache")
            src = str(quote.get("source") or "quote").lower()
            row["tech_source"] = f"{base}+{src}"
    return out
