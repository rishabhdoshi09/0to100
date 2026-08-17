"""Live order book for the buy-thesis sheet.

Priority (never invents a 5-level book):
  1. Kite depth — real exchange-side levels when the user is logged in
  2. NSE quote-equity — official book; often HTTP 403 from datacenter IPs
  3. Groww public tape — last print + aggregate buy/sell qty (not 5-level)

Google Finance is last-price only and is not used here. Missing stays missing.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Mapping

import requests

from logger import get_logger

log = get_logger(__name__)

_GROWW_TTL_S = 30.0
_groww_cache: dict[str, tuple[float, dict[str, Any]]] = {}

_GROWW_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://groww.in/",
    "Origin": "https://groww.in",
}


def empty_book(
    *,
    note: str,
    source: str = "",
    last_price: float | None = None,
    as_of: str = "",
) -> dict[str, Any]:
    return {
        "available": False,
        "status": "unavailable",
        "note": note,
        "source": source,
        "bids": [],
        "asks": [],
        "bid_qty": 0,
        "ask_qty": 0,
        "last_price": last_price,
        "as_of": as_of,
    }


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        n = float(value)
        return n if n == n else None
    except (TypeError, ValueError):
        return None


def _level_rows(raw: Any) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for level in list(raw or [])[:5]:
        if not isinstance(level, Mapping):
            continue
        price = _f(level.get("price"))
        qty = _f(level.get("quantity") if level.get("quantity") is not None else level.get("qty"))
        if price is None or qty is None:
            continue
        if price <= 0 and qty <= 0:
            continue
        out.append({"price": price, "quantity": qty})
    return out


def _session_note() -> str:
    try:
        from core.market_session import in_market_open
        if not in_market_open():
            return " NSE is closed — leftover tape, not a live 5-level book."
    except Exception:
        pass
    return ""


def book_from_levels(
    bids: list[dict[str, float]],
    asks: list[dict[str, float]],
    *,
    source: str,
    last_price: float | None = None,
    as_of: str = "",
    extra_note: str = "",
) -> dict[str, Any]:
    bid_qty = sum(float(lv.get("quantity") or 0) for lv in bids)
    ask_qty = sum(float(lv.get("quantity") or 0) for lv in asks)
    if bid_qty <= 0 and ask_qty <= 0:
        note = extra_note or "No live depth."
        return empty_book(note=note + _session_note(), source=source, last_price=last_price, as_of=as_of)
    imbalance = (bid_qty - ask_qty) / max(bid_qty + ask_qty, 1.0)
    if imbalance >= 0.15:
        status, note = "bid_heavy", f"Top-5 bid qty {bid_qty:,.0f} > ask {ask_qty:,.0f}"
    elif imbalance <= -0.15:
        status, note = "ask_heavy", f"Top-5 ask qty {ask_qty:,.0f} > bid {bid_qty:,.0f}"
    else:
        status, note = "balanced", f"Top-5 bid {bid_qty:,.0f} ≈ ask {ask_qty:,.0f}"
    if extra_note:
        note = f"{note}. {extra_note}"
    note += _session_note()
    return {
        "available": True,
        "status": status,
        "note": note,
        "source": source,
        "bids": bids,
        "asks": asks,
        "bid_qty": round(bid_qty),
        "ask_qty": round(ask_qty),
        "imbalance": round(imbalance, 3),
        "last_price": last_price,
        "as_of": as_of,
    }


def fetch_kite_depth(symbol: str) -> dict[str, Any]:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return empty_book(note="Empty symbol.", source="kite")
    try:
        from data.kite_client import KiteClient, _fresh_env
        if not _fresh_env("KITE_ACCESS_TOKEN"):
            return empty_book(note="Kite not logged in — no exchange-side depth.", source="kite")
        from config import settings
        kite = KiteClient()
        key = f"{settings.exchange}:{sym}"
        raw = kite.raw.quote([key]) or {}
        quote = raw.get(key) or next(iter(raw.values()), {}) or {}
        depth = quote.get("depth") or {}
        bids = _level_rows(depth.get("buy"))
        asks = _level_rows(depth.get("sell"))
        last = _f(quote.get("last_price") or quote.get("last_traded_price"))
        return book_from_levels(bids, asks, source="kite", last_price=last)
    except Exception as exc:
        log.debug("kite_depth_failed", symbol=sym, error=str(exc))
        return empty_book(note=f"Kite depth failed ({type(exc).__name__})", source="kite")


def fetch_nse_depth(symbol: str) -> dict[str, Any]:
    try:
        from data.nse_live import fetch_market_depth
        payload = dict(fetch_market_depth(symbol) or {})
        payload.setdefault("bids", [])
        payload.setdefault("asks", [])
        payload.setdefault("source", "nse")
        return payload
    except Exception as exc:
        return empty_book(note=f"NSE quote failed ({type(exc).__name__})", source="nse")


def _groww_as_of(ts: Any) -> str:
    n = _f(ts)
    if n is None or n <= 0:
        return ""
    if n > 10_000_000_000:
        n = n / 1000.0
    try:
        return datetime.fromtimestamp(n, timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return ""


def fetch_groww_tape(symbol: str) -> dict[str, Any]:
    """Last-resort public tape. Aggregate buy/sell qty — never a fabricated 5-level book."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return empty_book(note="Empty symbol.", source="groww")
    now = time.time()
    cached = _groww_cache.get(sym)
    if cached and now - cached[0] < _GROWW_TTL_S:
        return dict(cached[1])
    url = (
        "https://groww.in/v1/api/stocks_data/v1/tr_live_prices/"
        f"exchange/NSE/segment/CASH/{sym}/latest"
    )
    try:
        resp = requests.get(url, headers=_GROWW_HEADERS, timeout=10)
        if resp.status_code != 200:
            payload = empty_book(
                note=f"Groww public tape HTTP {resp.status_code}",
                source="groww",
            )
            _groww_cache[sym] = (now, payload)
            return payload
        data = resp.json() if resp.content else {}
        if not isinstance(data, Mapping):
            return empty_book(note="Groww public tape was not an object.", source="groww")
        last = _f(data.get("ltp") or data.get("close"))
        bid_qty = _f(data.get("totalBuyQty") if data.get("totalBuyQty") is not None else data.get("cumulativeBuyQty")) or 0.0
        ask_qty = _f(data.get("totalSellQty") if data.get("totalSellQty") is not None else data.get("cumulativeSellQty")) or 0.0
        as_of = _groww_as_of(data.get("lastTradeTime") or data.get("tsInMillis"))
        bids = _level_rows(data.get("buy") or data.get("bids") or data.get("marketDepthBuy"))
        asks = _level_rows(data.get("sell") or data.get("asks") or data.get("marketDepthSell"))
        if bids or asks:
            out = book_from_levels(
                bids, asks, source="groww", last_price=last, as_of=as_of,
                extra_note="Groww public tape",
            )
            _groww_cache[sym] = (now, out)
            return out
        if bid_qty > 0 or ask_qty > 0:
            imbalance = (bid_qty - ask_qty) / max(bid_qty + ask_qty, 1.0)
            if imbalance >= 0.15:
                status = "bid_heavy"
            elif imbalance <= -0.15:
                status = "ask_heavy"
            else:
                status = "balanced"
            note = (
                f"Groww public tape (not NSE 5-level): buy qty {bid_qty:,.0f} vs sell qty {ask_qty:,.0f}."
            )
            note += _session_note()
            out = {
                "available": True,
                "status": status,
                "note": note,
                "source": "groww",
                "bids": [],
                "asks": [],
                "bid_qty": round(bid_qty),
                "ask_qty": round(ask_qty),
                "imbalance": round(imbalance, 3),
                "last_price": last,
                "as_of": as_of,
            }
            _groww_cache[sym] = (now, out)
            return out
        note = "Groww public tape has no bid/ask rows."
        if last is not None:
            note += f" Last print ₹{last}."
        if as_of:
            note += f" As of {as_of}."
        note += _session_note()
        payload = empty_book(note=note.strip(), source="groww", last_price=last, as_of=as_of)
        _groww_cache[sym] = (now, payload)
        return payload
    except Exception as exc:
        log.debug("groww_tape_failed", symbol=sym, error=str(exc))
        return empty_book(note=f"Groww public tape failed ({type(exc).__name__})", source="groww")


def fetch_order_book(symbol: str) -> dict[str, Any]:
    kite = fetch_kite_depth(symbol)
    if kite.get("available"):
        return kite
    nse = fetch_nse_depth(symbol)
    if nse.get("available"):
        return nse
    groww = fetch_groww_tape(symbol)
    if groww.get("available") or groww.get("last_price"):
        return groww
    if nse.get("note"):
        return nse
    return kite if kite.get("note") else groww
