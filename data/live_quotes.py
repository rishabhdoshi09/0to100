"""
Unified live-quote source — ONE reliable path for "today's price".

Priority (most reliable first):
  1. Kite (Zerodha)   — real-time LTP, up to 500 symbols/call. Used
                         whenever the user is logged in. This is the
                         source to trust during and after market hours.
  2. NSE snapshot     — one bulk call, live pChange for ~750 stocks.
  3. Google Finance   — per-stock scrape, fragile (often blocked). Last
                         resort only.

Returns {symbol: {price, chg_pct, source}}. A symbol missing from all
three simply isn't included — the caller shows an honest EOD tag.
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)


def _kite_quotes(symbols: list[str]) -> dict[str, dict]:
    try:
        from config import settings
        if not settings.kite_access_token:
            return {}
        from data.kite_client import KiteClient
        kite = KiteClient()
        out: dict[str, dict] = {}
        # Kite allows up to 500 instruments per quote() call
        for i in range(0, len(symbols), 500):
            chunk = symbols[i:i + 500]
            raw = kite.batch_quotes(chunk)
            for sym, q in raw.items():
                ltp = float(q.get("ltp") or 0)
                prev = float(q.get("close") or 0)
                if ltp <= 0:
                    continue
                chg = float(q.get("change") or ((ltp - prev) / prev * 100 if prev else 0))
                out[sym] = {"price": ltp, "chg_pct": round(chg, 2), "source": "kite"}
        return out
    except Exception as exc:
        log.debug("kite_quotes_failed", error=str(exc))
        return {}


def _nse_quotes(symbols: list[str]) -> dict[str, dict]:
    try:
        from data.nse_live import live_quotes as _nse_lq
        out = _nse_lq(symbols)
        for v in out.values():
            v["source"] = "nse"
        return out
    except Exception:
        return {}


def _google_quotes(symbols: list[str]) -> dict[str, dict]:
    try:
        from data.google_finance import get_quotes
        out = get_quotes(symbols)
        for v in out.values():
            v["source"] = "google"
        return out
    except Exception:
        return {}


def get_live_quotes(symbols: list[str]) -> dict[str, dict]:
    """
    {symbol: {price, chg_pct, source}} — Kite → NSE → Google, each only
    filling what the previous missed. Logs which source covered how many.
    """
    if not symbols:
        return {}
    quotes: dict[str, dict] = {}

    for name, fn in (("kite", _kite_quotes), ("nse", _nse_quotes),
                     ("google", _google_quotes)):
        missing = [s for s in symbols if s not in quotes]
        if not missing:
            break
        got = fn(missing)
        quotes.update(got)
        if got:
            log.info("live_quotes_source", source=name, filled=len(got),
                     still_missing=len(symbols) - len(quotes))
    return quotes


_KITE_INDEX_KEYS = {
    "NIFTY":     "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "VIX":       "NSE:INDIA VIX",
    "SENSEX":    "BSE:SENSEX",
}


def get_index_quotes(names: list[str]) -> dict[str, dict]:
    """
    {name: {price, chg_pct, source}} for NIFTY/BANKNIFTY/VIX/SENSEX.
    Kite first (real-time indices), Google Finance fallback.
    """
    out: dict[str, dict] = {}
    wanted = [n.upper() for n in names]
    # 1. Kite — one ltp-style quote call for all indices
    try:
        from config import settings
        if settings.kite_access_token:
            from data.kite_client import KiteClient
            kite = KiteClient()
            keys = [_KITE_INDEX_KEYS[n] for n in wanted if n in _KITE_INDEX_KEYS]
            raw = kite.raw.quote(keys)
            for name in wanted:
                key = _KITE_INDEX_KEYS.get(name)
                d = raw.get(key) if key else None
                if not d:
                    continue
                price = float(d.get("last_price") or 0)
                prev = float((d.get("ohlc") or {}).get("close") or 0)
                if price > 0:
                    chg = (price - prev) / prev * 100 if prev else 0.0
                    out[name] = {"price": price, "chg_pct": round(chg, 2),
                                 "source": "kite"}
    except Exception as exc:
        log.debug("kite_index_quotes_failed", error=str(exc))
    # 2. Google Finance for what Kite missed
    missing = [n for n in wanted if n not in out]
    for name in missing:
        try:
            from data.google_finance import get_quote
            q = get_quote(name)
            if q and q.get("price"):
                out[name] = {**q, "source": "google"}
        except Exception:
            pass
    return out


def source_health(sample: str = "RELIANCE") -> tuple[bool, str]:
    """(ok, 'kite ₹1,390') — which source answers for a probe symbol."""
    q = get_live_quotes([sample]).get(sample)
    if q:
        return True, f"{q['source']} ₹{q['price']:,.0f}"
    return False, "kite/NSE/Google sabhi fail"
