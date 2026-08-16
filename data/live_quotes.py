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

import threading
import time

from logger import get_logger

log = get_logger(__name__)

# ── Micro-cache: same quotes, one network trip ────────────────────────────────
# Within one 15-min cycle the position manager, watchlist watcher, autopilot
# (×3 internal passes) and the UI overlay all ask for overlapping symbols —
# previously ~6 full Kite/NSE round-trips. 8 seconds is far fresher than any
# consumer needs (the sniper has its own tick stream and does NOT use this
# path), so correctness is untouched while redundant calls collapse to one.
_QUOTE_TTL_S = 8.0
_qcache: dict[str, tuple[float, dict]] = {}      # symbol -> (fetched_ts, quote)
_qcache_lock = threading.Lock()


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


def get_live_quotes(symbols: list[str], ttl: float = _QUOTE_TTL_S) -> dict[str, dict]:
    """
    {symbol: {price, chg_pct, source}} — Kite → NSE → Google, each only
    filling what the previous missed. Logs which source covered how many.
    Offline short-circuit: when Kite AND NSE both return nothing (DNS
    down / no internet), skip the per-symbol Google pass entirely —
    60 doomed requests add minutes of latency and pure log spam.

    A symbol fetched within `ttl` seconds is served from the micro-cache
    (pass ttl=0 to force a fresh fetch). Only found quotes are cached —
    a missing symbol stays missing and is retried on the next call.
    """
    if not symbols:
        return {}
    quotes: dict[str, dict] = {}

    # 1. Micro-cache pass
    if ttl > 0:
        now = time.time()
        with _qcache_lock:
            for s in symbols:
                hit = _qcache.get(s)
                if hit and now - hit[0] < ttl:
                    quotes[s] = dict(hit[1])
        if quotes:
            try:
                from core.health import count
                count("quote_cache", "hits", len(quotes))
            except Exception:
                pass
        if len(quotes) == len(symbols):
            return quotes

    # 2. Network chain for the rest
    t0 = time.perf_counter()
    fetched: dict[str, dict] = {}
    for name, fn in (("kite", _kite_quotes), ("nse", _nse_quotes),
                     ("google", _google_quotes)):
        missing = [s for s in symbols if s not in quotes and s not in fetched]
        if not missing:
            break
        if name == "google" and not quotes and not fetched and len(missing) > 5:
            log.info("live_quotes_offline_skip", missing=len(missing))
            break
        if name == "google":
            # per-symbol scrape pass — registered-dead symbols skip karo
            try:
                from data.dead_symbols import is_dead
                missing = [s for s in missing if not is_dead(s)]
            except Exception:
                pass
            if not missing:
                break
        got = fn(missing)
        fetched.update(got)
        if got:
            log.info("live_quotes_source", source=name, filled=len(got),
                     still_missing=len(symbols) - len(quotes) - len(fetched))

    if fetched:
        now = time.time()
        with _qcache_lock:
            for s, q in fetched.items():
                _qcache[s] = (now, dict(q))
            # bounded: drop expired entries once the cache grows big
            if len(_qcache) > 4000:
                cutoff = now - max(_QUOTE_TTL_S, ttl)
                for k in [k for k, (ts, _) in _qcache.items() if ts < cutoff]:
                    del _qcache[k]
        quotes.update(fetched)
    try:
        from core.health import count, record_latency, beat
        count("quote_cache", "misses", len(fetched))
        record_latency("quote_fetch", time.perf_counter() - t0)
        if fetched:
            src = next(iter(fetched.values())).get("source", "?")
            beat("quotes", note=f"{src} · {len(fetched)} syms")
    except Exception:
        pass
    return quotes


_KITE_INDEX_KEYS = {
    "NIFTY":      "NSE:NIFTY 50",
    "BANKNIFTY":  "NSE:NIFTY BANK",
    "BANK":       "NSE:NIFTY BANK",
    "FINNIFTY":   "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
    "VIX":        "NSE:INDIA VIX",
    "SENSEX":     "BSE:SENSEX",
    "IT":         "NSE:NIFTY IT",
    "PHARMA":     "NSE:NIFTY PHARMA",
    "AUTO":       "NSE:NIFTY AUTO",
    "FMCG":       "NSE:NIFTY FMCG",
    "METAL":      "NSE:NIFTY METAL",
    "ENERGY":     "NSE:NIFTY ENERGY",
    "REALTY":     "NSE:NIFTY REALTY",
}

_INDEX_STORE_NAMES = {
    "NIFTY": "Nifty 50",
    "VIX": "India VIX",
    "BANK": "Nifty Bank",
    "BANKNIFTY": "Nifty Bank",
    "IT": "Nifty IT",
    "PHARMA": "Nifty Pharma",
    "AUTO": "Nifty Auto",
    "FMCG": "Nifty FMCG",
    "METAL": "Nifty Metal",
    "ENERGY": "Nifty Energy",
    "REALTY": "Nifty Realty",
}


def get_index_quotes(names: list[str]) -> dict[str, dict]:
    """
    {name: {price, chg_pct, source}} for NIFTY / VIX / sectors.
    Kite first, official NSE index store next, Google last.
    """
    out: dict[str, dict] = {}
    wanted = [n.upper() for n in names]
    # 1. Kite — one quote call for all requested indices
    try:
        from data.kite_client import KiteClient, _fresh_env
        if (_fresh_env("KITE_ACCESS_TOKEN") or "").strip():
            kite = KiteClient()
            keys = [_KITE_INDEX_KEYS[n] for n in wanted if n in _KITE_INDEX_KEYS]
            raw = kite.raw.quote(keys) if keys else {}
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
    # 2. Official NSE index store for what Kite missed
    missing = [n for n in wanted if n not in out]
    if missing:
        try:
            from data.index_store import TICKER_MAP, get_index_ohlcv
            for name in missing:
                official = _INDEX_STORE_NAMES.get(name)
                if not official:
                    continue
                ticker = next((t for t, n in TICKER_MAP.items() if n == official), "")
                frame = get_index_ohlcv(ticker) if ticker else None
                if frame is None or getattr(frame, "empty", True) or len(frame) < 2:
                    continue
                close = frame["Close"] if "Close" in frame.columns else frame["close"]
                last = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                if last > 0 and prev:
                    out[name] = {
                        "price": last,
                        "chg_pct": round((last / prev - 1.0) * 100.0, 2),
                        "source": "official_nse",
                    }
        except Exception as exc:
            log.debug("index_store_quotes_failed", error=str(exc))
    # 3. Google Finance last resort
    still = [n for n in wanted if n not in out]
    for name in still:
        try:
            from data.google_finance import get_quote
            q = get_quote(name)
            if q and q.get("price"):
                out[name] = {**q, "source": "google"}
        except Exception:
            pass
    return out


_kite_health_cache: tuple[float, dict] | None = None
_KITE_HEALTH_TTL_S = 30.0


def kite_quote_health() -> dict:
    """Live probe: token on file is not enough — Kite must answer a quote."""
    global _kite_health_cache
    now = time.time()
    if _kite_health_cache and now - _kite_health_cache[0] < _KITE_HEALTH_TTL_S:
        return dict(_kite_health_cache[1])
    try:
        from data.kite_client import _fresh_env
        if not (_fresh_env("KITE_ACCESS_TOKEN") or "").strip():
            payload = {"ok": False, "status": "missing", "note": "No Kite access token — python main.py login"}
        else:
            quotes = get_index_quotes(["NIFTY"])
            nifty = quotes.get("NIFTY") or {}
            if nifty.get("source") == "kite" and nifty.get("price"):
                payload = {
                    "ok": True,
                    "status": "live",
                    "note": "Kite last print",
                    "nifty": float(nifty["price"]),
                    "chg_pct": nifty.get("chg_pct"),
                }
            else:
                payload = {
                    "ok": False,
                    "status": "stale",
                    "note": "Kite token rejected — run python main.py login",
                }
    except Exception as exc:
        payload = {"ok": False, "status": "error", "note": str(exc)[:160]}
    _kite_health_cache = (now, payload)
    return dict(payload)


def source_health(sample: str = "RELIANCE") -> tuple[bool, str]:
    """(ok, 'kite ₹1,390') — which source answers for a probe symbol."""
    q = get_live_quotes([sample]).get(sample)
    if q:
        return True, f"{q['source']} ₹{q['price']:,.0f}"
    return False, "kite/NSE/Google sabhi fail"
