"""
⚡ Live Ticker — Kite WebSocket tick stream for the Live Watch board.

Second WebSocket connection alongside the sniper's (Kite allows 3 per
app). MODE_QUOTE ticks (LTP, change, volume, OHLC) land in an
in-memory store; the UI reads snapshots with an honest tick-age so
stale can never masquerade as live.

Design mirrors the sniper's battle-tested shell: daemon thread,
on_close marks dead so the next watch() call reconnects with a fresh
token (daily token expiry survives without an app restart).
"""
from __future__ import annotations

import threading
import time

from logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_store: dict[str, dict] = {}          # symbol -> {price, chg_pct, volume, high, low, ts}
_tok2sym: dict[int, str] = {}
_desired: set[int] = set()
_ticker = None
_started = False


def update_store(ticks: list[dict], tok_map: dict[int, str],
                 now: float | None = None) -> int:
    """Pure tick→store fold (unit-tested). Returns symbols updated."""
    now = now or time.time()
    n = 0
    for t in ticks:
        sym = tok_map.get(t.get("instrument_token"))
        ltp = t.get("last_price")
        if not sym or not ltp:
            continue
        ohlc = t.get("ohlc") or {}
        prev = float(ohlc.get("close") or 0)
        with _lock:
            _store[sym] = {
                "price": float(ltp),
                "chg_pct": round((float(ltp) - prev) / prev * 100, 2) if prev else 0.0,
                "volume": int(t.get("volume_traded") or t.get("volume") or 0),
                "high": float(ohlc.get("high") or 0),
                "low": float(ohlc.get("low") or 0),
                "ts": now,
            }
        n += 1
    return n


def _start() -> bool:
    global _ticker, _started
    with _lock:
        if _started:
            return True
    try:
        from execution.trade_executor import kite_ready
        if not kite_ready():
            return False
        from data.kite_client import KiteClient

        def on_ticks(ws, ticks):
            with _lock:
                tok_map = dict(_tok2sym)
            update_store(ticks, tok_map)
            try:
                from core.health import beat as _hb
                _hb("live_ticker", note=f"{len(ticks)} ticks")
            except Exception:
                pass

        def on_connect(ws, response):
            with _lock:
                tokens = list(_desired)
            if tokens:
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_QUOTE, tokens)
            log.info("live_ticker_connected", watching=len(tokens))

        def on_close(ws, code, reason):
            global _started, _ticker
            with _lock:
                _started = False
                _ticker = None
            log.info("live_ticker_closed_will_restart", code=code,
                     reason=str(reason)[:80])

        kws = KiteClient().get_ticker(on_ticks, on_connect, on_close)
        kws.connect(threaded=True)
        with _lock:
            _ticker = kws
            _started = True
        return True
    except Exception as exc:
        log.debug("live_ticker_start_failed", error=str(exc))
        return False


def watch(symbols: list[str]) -> bool:
    """Idempotent: map symbols→tokens, start/extend the stream.
    False = stream unavailable (not logged in) — caller falls back to
    REST polling and SAYS so."""
    if not symbols:
        return _started
    try:
        from data.instruments import InstrumentManager
        sym2tok = InstrumentManager().tokens_for(symbols)
    except Exception as exc:
        log.debug("live_ticker_tokens_failed", error=str(exc))
        return False
    if not sym2tok:
        return False
    new: list[int] = []
    with _lock:
        for sym, tok in sym2tok.items():
            _tok2sym[tok] = sym.upper()
            if tok not in _desired:
                _desired.add(tok)
                new.append(tok)
    if not _start():
        return False
    if new:
        with _lock:
            kws = _ticker
        try:
            if kws and kws.is_connected():
                kws.subscribe(new)
                kws.set_mode(kws.MODE_QUOTE, new)
        except Exception as exc:
            log.debug("live_ticker_subscribe_failed", error=str(exc))
    return True


def get_ticks(symbols: list[str] | None = None) -> dict[str, dict]:
    """Snapshot with age_s per symbol — the UI shows the age, always."""
    now = time.time()
    with _lock:
        items = dict(_store)
    out = {}
    for sym, d in items.items():
        if symbols and sym not in symbols:
            continue
        out[sym] = {**d, "age_s": round(now - d["ts"], 1)}
    return out


def status() -> dict:
    with _lock:
        started, n_watch = _started, len(_desired)
        last = max((d["ts"] for d in _store.values()), default=0.0)
    return {"streaming": started, "watching": n_watch,
            "last_tick_age_s": round(time.time() - last, 1) if last else None}
