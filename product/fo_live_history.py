"""Build F&O scan frames from Kite history and validated current Kite OHLCV."""
from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd

from data.nfo_market import IST, quote_provenance, read_market_quotes


def current_frame(history, quote, *, as_of: date, now: datetime):
    state = quote_provenance(quote, as_of=as_of, now=now)
    if not state.get("ok"):
        return None, str(state.get("reason"))
    if history is None or history.empty:
        return None, "HISTORY_UNAVAILABLE"
    if not isinstance(history.index, pd.DatetimeIndex):
        return None, "HISTORY_SESSION_INDEX_UNAVAILABLE"
    ohlc = quote.get("ohlc") or {}
    try:
        bar = {k: float(ohlc[k]) for k in ("open", "high", "low")}
        bar.update(close=float(quote["last_price"]), volume=float(quote["volume"]))
        if (not all(math.isfinite(v) for v in bar.values())
                or min(bar[k] for k in ("open", "high", "low", "close")) <= 0
                or bar["volume"] < 0
                or bar["low"] > min(bar["open"], bar["close"])
                or bar["high"] < max(bar["open"], bar["close"])):
            return None, "CURRENT_OHLCV_INVALID"
    except (KeyError, TypeError, ValueError):
        return None, "CURRENT_OHLCV_UNAVAILABLE"
    frame = history.copy()
    idx = frame.index
    if idx.tz is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    frame.index = idx.normalize()
    # Never carry a future or partial current-session history bar into the
    # prior-window breakout baseline. Replace today's bar, do not append twice.
    frame = frame.loc[frame.index < pd.Timestamp(as_of)].sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    if frame.empty:
        return None, "COMPLETED_HISTORY_UNAVAILABLE"
    frame.loc[pd.Timestamp(as_of), list(bar)] = list(bar.values())
    frame.attrs["quantterm_live_overlay_date"] = as_of.isoformat()
    frame.attrs["quantterm_live_overlay_source"] = "kite_quotes"
    return frame, "CURRENT_SESSION_READY"


def prepare_history(symbols, *, history_getter, client, as_of, now):
    symbols = list(dict.fromkeys(str(s).upper() for s in symbols if str(s)))
    frames, rejected = {}, {}
    try:
        quotes = read_market_quotes([f"NSE:{s}" for s in symbols], client=client)
    except Exception as exc:
        quotes = {}
        rejected = {s: f"CURRENT_QUOTE_READ_FAILED:{type(exc).__name__}" for s in symbols}
    for symbol in symbols:
        if symbol in rejected:
            continue
        try:
            frame, reason = current_frame(history_getter(symbol), quotes.get(f"NSE:{symbol}"),
                                          as_of=as_of, now=now)
        except Exception as exc:
            frame, reason = None, f"HISTORY_ERROR:{type(exc).__name__}"
        if frame is None:
            rejected[symbol] = reason
        else:
            frames[symbol] = frame
    return frames, {
        "ready": bool(frames), "symbols": len(frames), "requested_symbols": len(symbols),
        "source": "kite_quotes", "history_source": "kite_snapshot",
        "session_date": as_of.isoformat(), "rejected": rejected,
    }
