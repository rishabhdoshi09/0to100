"""
NSE Live Snapshot — today's intraday bar for the whole universe.

The bhavcopy is EOD (published ~6 PM), so during market hours the
store is one session behind. This module fetches TODAY's live
open/high/low/last/volume for ~750 stocks in 1-2 requests from NSE's
own index API and overlays it as today's (partial) bar on the store —
scans then see today's move, not yesterday's close.

Endpoints (cookie-primed session required):
  https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20TOTAL%20MARKET
  fallback: index=NIFTY%20500
"""
from __future__ import annotations

import time
from datetime import date, datetime
from typing import Optional
from urllib.parse import quote

from logger import get_logger

log = get_logger(__name__)

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}

_INDICES = ("NIFTY TOTAL MARKET", "NIFTY 500")

_last_fetch_ts: float = 0.0
_last_snapshot: dict = {}
_last_kite_ts: float = 0.0
_last_kite_snap: dict = {}
_MIN_GAP_S = 120          # don't hammer NSE/Kite quotes more than once per 2 min


def fetch_live_snapshot() -> dict[str, dict]:
    """
    {symbol: {open, high, low, close, volume}} for today, live from NSE.
    Empty dict on failure (offline / blocked / weekend pre-open).
    """
    global _last_fetch_ts, _last_snapshot
    now = time.time()
    if now - _last_fetch_ts < _MIN_GAP_S and _last_snapshot:
        return dict(_last_snapshot)

    import requests
    s = requests.Session()
    try:
        s.headers.update(_HEADERS)
        # Prime cookies — NSE rejects cookieless API hits
        s.get("https://www.nseindia.com", timeout=10)

        for idx in _INDICES:
            try:
                r = s.get(
                    f"https://www.nseindia.com/api/equity-stockIndices?index={quote(idx)}",
                    timeout=12)
                if r.status_code != 200:
                    continue
                data = r.json().get("data", [])
                snap: dict[str, dict] = {}
                for row in data:
                    sym = str(row.get("symbol", "")).strip().upper()
                    last = row.get("lastPrice")
                    if not sym or last in (None, 0):
                        continue
                    snap[sym] = {
                        "open":   float(row.get("open") or last),
                        "high":   float(row.get("dayHigh") or last),
                        "low":    float(row.get("dayLow") or last),
                        "close":  float(last),
                        "volume": float(row.get("totalTradedVolume") or 0),
                        "prev_close": float(row.get("previousClose") or 0),
                        "pchange": float(row.get("pChange") or 0),
                    }
                if len(snap) >= 100:
                    _last_snapshot = snap
                    _last_fetch_ts = now
                    log.info("nse_live_snapshot", index=idx, symbols=len(snap))
                    return dict(snap)
            except Exception as exc:
                log.debug("nse_live_index_failed", index=idx, error=str(exc))
    except Exception as exc:
        log.debug("nse_live_failed", error=str(exc))
    finally:
        try:
            s.close()
        except Exception:
            pass
    return {}


def _is_trading_now() -> bool:
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        now = datetime.now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= minutes <= 16 * 60   # till ~4 PM (post-close prints)


def _kite_snapshot(symbols: list[str]) -> dict[str, dict]:
    """
    Today's OHLCV bar from Kite quotes — the reliable path (user's Kite
    works even when NSE's cookie-fussy public API refuses requests).
    500 symbols per call.
    """
    global _last_kite_ts, _last_kite_snap
    now = time.time()
    if now - _last_kite_ts < _MIN_GAP_S and _last_kite_snap:
        return dict(_last_kite_snap)
    try:
        from execution.trade_executor import kite_ready
        if not kite_ready():
            return {}
        from data.kite_client import KiteClient
        kite = KiteClient()
        snap: dict[str, dict] = {}
        for i in range(0, len(symbols), 500):
            chunk = symbols[i:i + 500]
            raw = kite.batch_quotes(chunk)
            for sym, q in raw.items():
                ltp = float(q.get("ltp") or 0)
                if ltp <= 0:
                    continue
                snap[sym] = {
                    "open":   float(q.get("open") or ltp),
                    "high":   float(q.get("high") or ltp),
                    "low":    float(q.get("low") or ltp),
                    "close":  ltp,
                    "volume": float(q.get("volume") or 0),
                }
        if snap:
            _last_kite_snap = snap
            _last_kite_ts = now
            log.info("kite_intraday_snapshot", symbols=len(snap))
        return snap
    except Exception as exc:
        log.debug("kite_snapshot_failed", error=str(exc))
        return {}


def apply_live_to_store() -> int:
    """
    Overlay today's live bar onto the bhavcopy store during market hours.
    Source order: Kite (reliable) → NSE public API (fallback).
    Returns number of symbols updated. No-op when the store already has
    today's session (evening bhavcopy landed) or market is closed.
    """
    if not _is_trading_now():
        return 0
    try:
        import pandas as pd
        from data import bhavcopy_store as bs
        with bs._lock:
            last_day = bs._store_last_day
        if last_day == date.today():
            return 0            # today's official bhavcopy already in store

        # Kite first — proven working; NSE public API only as fallback
        try:
            from data.nse_universe import get_nse_universe
            _uni = get_nse_universe()
        except Exception:
            _uni = []
        snap = _kite_snapshot(_uni) if _uni else {}
        if not snap:
            snap = fetch_live_snapshot()
        if not snap:
            log.info("live_bar_unavailable",
                     hint="Kite login + NSE API dono se intraday nahi mila")
            return 0

        today_ts = pd.Timestamp(date.today())
        updated = 0
        with bs._lock:
            for sym, bar in snap.items():
                df = bs._store.get(sym)
                if df is None:
                    continue
                if len(df) and df.index[-1] == today_ts:
                    df.loc[today_ts, ["open", "high", "low", "close", "volume"]] = [
                        bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]]
                else:
                    new_row = {c: bar.get(c) for c in ("open", "high", "low", "close", "volume")
                               if c in df.columns}
                    bs._store[sym] = pd.concat(
                        [df, pd.DataFrame([new_row], index=[today_ts])])
                updated += 1
        if updated:
            log.info("live_bar_overlaid", symbols=updated)
        return updated
    except Exception as exc:
        log.warning("live_overlay_store_failed", error=str(exc))
        return 0


def live_session_ready(*, apply: bool = True) -> dict:
    """True when official history plus today's Kite quotes can drive a scan.

    The EOD bhavcopy is one session behind during market hours. Kite already
    returns today's OHLCV — that is the latest session, and a cash scan must
    not wait for options history or a full Kite historical rewrite.
    """
    out = {
        "ready": False,
        "symbols": 0,
        "source": "",
        "session_date": "",
        "sessions": 0,
    }
    try:
        try:
            from zoneinfo import ZoneInfo
            today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        except Exception:
            today = date.today()
        from data import bhavcopy_store as bs
        if not bs.is_ready():
            try:
                bs.build_store(days=260)
            except Exception:
                pass
        if not bs.is_ready():
            return out
        with bs._lock:
            sessions = int(bs._store_sessions or 0)
            last = bs._store_last_day
            stored = len(bs._store)
        out["sessions"] = sessions
        out["session_date"] = last.isoformat() if last else ""
        if _is_trading_now():
            overlaid = apply_live_to_store() if apply else 0
            if overlaid <= 0:
                with bs._lock:
                    last = bs._store_last_day
                    stored = len(bs._store)
                if last == today:
                    overlaid = stored
                elif _last_kite_snap:
                    overlaid = len(_last_kite_snap)
            if overlaid <= 0:
                return out
            out.update(
                ready=True,
                symbols=int(overlaid),
                source="kite_quotes",
                session_date=today.isoformat(),
            )
            return out
        out.update(
            ready=sessions >= 60,
            symbols=stored,
            source="official_bhavcopy",
        )
        return out
    except Exception as exc:
        log.debug("live_session_ready_failed", error=str(exc))
        return out


def live_quotes(symbols: list[str]) -> dict[str, dict]:
    """
    {symbol: {price, chg_pct}} from today's NSE snapshot — one bulk call
    covers the whole list. Empty for symbols not in the snapshot (caller
    falls back to Google Finance / EOD). Uses NSE's own pChange field.
    """
    snap = fetch_live_snapshot()
    out: dict[str, dict] = {}
    for sym in symbols:
        bar = snap.get(sym.upper())
        if bar and bar.get("close"):
            out[sym] = {"price": bar["close"], "chg_pct": round(bar.get("pchange", 0), 2)}
    return out
