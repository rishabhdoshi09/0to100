"""
Watchlist Watcher — watchlist ab sirf list nahi, chowkidaar hai.

Every scan cycle during market hours, watchlist entries are checked
against live prices:

  price enters buy zone      → 🎯 "entry zone mein aa gaya — plan ready?"
  price breaks below stop    → 🔴 "setup toot gaya — watchlist se hatao"
  price runs past target     → 🚀 "bina tumhare bhaag gaya — chase mat
                                    karo, agla setup aayega"

Alerts push to Telegram once per symbol+event per day.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from core.market_clock import IST
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)

_WL_DB = Path(__file__).resolve().parent.parent / "logs" / "watchlist.db"
_lock = threading.Lock()
_alerted: dict[str, set] = {}


def _watchlist_rows() -> list[dict]:
    try:
        if not _WL_DB.exists():
            return []
        conn = sqlite3.connect(_WL_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT symbol, buy_zone_low, buy_zone_high, target_price, "
            "stop_price FROM watchlist").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("watchlist_read_failed", error=str(exc))
        return []


def check_watchlist() -> list[dict]:
    """[{symbol, event, message}] — every actionable watchlist event now."""
    rows = _watchlist_rows()
    if not rows:
        return []
    try:
        from data.live_quotes import get_live_quotes
        live = get_live_quotes(sorted({r["symbol"] for r in rows}))
    except Exception:
        return []

    events: list[dict] = []
    for r in rows:
        sym = r["symbol"]
        q = live.get(sym)
        if not (q and q.get("price")):
            continue
        px = float(q["price"])
        lo = float(r["buy_zone_low"] or 0)
        hi = float(r["buy_zone_high"] or 0)
        stop = float(r["stop_price"] or 0)
        target = float(r["target_price"] or 0)

        if stop and px < stop:
            events.append({"symbol": sym, "event": "BROKEN",
                           "message": (f"🔴 <b>{sym}</b> ₹{px:,.1f} — stop "
                                       f"₹{stop:,.1f} ke neeche. Setup toot "
                                       f"gaya, watchlist se hatao.")})
        elif target and px > target:
            events.append({"symbol": sym, "event": "RAN_AWAY",
                           "message": (f"🚀 <b>{sym}</b> ₹{px:,.1f} — target "
                                       f"₹{target:,.1f} paar, bina entry ke "
                                       f"bhaag gaya. Chase MAT karo — agla "
                                       f"setup aayega.")})
        elif lo and hi and lo <= px <= hi:
            events.append({"symbol": sym, "event": "IN_ZONE",
                           "message": (f"🎯 <b>{sym}</b> ₹{px:,.1f} — buy zone "
                                       f"(₹{lo:,.0f}–₹{hi:,.0f}) mein hai! "
                                       f"Plan: stop ₹{stop:,.0f}"
                                       + (f" / target ₹{target:,.0f}" if target else "")
                                       + ". App kholke ticket se lo.")})
    return events


def push_watchlist_alerts() -> int:
    """Telegram push — once per symbol+event per day."""
    try:
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if not engine.is_configured():
            return 0
        events = check_watchlist()
        if not events:
            return 0
        today = datetime.now(IST).strftime("%Y-%m-%d")
        with _lock:
            _alerted.setdefault(today, set())
            for k in list(_alerted):
                if k != today:
                    del _alerted[k]
            fresh = [e for e in events
                     if f"{e['symbol']}:{e['event']}" not in _alerted[today]]
            if not fresh:
                return 0
            lines = ["👁 <b>Watchlist Update</b>"]
            lines += [f"\n{e['message']}" for e in fresh[:6]]
            if engine.send("\n".join(lines)):
                _alerted[today].update(f"{e['symbol']}:{e['event']}" for e in fresh)
                log.info("watchlist_alerts_pushed", count=len(fresh))
                return len(fresh)
    except Exception as exc:
        log.debug("watchlist_alerts_skip", error=str(exc))
    return 0
