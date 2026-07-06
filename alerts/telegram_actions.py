"""
Telegram two-way — phone se hi action, laptop ki zaroorat nahi.

Setup alerts carry inline buttons:
  📝 Paper trade  → records the trade (ALWAYS paper from Telegram —
                    live orders need the app's full ticket, deliberately)
  👁 Watchlist    → adds to the watchlist with the setup's levels

A daemon thread long-polls getUpdates and handles callback taps.
Security: only the configured TELEGRAM_CHAT_ID may trigger actions —
anyone else tapping gets ignored.

Callback data format (Telegram cap: 64 bytes):
  pt|SYM|entry|stop|target      → paper trade
  wl|SYM|entry|stop|target      → watchlist add
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime

from logger import get_logger

log = get_logger(__name__)

_started = False
_lock = threading.Lock()


def build_setup_keyboard(setups: list[dict]) -> dict:
    """Inline keyboard: one row per setup with paper-trade + watchlist."""
    rows = []
    for p in setups[:5]:
        sym = p["symbol"]
        e, s, t = p.get("entry", 0), p.get("stop", 0), p.get("target", 0)
        rows.append([
            {"text": f"📝 {sym} paper", "callback_data": f"pt|{sym}|{e:.0f}|{s:.0f}|{t:.0f}"},
            {"text": f"👁 watch", "callback_data": f"wl|{sym}|{e:.0f}|{s:.0f}|{t:.0f}"},
        ])
    return {"inline_keyboard": rows}


def _do_paper_trade(sym: str, entry: float, stop: float, target: float) -> str:
    from execution.trade_executor import place_trade
    from risk.position_sizer import size_position
    ps = size_position(entry, stop)
    qty = max(1, int(ps.get("qty") or 1))
    res = place_trade(symbol=sym, qty=qty, entry_type="LIMIT",
                      entry_price=entry, stop=stop, target=target,
                      paper=True)
    if res["ok"]:
        return (f"📝 Paper trade: {qty} × {sym} @ ₹{entry:,.0f} "
                f"(stop ₹{stop:,.0f} / target ₹{target:,.0f})")
    return f"❌ {res['message']}"


def _do_watchlist(sym: str, entry: float, stop: float, target: float) -> str:
    import sqlite3
    from pathlib import Path
    db = Path(__file__).resolve().parent.parent / "logs" / "watchlist.db"
    try:
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db)
        conn.execute("""CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
            added_date TEXT NOT NULL, buy_zone_low REAL, buy_zone_high REAL,
            target_price REAL, stop_price REAL, notes TEXT, added_price REAL)""")
        dup = conn.execute(
            "SELECT id FROM watchlist WHERE symbol=?", (sym,)).fetchone()
        if dup:
            conn.close()
            return f"👁 {sym} pehle se watchlist mein hai"
        conn.execute(
            "INSERT INTO watchlist (symbol, added_date, buy_zone_low, "
            "buy_zone_high, target_price, stop_price, notes, added_price) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (sym, datetime.now().strftime("%Y-%m-%d"), entry * 0.99, entry,
             target, stop, "Telegram se add kiya", entry))
        conn.commit()
        conn.close()
        return f"👁 {sym} watchlist mein add — entry zone ₹{entry:,.0f}"
    except Exception as exc:
        return f"❌ Watchlist add fail: {str(exc)[:60]}"


def _handle_callback(cb: dict, token: str, allowed_chat: str) -> None:
    import requests
    cb_id = cb.get("id", "")
    chat_id = str(((cb.get("message") or {}).get("chat") or {}).get("id", ""))
    data = cb.get("data", "")

    def answer(text: str) -> None:
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/answerCallbackQuery",
                json={"callback_query_id": cb_id, "text": text[:190]}, timeout=8)
        except Exception:
            pass

    if chat_id != str(allowed_chat):
        answer("Not authorised.")
        log.warning("telegram_unauthorised_tap", chat=chat_id)
        return

    try:
        action, sym, e, s, t = data.split("|")
        entry, stop, target = float(e), float(s), float(t)
    except Exception:
        answer("Button data samajh nahi aaya.")
        return

    if action == "pt":
        msg = _do_paper_trade(sym, entry, stop, target)
    elif action == "wl":
        msg = _do_watchlist(sym, entry, stop, target)
    else:
        msg = "Unknown action."
    answer(msg)
    # Also send as a proper message so it stays in the chat history
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": allowed_chat, "text": msg}, timeout=8)
    except Exception:
        pass
    log.info("telegram_action_done", action=action, symbol=sym)


def _listener() -> None:
    import requests
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    offset = 0
    while True:
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"offset": offset, "timeout": 25,
                        "allowed_updates": json.dumps(["callback_query"])},
                timeout=35)
            for upd in (r.json().get("result") or []):
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                if "callback_query" in upd:
                    _handle_callback(upd["callback_query"], token, chat)
        except Exception as exc:
            log.debug("telegram_listener_hiccup", error=str(exc))
            time.sleep(10)


def start_telegram_listener() -> None:
    """Idempotent — starts the button-tap listener if Telegram is configured."""
    global _started
    with _lock:
        if _started:
            return
        if not (os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
                and os.environ.get("TELEGRAM_CHAT_ID", "").strip()):
            return
        _started = True
    threading.Thread(target=_listener, name="tg-actions", daemon=True).start()
    log.info("telegram_listener_started")
