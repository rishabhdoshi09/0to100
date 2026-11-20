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
from core.runtime_paths import logs_dir

log = get_logger(__name__)

_started = False
_lock = threading.Lock()


def _safe_error(exc: BaseException) -> str:
    """Exception text with the bot token masked (requests embeds the URL)."""
    from core.redact import redact_secrets

    return redact_secrets(str(exc))[:160]


def build_setup_keyboard(setups: list[dict]) -> dict:
    """Inline keyboard: one row per setup with paper-trade + watchlist."""
    rows = []
    for p in setups[:5]:
        sym = p["symbol"]
        e, s, t = p.get("entry", 0), p.get("stop", 0), p.get("target", 0)
        rows.append([
            # Paise precision: rounding a ₹48.60/₹46.10 plan to whole rupees
            # moved its risk by ~40% (GPT cross-review C-9).
            {"text": f"📝 {sym} paper", "callback_data": f"pt|{sym}|{e:.2f}|{s:.2f}|{t:.2f}"},
            {"text": f"👁 watch", "callback_data": f"wl|{sym}|{e:.2f}|{s:.2f}|{t:.2f}"},
        ])
    return {"inline_keyboard": rows}


#: Hard per-trade ceilings for a Telegram paper tap (CLAUDE.md invariant 5).
HARD_RISK_PCT = 0.01
HARD_POSITION_PCT = 0.10


def _do_paper_trade(sym: str, entry: float, stop: float, target: float) -> str:
    """A tapped PAPER trade, under the same account limits as every paper engine.

    Until 2026-11-20 a tap forced at least one share when the 1% / 10% sizer
    said zero (a ₹15,000 share risked 1.5% and deployed 15% of ₹1 lakh), and
    never checked the account: a second tap opened the same symbol again, past
    the 5-position and 5% open-risk caps and across the modern book (GPT
    cross-review C-9). Now a zero size is refused, and the account exposure
    gate (duplicates across every paper book, position count, sector, total
    open risk) is re-checked under the shared account lock before the journal
    write. Never a live order.
    """
    from execution.trade_executor import place_trade
    from risk.position_sizer import size_position
    # The 1% / 10% invariants are HARD ceilings for a tap, whatever looser
    # values the settings carry (cross-review verification QT-03).
    try:
        from config import settings
        capital = float(settings.trading_capital or 0.0)
        risk_pct = min(float(settings.risk_per_trade_pct or HARD_RISK_PCT), HARD_RISK_PCT)
    except Exception:
        capital, risk_pct = 0.0, HARD_RISK_PCT
    ps = size_position(entry, stop, capital=capital or None, risk_pct=risk_pct)
    qty = int(ps.get("qty") or 0)
    if capital > 0 and entry > 0:
        qty = min(qty, int((capital * HARD_POSITION_PCT) // entry))
    if qty < 1:
        return (f"❌ {sym}: the 1% risk / 10% position limits allow 0 shares at "
                f"₹{entry:,.2f} (stop ₹{stop:,.2f}) — no paper trade.")
    try:
        from risk.portfolio_risk import account_exposure_gate, account_mutation_lock
    except Exception as exc:
        return f"❌ {sym}: account risk check unavailable ({type(exc).__name__}) — no paper trade."
    with account_mutation_lock():
        gate = account_exposure_gate(sym, qty=qty, entry=entry, stop=stop)
        if not gate.get("ok"):
            return f"❌ {sym}: {gate.get('detail') or gate.get('reason_code') or 'account limit'} — no paper trade."
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
    db = logs_dir() / "watchlist.db"
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


def _handle_message(msg: dict, token: str, allowed_chat: str) -> None:
    """📱 Text commands (/status, /pause, /aggressive …) — ADDITIVE layer,
    button-tap/alerts logic untouched. Wahi chat-id guard jo taps par hai:
    kisi aur ka message = silently ignore + log."""
    import requests
    chat_id = str(((msg.get("chat") or {}).get("id", "")))
    text = str(msg.get("text") or "")
    if not text.startswith("/"):
        return
    if chat_id != str(allowed_chat):
        log.warning("telegram_unauthorised_command", chat=chat_id)
        return
    try:
        from alerts.telegram_commands import handle_command
        reply = handle_command(text)
        if reply:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": allowed_chat, "text": reply,
                                "parse_mode": "HTML"}, timeout=8)
            log.info("telegram_command", cmd=text.split()[0])
    except Exception as exc:
        log.debug("telegram_command_error", error=_safe_error(exc))


def _listener() -> None:
    import requests
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    offset = 0
    fails = 0
    while True:
        try:
            from core.health import beat as _hb
            _hb("telegram_listener",
                note="ok" if fails == 0 else f"{fails} fails")
        except Exception:
            pass
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"offset": offset, "timeout": 25,
                        "allowed_updates": json.dumps(
                            ["callback_query", "message"])},
                timeout=35)
            for upd in (r.json().get("result") or []):
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                if "callback_query" in upd:
                    _handle_callback(upd["callback_query"], token, chat)
                elif "message" in upd:
                    _handle_message(upd["message"], token, chat)
            if fails >= 3:
                log.info("telegram_listener_recovered")
            fails = 0
        except Exception as exc:
            fails += 1
            # Backoff during outages: 10s → 60s after 3 straight fails.
            # One warning when entering backoff, then silence — a dead
            # network must not fill the log at 6 lines/minute.
            if fails == 3:
                log.warning("telegram_listener_offline_backoff",
                            error=_safe_error(exc))
            elif fails < 3:
                log.debug("telegram_listener_hiccup", error=_safe_error(exc))
            time.sleep(60 if fails >= 3 else 10)


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
