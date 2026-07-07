"""
Trade Executor — scan se execution tak, one click.

place_trade() does the full professional sequence:
  1. Entry order  — MARKET or LIMIT buy via Kite
  2. GTT OCO      — one-cancels-other exit: stop-loss + target sit at
                    the exchange (survives app/laptop shutdown)
  3. Journal      — every trade recorded in logs/trades.db

Paper mode: when Kite isn't connected, the same flow records a paper
trade instead — the buttons work identically, money doesn't move.

Safety rails:
  - quantity/price sanity checks before anything is sent
  - stop must be below entry, target above (long-only for now)
  - never places anything without an explicit user click upstream
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

_DB = Path(__file__).resolve().parent.parent / "logs" / "trades.db"
_db_lock = threading.Lock()

_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at TEXT NOT NULL,
    mode TEXT NOT NULL,              -- LIVE | PAPER
    symbol TEXT NOT NULL,
    qty INTEGER NOT NULL,
    entry_type TEXT NOT NULL,        -- MARKET | LIMIT
    entry_price REAL,
    stop_price REAL,
    target_price REAL,
    product TEXT,
    entry_order_id TEXT,
    gtt_id TEXT,
    status TEXT,
    note TEXT
);
"""


def _journal(row: dict) -> None:
    try:
        with _db_lock:
            _DB.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(_DB)
            conn.execute(_DDL)
            conn.execute(
                """INSERT INTO trades (placed_at, mode, symbol, qty, entry_type,
                   entry_price, stop_price, target_price, product,
                   entry_order_id, gtt_id, status, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.now().isoformat(timespec="seconds"), row.get("mode"),
                 row.get("symbol"), row.get("qty"), row.get("entry_type"),
                 row.get("entry_price"), row.get("stop_price"),
                 row.get("target_price"), row.get("product"),
                 str(row.get("entry_order_id") or ""), str(row.get("gtt_id") or ""),
                 row.get("status"), row.get("note", "")))
            conn.commit()
            conn.close()
    except Exception as exc:
        log.warning("trade_journal_failed", error=str(exc))


def _validate(symbol: str, qty: int, entry_price: float,
              stop: float, target: float) -> Optional[str]:
    if not symbol or qty < 1:
        return "Quantity kam se kam 1 honi chahiye"
    if entry_price <= 0:
        return "Entry price invalid hai"
    if stop >= entry_price:
        return f"Stop (₹{stop:,.1f}) entry (₹{entry_price:,.1f}) se NEECHE hona chahiye"
    if target <= entry_price:
        return f"Target (₹{target:,.1f}) entry se UPAR hona chahiye"
    if (entry_price - stop) / entry_price > 0.20:
        return "Stop entry se 20%+ door hai — galat lag raha hai, check karo"
    return None


def kite_ready() -> bool:
    try:
        from config import settings
        if not settings.kite_access_token:
            return False
        from data.kite_client import KiteClient
        return KiteClient().is_connected()
    except Exception:
        return False


def place_trade(symbol: str, qty: int, entry_type: str, entry_price: float,
                stop: float, target: float, product: str = "CNC",
                paper: bool = False) -> dict:
    """
    Full trade: entry + GTT OCO exits + journal.
    paper=True forces paper mode even when Kite is connected — Telegram
    buttons use this (live orders need the app's full ticket, on purpose).
    Returns {ok, mode, entry_order_id, gtt_id, message}.
    """
    err = _validate(symbol, qty, entry_price, stop, target)
    if err:
        return {"ok": False, "mode": "-", "message": err}

    if paper or not kite_ready():
        # ── Paper mode — same flow, no money ─────────────────────────────
        _journal({"mode": "PAPER", "symbol": symbol, "qty": qty,
                  "entry_type": entry_type, "entry_price": entry_price,
                  "stop_price": stop, "target_price": target,
                  "product": product, "status": "PAPER_OPEN"})
        log.info("paper_trade_placed", symbol=symbol, qty=qty)
        return {"ok": True, "mode": "PAPER",
                "message": f"📝 Paper trade recorded: {qty} × {symbol} @ "
                           f"₹{entry_price:,.1f} (stop ₹{stop:,.1f} / target "
                           f"₹{target:,.1f}). Kite login karke real trade kar "
                           f"sakte ho."}

    # ── LIVE mode ─────────────────────────────────────────────────────────
    from data.kite_client import KiteClient
    kite = KiteClient()
    result = {"ok": False, "mode": "LIVE", "entry_order_id": None,
              "gtt_id": None, "message": ""}
    try:
        # 1. Entry order
        entry_order_id = kite.place_order(
            symbol=symbol, transaction_type="BUY", quantity=qty,
            order_type=entry_type.upper(),
            price=entry_price if entry_type.upper() == "LIMIT" else None,
            product=product, tag="quantterm",
        )
        result["entry_order_id"] = entry_order_id
    except Exception as exc:
        result["message"] = f"Entry order fail: {exc}"
        _journal({"mode": "LIVE", "symbol": symbol, "qty": qty,
                  "entry_type": entry_type, "entry_price": entry_price,
                  "stop_price": stop, "target_price": target,
                  "product": product, "status": "ENTRY_FAILED",
                  "note": str(exc)[:200]})
        return result

    # 2. Wait briefly for the fill. GTT before fill is DANGEROUS: a stop
    #    trigger on an unfilled entry fires a sell with no shares, gets
    #    rejected, consumes the GTT — and the later-filled position sits
    #    UNPROTECTED while looking protected.
    filled = _entry_filled(kite, result["entry_order_id"], wait_s=8)

    gtt_msg = ""
    if filled:
        gtt_id, gtt_msg = _place_gtt(kite, symbol, qty, entry_price, stop,
                                     target, product)
        result["gtt_id"] = gtt_id
        status = "PLACED" + ("" if gtt_id else "_NO_GTT")
    else:
        status = "PENDING_GTT"
        gtt_msg = (" · ⏳ Entry abhi fill nahi hua — GTT fill hone ke BAAD "
                   "khud lag jayegi (system har scan cycle pe check karta hai)")

    result["ok"] = True
    result["message"] = (f"✅ Order placed: {qty} × {symbol} "
                         f"({entry_type} @ ₹{entry_price:,.1f}){gtt_msg}")
    _journal({"mode": "LIVE", "symbol": symbol, "qty": qty,
              "entry_type": entry_type, "entry_price": entry_price,
              "stop_price": stop, "target_price": target, "product": product,
              "entry_order_id": result["entry_order_id"],
              "gtt_id": result["gtt_id"],
              "status": status})
    log.info("live_trade_placed", symbol=symbol, qty=qty,
             order_id=result["entry_order_id"], gtt=result["gtt_id"],
             status=status)
    return result


def _entry_filled(kite, order_id: str, wait_s: int = 8) -> bool:
    """Poll order status briefly. True only on COMPLETE."""
    import time as _time
    deadline = _time.time() + wait_s
    while _time.time() < deadline:
        try:
            hist = kite.raw.order_history(order_id)
            status = str(hist[-1].get("status", "")).upper() if hist else ""
            if status == "COMPLETE":
                return True
            if status in ("REJECTED", "CANCELLED"):
                return False
        except Exception:
            pass
        _time.sleep(1.5)
    return False


def _place_gtt(kite, symbol: str, qty: int, entry_price: float,
               stop: float, target: float, product: str) -> tuple:
    """(gtt_id|None, message). Loud warning string on failure."""
    try:
        raw = kite.raw
        gtt = raw.place_gtt(
            trigger_type=raw.GTT_TYPE_OCO,
            tradingsymbol=symbol, exchange="NSE",
            trigger_values=[round(stop, 1), round(target, 1)],
            last_price=entry_price,
            orders=[
                {"transaction_type": "SELL", "quantity": qty,
                 "order_type": "LIMIT", "product": product,
                 "price": round(stop * 0.995, 1)},      # small buffer for fill
                {"transaction_type": "SELL", "quantity": qty,
                 "order_type": "LIMIT", "product": product,
                 "price": round(target, 1)},
            ])
        gtt_id = gtt.get("trigger_id") if isinstance(gtt, dict) else gtt
        return gtt_id, f" · GTT OCO set (stop ₹{stop:,.1f} / target ₹{target:,.1f})"
    except Exception as exc:
        log.warning("gtt_failed", symbol=symbol, error=str(exc))
        return None, (f" · ⚠ GTT nahi laga ({str(exc)[:80]}) — stop-loss "
                      f"MANUALLY lagao Kite app mein!")


def ensure_pending_gtts() -> int:
    """
    Babysit PENDING_GTT trades: once the entry fills, place the GTT and
    upgrade the journal row. Rejected/cancelled entries get closed out.
    Called every scan cycle during market hours. Returns GTTs placed.
    """
    import sqlite3
    if not kite_ready():
        return 0
    try:
        with _db_lock:
            if not _DB.exists():
                return 0
            conn = sqlite3.connect(_DB)
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM trades WHERE status='PENDING_GTT'").fetchall()]
            conn.close()
        if not rows:
            return 0

        from data.kite_client import KiteClient
        kite = KiteClient()
        placed = 0
        for t in rows:
            oid = t["entry_order_id"]
            try:
                hist = kite.raw.order_history(oid)
                status = str(hist[-1].get("status", "")).upper() if hist else ""
            except Exception:
                continue
            new_status, gtt_id = None, None
            if status == "COMPLETE":
                gtt_id, _msg = _place_gtt(
                    kite, t["symbol"], int(t["qty"]), float(t["entry_price"]),
                    float(t["stop_price"]), float(t["target_price"]),
                    t["product"] or "CNC")
                new_status = "PLACED" if gtt_id else "PLACED_NO_GTT"
                if gtt_id:
                    placed += 1
            elif status in ("REJECTED", "CANCELLED"):
                new_status = "ENTRY_FAILED"
            if new_status:
                with _db_lock:
                    conn = sqlite3.connect(_DB)
                    conn.execute(
                        "UPDATE trades SET status=?, gtt_id=? WHERE id=?",
                        (new_status, str(gtt_id or ""), t["id"]))
                    conn.commit()
                    conn.close()
                log.info("pending_gtt_resolved", symbol=t["symbol"],
                         status=new_status)
        return placed
    except Exception as exc:
        log.debug("ensure_pending_gtts_skip", error=str(exc))
        return 0


def recent_trades(limit: int = 20) -> list[dict]:
    """Last N journal entries for display."""
    try:
        with _db_lock:
            if not _DB.exists():
                return []
            conn = sqlite3.connect(_DB)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []
