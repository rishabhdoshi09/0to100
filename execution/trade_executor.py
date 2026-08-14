"""
Legacy Trade Executor — manually clicked compatibility path.

This module is NOT the institutional OMS. Its live branch is locked by default while
QuantTerm builds the durable Target Portfolio → Risk Governor → OMS → reconciliation
chain. PAPER behaviour remains available for existing UI and Telegram flows.

Safety rails:
  - quantity/price sanity checks before anything is sent
  - stop must be below entry, target above (long-only for now)
  - connected-broker execution requires an explicit unsafe legacy override
  - governance uncertainty blocks the order instead of failing open
  - never places anything without an explicit user click upstream
"""
from __future__ import annotations

import os
import sqlite3
import threading
from core.market_clock import now_ist_naive
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


def connect():
    """Concurrency-safe SQLite connection for the shared trades DB.

    Two autopilots (NSE + US) and the position manager write this DB from
    separate daemon threads. WAL lets readers not block the writer, and a
    10s busy timeout makes a second writer WAIT for the lock instead of
    instantly raising 'database is locked'. Every writer must use this."""
    conn = sqlite3.connect(_DB, timeout=10.0)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return conn


def _journal(row: dict) -> None:
    try:
        with _db_lock:
            _DB.parent.mkdir(parents=True, exist_ok=True)
            conn = connect()
            conn.execute(_DDL)
            conn.execute(
                """INSERT INTO trades (placed_at, mode, symbol, qty, entry_type,
                   entry_price, stop_price, target_price, product,
                   entry_order_id, gtt_id, status, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now_ist_naive().isoformat(timespec="seconds"), row.get("mode"),
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
    """True when a usable Kite session exists.

    Re-reads ``KITE_ACCESS_TOKEN`` from ``.env`` so ``python main.py login``
    wakes the sniper without requiring an autonomy process restart.
    """
    try:
        from data.kite_client import KiteClient, _fresh_env
        token = (_fresh_env("KITE_ACCESS_TOKEN") or "").strip()
        key = (_fresh_env("KITE_API_KEY") or "").strip()
        if not token or not key:
            return False
        return KiteClient().is_connected()
    except Exception:
        return False


def legacy_live_enabled() -> bool:
    """Emergency compatibility override; false by default and never set by the UI."""
    return os.getenv("QT_ENABLE_UNSAFE_LEGACY_LIVE", "0").strip() == "1"


def place_trade(symbol: str, qty: int, entry_type: str, entry_price: float,
                stop: float, target: float, product: str = "CNC",
                paper: bool = False, note: str = "") -> dict:
    """Record a PAPER trade or, only under an explicit unsafe override, use legacy LIVE.

    The institutional execution path must consume a durable TradeIntent through the future
    OMS. This function remains only for backwards-compatible manually clicked flows.
    """
    err = _validate(symbol, qty, entry_price, stop, target)
    if err:
        return {"ok": False, "mode": "-", "message": err}

    if paper or not kite_ready():
        # Existing compatibility behaviour. The response explicitly identifies PAPER.
        _journal({"mode": "PAPER", "symbol": symbol, "qty": qty,
                  "entry_type": entry_type, "entry_price": entry_price,
                  "stop_price": stop, "target_price": target,
                  "product": product, "status": "PAPER_OPEN", "note": note})
        log.info("paper_trade_placed", symbol=symbol, qty=qty)
        return {"ok": True, "mode": "PAPER", "fallback": not paper,
                "message": f"📝 Paper trade recorded: {qty} × {symbol} @ "
                           f"₹{entry_price:,.1f} (stop ₹{stop:,.1f} / target "
                           f"₹{target:,.1f}). No live order was sent."}

    if not legacy_live_enabled():
        reason = ("Live execution is locked: the legacy executor has no durable OMS, "
                  "broker reconciliation, or verified protection recovery. Use PAPER until "
                  "the institutional execution chain is certified.")
        _journal({"mode": "LIVE", "symbol": symbol, "qty": qty,
                  "entry_type": entry_type, "entry_price": entry_price,
                  "stop_price": stop, "target_price": target,
                  "product": product, "status": "BLOCKED_LEGACY_LIVE_LOCK",
                  "note": reason})
        return {"ok": False, "mode": "LIVE", "message": reason}

    # Governance must be available and affirmative. Unknown state fails closed.
    try:
        from core.governance import can_place_order
        allowed, size_mult, gov_reason = can_place_order()
    except Exception as exc:
        reason = f"Governance unavailable; live order blocked: {exc}"
        _journal({"mode": "LIVE", "symbol": symbol, "qty": qty,
                  "entry_type": entry_type, "entry_price": entry_price,
                  "stop_price": stop, "target_price": target,
                  "product": product, "status": "BLOCKED_GOVERNANCE_UNAVAILABLE",
                  "note": reason[:200]})
        return {"ok": False, "mode": "LIVE", "message": reason}
    if not allowed:
        _journal({"mode": "LIVE", "symbol": symbol, "qty": qty,
                  "entry_type": entry_type, "entry_price": entry_price,
                  "stop_price": stop, "target_price": target,
                  "product": product, "status": "BLOCKED_GOVERNANCE",
                  "note": gov_reason[:200]})
        return {"ok": False, "mode": "LIVE", "message": gov_reason}

    approved_qty = int(qty * float(size_mult))
    if approved_qty < 1:
        reason = "Governance reduced the approved quantity below one share; order blocked."
        _journal({"mode": "LIVE", "symbol": symbol, "qty": qty,
                  "entry_type": entry_type, "entry_price": entry_price,
                  "stop_price": stop, "target_price": target,
                  "product": product, "status": "BLOCKED_GOVERNANCE_SIZE",
                  "note": reason})
        return {"ok": False, "mode": "LIVE", "message": reason}
    qty = approved_qty
    if gov_reason:
        note = f"{note} | {gov_reason}".strip(" |")

    # ── LEGACY LIVE mode — emergency override only ─────────────────────────
    from data.kite_client import KiteClient
    kite = KiteClient()
    result = {"ok": False, "mode": "LIVE", "entry_order_id": None,
              "gtt_id": None, "message": ""}
    try:
        entry_order_id = kite.place_order(
            symbol=symbol, transaction_type="BUY", quantity=qty,
            order_type=entry_type.upper(),
            price=entry_price if entry_type.upper() == "LIMIT" else None,
            product=product, tag="quantterm-legacy",
        )
        result["entry_order_id"] = entry_order_id
    except Exception as exc:
        result["message"] = f"Entry order state uncertain or failed: {exc}"
        _journal({"mode": "LIVE", "symbol": symbol, "qty": qty,
                  "entry_type": entry_type, "entry_price": entry_price,
                  "stop_price": stop, "target_price": target,
                  "product": product, "status": "RECOVERY_REQUIRED",
                  "note": str(exc)[:200]})
        try:
            from core.governance import record_order_result
            record_order_result(ok=False)
        except Exception:
            pass
        return result

    filled = _entry_filled(kite, result["entry_order_id"], wait_s=8)

    gtt_msg = ""
    if filled:
        gtt_id, gtt_msg = _place_gtt(kite, symbol, qty, entry_price, stop,
                                     target, product)
        result["gtt_id"] = gtt_id
        status = "PLACED" + ("" if gtt_id else "_NO_GTT")
    else:
        status = "PENDING_GTT"
        gtt_msg = (" · Entry not confirmed filled; protection remains pending and "
                   "requires reconciliation before any retry.")

    result["ok"] = True
    result["message"] = (f"Order accepted by legacy path: {qty} × {symbol} "
                         f"({entry_type} @ ₹{entry_price:,.1f}){gtt_msg}")
    _journal({"mode": "LIVE", "symbol": symbol, "qty": qty,
              "entry_type": entry_type, "entry_price": entry_price,
              "stop_price": stop, "target_price": target, "product": product,
              "entry_order_id": result["entry_order_id"],
              "gtt_id": result["gtt_id"],
              "status": status, "note": note})
    log.info("legacy_live_trade_placed", symbol=symbol, qty=qty,
             order_id=result["entry_order_id"], gtt=result["gtt_id"],
             status=status)
    try:
        from core.governance import record_order_result
        record_order_result(ok=True, gtt_ok=bool(result["gtt_id"]) if filled else None)
    except Exception:
        pass
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
                 "price": round(stop * 0.995, 1)},
                {"transaction_type": "SELL", "quantity": qty,
                 "order_type": "LIMIT", "product": product,
                 "price": round(target, 1)},
            ])
        gtt_id = gtt.get("trigger_id") if isinstance(gtt, dict) else gtt
        return gtt_id, f" · GTT OCO set (stop ₹{stop:,.1f} / target ₹{target:,.1f})"
    except Exception as exc:
        log.warning("gtt_failed", symbol=symbol, error=str(exc))
        return None, (f" · GTT failed ({str(exc)[:80]}); position is unprotected and "
                      f"requires immediate recovery.")


def ensure_pending_gtts() -> int:
    """Resolve legacy PENDING_GTT rows after entry status becomes authoritative."""
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
