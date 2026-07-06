"""
Position Manager — trade lene ke BAAD ka dimaag.

Every 15 minutes during market hours, open trades from the journal
are reviewed against live prices. Each position gets an R-progress
reading (profit measured in units of initial risk) and an action:

  r >= 2.0        → 🎯 TARGET ZONE  — "book karo ya trail karo, gift wapas mat do"
  r >= 1.0        → 💰 BOOK HALF    — "aadha profit book, stop ko entry pe le aao
                                       (ab yeh FREE trade hai)"
  price near stop → ⚠️ STOP KAREEB  — "discipline: stop hit ho toh exit, average mat karo"
  stop crossed    → 🔴 STOP HIT     — paper trades auto-close as LOSS
  target crossed  → 🟢 TARGET HIT   — paper trades auto-close as WIN

Live trades: GTT already sits at the exchange, so alerts here are
guidance (book-half/trail) — the hard exits are enforced by NSE.
Alerts push to Telegram once per symbol+type per day.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime

from logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_alerted: dict[str, set] = {}      # {YYYY-MM-DD: {"SYM:TYPE", ...}}


def _open_trades() -> list[dict]:
    from execution.trade_executor import _DB
    try:
        if not _DB.exists():
            return []
        conn = sqlite3.connect(_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trades WHERE status IN "
            "('PAPER_OPEN','PLACED','PLACED_NO_GTT') ORDER BY id DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("open_trades_read_failed", error=str(exc))
        return []


def _close_paper(trade_id: int, outcome: str, exit_price: float) -> None:
    from execution.trade_executor import _DB
    try:
        conn = sqlite3.connect(_DB)
        conn.execute(
            "UPDATE trades SET status=?, note=? WHERE id=?",
            (f"PAPER_{outcome}",
             f"closed @ ₹{exit_price:,.1f} on "
             f"{datetime.now().strftime('%d %b %H:%M')}", trade_id))
        conn.commit()
        conn.close()
    except Exception as exc:
        log.debug("paper_close_failed", error=str(exc))


def review_positions() -> list[dict]:
    """
    [{symbol, qty, entry, stop, target, live, r_progress, pnl, alert,
      advice, mode}] for every open trade. Auto-closes paper trades
    whose stop/target has been crossed.
    """
    trades = _open_trades()
    if not trades:
        return []

    try:
        from data.live_quotes import get_live_quotes
        live = get_live_quotes(sorted({t["symbol"] for t in trades}))
    except Exception:
        live = {}

    out: list[dict] = []
    for t in trades:
        sym = t["symbol"]
        entry = float(t["entry_price"] or 0)
        stop = float(t["stop_price"] or 0)
        target = float(t["target_price"] or 0)
        qty = int(t["qty"] or 0)
        risk = entry - stop
        q = live.get(sym)
        px = float(q["price"]) if q and q.get("price") else None
        is_paper = str(t["mode"]) == "PAPER"

        row = {"symbol": sym, "qty": qty, "entry": entry, "stop": stop,
               "target": target, "live": px, "mode": t["mode"],
               "r_progress": None, "pnl": None, "alert": "", "advice": ""}

        if px is None or risk <= 0:
            row["advice"] = "live price nahi mila — Kite login check karo"
            out.append(row)
            continue

        r = (px - entry) / risk
        row["r_progress"] = round(r, 2)
        row["pnl"] = round((px - entry) * qty, 0)

        if px <= stop:
            row["alert"] = "STOP_HIT"
            row["advice"] = (f"🔴 Stop hit (₹{px:,.1f} ≤ ₹{stop:,.1f}). "
                             + ("Paper trade LOSS mark ho gaya."
                                if is_paper else
                                "GTT ne exit kar diya hoga — Kite mein confirm karo."))
            if is_paper:
                _close_paper(t["id"], "LOSS", px)
        elif px >= target:
            row["alert"] = "TARGET_HIT"
            row["advice"] = (f"🟢 Target hit (₹{px:,.1f} ≥ ₹{target:,.1f})! "
                             + ("Paper trade WIN mark ho gaya."
                                if is_paper else
                                "GTT ne book kar liya hoga — Kite mein confirm karo."))
            if is_paper:
                _close_paper(t["id"], "WIN", px)
        elif r >= 2.0:
            row["alert"] = "TARGET_ZONE"
            row["advice"] = (f"🎯 +{r:.1f}R chal raha hai — target zone. Book karo "
                             f"ya stop ko ₹{entry + risk:,.0f} (+1R) pe trail karo. "
                             f"Gift wapas mat do.")
        elif r >= 1.0:
            row["alert"] = "BOOK_HALF"
            row["advice"] = (f"💰 +{r:.1f}R profit — aadha book karo aur stop ko "
                             f"entry ₹{entry:,.0f} pe le aao. Uske baad yeh FREE "
                             f"trade hai, tension khatam.")
        elif px <= stop * 1.01:
            row["alert"] = "NEAR_STOP"
            row["advice"] = (f"⚠️ Stop se sirf {((px - stop) / px * 100):.1f}% upar. "
                             f"Hit ho toh exit — average down KABHI nahi.")
        else:
            row["advice"] = f"👍 Theek chal raha hai ({r:+.1f}R). Plan pe tike raho."
        out.append(row)
    return out


def push_position_alerts() -> int:
    """Telegram push for actionable alerts — once per symbol+type per day."""
    try:
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if not engine.is_configured():
            return 0
        rows = [r for r in review_positions() if r["alert"]]
        if not rows:
            return 0
        today = datetime.now().strftime("%Y-%m-%d")
        with _lock:
            _alerted.setdefault(today, set())
            for k in list(_alerted):
                if k != today:
                    del _alerted[k]
            fresh = [r for r in rows
                     if f"{r['symbol']}:{r['alert']}" not in _alerted[today]]
            if not fresh:
                return 0
            lines = ["📍 <b>Position Update</b>"]
            for r in fresh[:6]:
                pnl = f" · P&L ₹{r['pnl']:+,.0f}" if r["pnl"] is not None else ""
                lines.append(f"\n<b>{r['symbol']}</b> ({r['mode']}){pnl}\n{r['advice']}")
            if engine.send("\n".join(lines)):
                _alerted[today].update(f"{r['symbol']}:{r['alert']}" for r in fresh)
                log.info("position_alerts_pushed", count=len(fresh))
                return len(fresh)
    except Exception as exc:
        log.debug("position_alerts_skip", error=str(exc))
    return 0
