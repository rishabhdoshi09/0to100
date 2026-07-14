"""
🇺🇸 US Autopilot — paper-only autonomous trading for US equities.

Same discipline as the NSE autopilot (gates → conviction sizing →
protected exit → honest, cost-net accounting → evidence-gated Report
Card), but self-contained so the working Kite money-path is never
touched. PAPER ONLY: no US broker, no real orders — trades are
simulated and journaled, closed on stop/target against live US prices.
Going LIVE on US would need a US broker adapter (Alpaca); this module
stops at paper by design.

State: logs/us_autopilot.json. Trades tagged US_AUTOPILOT in the shared
journal so they never collide with the NSE ledger or Report Card.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, date

from logger import get_logger

log = get_logger(__name__)

_STATE_FILE = __import__("pathlib").Path(__file__).resolve().parent.parent \
    / "logs" / "us_autopilot.json"
_lock = threading.RLock()
_consider_lock = threading.Lock()

TAG = "US_AUTOPILOT"
_OPEN = ("PAPER_OPEN",)
_WIN = ("PAPER_WIN",)
_LOSS = ("PAPER_LOSS",)

_DEFAULTS = {
    "armed": False,
    "allocation": 0.0,             # USD the user assigns
    "realized_pnl": 0.0,
    "cash_reserve_pct": 0.10,
    "per_trade_cap_pct": 0.20,
    "risk_per_trade_pct": 0.01,
    "max_open_positions": 3,
    "max_trades_per_day": 4,
    "daily_loss_limit_pct": 0.03,
    "min_score": 60.0,
    "min_conviction": 50.0,
    "start_time": "10:00",         # ET — let the US open settle
    "end_time": "15:30",           # ET — no entries in the last 30 min
    "target_pct": 4.0,             # US moves bigger; 4% default
    "max_chase_pct": 1.5,
    "max_hold_days": 5,
    "trailing_enabled": True,
    "breakeven_trigger_pct": 2.0,
    "trades_today": {},
    "traded_symbols": {},
    "accounted_ids": [],
    "activity": [],
    "disarmed_reason": "",
}

_state: dict = {}


# ── Persistence ───────────────────────────────────────────────────────────────

def _load() -> dict:
    global _state
    with _lock:
        if _state:
            return _state
        _state = dict(_DEFAULTS)
        try:
            if _STATE_FILE.exists():
                on_disk = json.loads(_STATE_FILE.read_text())
                _state.update({k: v for k, v in on_disk.items() if k in _DEFAULTS})
        except Exception as exc:
            log.warning("us_autopilot_load_failed", error=str(exc))
        return _state


def _save() -> None:
    try:
        with _lock:
            data = dict(_load())
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(_STATE_FILE)
    except Exception as exc:
        log.warning("us_autopilot_save_failed", error=str(exc))


def _log_activity(msg: str) -> None:
    with _lock:
        s = _load()
        s["activity"] = ([f"{datetime.now().strftime('%d %b %H:%M')} · {msg}"]
                         + s.get("activity", []))[:40]
    _save()
    log.info("us_autopilot", msg=msg)


def _notify(msg: str) -> None:
    try:
        from alerts.telegram_alerts import AlertEngine
        AlertEngine().send(f"🇺🇸🤖 <b>US Autopilot</b>\n{msg}")
    except Exception:
        pass


# ── Journal (shared DB, US_AUTOPILOT tag) ─────────────────────────────────────

def _trades(statuses: tuple) -> list[dict]:
    conn = None
    try:
        from execution.trade_executor import _DB, connect as _te_connect
        if not _DB.exists():
            return []
        conn = _te_connect()
        conn.row_factory = sqlite3.Row
        q = ("SELECT * FROM trades WHERE note LIKE ? AND status IN (%s) "
             "ORDER BY id DESC" % ",".join("?" * len(statuses)))
        rows = conn.execute(q, (f"%{TAG}%", *statuses)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _open_trades() -> list[dict]:
    return _trades(_OPEN)


def _update(trade_id: int, sets: str, params: tuple) -> None:
    conn = None
    try:
        from execution.trade_executor import connect as _te_connect
        conn = _te_connect()
        conn.execute(f"UPDATE trades SET {sets} WHERE id=?", (*params, trade_id))
        conn.commit()
    except Exception as exc:
        log.debug("us_autopilot_update_failed", error=str(exc))
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ── Public config / status ────────────────────────────────────────────────────

def get_status() -> dict:
    with _lock:
        s = dict(_load())
    opens = _open_trades()
    s["open_trades"] = opens
    s["deployed"] = round(sum(float(t["entry_price"] or 0) * int(t["qty"] or 0)
                              for t in opens), 2)
    s["pool"] = round(s["allocation"] + s["realized_pnl"], 2)
    s["available"] = round(
        max(0.0, s["pool"] * (1 - s["cash_reserve_pct"]) - s["deployed"]), 2)
    today = date.today().isoformat()
    s["trades_today_count"] = int(s.get("trades_today", {}).get(today, 0))
    return s


def set_config(**kwargs) -> None:
    clamps = {
        "allocation": (0.0, 1e9), "cash_reserve_pct": (0.05, 0.50),
        "per_trade_cap_pct": (0.05, 0.50), "risk_per_trade_pct": (0.0025, 0.02),
        "max_open_positions": (1, 10), "max_trades_per_day": (1, 15),
        "daily_loss_limit_pct": (0.01, 0.10), "min_score": (40.0, 95.0),
        "min_conviction": (0.0, 90.0), "target_pct": (1.0, 15.0),
        "max_chase_pct": (0.25, 5.0), "max_hold_days": (2, 15),
        "breakeven_trigger_pct": (0.5, 8.0),
    }
    with _lock:
        s = _load()
        for k, v in kwargs.items():
            if k == "trailing_enabled":
                s[k] = bool(v)
            elif k in ("start_time", "end_time") and isinstance(v, str):
                try:
                    datetime.strptime(v.strip(), "%H:%M")
                    s[k] = v.strip()
                except ValueError:
                    pass
            elif k in clamps:
                lo, hi = clamps[k]
                s[k] = type(_DEFAULTS[k])(min(hi, max(lo, v)))
    _save()


def arm() -> tuple[bool, str]:
    s = _load()
    if s["allocation"] < 1000:
        return False, "Pehle allocation set karo (min $1,000)"
    with _lock:
        s["armed"] = True
        s["disarmed_reason"] = ""
    _save()
    pool = s["allocation"] + s["realized_pnl"]
    _log_activity(f"ARMED (PAPER) — pool ${pool:,.0f}")
    _notify(f"ARMED in PAPER.\nPool: ${pool:,.0f} · entries "
            f"{s['start_time']}–{s['end_time']} ET · target +{s['target_pct']}% · "
            f"max {s['max_trades_per_day']} trades/day")
    return True, "Armed (PAPER)"


def disarm(reason: str = "user") -> None:
    with _lock:
        s = _load()
        s["armed"] = False
        s["disarmed_reason"] = reason
    _save()
    _log_activity(f"DISARMED ({reason})")
    if reason != "user":
        _notify(f"⛔ DISARMED — {reason}")


# ── Gates ─────────────────────────────────────────────────────────────────────

def _in_window(now_et=None) -> bool:
    s = _load()
    try:
        import pytz
        now_et = now_et or datetime.now(pytz.timezone("America/New_York"))
    except Exception:
        return False
    if now_et.weekday() >= 5:
        return False
    hm = now_et.strftime("%H:%M")
    return s["start_time"] <= hm <= s["end_time"]


def _passes_gates(symbol: str, score: float, conviction: float) -> str | None:
    s = _load()
    if not s["armed"]:
        return "not armed"
    if not _in_window():
        return f"US window ({s['start_time']}-{s['end_time']} ET) ke bahar"
    today = date.today().isoformat()
    if int(s.get("trades_today", {}).get(today, 0)) >= s["max_trades_per_day"]:
        return "daily trade limit reached"
    if len(_open_trades()) >= s["max_open_positions"]:
        return "max open positions reached"
    if symbol in s.get("traded_symbols", {}).get(today, []):
        return "symbol already traded today"
    if score < s["min_score"]:
        return f"score {score:.0f} < min {s['min_score']:.0f}"
    if conviction and conviction < s["min_conviction"]:
        return f"conviction {conviction:.0f} < min {s['min_conviction']:.0f}"
    return None


# ── Sizing (shared philosophy) ────────────────────────────────────────────────

def _conviction_mult(score: float, conviction: float) -> float:
    m = 1.0
    if score >= 80:
        m += 0.25
    if conviction >= 70:
        m += 0.25
    elif conviction and conviction < 55:
        m -= 0.25
    return min(1.5, max(0.5, m))


def _size(entry: float, stop: float, mult: float) -> int:
    st = get_status()
    pool, available = st["pool"], st["available"]
    if pool <= 0 or available <= 0 or entry <= stop or entry <= 0:
        return 0
    risk_qty = int((pool * _load()["risk_per_trade_pct"] * mult) / (entry - stop))
    cap_qty = int((pool * _load()["per_trade_cap_pct"]) / entry)
    avail_qty = int(available / entry)
    return max(0, min(risk_qty, cap_qty, avail_qty))


def _anchor_live(symbol: str, entry: float, stop: float,
                 max_chase_pct: float) -> tuple[float | None, str]:
    try:
        from data.us_data import us_live_prices
        q = us_live_prices([symbol]).get(symbol) or {}
    except Exception:
        q = {}
    px = float(q.get("price") or 0)
    if px <= 0:
        return None, "live US quote nahi — stale price pe trade NAHI"
    if px <= stop:
        return None, f"live ${px:,.2f} stop ${stop:,.2f} ke neeche — setup toota"
    if entry > 0 and px > entry * (1 + max_chase_pct / 100):
        return None, f"live ${px:,.2f} entry ${entry:,.2f} se upar — chase nahi"
    return px, ""


# ── Entry ─────────────────────────────────────────────────────────────────────

def consider(symbol: str, entry: float, stop: float, score: float,
             conviction: float, source: str = "us_scanner") -> bool:
    with _consider_lock:
        return _consider_locked(symbol, entry, stop, score, conviction, source)


def _consider_locked(symbol, entry, stop, score, conviction, source) -> bool:
    try:
        s = _load()
        entry, stop = float(entry), float(stop)
        if entry <= 0 or stop <= 0 or stop >= entry:
            return False
        reject = _passes_gates(symbol, score, conviction)
        if reject:
            log.debug("us_autopilot_reject", symbol=symbol, reason=reject)
            return False
        live_entry, why = _anchor_live(symbol, entry, stop, s["max_chase_pct"])
        if live_entry is None:
            _log_activity(f"SKIP {symbol}: {why}")
            return False
        entry = live_entry
        target = round(entry * (1 + s["target_pct"] / 100), 2)
        qty = _size(entry, stop, _conviction_mult(score, conviction))
        if qty < 1:
            _log_activity(f"SKIP {symbol}: pool/limits mein 1 share nahi aata")
            return False

        from execution.trade_executor import place_trade
        res = place_trade(symbol=symbol, qty=qty, entry_type="MARKET",
                          entry_price=entry, stop=round(stop, 2), target=target,
                          product="CNC", paper=True, note=f"{TAG}:{source}")
        if not res.get("ok"):
            _log_activity(f"FAIL {symbol}: {res.get('message','')[:80]}")
            return False
        today = date.today().isoformat()
        with _lock:
            s = _load()
            s["trades_today"] = {today: int(s.get("trades_today", {}).get(today, 0)) + 1}
            s["traded_symbols"] = {
                today: s.get("traded_symbols", {}).get(today, []) + [symbol]}
        _save()
        _log_activity(f"BUY {qty}×{symbol} @ ${entry:,.2f} "
                      f"(stop ${stop:,.2f} / target ${target:,.2f}) [PAPER]")
        _notify(f"BUY <b>{qty} × {symbol}</b> @ ${entry:,.2f}\n"
                f"stop ${stop:,.2f} · target ${target:,.2f} (+{s['target_pct']}%)")
        return True
    except Exception as exc:
        log.warning("us_autopilot_consider_failed", symbol=symbol, error=str(exc))
        return False


def on_setups(results: list[dict]) -> None:
    s = _load()
    if not s["armed"]:
        return
    ranked = sorted(
        [r for r in results if r.get("verdict") in ("STRONG BUY", "BUY")],
        key=lambda r: float(r.get("conviction_rank") or r.get("score", 0) or 0),
        reverse=True)
    for r in ranked[:15]:
        consider(symbol=r["symbol"],
                 entry=float(r.get("entry") or r.get("price") or 0),
                 stop=float(r.get("stop") or 0),
                 score=float(r.get("score") or 0),
                 conviction=float(r.get("breakout_conviction") or 0),
                 source="us_scanner")


# ── Cost-net P&L (single source of truth) ─────────────────────────────────────

def _net_pnl(t: dict) -> float:
    entry = float(t.get("entry_price") or 0)
    qty = int(t.get("qty") or 0)
    exit_px = (float(t.get("exit_price") or 0) or
               (float(t["target_price"] or 0) if t["status"] in _WIN
                else float(t["stop_price"] or 0)))
    if entry <= 0 or qty <= 0 or exit_px <= 0:
        return 0.0
    try:
        from execution.cost_model import net_result
        return net_result(entry, exit_px, qty, exit_is_stop=(t["status"] in _LOSS),
                          paper=True, market="US")["net"]
    except Exception:
        return (exit_px - entry) * qty


# ── Review: closes, compounding, circuit breaker, time-stop ───────────────────

def review_cycle() -> None:
    s = _load()
    if s["allocation"] <= 0:
        return
    try:
        _close_positions()
        _account_closed()
        _circuit_breaker()
    except Exception as exc:
        log.debug("us_autopilot_review_failed", error=str(exc))


def _close_positions() -> None:
    """Paper positions close on stop/target vs live US price; +breakeven
    trail; time-stop for stale flats."""
    opens = _open_trades()
    if not opens:
        return
    try:
        from data.us_data import us_live_prices
        live = us_live_prices(sorted({t["symbol"] for t in opens}))
    except Exception:
        return
    s = _load()
    trig = 1 + s.get("breakeven_trigger_pct", 2.0) / 100
    max_days = int(s.get("max_hold_days", 5))
    for t in opens:
        q = live.get(t["symbol"])
        if not (q and q.get("price")):
            continue
        px = float(q["price"])
        entry = float(t["entry_price"] or 0)
        stop = float(t["stop_price"] or 0)
        target = float(t["target_price"] or 0)
        if target and px >= target:
            _update(t["id"], "status=?, exit_price=?, note=note||?",
                    ("PAPER_WIN", px, f" | target ${px:,.2f}"))
            continue
        if stop and px <= stop:
            _update(t["id"], "status=?, exit_price=?, note=note||?",
                    ("PAPER_LOSS", px, f" | stop ${px:,.2f}"))
            continue
        # breakeven trail
        if s.get("trailing_enabled", True) and 0 < stop < entry and px >= entry * trig:
            _update(t["id"], "stop_price=?, note=note||?",
                    (round(entry, 2), f" | breakeven trail @ ${px:,.2f}"))
        # time-stop
        try:
            placed = datetime.fromisoformat(str(t["placed_at"]))
            if (datetime.now() - placed).days >= max_days:
                _update(t["id"], "status=?, exit_price=?, note=note||?",
                        ("PAPER_WIN" if px >= entry else "PAPER_LOSS", px,
                         f" | time-stop day {max_days} @ ${px:,.2f}"))
        except Exception:
            pass


def _account_closed() -> None:
    with _lock:
        s = _load()
        seen = set(s.get("accounted_ids", []))
        changed = False
        for t in _trades(_WIN + _LOSS):
            if t["id"] in seen:
                continue
            pnl = round(_net_pnl(t), 2)
            s["realized_pnl"] = round(s.get("realized_pnl", 0.0) + pnl, 2)
            seen.add(t["id"])
            changed = True
            _notify(f"CLOSED <b>{t['symbol']}</b> {'🟢' if pnl >= 0 else '🔴'} "
                    f"${pnl:+,.2f} (net)\nPool: ${s['allocation'] + s['realized_pnl']:,.0f}")
        s["accounted_ids"] = sorted(seen)[-500:]
    if changed:
        _save()


def _circuit_breaker() -> None:
    s = _load()
    if not s["armed"]:
        return
    pool = s["allocation"] + s["realized_pnl"]
    if pool <= 0:
        return
    today = date.today().isoformat()
    day = sum(_net_pnl(t) for t in _trades(_WIN + _LOSS)
              if str(t["placed_at"])[:10] == today)
    # Include UNREALIZED on open positions — else three open trades could
    # bleed −8% and never trip the breaker until they close (NSE-parity).
    opens = _open_trades()
    if opens:
        try:
            from data.us_data import us_live_prices
            live = us_live_prices(sorted({t["symbol"] for t in opens}))
            for t in opens:
                q = live.get(t["symbol"])
                if q and q.get("price"):
                    day += ((float(q["price"]) - float(t["entry_price"] or 0))
                            * int(t["qty"] or 0))
        except Exception:
            pass
    if day <= -pool * s["daily_loss_limit_pct"]:
        disarm(f"circuit breaker: day P&L ${day:,.0f} (realized+open)")


# ── Report Card ───────────────────────────────────────────────────────────────

def report_card() -> dict:
    closed = _trades(_WIN + _LOSS)
    trades, equity, peak, dd = [], 0.0, 0.0, 0.0
    for t in reversed(closed):
        entry = float(t["entry_price"] or 0)
        stop = float(t["stop_price"] or 0)
        pnl = round(_net_pnl(t), 2)
        risk = (entry - stop) * int(t["qty"] or 0)
        equity = round(equity + pnl, 2)
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
        trades.append({"symbol": t["symbol"], "pnl": pnl, "win": pnl > 0,
                       "r": round(pnl / risk, 2) if risk > 0 else 0.0,
                       "equity": equity})
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    stats = {
        "n": n, "wins": len(wins), "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
        "total_pnl": round(sum(t["pnl"] for t in trades), 2),
        "expectancy_r": round(sum(t["r"] for t in trades) / n, 2) if n else 0.0,
        "profit_factor": round(gw / gl, 2) if gl else (99.0 if gw else 0.0),
        "max_drawdown": round(dd, 2),
    }
    if n < 30:
        verdict = ("COLLECTING_EVIDENCE",
                   f"{n}/30 closed US trades — PAPER mein chalne do")
    elif stats["expectancy_r"] > 0 and stats["profit_factor"] >= 1.3:
        verdict = ("READY_CANDIDATE",
                   f"{n} trades: exp {stats['expectancy_r']:+.2f}R, PF "
                   f"{stats['profit_factor']:.2f} — par US LIVE ke liye broker chahiye")
    else:
        verdict = ("NOT_READY", f"{n} trades: exp {stats['expectancy_r']:+.2f}R "
                                f"— abhi paisa deserve nahi karta")
    return {"trades": trades, "stats": stats,
            "verdict": verdict[0], "verdict_reason": verdict[1]}
