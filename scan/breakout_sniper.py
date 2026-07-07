"""
Breakout Sniper — pivot cross hote hi USI SECOND alert.

The 15-min scan cycle can be up to 15 minutes late on a breakout.
This module subscribes a Kite WebSocket tick stream to the stocks
that matter — pre-breakout candidates (within 2.5%% of pivot) and
watchlist entries — and fires the moment a tick crosses the level:

  🚨 KAYNES ne ₹4,250 ka pivot ABHI toda (₹4,251.5) —
     plan: stop ₹4,100 / target ₹4,550

Design for safety:
  - pure trigger logic (process_ticks) is unit-tested; the WebSocket
    wrapper is a thin shell around it
  - one alert per symbol per day; auto-rebuilds the watch map after
    every scan; inert without Kite login or outside market hours
"""
from __future__ import annotations

import threading
from datetime import datetime

from logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_watch: dict[int, dict] = {}       # token -> {symbol, trigger, stop, target}
_fired: dict[str, set] = {}        # {YYYY-MM-DD: {symbols}}
_ticker = None
_started = False


# ── Pure, testable core ───────────────────────────────────────────────────────

def process_ticks(ticks: list[dict], watch: dict[int, dict],
                  already_fired: set) -> list[dict]:
    """[{symbol, trigger, ltp, stop, target}] for every fresh breakout tick."""
    out = []
    for t in ticks:
        token = t.get("instrument_token")
        ltp = t.get("last_price")
        w = watch.get(token)
        if not w or not ltp:
            continue
        if w["symbol"] in already_fired:
            continue
        if float(ltp) >= float(w["trigger"]):
            out.append({**w, "ltp": float(ltp)})
    return out


def build_watch_map(results: list[dict]) -> dict[int, dict]:
    """Token→level map from scan results (pre-breakout ≤2.5%%) + watchlist."""
    targets: dict[str, dict] = {}
    for r in results:
        if "PreBreakout" in (r.get("categories") or []) \
                and 0 < (r.get("pivot_distance_pct") or 99) <= 2.5:
            targets[r["symbol"]] = {"trigger": float(r.get("entry") or 0),
                                    "stop": float(r.get("stop") or 0),
                                    "target": float(r.get("target") or 0)}
    try:
        import sqlite3
        from pathlib import Path
        db = Path(__file__).resolve().parent.parent / "logs" / "watchlist.db"
        if db.exists():
            conn = sqlite3.connect(db)
            for sym, hi, stp, tgt in conn.execute(
                    "SELECT symbol, buy_zone_high, stop_price, target_price "
                    "FROM watchlist"):
                if hi and sym not in targets:
                    targets[sym] = {"trigger": float(hi),
                                    "stop": float(stp or 0),
                                    "target": float(tgt or 0)}
            conn.close()
    except Exception:
        pass
    if not targets:
        return {}
    try:
        from data.instruments import InstrumentManager
        tokens = InstrumentManager().tokens_for(list(targets))
        return {tok: {"symbol": sym, **targets[sym]}
                for sym, tok in tokens.items() if tok}
    except Exception as exc:
        log.debug("sniper_tokens_failed", error=str(exc))
        return {}


def _alert(hits: list[dict]) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    with _lock:
        _fired.setdefault(today, set())
        for k in list(_fired):
            if k != today:
                del _fired[k]
        fresh = [h for h in hits if h["symbol"] not in _fired[today]]
        _fired[today].update(h["symbol"] for h in fresh)
    if not fresh:
        return
    try:
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if not engine.is_configured():
            return
        lines = ["🚨 <b>BREAKOUT ABHI HUA</b>"]
        for h in fresh[:5]:
            plan = ""
            if h.get("stop") and h.get("target"):
                plan = f"\n   plan: stop ₹{h['stop']:,.0f} / target ₹{h['target']:,.0f}"
            lines.append(f"\n<b>{h['symbol']}</b> ne ₹{h['trigger']:,.0f} toda "
                         f"(₹{h['ltp']:,.1f}){plan}")
        engine.send("\n".join(lines))
        log.info("sniper_fired", symbols=[h["symbol"] for h in fresh])
    except Exception as exc:
        log.debug("sniper_alert_failed", error=str(exc))


# ── WebSocket shell ───────────────────────────────────────────────────────────

def refresh_watch(results: list[dict]) -> int:
    """Rebuild the watch map after each scan; (re)subscribe if live."""
    global _watch
    new_map = build_watch_map(results)
    with _lock:
        _watch = new_map
    if _ticker is not None and new_map:
        try:
            _ticker.subscribe(list(new_map))
            _ticker.set_mode(_ticker.MODE_LTP, list(new_map))
        except Exception:
            pass
    if new_map:
        log.info("sniper_watching", symbols=len(new_map))
    return len(new_map)


def start_sniper() -> bool:
    """Start the tick stream (idempotent). False when Kite/ws unavailable."""
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
                watch = dict(_watch)
                today = datetime.now().strftime("%Y-%m-%d")
                fired = set(_fired.get(today, set()))
            hits = process_ticks(ticks, watch, fired)
            if hits:
                _alert(hits)

        def on_connect(ws, response):
            with _lock:
                tokens = list(_watch)
            if tokens:
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_LTP, tokens)
            log.info("sniper_connected", watching=len(tokens))

        def on_close(ws, code, reason):
            # Mark dead so the next scan cycle restarts with a FRESH token —
            # otherwise a daily token expiry kills the sniper until app restart.
            global _started, _ticker
            with _lock:
                _started = False
                _ticker = None
            log.info("sniper_ws_closed_will_restart", code=code,
                     reason=str(reason)[:80])

        kws = KiteClient().get_ticker(on_ticks, on_connect, on_close)
        kws.connect(threaded=True)
        _ticker = kws
        with _lock:
            _started = True
        log.info("sniper_started")
        return True
    except Exception as exc:
        log.debug("sniper_start_failed", error=str(exc))
        return False
