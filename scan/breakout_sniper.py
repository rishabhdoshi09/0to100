"""
Breakout Sniper — pivot cross hote hi USI SECOND alert.

The 15-min scan cycle can be up to 15 minutes late on a breakout.
This module subscribes a Kite WebSocket tick stream to the stocks
that matter — pre-breakout candidates (within 2.5%% of pivot) and
watchlist entries — and fires when a break CLEARS the level and HOLDS:

  🚨 KAYNES ne ₹4,250 ka pivot toda (₹4,268, 45s tak upar ruka) —
     plan: stop ₹4,100 / target ₹4,550

Design for safety:
  - two-stage confirmation kills the intraday false-break: a wick that
    pokes the level then reverses disarms and never alerts; only a
    break that clears the level and holds for the dwell window fires
  - pure trigger logic (process_ticks) is unit-tested; the WebSocket
    wrapper is a thin shell around it
  - one alert per symbol per day; auto-rebuilds the watch map after
    every scan; inert without Kite login or outside market hours
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime

from logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_watch: dict[int, dict] = {}       # token -> {symbol, trigger, stop, target}
_fired: dict[str, set] = {}        # {YYYY-MM-DD: {symbols}}
_arm: dict[str, float] = {}        # symbol -> ts when it first CLEARED the level
_ticker = None
_started = False

# ── Confirmation thresholds (the false-break killer) ─────────────────────────
# A tick TOUCHING the level is not a breakout — the intraday false-break is
# what burns breakout traders (e.g. a wick to ₹182.5 that then keeps falling).
# We fire only when the break (a) CLEARS the level by a buffer and (b) HOLDS
# above it for a dwell window. A wick that reverses within the window disarms
# and never alerts. Both tunable via .env.
_CLEARANCE_PCT = float(os.getenv("QT_SNIPER_CLEARANCE_PCT", "0.0015") or 0.0015)
_HOLD_SECONDS = float(os.getenv("QT_SNIPER_HOLD_SECONDS", "45") or 45)
# Volume pacing — a break on dead volume is a trap even if it holds. We
# require the day's cumulative volume to run AHEAD of its normal pace for
# the time of day. Fail-open: unknown avg volume or very early session →
# lean on the hold alone (never block for missing data).
_VOL_SURGE = float(os.getenv("QT_SNIPER_VOL_SURGE", "1.2") or 1.2)
_MKT_OPEN_MIN = 9 * 60 + 15
_MKT_CLOSE_MIN = 15 * 60 + 30
_MKT_MINUTES = _MKT_CLOSE_MIN - _MKT_OPEN_MIN


def day_fraction(now_dt=None) -> float:
    """Fraction of the trading session elapsed (0..1), IST."""
    try:
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        now_dt = now_dt or datetime.now(ist)
        mins = now_dt.hour * 60 + now_dt.minute
        return max(0.0, min(1.0, (mins - _MKT_OPEN_MIN) / _MKT_MINUTES))
    except Exception:
        return 1.0                    # unknown → don't gate on volume pace


def volume_confirms(cum_vol: float, avg_daily_vol: float, frac: float,
                    surge: float = _VOL_SURGE) -> bool:
    """Is today's volume running hot enough for a real breakout? Compares
    cumulative volume to the pace expected by this point in the day.
    Fail-open when data is missing or it's too early to judge."""
    if not avg_daily_vol or avg_daily_vol <= 0 or frac < 0.05:
        return True                   # not enough info → don't block
    expected_by_now = avg_daily_vol * frac
    return cum_vol >= surge * expected_by_now


# ── Pure, testable core ───────────────────────────────────────────────────────

def process_ticks(ticks: list[dict], watch: dict[int, dict],
                  already_fired: set, arm_state: dict | None = None,
                  now: float | None = None,
                  hold_seconds: float = _HOLD_SECONDS,
                  frac: float | None = None) -> list[dict]:
    """Confirmed breakouts only. Three checks per symbol:

      ARM     — price clears trigger × (1 + clearance). Timestamp recorded.
      DISARM  — price falls back below trigger → false poke, forgotten.
      CONFIRM — still cleared AND held ≥ hold_seconds AND volume running
                ahead of pace (dead-volume breaks rejected; fail-open when
                avg volume unknown).

    arm_state persists across calls (like already_fired). Returns the
    confirmed breakouts only — a wick or a low-volume poke never appears."""
    arm_state = _arm if arm_state is None else arm_state
    now = time.time() if now is None else now
    frac = day_fraction() if frac is None else frac
    out = []
    for t in ticks:
        token = t.get("instrument_token")
        ltp = t.get("last_price")
        w = watch.get(token)
        if not w or not ltp:
            continue
        sym = w["symbol"]
        if sym in already_fired:
            continue
        ltp = float(ltp)
        trigger = float(w["trigger"])
        if trigger <= 0:
            continue
        cleared = trigger * (1 + _CLEARANCE_PCT)
        if ltp < trigger:
            arm_state.pop(sym, None)        # fell back below → false poke, reset
            continue
        if ltp < cleared:
            continue                        # above level but not cleared yet
        first = arm_state.get(sym)
        if first is None:
            arm_state[sym] = now            # armed — start the hold clock
            continue
        if now - first < hold_seconds:      # not held long enough yet
            continue
        cum_vol = float(t.get("volume_traded") or t.get("volume") or 0)
        if not volume_confirms(cum_vol, float(w.get("avg_vol") or 0), frac):
            continue                        # break on dead volume → skip
        out.append({**w, "ltp": ltp, "held_s": round(now - first),
                    "cum_vol": cum_vol})
    return out


def build_watch_map(results: list[dict]) -> dict[int, dict]:
    """Token→level map from scan results (pre-breakout ≤2.5%%) + watchlist."""
    targets: dict[str, dict] = {}
    for r in results:
        if "PreBreakout" in (r.get("categories") or []) \
                and 0 < (r.get("pivot_distance_pct") or 99) <= 2.5:
            targets[r["symbol"]] = {"trigger": float(r.get("entry") or 0),
                                    "stop": float(r.get("stop") or 0),
                                    "target": float(r.get("target") or 0),
                                    "avg_vol": float(r.get("avg_vol20") or 0)}
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
        lines = ["🚨 <b>BREAKOUT CONFIRMED</b>"]
        for h in fresh[:5]:
            plan = ""
            if h.get("stop") and h.get("target"):
                plan = f"\n   plan: stop ₹{h['stop']:,.0f} / target ₹{h['target']:,.0f}"
            held = h.get("held_s")
            hold_bit = f", {held}s tak upar ruka" if held else ""
            vol_bit = ""
            if h.get("cum_vol") and h.get("avg_vol"):
                vr = h["cum_vol"] / h["avg_vol"]
                vol_bit = f", volume {vr:.1f}× (pace se aage)"
            lines.append(f"\n<b>{h['symbol']}</b> ne ₹{h['trigger']:,.0f} toda "
                         f"(₹{h['ltp']:,.1f}{hold_bit}{vol_bit}){plan}")
        engine.send("\n".join(lines))
        log.info("sniper_fired", symbols=[h["symbol"] for h in fresh])
        # 🤖 Autopilot hook — off-thread, tick stream kabhi block nahi hota
        def _feed_autopilot(hits_copy=list(fresh)):
            try:
                from execution.autopilot import on_breakout
                for h in hits_copy:
                    on_breakout(h)
            except Exception as exc:
                log.debug("autopilot_breakout_skip", error=str(exc))
        threading.Thread(target=_feed_autopilot, daemon=True).start()
    except Exception as exc:
        log.debug("sniper_alert_failed", error=str(exc))


# ── WebSocket shell ───────────────────────────────────────────────────────────

def refresh_watch(results: list[dict]) -> int:
    """Rebuild the watch map after each scan; (re)subscribe if live."""
    global _watch
    new_map = build_watch_map(results)
    with _lock:
        _watch = new_map
        # prune arm-state for symbols no longer watched
        _watched_syms = {v["symbol"] for v in new_map.values()}
        for _s in [s for s in _arm if s not in _watched_syms]:
            _arm.pop(_s, None)
    if _ticker is not None and new_map:
        try:
            _ticker.subscribe(list(new_map))
            _ticker.set_mode(_ticker.MODE_QUOTE, list(new_map))
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
            try:
                from core.health import beat as _hb
                _hb("sniper", note=f"{len(ticks)} ticks")
            except Exception:
                pass
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
                ws.set_mode(ws.MODE_QUOTE, tokens)
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
