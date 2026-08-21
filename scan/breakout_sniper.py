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
from core.market_clock import IST

from logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_watch: dict[int, dict] = {}       # token -> {symbol, trigger, stop, target}
_fired: dict[str, set] = {}        # {YYYY-MM-DD: {symbols}} telegram-acked
_autopilot_fed: dict[str, set] = {}  # symbol/day already handed to autopilot
_arm: dict[str, float] = {}        # symbol -> ts when it first CLEARED the level
_ticker = None
_owned_tickers: list = []
_started = False
_mode = "off"                 # off | owner | attached
_stopping = False
_last_tick_ts = 0.0
_ws_forbidden_until = 0.0
_SNIPER_STALE_S = 90.0
_WS_FORBIDDEN_BACKOFF_S = 120.0

# ── Confirmation thresholds (the false-break killer) ─────────────────────────
# A tick TOUCHING the level is not a breakout — the intraday false-break is
# what burns breakout traders (e.g. a wick to ₹182.5 that then keeps falling).
# We fire only when the break (a) CLEARS the level by a buffer and (b) HOLDS
# above it for a dwell window. A wick that reverses within the window disarms
# and never alerts. Both tunable via .env.
# Eased defaults (still block AUROPHARMA-style 0.1× thin tape / RSI blow-off).
# Override via .env if you want the older stricter desk.
_CLEARANCE_PCT = float(os.getenv("QT_SNIPER_CLEARANCE_PCT", "0.0010") or 0.0010)
_HOLD_SECONDS = float(os.getenv("QT_SNIPER_HOLD_SECONDS", "20") or 20)
# Volume pacing — a break on dead volume is a trap even if it holds. We
# require BOTH: (a) pace-aware absolute floor (default 0.7× scaled by session),
# and (b) at least keeping up with time-of-day pace. Zero / missing volume is
# NEVER suggested. 0.1× prints still fail closed.
_VOL_SURGE = float(os.getenv("QT_SNIPER_VOL_SURGE", "1.0") or 1.0)
_VOL_ABS_MIN = float(os.getenv("QT_SNIPER_VOL_ABS_MIN", "0.7") or 0.7)
_VOL_EARLY_FRAC = float(os.getenv("QT_SNIPER_VOL_EARLY_FRAC", "0.15") or 0.15)
# Quality gate — the sniper is a SEPARATE path from the main scanner.
# Blow-off RSI still hard-skips. Chase/extension is SOFT by default (watch
# still arms; scanner BUY demote remains) so Telegram fires like the earlier
# desk — set QT_SNIPER_SKIP_CHASE=1 to restore hard skip.
_RSI_BLOWOFF = float(os.getenv("QT_SNIPER_RSI_BLOWOFF", "82") or 82)
_SKIP_CHASE = (os.getenv("QT_SNIPER_SKIP_CHASE", "0") or "0").strip().lower() in (
    "1", "true", "yes", "on",
)
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
                    surge: float = _VOL_SURGE,
                    abs_min: float = _VOL_ABS_MIN,
                    early_frac: float = _VOL_EARLY_FRAC,
                    scan_volume_ratio: float = 0.0) -> bool:
    """Is today's volume strong enough for a real breakout?

    Fail-closed on zero/missing tick volume. Requires:
      1. Pace-aware absolute floor — need ≥ abs_min × max(session_frac, early_frac)
         of a full average day. Blocks AUROPHARMA-style 0.1× prints, but does
         NOT demand a full day's volume by 11am (that silenced every alert).
      2. After the open (frac ≥ 5%), also keep up with time-of-day pace
         (surge × frac; default surge 1.0 = on-pace is enough).

    If avg-day volume is missing but the scan already printed a real
    volume_ratio ≥ floor, that ratio is enough — including when the tick
    itself has no volume field (shared live-feed MODE_LTP).
    """
    if cum_vol <= 0:
        # Shared live-feed socket is MODE_LTP: last_price only, no
        # volume_traded. That is "field absent", not a 0-volume print.
        # Scan-day ratio (already required to arm the watch) is the tape.
        return float(scan_volume_ratio or 0) >= float(abs_min)
    if not avg_daily_vol or avg_daily_vol <= 0:
        return float(scan_volume_ratio or 0) >= float(abs_min)
    session_frac = max(0.0, min(1.0, float(frac)))
    day_ratio = cum_vol / avg_daily_vol
    # Early bar: insist on a meaningful open surge (≥ early_frac × abs_min of
    # avg day). Later: scale the floor with session progress up to abs_min.
    floor_frac = max(session_frac, max(0.05, float(early_frac)))
    abs_floor = float(abs_min) * floor_frac
    if day_ratio < abs_floor:
        return False
    if session_frac < 0.05:
        return True                   # absolute floor already cleared
    expected_by_now = avg_daily_vol * session_frac
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
                ahead of pace (0 / missing volume rejected; no fail-open).

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
        if not volume_confirms(
            cum_vol, float(w.get("avg_vol") or 0), frac,
            scan_volume_ratio=float(w.get("volume_ratio") or 0),
        ):
            continue                        # break on dead volume → skip
        out.append({**w, "ltp": ltp, "held_s": round(now - first),
                    "cum_vol": cum_vol})
    return out


def _quality_skip(r: dict) -> str:
    """Non-empty reason if this scan result is too low-quality to snipe —
    the sniper's version of the scanner's demote gates. '' = fine to watch.

    Known-thin tape (0 < volume_ratio < floor) and RSI blow-off still skip.
    Missing avg_vol20 is OK when scan-day volume_ratio already clears the
    floor — those "empty" product rows must still arm so Telegram fires.
    """
    if r.get("chase_risk") and _SKIP_CHASE:
        return "extended/chase-risk (already run hard, no clean base)"
    rsi = float(r.get("rsi") or 0)
    if rsi > _RSI_BLOWOFF:
        return f"RSI {rsi:.0f} — blow-off-top overbought"
    try:
        vratio = float(r.get("volume_ratio") or r.get("rvol") or 0)
    except (TypeError, ValueError):
        vratio = 0.0
    # Scan-day relative volume already thin → do not arm for a "confirmed" fire.
    if 0 < vratio < _VOL_ABS_MIN:
        return f"volume {vratio:.2f}× < {_VOL_ABS_MIN:.1f}× — thin tape, skip"
    avg_vol = float(r.get("avg_vol20") or 0)
    if avg_vol <= 0 and vratio < _VOL_ABS_MIN:
        return "0 / unknown volume — sniper will not watch"
    return ""


def build_watch_map(results: list[dict]) -> dict[int, dict]:
    """Token→level map from scan results (pre-breakout ≤2.5%%) + watchlist.
    Blow-off-top / thin-or-zero-volume names are skipped. Chase/extension is
    soft by default (still watched) so alerts fire; set QT_SNIPER_SKIP_CHASE=1
    to hard-skip extended names.

    Accepts both unified-scanner rows (categories/PreBreakout) and product
    scan-store records (signals/PRE_BREAKOUT, status Watch for breakout).
    """
    targets: dict[str, dict] = {}
    for r in results:
        cats = set(r.get("categories") or [])
        sigs = [str(x) for x in (r.get("signals") or [])]
        is_pre = (
            "PreBreakout" in cats
            or "PRE_BREAKOUT" in sigs
            or str(r.get("status") or "") == "Watch for breakout"
        )
        dist = r.get("pivot_distance_pct")
        try:
            dist_f = float(dist) if dist is not None else 99.0
        except (TypeError, ValueError):
            dist_f = 99.0
        # Product records without an explicit distance still qualify when the
        # scanner already labelled them pre-breakout / watch-for-breakout.
        if dist is None and is_pre:
            dist_f = 0.0
        if not (is_pre and 0 <= dist_f <= 2.5):
            continue
        if float(r.get("entry") or 0) <= 0:
            continue
        skip = _quality_skip(r)
        if skip:
            log.debug("sniper_quality_skip", symbol=r.get("symbol"), why=skip)
            continue
        sym = str(r.get("symbol") or "").upper()
        if not sym:
            continue
        try:
            vratio = float(r.get("volume_ratio") or r.get("rvol") or 0)
        except (TypeError, ValueError):
            vratio = 0.0
        targets[sym] = {
            "trigger": float(r.get("entry") or 0),
            "stop": float(r.get("stop") or 0),
            "target": float(r.get("target") or 0),
            "avg_vol": float(r.get("avg_vol20") or 0),
            "volume_ratio": vratio,
        }
    try:
        import sqlite3
        from pathlib import Path
        db = Path(__file__).resolve().parent.parent / "logs" / "watchlist.db"
        if db.exists():
            conn = sqlite3.connect(db)
            for sym, hi, stp, tgt in conn.execute(
                    "SELECT symbol, buy_zone_high, stop_price, target_price "
                    "FROM watchlist"):
                # Watchlist rows have no avg volume — do not sniper-suggest
                # them until a scan result supplies real volume evidence.
                if hi and sym not in targets:
                    log.debug("sniper_watchlist_skip_no_volume", symbol=sym)
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


def _sniper_telegram_text(hits: list[dict]) -> str:
    """Plain BREAKOUT CONFIRMED — extra fields optional, never required."""
    lines = ["🚨 <b>BREAKOUT CONFIRMED</b>"]
    for h in hits[:5]:
        plan = ""
        if h.get("stop") and h.get("target"):
            plan = f"\n   plan: stop ₹{h['stop']:,.0f} / target ₹{h['target']:,.0f}"
        held = h.get("held_s")
        hold_bit = f", {held}s tak upar ruka" if held else ""
        vol_bit = ""
        if h.get("cum_vol") and h.get("avg_vol"):
            vr = h["cum_vol"] / h["avg_vol"]
            vol_bit = f", volume {vr:.1f}× avg-day"
        elif h.get("volume_ratio"):
            vol_bit = f", volume {float(h['volume_ratio']):.1f}×"
        lines.append(f"\n<b>{h['symbol']}</b> ne ₹{h['trigger']:,.0f} toda "
                     f"(₹{h['ltp']:,.1f}{hold_bit}{vol_bit}){plan}")
    return "\n".join(lines)


def _send_sniper_telegram(fresh: list[dict]) -> bool:
    """Autonomy already delivers on this bot — never mute sniper on a stale
    'not configured' snapshot. Reload credentials and try the send."""
    msg = _sniper_telegram_text(fresh)
    try:
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        except Exception:
            pass
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if engine.send(msg):
            log.info("sniper_fired", symbols=[h["symbol"] for h in fresh])
            return True
        if engine.is_configured():
            log.warning("sniper_telegram_send_failed",
                        symbols=[h["symbol"] for h in fresh])
        else:
            log.warning("sniper_telegram_not_configured")
    except Exception as exc:
        log.warning("sniper_telegram_send_failed", error=str(exc),
                    symbols=[h["symbol"] for h in fresh])
    return False


def _alert(hits: list[dict] | None) -> None:
    if _stopping:
        return
    today = datetime.now(IST).strftime("%Y-%m-%d")
    with _lock:
        _fired.setdefault(today, set())
        _autopilot_fed.setdefault(today, set())
        for store in (_fired, _autopilot_fed):
            for k in list(store):
                if k != today:
                    del store[k]
        fresh = [h for h in hits if h["symbol"] not in _fired[today]]
    if not fresh:
        return
    try:
        sent = _send_sniper_telegram(fresh)
        if sent:
            with _lock:
                _fired.setdefault(today, set())
                _fired[today].update(h["symbol"] for h in fresh)
        ap_fresh = []
        with _lock:
            _autopilot_fed.setdefault(today, set())
            ap_fresh = [h for h in fresh if h["symbol"] not in _autopilot_fed[today]]
            _autopilot_fed[today].update(h["symbol"] for h in ap_fresh)
        if not ap_fresh:
            return

        def _feed_autopilot(hits_copy=list(ap_fresh)):
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


def ingest_ticks(ticks: list[dict] | None) -> None:
    """Fold ticks from whoever owns the process KiteTicker (live feed or sniper)."""
    global _last_tick_ts
    if _stopping:
        return
    batch = list(ticks or [])
    if not batch:
        return
    _last_tick_ts = time.time()
    try:
        from core.health import beat as _hb
        _hb("sniper", note=f"{len(batch)} ticks")
    except Exception:
        pass
    with _lock:
        watch = dict(_watch)
        today = datetime.now(IST).strftime("%Y-%m-%d")
        fired = set(_fired.get(today, set()))
    hits = process_ticks(batch, watch, fired)
    if hits:
        _alert(hits)


def _ws_forbidden(code, reason) -> bool:
    text = f"{code} {reason}".lower()
    return "403" in text or "forbidden" in text


def remember_ws_forbidden(code=None, reason: str = "") -> None:
    """Stop hammering Kite after a 403 upgrade. Attach to the live feed instead."""
    global _ws_forbidden_until, _started, _ticker, _mode
    _ws_forbidden_until = time.time() + _WS_FORBIDDEN_BACKOFF_S
    try:
        from data.kite_ws_slot import release_ticker
        release_ticker("sniper")
    except Exception:
        pass
    with _lock:
        _started = False
        _ticker = None
        _mode = "off"
    log.warning(
        "sniper_ws_403_backing_off",
        retry_in_s=int(_WS_FORBIDDEN_BACKOFF_S),
        code=code,
        reason=str(reason)[:80],
    )


def _silence_ticker(ticker) -> None:
    """Stop Autobahn retry then close. Never reactor.stop() — that kills the process loop."""
    if ticker is None:
        return
    for name in ("stop_retry", "close"):
        try:
            fn = getattr(ticker, name, None)
            if callable(fn):
                fn()
        except Exception:
            pass


def handle_ws_close(code=None, reason: str = "") -> None:
    """on_close from KiteTicker. Shutdown and 403 must not schedule a reconnect storm."""
    global _started, _ticker, _mode
    if _stopping:
        log.info("sniper_ws_closed_on_shutdown", code=code, reason=str(reason)[:80])
        return
    if _ws_forbidden(code, reason):
        remember_ws_forbidden(code, reason)
        return
    with _lock:
        _started = False
        _ticker = None
        _mode = "off"
    try:
        from data.kite_ws_slot import release_ticker
        release_ticker("sniper")
    except Exception:
        pass
    log.info("sniper_ws_closed_will_restart", code=code, reason=str(reason)[:80])


def stop_sniper() -> None:
    """Owner is stopping the stack. No new sockets, no Telegram, no kite reconnect."""
    global _stopping, _started, _ticker, _mode
    _stopping = True
    owned = []
    with _lock:
        if _ticker is not None:
            owned.append(_ticker)
        owned.extend(_owned_tickers)
        _owned_tickers.clear()
        _ticker = None
        _started = False
        _mode = "off"
    for ticker in owned:
        _silence_ticker(ticker)
    try:
        from data.kite_ws_slot import release_ticker
        release_ticker("sniper")
    except Exception:
        pass
    log.info("sniper_stopped")


def _attach_to_owner(owner: str) -> bool:
    global _ticker, _started, _mode, _last_tick_ts
    if _stopping:
        return False
    with _lock:
        _ticker = None
        _started = True
        _mode = "attached"
        _last_tick_ts = time.time()
    log.info("sniper_attached_to_existing_ticker", owner=owner)
    return True


def start_sniper() -> bool:
    """Start the tick stream (idempotent). False when Kite/ws unavailable.

    A socket that stays "started" but stops ticking during market hours is
    treated as dead — daily token expiry and silent WS hangs used to mute
    Telegram until process restart.

    Never opens a second KiteTicker while the autonomy live feed (or anyone
    else in this process) already owns the slot — that upgrade fails 403.

    ``stop_sniper()`` is for THIS shutdown (no reconnect storm). A later
    explicit start — Streamlit rerun, uvicorn reload, supervisor tick —
    must arm again. Otherwise Telegram breakouts die after every edit.
    """
    global _ticker, _started, _mode, _last_tick_ts, _stopping
    _stopping = False
    stale_ticker = None
    with _lock:
        if _started and _mode == "attached":
            trading = False
            try:
                from data.nse_live import _is_trading_now
                trading = bool(_is_trading_now())
            except Exception:
                trading = False
            if (
                trading
                and _last_tick_ts > 0
                and (time.time() - _last_tick_ts) > _SNIPER_STALE_S
            ):
                log.warning(
                    "sniper_attached_stale",
                    idle_s=int(time.time() - _last_tick_ts),
                )
            return True
        if _started:
            trading = False
            try:
                from data.nse_live import _is_trading_now
                trading = bool(_is_trading_now())
            except Exception:
                trading = False
            hung = _ticker is None or (
                trading
                and _last_tick_ts > 0
                and (time.time() - _last_tick_ts) > _SNIPER_STALE_S
            )
            if not hung:
                return True
            stale_ticker = _ticker
            _ticker = None
            _started = False
            _mode = "off"
            log.warning("sniper_stale_restart", idle_s=int(time.time() - _last_tick_ts) if _last_tick_ts else 0)
    if stale_ticker is not None:
        try:
            stale_ticker.close()
        except Exception:
            pass
        try:
            from data.kite_ws_slot import release_ticker
            release_ticker("sniper")
        except Exception:
            pass
    try:
        from execution.trade_executor import kite_ready
        if not kite_ready():
            return False
        from data.kite_ws_slot import claim_ticker, ticker_owner

        owner = ticker_owner()
        if owner and owner != "sniper":
            return _attach_to_owner(owner)
        if time.time() < _ws_forbidden_until:
            log.info("sniper_ws_403_wait", retry_in_s=int(_ws_forbidden_until - time.time()))
            return False
        if not claim_ticker("sniper"):
            return _attach_to_owner(ticker_owner() or "other")
        from data.kite_client import KiteClient

        def on_connect(ws, response):
            if _stopping:
                _silence_ticker(ws)
                return
            with _lock:
                tokens = list(_watch)
            if tokens:
                ws.subscribe(tokens)
                ws.set_mode(ws.MODE_QUOTE, tokens)
            log.info("sniper_connected", watching=len(tokens))

        def on_close(ws, code, reason):
            handle_ws_close(code, reason)

        def on_error(ws, code, reason):
            if _stopping:
                return
            if _ws_forbidden(code, reason):
                remember_ws_forbidden(code, reason)

        def on_ticks(ws, ticks):
            ingest_ticks(ticks)

        kws = KiteClient().get_ticker(
            on_ticks, on_connect, on_close, on_error, reconnect=False,
        )
        kws.connect(threaded=True)
        if _stopping:
            _silence_ticker(kws)
            return False
        _ticker = kws
        _owned_tickers.append(kws)
        with _lock:
            _started = True
            _mode = "owner"
            _last_tick_ts = time.time()
        log.info("sniper_started")
        return True
    except Exception as exc:
        log.debug("sniper_start_failed", error=str(exc))
        try:
            from data.kite_ws_slot import release_ticker
            release_ticker("sniper")
        except Exception:
            pass
        return False


try:
    import atexit
    atexit.register(stop_sniper)
except Exception:
    pass
