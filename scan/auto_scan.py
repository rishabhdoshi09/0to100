"""
Auto-Scan — background full-market scanner.

Starts once per process. A daemon thread scans the ENTIRE NSE universe
with the UnifiedScanner, stores results in a module-level store, and
refreshes every 15 minutes during market hours (hourly otherwise).

The UI reads the store instantly — no waiting. Every BUY signal is
logged to the signal-outcome tracker so accuracy is measured on real
outcomes, not opinions.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_results: list[dict] = []
_scanned_count: int = 0
_last_scan_ts: float = 0.0
_status: str = "idle"          # idle | scanning | ready | error
_thread_started: bool = False

_MARKET_REFRESH_S = 900        # 15 min during market hours
_OFFHOURS_REFRESH_S = 3600     # hourly otherwise


def _is_market_hours() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= minutes <= 15 * 60 + 30


def _serialize(r) -> dict:
    return {
        "symbol": r.symbol, "price": r.price, "change_pct": r.change_pct,
        "momentum_5d": r.momentum_5d, "volume_ratio": r.volume_ratio,
        "signals": r.signal_labels, "categories": sorted(r.categories),
        "reasons": r.reasons, "score": r.score, "verdict": r.verdict,
        "entry": r.entry, "stop": r.stop, "target": r.target,
        "rr": round(r.risk_reward, 1),
        "pivot_distance_pct": getattr(r, "pivot_distance_pct", 0.0),
    }


# {YYYY-MM-DD: set of symbols already pushed} — one alert per stock per day
_pushed: dict[str, set] = {}


def _push_new_setups(picks: list[dict]) -> None:
    """Proactively send fresh BUY/STRONG BUY setups to Telegram. Silent if unconfigured."""
    try:
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if not engine.is_configured():
            return
        today = datetime.now().strftime("%Y-%m-%d")
        _pushed.setdefault(today, set())
        # Drop stale day keys
        for k in list(_pushed):
            if k != today:
                del _pushed[k]

        fresh = [p for p in picks
                 if p["verdict"] in ("STRONG BUY", "BUY")
                 and p["symbol"] not in _pushed[today]][:5]

        # Pre-breakout watch: stock pivot ke 2.5% andar, breakout abhi hua nahi
        pre = [p for p in picks
               if "PreBreakout" in p.get("categories", [])
               and 0 < p.get("pivot_distance_pct", 0) <= 2.5
               and p["symbol"] not in _pushed[today]
               and p not in fresh][:4]

        if not fresh and not pre:
            return

        lines = []
        if fresh:
            lines.append("🎯 <b>QuantTerm — naye setups mile</b>")
            for p in fresh:
                emoji = "🔥" if p["verdict"] == "STRONG BUY" else "⚡"
                why = p.get("checks") or [f"✓ {r}" for r in p.get("reasons", [])]
                why_txt = "\n".join(f"   {w}" for w in why[:3])
                size_line = ""
                try:
                    from risk.position_sizer import size_position
                    ps = size_position(float(p["entry"]), float(p["stop"]))
                    if ps["qty"] >= 1:
                        size_line = (f"\n   📏 {ps['qty']} shares · max loss "
                                     f"₹{ps['max_loss']:,.0f} (1% rule)")
                except Exception:
                    pass
                lines.append(
                    f"\n{emoji} <b>{p['symbol']}</b> ₹{p['price']:,.1f} — {p['verdict']}\n"
                    f"{why_txt}\n"
                    f"   Entry ₹{p['entry']:,.0f} · Stop ₹{p['stop']:,.0f} · Target ₹{p['target']:,.0f}"
                    f"{size_line}"
                )
        if pre:
            lines.append("\n⏳ <b>Breakout ke kareeb — nazar rakho</b>")
            for p in pre:
                first_reason = (p.get("reasons") or [""])[0]
                lines.append(
                    f"\n👀 <b>{p['symbol']}</b> ₹{p['price']:,.1f} — "
                    f"pivot ₹{p['entry']:,.0f} se {p['pivot_distance_pct']:.1f}% neeche\n"
                    f"   {first_reason}\n"
                    f"   Breakout confirm hone par hi entry — stop ₹{p['stop']:,.0f}"
                )
        if engine.send("\n".join(lines)):
            _pushed[today].update(p["symbol"] for p in fresh + pre)
            log.info("setups_pushed_to_telegram", buys=len(fresh), prebreakout=len(pre))
    except Exception as exc:
        log.debug("push_setups_skip", error=str(exc))


def _log_buys_for_tracking(results) -> None:
    """Feed BUY signals into the outcome tracker (dedupes per day itself)."""
    try:
        from core.signal_outcome_tracker import log_signal
        for r in results:
            if r.verdict != "BUY":
                continue
            log_signal(
                symbol=r.symbol, signal_type="UNIFIED_BUY",
                entry_price=r.entry, pivot_price=r.entry,
                stop_price=r.stop, target_price=r.target,
                quality_score=r.score, accum_score=0.0,
                archetype="|".join(r.signals), regime="",
            )
    except Exception as exc:
        log.debug("auto_scan_tracking_skip", error=str(exc))


def _scan_once(universe: Optional[list[str]] = None, progress=None) -> None:
    global _results, _scanned_count, _last_scan_ts, _status
    with _lock:
        _status = "scanning"
    try:
        from scan.unified_scanner import UnifiedScanner
        if universe is None:
            from data.nse_universe import get_nse_universe
            universe = get_nse_universe()
        raw = UnifiedScanner(max_workers=8).scan(universe, progress=progress)
        _log_buys_for_tracking(raw)
        serialized = [_serialize(r) for r in raw]
        # Sector heat — packs of 3+ signals in one sector boost each other
        try:
            from scan.sector_heat import apply_sector_heat
            apply_sector_heat(serialized)
            serialized.sort(key=lambda r: r.get("score", 0), reverse=True)
        except Exception as exc:
            log.debug("sector_heat_skip", error=str(exc))
        # JARVIS conviction layer — news buzz + earnings evidence on top picks
        try:
            from scan.conviction import build_conviction
            serialized = build_conviction(serialized)
        except Exception as exc:
            log.debug("conviction_skip", error=str(exc))
        # Live price overlay from Google Finance (history is official NSE EOD)
        try:
            from data.google_finance import get_quotes
            top_syms = [r["symbol"] for r in serialized[:60]]
            live = get_quotes(top_syms)
            overlaid = 0
            for r in serialized[:60]:
                q = live.get(r["symbol"])
                if not (q and q.get("price")):
                    continue
                r["price"] = q["price"]
                r["change_pct"] = q["chg_pct"]
                overlaid += 1
                # EOD signal sanity vs live price: if the stock has already
                # slipped well below its entry/pivot, the setup is broken —
                # demote and warn instead of showing a stale Buy.
                entry = float(r.get("entry") or 0)
                if entry and q["price"] < entry * 0.97:
                    slip = (entry - q["price"]) / entry * 100
                    r.setdefault("reasons", []).insert(
                        0, f"⚠ Live price ₹{q['price']:,.0f} — entry ₹{entry:,.0f} "
                           f"se {slip:.0f}% neeche aa gaya, setup abhi valid nahi")
                    if r.get("verdict") in ("STRONG BUY", "BUY"):
                        r["verdict"] = "WATCH"
                    if r.get("checks"):
                        r["checks"].insert(
                            0, f"⚠ Live ₹{q['price']:,.0f} entry se {slip:.0f}% neeche "
                               f"— pullback mein hai, chase mat karo")
            log.info("live_overlay_done", overlaid=overlaid, of=min(60, len(serialized)))
        except Exception as exc:
            log.warning("live_overlay_failed", error=str(exc))
        # Proactive delivery — user ko dhundhna na pade, setups khud pahunchein
        _push_new_setups(serialized[:15])
        with _lock:
            _results = serialized
            _scanned_count = len(universe)
            _last_scan_ts = time.time()
            _status = "ready"
        log.info("auto_scan_complete", universe=len(universe), signals=len(serialized))
    except Exception as exc:
        with _lock:
            _status = "error" if not _results else "ready"
        log.warning("auto_scan_failed", error=str(exc))


_auto_enabled: bool = True


def set_auto_enabled(enabled: bool) -> None:
    """User control: pause/resume the automatic background refresh."""
    global _auto_enabled
    with _lock:
        _auto_enabled = enabled
    log.info("auto_scan_toggled", enabled=enabled)


def is_auto_enabled() -> bool:
    with _lock:
        return _auto_enabled


_pulse_pushed_date: str = ""


def _maybe_push_morning_pulse() -> None:
    """Once per weekday, 8:30-10:00 window: full Daily Pulse to Telegram."""
    global _pulse_pushed_date
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if now.weekday() >= 5 or _pulse_pushed_date == today:
        return
    if not (8 * 60 + 30 <= now.hour * 60 + now.minute <= 10 * 60):
        return
    try:
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if not engine.is_configured():
            return
        from reports.street_pulse import build_pulse, pulse_to_telegram
        if engine.send(pulse_to_telegram(build_pulse())):
            _pulse_pushed_date = today
            log.info("morning_pulse_pushed")
    except Exception as exc:
        log.debug("morning_pulse_skip", error=str(exc))


def _worker() -> None:
    while True:
        if is_auto_enabled():
            _scan_once()
            _maybe_push_morning_pulse()
            time.sleep(_MARKET_REFRESH_S if _is_market_hours() else _OFFHOURS_REFRESH_S)
        else:
            time.sleep(30)   # paused by user — just idle and re-check


def start_background_scan() -> None:
    """Idempotent — starts the daemon scanner thread once per process."""
    global _thread_started
    with _lock:
        if _thread_started:
            return
        _thread_started = True
    t = threading.Thread(target=_worker, name="auto-scan", daemon=True)
    t.start()
    log.info("auto_scan_started")


def force_rescan() -> None:
    """Trigger an immediate rescan in the background (non-blocking)."""
    threading.Thread(target=_scan_once, name="auto-scan-force", daemon=True).start()


def run_manual_scan(universe: Optional[list[str]] = None, progress=None) -> list[dict]:
    """
    User-controlled scan: runs synchronously (blocks until done) so the UI
    can show live progress, then returns the fresh results. Also updates
    the shared store so Dashboard/Telegram stay in sync.
    """
    _scan_once(universe=universe, progress=progress)
    with _lock:
        return list(_results)


def get_results() -> tuple[list[dict], int, float, str]:
    """Returns (results, universe_size, last_scan_unix_ts, status)."""
    with _lock:
        return list(_results), _scanned_count, _last_scan_ts, _status
