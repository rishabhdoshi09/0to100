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

import os
import threading
import time
from datetime import datetime

# Edge VETO threshold — only demote a BUY→WATCH when the backtest says the
# combo is a genuine LOSER, not merely NEUTRAL. The old -0.02 cutoff sat INSIDE
# the backtest's own NEUTRAL band (signal_verdict: -0.05..+0.05 = NEUTRAL, only
# ≤-0.05 = LOSER), so it vetoed noise-level-negative combos — and since most
# breakout/momentum combos measure in that band, nearly EVERY setup got
# stamped "negative edge — skip". A point estimate of -0.04R over ~30-100
# trades is within one standard error of zero — indistinguishable from
# breakeven. So veto only a PROVEN loser (≤ LOSER line). Env-tunable.
_EDGE_VETO_R = float(os.getenv("QT_EDGE_VETO_R", "-0.05") or -0.05)


def _edge_vetoes(edge, verdict: str, threshold: float = _EDGE_VETO_R) -> bool:
    """True iff a measured combo edge is a PROVEN loser that should demote a
    BUY→WATCH. A NEUTRAL/breakeven edge (within measurement noise of zero, i.e.
    above the LOSER line) never vetoes — it just doesn't earn a BUY on its own.
    None edge (no evidence) never vetoes either."""
    return (edge is not None and float(edge) <= threshold
            and verdict in ("STRONG BUY", "BUY"))

from core.market_clock import IST
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_results: list[dict] = []
_scanned_count: int = 0
_last_scan_ts: float = 0.0
_status: str = "idle"          # idle | scanning | ready | error
_thread_started: bool = False

# 15 min is a DECISION, not a leftover: instant breakouts are the
# sniper's job (WebSocket ticks); this scan's job is structural —
# patterns, sector heat, conviction — and patterns don't form in 5
# minutes. Faster cycles just 3× the NSE-snapshot/news load (public
# API block risk) and the laptop's duty cycle for marginal value.
# Tune via .env QT_SCAN_REFRESH_S if ever needed — not via code edits.
import os as _os
_MARKET_REFRESH_S = int(_os.getenv("QT_SCAN_REFRESH_S", "900") or 900)
_OFFHOURS_REFRESH_S = 3600     # hourly otherwise


def _is_market_hours() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= minutes <= 15 * 60 + 30


def tag_conviction(serialized: list[dict]) -> None:
    """In-place: har result ko conviction_rank + high_conviction tag.

    high_conviction is an EVIDENCE tier, not a vibe tier: buy verdict
    AND calibrated score ≥ 75 AND measured backtest edge ≥ +0.10R.
    No measured edge (edge_r missing) = not high conviction, period.
    """
    _vr = {"STRONG BUY": 2, "BUY": 1}
    for r in serialized:
        edge = float(r.get("edge_r", 0) or 0)
        score = float(r.get("score", 0) or 0)
        r["conviction_rank"] = round(
            _vr.get(r.get("verdict"), 0) * 100 + score + edge * 40, 1)
        r["high_conviction"] = bool(
            r.get("verdict") in ("STRONG BUY", "BUY")
            and score >= 75
            and float(r.get("edge_r") or 0) >= 0.10)


def _serialize(r) -> dict:
    return {
        "symbol": r.symbol, "price": r.price, "change_pct": r.change_pct,
        "momentum_5d": r.momentum_5d, "volume_ratio": r.volume_ratio,
        "rsi": getattr(r, "rsi", 0.0),
        "signals": r.signal_labels, "categories": sorted(r.categories),
        "reasons": r.reasons, "score": r.score, "verdict": r.verdict,
        "entry": r.entry, "stop": r.stop, "target": r.target,
        "rr": round(r.risk_reward, 1),
        "pivot_distance_pct": getattr(r, "pivot_distance_pct", 0.0),
        "breakout_grade": getattr(r, "breakout_grade", ""),
        "breakout_conviction": getattr(r, "breakout_conviction", 0.0),
        "avg_vol20": getattr(r, "avg_vol20", 0.0),
        "above_sma50": bool(getattr(r, "above_sma50", False)),
        "above_sma200": bool(getattr(r, "above_sma200", False)),
        "chase_risk": bool(getattr(r, "chase_risk", False)),
    }


# {YYYY-MM-DD: set of symbols already pushed} — one alert per stock per day
_pushed: dict[str, set] = {}

# ── Restart persistence — scan results + push-dedupe survive restarts ─────────
from pathlib import Path as _Path
_STATE_FILE = _Path(__file__).resolve().parent.parent / "logs" / "scan_store.json"


def _save_state() -> None:
    """Atomic dump of results + dedupe so a restart resumes, not forgets."""
    import json
    try:
        with _lock:
            data = {
                "ts": _last_scan_ts,
                "count": _scanned_count,
                "results": _results[:400],          # cap file size
                "pushed": {k: sorted(v) for k, v in _pushed.items()},
            }
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(_STATE_FILE)
    except Exception as exc:
        log.debug("scan_state_save_failed", error=str(exc))


def _load_state() -> None:
    """Warm the store from disk at startup — results show instantly with
    their honest age instead of a blank 'pehla scan chal raha hai'."""
    global _results, _scanned_count, _last_scan_ts, _status, _pushed
    import json
    try:
        if not _STATE_FILE.exists():
            return
        data = json.loads(_STATE_FILE.read_text())
        if not data.get("results"):
            return
        # Stale beyond 3 days → ignore (weekend ke baad Monday tak theek hai)
        if time.time() - float(data.get("ts") or 0) > 3 * 86400:
            return
        with _lock:
            if _results:
                return                      # live scan already beat us to it
            _results = data["results"]
            _scanned_count = int(data.get("count") or 0)
            _last_scan_ts = float(data.get("ts") or 0)
            _status = "ready"
            _pushed = {k: set(v) for k, v in (data.get("pushed") or {}).items()}
        log.info("scan_state_restored", results=len(data["results"]),
                 age_min=int((time.time() - _last_scan_ts) / 60))
    except Exception as exc:
        log.debug("scan_state_load_failed", error=str(exc))


def _push_new_setups(picks: list[dict]) -> None:
    """Proactively send fresh BUY/STRONG BUY setups to Telegram. Silent if unconfigured."""
    try:
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if not engine.is_configured():
            return
        today = datetime.now(IST).strftime("%Y-%m-%d")
        _pushed.setdefault(today, set())
        # Drop stale day keys
        for k in list(_pushed):
            if k != today:
                del _pushed[k]

        _cands = [p for p in picks
                  if p["verdict"] in ("STRONG BUY", "BUY")
                  and p["symbol"] not in _pushed[today]]
        # 💎 PRIME first — har data-layer ke survivors sabse upar
        _cands.sort(key=lambda p: (bool(p.get("prime")),
                                   float(p.get("conviction_rank")
                                         or p.get("score") or 0)),
                    reverse=True)
        fresh = _cands[:5]

        # Pre-breakout watch: stock pivot ke 2.5% andar, breakout abhi hua nahi
        pre = [p for p in picks
               if "PreBreakout" in p.get("categories", [])
               and 0 < p.get("pivot_distance_pct", 0) <= 2.5
               and p["symbol"] not in _pushed[today]
               and p not in fresh][:4]

        if not fresh and not pre:
            return

        # ₹-book plan context (autopilot ka profit_book_rupees, agar set hai)
        _book_amt = 0.0
        try:
            from execution.autopilot import get_status as _ap_status
            _book_amt = float(_ap_status().get("profit_book_rupees") or 0)
        except Exception:
            _book_amt = 0.0

        lines = []
        if fresh:
            lines.append("🎯 <b>QuantTerm — naye setups mile</b>")
            for p in fresh:
                emoji = "🔥" if p["verdict"] == "STRONG BUY" else "⚡"
                prime_bit = ""
                if p.get("prime"):
                    emoji = "💎"
                    prime_bit = (f"\n   💎 PRIME — {p.get('prime_why', '')}")
                why = p.get("checks") or [f"✓ {r}" for r in p.get("reasons", [])]
                why_txt = "\n".join(f"   {w}" for w in why[:3])
                size_line = ""
                try:
                    from risk.position_sizer import size_position
                    ps = size_position(float(p["entry"]), float(p["stop"]))
                    if ps["qty"] >= 1:
                        size_line = (f"\n   📏 {ps['qty']} shares · max loss "
                                     f"₹{ps['max_loss']:,.0f} (1% rule)")
                        # 💰 book-plan: NET ₹X (charges ke BAAD) ke liye level
                        if _book_amt > 0:
                            try:
                                from execution.cost_model import gross_for_net
                                _need = gross_for_net(_book_amt,
                                                      float(p["entry"]),
                                                      int(ps["qty"]))
                            except Exception:
                                _need = _book_amt
                            _mv = _need / (ps["qty"] * float(p["entry"])) * 100
                            _lvl = float(p["entry"]) * (1 + _mv / 100)
                            size_line += (f"\n   💰 NET ₹{_book_amt:,.0f} aim "
                                          f"@ ₹{_lvl:,.1f} (+{_mv:.1f}%), "
                                          f"phir TRAIL — floor se kam kabhi nahi")
                except Exception:
                    pass
                lines.append(
                    f"\n{emoji} <b>{p['symbol']}</b> ₹{p['price']:,.1f} — {p['verdict']}"
                    f"{prime_bit}\n"
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
        # Inline action buttons — phone se hi paper-trade / watchlist
        markup = None
        try:
            from alerts.telegram_actions import build_setup_keyboard
            if fresh:
                markup = build_setup_keyboard(fresh)
        except Exception:
            pass
        if engine.send("\n".join(lines), reply_markup=markup):
            _pushed[today].update(p["symbol"] for p in fresh + pre)
            _save_state()   # dedupe survives restarts — no duplicate pushes
            log.info("setups_pushed_to_telegram", buys=len(fresh), prebreakout=len(pre))
    except Exception as exc:
        log.debug("push_setups_skip", error=str(exc))


def _log_buys_for_tracking(results) -> None:
    """Feed BUY signals into the outcome tracker (dedupes per day itself)."""
    try:
        from core.signal_outcome_tracker import log_signal
        # Capture the CURRENT market tape ONCE (cached 15 min) — so we can
        # later measure which signals earn in which regime. Was logged empty;
        # that blinded the edge profiler to regime, the biggest context an
        # entry has. compute_regime is streamlit-free + cached → cheap here.
        regime = ""
        try:
            from core.regime_engine import compute_regime
            regime = str(getattr(compute_regime(), "market_regime", "") or "")
        except Exception:
            regime = ""
        for r in results:
            if r.verdict != "BUY":
                continue
            log_signal(
                symbol=r.symbol, signal_type="UNIFIED_BUY",
                entry_price=r.entry, pivot_price=r.entry,
                stop_price=r.stop, target_price=r.target,
                quality_score=r.score, accum_score=0.0,
                archetype="|".join(r.signals), regime=regime,
            )
    except Exception as exc:
        log.debug("auto_scan_tracking_skip", error=str(exc))


def _log_decisions_for_calibration(serialized) -> None:
    """📓 Journal EVERY scan verdict with its prediction — TAKEN (a buy) or
    REJECTED (a watch) — so Confidence Accuracy (calibration) and gate attribution
    get data for MANUAL users too, not only when autopilot is running. Deduped per
    symbol×day×decision by the journal; outcomes settle 5 days later. Fail-open."""
    try:
        from core.decision_journal import log_decision
        for r in serialized:
            entry = float(r.get("entry") or 0)
            if entry <= 0:
                continue                         # no reference price → no claim
            verdict = r.get("verdict", "")
            decision = "TAKEN" if verdict in ("BUY", "STRONG BUY") else "REJECTED"
            reason = "" if decision == "TAKEN" else (
                (r.get("reasons") or [""])[0][:120])
            log_decision(r["symbol"], decision, reason=reason, source="scanner",
                         entry_ref=entry, stop_ref=float(r.get("stop") or 0),
                         score=float(r.get("score") or 0),
                         ev_pct=r.get("ev_pct"), p_win=r.get("p_win"),
                         confidence=r.get("confidence"))
    except Exception as exc:
        log.debug("decision_log_skip", error=str(exc))


def _log_non_events_for_learning(results) -> None:
    """🕳️ Freeze the WATCH / rejected names as structured non-event observations
    in the Feature Platform — the control group that turns the P&L into a
    controlled experiment (which rejection reasons SAVE money vs are too
    conservative). Deduped per day by the store; fully fail-open."""
    try:
        from research.non_event import record_scan_batch
        regime = ""
        try:
            from core.regime_engine import compute_regime
            regime = str(getattr(compute_regime(), "market_regime", "") or "")
        except Exception:
            regime = ""
        breadth_pct = None
        try:
            breadth_pct = (_breadth or {}).get("pct_above_50")
        except Exception:
            breadth_pct = None
        record_scan_batch(results, regime=regime, breadth_pct=breadth_pct)
    except Exception as exc:
        log.debug("auto_scan_nonevent_skip", error=str(exc))


_scan_gate = threading.Lock()   # one scan at a time — no duplicate work


def _scan_once(universe: Optional[list[str]] = None, progress=None) -> None:
    global _results, _scanned_count, _last_scan_ts, _status
    # Non-blocking gate: worker + page-triggered scans were running the
    # whole pipeline TWICE in parallel (double Kite calls, double logs).
    # Second caller just waits for the running scan and uses its result.
    if not _scan_gate.acquire(blocking=False):
        log.debug("scan_already_running_skip")
        with _scan_gate:      # wait for the in-flight scan to finish
            return
    try:
        _scan_once_locked(universe, progress)
    finally:
        _scan_gate.release()


def _scan_once_locked(universe: Optional[list[str]] = None, progress=None) -> None:
    global _results, _scanned_count, _last_scan_ts, _status
    with _lock:
        _status = "scanning"
    try:
        from scan.unified_scanner import UnifiedScanner
        if universe is None:
            from data.nse_universe import get_nse_universe
            universe = get_nse_universe()
        from core.eco import workers as _eco_workers
        raw = UnifiedScanner(max_workers=_eco_workers(8)).scan(
            universe, progress=progress)
        _log_buys_for_tracking(raw)
        _log_non_events_for_learning(raw)
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
        # Live price overlay — Kite → NSE → Google (history is official NSE
        # EOD). STRICT: har signal-wale stock pe, sirf top-60 nahi — Kite
        # bulk quote 500/call hai, poora set 2-3 calls mein aa jata hai.
        try:
            from data.live_quotes import get_live_quotes
            top_syms = [r["symbol"] for r in serialized]
            live = get_live_quotes(top_syms)
            overlaid = 0
            for r in serialized:
                q = live.get(r["symbol"])
                if not (q and q.get("price")):
                    r["live"] = False
                    continue
                r["price"] = q["price"]
                r["change_pct"] = q["chg_pct"]
                r["live"] = True
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
            log.info("live_overlay_done", overlaid=overlaid, of=len(serialized))
        except Exception as exc:
            log.warning("live_overlay_failed", error=str(exc))
        # Measured edge on every result — backtest data working FOR the user
        try:
            from scan.signal_backtest import combo_edge
            from scan.unified_scanner import SIGNAL_META
            _key_by_label = {v[0]: k for k, v in SIGNAL_META.items()}
            for r in serialized:
                keys = [_key_by_label.get(lbl) for lbl in r.get("signals", [])]
                keys = [k for k in keys if k]
                edge = combo_edge(keys)
                if edge is None:
                    continue
                r["edge_r"] = edge
                # Veto ONLY a proven loser (≤ LOSER line), not a NEUTRAL/
                # breakeven combo whose slightly-negative point estimate is
                # within measurement noise. A NEUTRAL edge doesn't earn a BUY
                # on its own — but it must not BLOCK one; conviction / EV /
                # breadth / regime gates rank it from here.
                if _edge_vetoes(edge, r.get("verdict")):
                    r["verdict"] = "WATCH"
                    r.setdefault("reasons", []).insert(
                        0, f"⚠ Backtest LOSER: is pattern-combo ki measured edge "
                           f"{edge:+.2f}R hai — proven negative, skip")
            # Proven edge floats to the top; demoted setups sink below every
            # buy — in cards, Telegram, Dashboard and JARVIS alike.
            _vrank = {"STRONG BUY": 2, "BUY": 1}
            serialized.sort(
                key=lambda r: (_vrank.get(r.get("verdict"), 0),
                               float(r.get("score", 0)) + float(r.get("edge_r", 0) or 0) * 40),
                reverse=True)
        except Exception as exc:
            log.debug("edge_apply_skip", error=str(exc))
        # 🎯 Conviction tier — highest conviction sabse pehle, har surface pe
        tag_conviction(serialized)
        # 💰 EV tier — expected value from OUR outcomes (north star: rank by
        # expected return per unit risk, not points). Additive fields only;
        # conviction_rank stays the fallback when evidence is thin.
        try:
            from scan.ev_engine import tag_ev
            tag_ev(serialized)
        except Exception as exc:
            log.debug("ev_tag_skip", error=str(exc))
        # 📓 Journal each verdict + its prediction (now that p_win/ev are on the
        # results) so calibration & gate-attribution learn from manual use too.
        _log_decisions_for_calibration(serialized)
        # 📊 Breadth — poore bulk cache se (data is gold: already computed).
        # Index ke peechhe ki sachchai — Brain iska evidence stream padhta hai.
        try:
            from scan.breadth import breadth_from_cache
            global _breadth
            _breadth = breadth_from_cache()
            if _breadth.get("verdict"):
                log.info("breadth", verdict=_breadth["verdict"],
                         n=_breadth["n"], adv_ratio=_breadth["adv_ratio"])
        except Exception as exc:
            log.debug("breadth_skip", error=str(exc))
        # 💎 Prime tier — momentum/breakout survivors of EVERY data layer
        # (conviction + EV + liquidity + breadth + regime). Telegram pushes
        # these first; sabse sophisticated chhalni sabse upar.
        try:
            from scan.prime_filter import tag_prime
            demoted: set = set()
            try:
                from core.regime_engine import compute_regime
                from scan.live_edge import regime_calibration
                from scan.unified_scanner import SIGNAL_META
                _reg = str(getattr(compute_regime(), "market_regime", "") or "")
                _rc = regime_calibration(_reg)
                demoted = {SIGNAL_META[k][0] for k, m in _rc.items()
                           if m < 1.0 and k in SIGNAL_META}
            except Exception:
                demoted = set()
            n_prime = tag_prime(serialized,
                                (_breadth or {}).get("verdict", ""), demoted)
            if n_prime:
                log.info("prime_tagged", n=n_prime)
        except Exception as exc:
            log.debug("prime_skip", error=str(exc))
        # 🏦 Institutional footprint tag — bulk-deal buying wale naam
        # (context-tag, score nahi badalta — edge measure hone tak)
        try:
            from data.institutional_flows import get_flows
            _bulk = set(get_flows().get("bulk_buys") or [])
            if _bulk:
                for r in serialized:
                    if r["symbol"] in _bulk:
                        r["bulk_deal"] = True
        except Exception as exc:
            log.debug("bulk_tag_skip", error=str(exc))
        # Proactive delivery — user ko dhundhna na pade, setups khud pahunchein
        _push_new_setups(serialized[:15])
        # 🤖 Autopilot hook — same signals, alert logic untouched
        try:
            from execution.autopilot import on_setups as _ap_setups
            _ap_setups(serialized[:20])
        except Exception as exc:
            log.debug("autopilot_setups_skip", error=str(exc))
        with _lock:
            _results = serialized
            _scanned_count = len(universe)
            _last_scan_ts = time.time()
            _status = "ready"
        _save_state()   # restart pe results turant wapas milenge
        log.info("auto_scan_complete", universe=len(universe), signals=len(serialized))
    except Exception as exc:
        with _lock:
            _status = "error" if not _results else "ready"
        if "interpreter shutdown" in str(exc):
            log.info("scan_aborted_by_shutdown")   # Ctrl+C mid-cycle — benign
        else:
            log.warning("auto_scan_failed", error=str(exc))


_auto_enabled: bool = True


def set_auto_enabled(enabled: bool) -> None:
    """User control: pause/resume the automatic background refresh.
    Logs only on CHANGE — every page rerun calls this."""
    global _auto_enabled
    with _lock:
        changed = _auto_enabled != enabled
        _auto_enabled = enabled
    if changed:
        log.info("auto_scan_toggled", enabled=enabled)


def is_auto_enabled() -> bool:
    with _lock:
        return _auto_enabled


_pulse_pushed_date: str = ""


def _maybe_push_morning_pulse() -> None:
    """Once per weekday, 8:30-10:00 window: full Daily Pulse to Telegram."""
    global _pulse_pushed_date
    now = datetime.now(IST)
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


_brain_briefing_date: str = ""


def _maybe_push_brain_briefing() -> None:
    """Once per weekday, market-open window: the Brain's one-verdict briefing
    to Telegram — posture + directives + top pick. Additive; leads the morning
    ahead of the full pulse newsletter. Existing pulse push is untouched."""
    global _brain_briefing_date
    now = datetime.now(IST)
    today = now.strftime("%Y-%m-%d")
    if now.weekday() >= 5 or _brain_briefing_date == today:
        return
    if not (8 * 60 + 30 <= now.hour * 60 + now.minute <= 10 * 60):
        return
    try:
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if not engine.is_configured():
            return
        from core.brain import briefing_telegram
        if engine.send(briefing_telegram("IN")):
            _brain_briefing_date = today
            log.info("brain_briefing_pushed")
    except Exception as exc:
        log.debug("brain_briefing_skip", error=str(exc))


_backup_done_date: str = ""


def _maybe_backup_evidence() -> None:
    """Roz ek baar, off-hours: evidence DBs + autopilot state ka snapshot
    (7-din rotation). Poonji ka bima — core/backup.py."""
    global _backup_done_date
    today = datetime.now(IST).strftime("%Y-%m-%d")
    if _backup_done_date == today or _is_market_hours():
        return
    try:
        from core.backup import snapshot
        snapshot()
        _backup_done_date = today
    except Exception as exc:
        log.debug("evidence_backup_skip", error=str(exc))


_bt_done_date: str = ""


def _maybe_run_nightly_backtest() -> None:
    """Auto-refresh the signal backtest once per day, off-market hours —
    the calibration stays current without the user pressing anything."""
    global _bt_done_date
    today = datetime.now(IST).strftime("%Y-%m-%d")
    if _bt_done_date == today or _is_market_hours():
        return
    try:
        from data.bhavcopy_store import is_ready
        from scan.signal_backtest import run_backtest, get_state, load_report
        if not is_ready() or get_state()["running"]:
            return
        rep = load_report()
        if rep and str(rep.get("generated_at", "")).startswith(today.replace("-", "")[:4]):
            # already ran today? generated_at format is "YYYY-MM-DD HH:MM"
            if str(rep.get("generated_at", ""))[:10] == today:
                _bt_done_date = today
                return
        log.info("nightly_backtest_start")
        run_backtest(max_symbols=None)  # full bhav universe — not a capped sample
        _bt_done_date = today
        # 🗂️ Edge Timeline — once the day's outcomes are settled, record any
        # per-signal drift STATE TRANSITION so the system builds a permanent
        # history (cyclical vs dead vs durable). Fail-open, transition-only.
        try:
            from research.edge_timeline import record_snapshot
            moved = record_snapshot()
            if moved:
                log.info("edge_timeline_transitions", n=len(moved))
        except Exception as exc:
            log.debug("edge_timeline_skip", error=str(exc))
        # 🕳️ Non-event settlement — fill forward-return outcomes on the day's
        # matured REJECTION / NEAR_MISS observations (the counterfactual control
        # group), from official bhavcopy. Fail-open.
        try:
            from research.non_event import settle_outcomes
            n_settled = settle_outcomes()
            if n_settled:
                log.info("non_event_settled", n=n_settled)
        except Exception as exc:
            log.debug("non_event_settle_skip", error=str(exc))
        # 📚 Scientific Memory — promote any newly-validated experiments into
        # durable beliefs (with Evidence-Graph provenance). Idempotent, fail-open.
        try:
            from research.scientific_memory import sync_from_registry
            synced = sync_from_registry()
            if synced.get("synced"):
                log.info("scientific_memory_synced", n=synced["synced"])
        except Exception as exc:
            log.debug("scientific_memory_sync_skip", error=str(exc))
    except Exception as exc:
        log.debug("nightly_backtest_skip", error=str(exc))


_kite_reminder_date: str = ""


def _maybe_remind_kite_login() -> None:
    """8:30-9:15 weekday window: if Kite isn't usable, one Telegram nudge —
    warna poora din live data ke bina chalta hai aur pata bhi nahi chalta."""
    global _kite_reminder_date
    now = datetime.now(IST)
    today = now.strftime("%Y-%m-%d")
    if now.weekday() >= 5 or _kite_reminder_date == today:
        return
    if not (8 * 60 + 30 <= now.hour * 60 + now.minute <= 9 * 60 + 15):
        return
    try:
        from execution.trade_executor import kite_ready
        if kite_ready():
            _kite_reminder_date = today          # token fresh — no nudge needed
            return
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if not engine.is_configured():
            return
        if engine.send(
                "🔑 <b>Kite login pending</b>\n"
                "Market khulne wala hai aur token fresh nahi hai — live "
                "quotes/orders NSE fallback pe chalenge.\n\n"
                "<code>cd ~/0to100 && python main.py login</code>"):
            _kite_reminder_date = today
            log.info("kite_login_reminder_sent")
    except Exception as exc:
        log.debug("kite_reminder_skip", error=str(exc))


_outcomes_checked_date: str = ""


def _maybe_update_outcomes() -> None:
    """Daily: resolve pending signal outcomes so the Report Card and
    accuracy calibration keep learning WITHOUT anyone opening JARVIS."""
    global _outcomes_checked_date
    today = datetime.now(IST).strftime("%Y-%m-%d")
    if _outcomes_checked_date == today:
        return
    try:
        from core.signal_outcome_tracker import update_outcomes
        update_outcomes()
        _outcomes_checked_date = today
        log.info("signal_outcomes_updated")
    except Exception as exc:
        log.debug("outcomes_update_skip", error=str(exc))
    # ONE-TIME back-data correction: re-judge old crude-labelled outcomes by true
    # target-vs-stop first-touch so the learning stack stops inheriting the proxy.
    # Guarded by a marker file → runs exactly once, then never again.
    try:
        _marker = _Path(__file__).resolve().parent.parent / "logs" / ".outcomes_reresolved_v1"
        if not _marker.exists():
            from core.signal_outcome_tracker import reresolve_history
            n = reresolve_history()
            _marker.parent.mkdir(parents=True, exist_ok=True)
            _marker.write_text(str(n))
            log.info("outcome_history_reresolved", corrected=n)
    except Exception as exc:
        log.debug("reresolve_skip", error=str(exc))
    # Decision journal outcomes too — accepted AND rejected candidates get
    # resolved, so we later learn from decisions we DIDN'T take.
    try:
        from core.decision_journal import update_outcomes as _dec_update
        _dec_update()
    except Exception as exc:
        log.debug("decision_outcomes_skip", error=str(exc))


_coach_pushed_week: str = ""


def _maybe_push_weekly_coach() -> None:
    """Sunday evening: coach review to Telegram, once per week."""
    global _coach_pushed_week
    now = datetime.now(IST)
    week = now.strftime("%Y-W%W")
    if now.weekday() != 6 or now.hour < 17 or _coach_pushed_week == week:
        return
    try:
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if not engine.is_configured():
            return
        from reports.trade_coach import build_coach_review
        if engine.send(build_coach_review()):
            _coach_pushed_week = week
            log.info("weekly_coach_pushed")
    except Exception as exc:
        log.debug("weekly_coach_skip", error=str(exc))


_lt_run_week: str = ""


def _maybe_run_long_term() -> None:
    """💎 Weekly (Sat off-hours): screen the market for LONG-TERM picks, alert new
    ones on Telegram, and REVISE (exit-alert) any held pick whose thesis broke.
    Long-term calls move slowly — once a week is right. Needs the bhavcopy store.
    Fail-open."""
    global _lt_run_week
    now = datetime.now(IST)
    week = now.strftime("%Y-W%W")
    if now.weekday() != 5 or _lt_run_week == week:      # Saturday, once/week
        return
    try:
        from data.bhavcopy_store import is_ready
        if not is_ready() or _is_market_hours():
            return
        from scan.long_term import scan_long_term
        from core.long_term_tracker import record_picks, review_picks
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        # 1) revise held picks first (exit-alerts matter most)
        revisions = review_picks()
        # 2) fresh picks
        picks = scan_long_term(top=12)
        added = record_picks(picks)
        _lt_run_week = week
        if not engine.is_configured():
            return
        if revisions:
            lines = ["🔄 <b>Long-term picks — REVISED (exit)</b>", ""]
            for r in revisions[:8]:
                lines.append(f"• <b>{r['symbol']}</b> — {r['reason']} "
                             f"(return {r['return_pct']:+.1f}%)")
            engine.send("\n".join(lines))
        if added:
            lines = ["💎 <b>New Long-Term Picks</b> (hold for months, not days)", ""]
            for p in added[:10]:
                lines.append(f"• <b>{p['symbol']}</b> @ ₹{(p.get('price') or 0):,.0f} "
                             f"· score {p.get('score', 0):.0f}\n  {p.get('thesis', '')}")
            lines.append("\nInhe hum track karenge — thesis toote toh exit-alert milega.")
            engine.send("\n".join(lines))
        log.info("long_term_run", new=len(added), revised=len(revisions))
    except Exception as exc:
        log.debug("long_term_skip", error=str(exc))


def _worker() -> None:
    while True:
        if is_auto_enabled():
            try:
                from core.health import beat as _hb, timed as _ht
                _hb("auto_scan", note=_status)
                _cycle_timer = _ht("scan_cycle")
            except Exception:
                import contextlib
                _cycle_timer = contextlib.nullcontext()
            with _cycle_timer:
                # 🍃 Eco: off-hours full-market scan pure heat hai (EOD data
                # badalta nahi) — shared machine pe skip; housekeeping chalti
                # rehti hai (briefing, outcomes, backtest sab neeche hain).
                try:
                    from core.eco import should_scan_now
                    _do_scan = should_scan_now(_is_market_hours())
                except Exception:
                    _do_scan = True
                if _do_scan:
                    _scan_once()
            _maybe_remind_kite_login()
            _maybe_push_brain_briefing()      # Brain verdict leads the morning
            _maybe_push_morning_pulse()
            _maybe_update_outcomes()
            _maybe_run_nightly_backtest()
            _maybe_push_weekly_coach()
            _maybe_run_long_term()            # 💎 weekly long-term picks + revision
            # 🐕 dead-daemon sweep (throttled) + 🗄️ nightly evidence backup
            try:
                from core.watchdog import check as _wd_check
                _wd_check()
            except Exception:
                pass
            _maybe_backup_evidence()
            # Breakout sniper — tick-stream instant alerts on the hottest names
            if _is_market_hours():
                try:
                    from scan.breakout_sniper import start_sniper, refresh_watch
                    with _lock:
                        _cur = list(_results)
                    if start_sniper():
                        refresh_watch(_cur)
                except Exception as exc:
                    log.debug("sniper_skip", error=str(exc))
            # Position babysitting — exit alerts for open trades
            if _is_market_hours():
                try:
                    from risk.position_manager import push_position_alerts
                    push_position_alerts()
                except Exception as exc:
                    log.debug("position_review_skip", error=str(exc))
                # Pending GTTs — fill hone par exchange-exit lagao
                try:
                    from execution.trade_executor import ensure_pending_gtts
                    ensure_pending_gtts()
                except Exception as exc:
                    log.debug("pending_gtt_skip", error=str(exc))
                # 🤖 Autopilot — closes, compounding, circuit breaker
                try:
                    from execution.autopilot import review_cycle
                    review_cycle()
                except Exception as exc:
                    log.debug("autopilot_review_skip", error=str(exc))
                # Watchlist chowkidaari — buy-zone / broken-setup alerts
                try:
                    from risk.watchlist_watcher import push_watchlist_alerts
                    push_watchlist_alerts()
                except Exception as exc:
                    log.debug("watchlist_watch_skip", error=str(exc))
                # Active Buys — MA/support/volume health warnings
                try:
                    from risk.buy_book_watcher import push_buy_book_alerts
                    push_buy_book_alerts()
                except Exception as exc:
                    log.debug("buy_book_watch_skip", error=str(exc))
            # 🤖 Autopilot EOD digest — post-close din ka hisaab (once/day)
            try:
                from execution.autopilot import eod_digest
                eod_digest()
            except Exception as exc:
                log.debug("autopilot_digest_skip", error=str(exc))
            try:
                from core.eco import scan_interval as _eco_interval
                _mkt_sleep = _eco_interval(_MARKET_REFRESH_S)
            except Exception:
                _mkt_sleep = _MARKET_REFRESH_S
            time.sleep(_mkt_sleep if _is_market_hours() else _OFFHOURS_REFRESH_S)
        else:
            time.sleep(30)   # paused by user — just idle and re-check


def start_background_scan() -> None:
    """Idempotent — starts the daemon scanner thread once per process."""
    global _thread_started
    with _lock:
        if _thread_started:
            return
        _thread_started = True
    _load_state()   # warm from disk — instant results after restart
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


_breadth: dict = {}


def get_breadth() -> dict:
    """Latest full-market breadth (computed once per scan, zero fetch)."""
    with _lock:
        return dict(_breadth)
