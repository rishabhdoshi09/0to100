"""
🤖 Autopilot — autonomous trading with hard guardrails.

Design philosophy: the autopilot is a CONSUMER of the same signals the
user already gets on Telegram (setups + sniper breakouts) — it never
changes that alert logic. It sits behind a wall of gates, and every
gate must pass before a single rupee moves:

  GATE 1  armed?              OFF by default; LIVE arming needs typed
                              confirmation in the UI
  GATE 2  mode                PAPER by default — full loop, zero money
  GATE 3  time window         no entries before 09:30 (market settles),
                              none after end_time (default 14:45)
  GATE 4  daily trade limit   max_trades_per_day
  GATE 5  position limit      max_open_positions (autopilot's own)
  GATE 6  one-shot per symbol per day
  GATE 7  sector strength     symbol's sector must be in the top-N
                              positive sectors right now
  GATE 8  quality             score >= min_score AND measured edge not
                              negative
  GATE 9  capital             deploy <= available pool AND (LIVE) <=
                              broker margin; the pool is NEVER exhausted
                              (cash_reserve_pct stays untouched)
  GATE 10 circuit breaker     day's P&L <= -daily_loss_limit% → disarm
                              for the day + Telegram alert

Money model (compounding):
  pool      = allocation + realized_pnl (profits/losses compound)
  deployed  = Σ entry×qty of OPEN autopilot trades
  available = pool × (1 - cash_reserve_pct) - deployed

Exits: every entry ships with GTT OCO — stop from the signal; under
thesis_hold (default) the target is a wide runner ceiling and the
position is held while technicals + fundamentals look good. High RSI
on an OPEN trade is not an exit — tighten a protective GTT 2–3% below
LTP and hold until structure actually fails. Optional profit_book_*
remains available as an explicit scalp AIM.

State persists atomically to logs/autopilot.json.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, date

from core.market_clock import (IST, today_ist as _ist_today,
                                now_ist_naive as _now_ist_naive,
                                is_ist_today as _is_ist_today)
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)

_STATE_FILE = Path(__file__).resolve().parent.parent / "logs" / "autopilot.json"
_lock = threading.RLock()
# Serialises the whole gate→size→place path. Scanner aur sniper alag
# threads se consider() bulate hain — bina iske dono ek saath daily/position
# count gates paar kar sakte hain (double entry).
_consider_lock = threading.Lock()

TAG = "AUTOPILOT"
ARM_PHRASE = "ARM LIVE"        # must be typed exactly to arm LIVE mode


def _live_enabled() -> bool:
    """`QT_LIVE_ENABLED` is a TEMPORARY MIGRATION INTERLOCK, not a graduation switch.
    LIVE arming fails closed while the Evidence Lab overhaul is in progress (ADR-001,
    directive §15). The flag removes ONLY this migration lock — it does NOT assert any
    strategy has earned live capital; the graduation criteria (and a future deployment
    manifest) in docs/architecture/EXECUTION_SAFETY.md are a separate, mandatory gate.
    Paper mode is unaffected. Fail closed: anything but an explicit truthy flag =
    disabled."""
    return os.getenv("QT_LIVE_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")

_DEFAULTS = {
    "armed": False,
    "mode": "PAPER",               # PAPER | LIVE — PAPER by default, always
    "allocation": 0.0,             # X — user sets this explicitly
    "realized_pnl": 0.0,           # compounds into the pool
    "cash_reserve_pct": 0.10,      # never deploy the last 10% of the pool
    "per_trade_cap_pct": 0.20,     # one trade ≤ 20% of pool
    "risk_per_trade_pct": 0.01,    # 1% rule within the pool
    "max_open_positions": 3,
    "max_per_sector": 2,           # correlation cap — ek sector mein zyada nahi
    "max_trades_per_day": 4,
    "daily_loss_limit_pct": 0.03,  # -3% day → circuit breaker
    "min_score": 60.0,
    "sector_top_n": 3,
    "start_time": "09:30",
    "end_time": "14:45",
    "target_pct": 3.0,             # scalp GTT % when thesis_hold is OFF
    "trailing_enabled": True,      # +breakeven_trigger% → stop moves to entry
    "breakeven_trigger_pct": 2.0,  # profit level that arms the breakeven trail
    "regime_gate": True,           # no new entries in DISTRIBUTION / BEAR tape
    "digest_date": "",             # last day the EOD digest was sent
    "max_chase_pct": 1.0,          # live price entry se itna upar → chase NAHI
    "conviction_sizing": True,     # risk scales 0.5×–1.5× with measured edge
    "adaptive_source_gate": True,  # pause a source its OWN record proves negative
    "max_hold_days": 15,           # soft ceiling; thesis_hold skips if still good
    "time_nudges": {},             # {date: [symbols]} LIVE stale-nudge dedupe
    "trades_today": {},            # {YYYY-MM-DD: count}
    "traded_symbols": {},          # {YYYY-MM-DD: [symbols]}
    "disarmed_reason": "",
    "max_accounted_id": 0,         # high-water mark — trade ids already
                                    # folded into realized_pnl (ids are
                                    # monotonic, so this never needs pruning)
    "activity": [],                # last N activity lines for the UI
    "reject_stats": {},            # {YYYY-MM-DD: {category: count}} — the funnel
    "considered": {},              # {YYYY-MM-DD: candidates seen}
    "preset": "Balanced",          # aggressiveness dial (frequency, not safety)
    "brain_gate": True,            # Brain STAND_ASIDE → pause new entries (survival)
    # Hold while technicals + fundamentals look good (NOT a fixed ₹/3% scalp).
    # Optional profit_book_* remains available if the user wants a scalp AIM.
    "thesis_hold": True,
    "runner_target_pct": 10.0,     # wide GTT ceiling under thesis_hold
    "rsi_protect_pct": 2.5,        # open-trade RSI extension → GTT 2–3% below LTP
    "profit_book_pct": 0.0,        # OFF by default — thesis hold, not scalp
    "profit_book_rupees": 0.0,     # >0 → absolute ₹ AIM (legacy /book 1500)
    "profit_book_min_pct": 1.5,    # floor as % of pool (pct mode)
    "profit_book_min_rupees": 1000.0,  # hard floor when absolute ₹ mode
    "profit_trail_giveback_pct": 0.6,  # giveback as % of pool (pct mode)
    "profit_trail_giveback_rupees": 300.0,  # giveback when absolute ₹ mode
    "book_nudges": {},             # {date: [symbols]} LIVE book-alert dedupe
    "trail_peaks": {},             # {trade_id: peak NET ₹} — trailing state
    "book_plans": {},              # {trade_id: {aim,floor,giveback,quality_mult}}
    "thesis_nudges": {},           # {date: [symbols]} LIVE thesis-exit dedupe
    "symbol_memory_gate": True,    # 🧠 serial false-breaker naam dobara nahi
}


# ── Rejection funnel — WHY so few trades (transparency, not blind loosening) ──

def _reject_category(reason: str) -> str:
    r = (reason or "").lower()
    if "brain" in r or "stand_aside" in r: return "Brain STAND_ASIDE (survival)"
    if "symbol memory" in r:      return "symbol memory (serial false-breaker)"
    if "rsi" in r or "blow-off" in r: return "RSI blow-off (ignored)"
    if "volume" in r and "1×" in (reason or ""): return "volume < 1× (priority niche)"
    if "time window" in r:        return "time window (market band/settle)"
    if "daily trade limit" in r:  return "daily limit hit (max/day)"
    if "max open positions" in r: return "position limit full"
    if "already traded" in r:     return "symbol already traded today"
    if "concentration" in r:      return "sector concentration cap"
    if "score" in r or "conviction" in r: return "score/conviction kam"
    if "edge" in r:               return "negative measured edge"
    if "regime" in r:             return "market regime kharab"
    if "sector" in r:             return "sector strong nahi"
    if "quote" in r:              return "live quote nahi mila"
    if "stop" in r and "neeche" in r: return "setup toot gaya (live < stop)"
    if "chase" in r:              return "chase (live entry se upar)"
    if "share" in r or "pool" in r:   return "capital/1-share nahi aata"
    if "source" in r:             return "source apne record se paused"
    if "stop" in r:               return "invalid stop (data)"
    return "other"


def _note_reject(reason: str) -> None:
    today = _ist_today().isoformat()
    with _lock:
        s = _load()
        # copy (not setdefault-in-place) — _DEFAULTS' nested dict is shared via
        # the shallow dict(_DEFAULTS) load, so never mutate it directly
        rs = dict(s.get("reject_stats", {}).get(today, {}))
        cat = _reject_category(reason)
        rs[cat] = rs.get(cat, 0) + 1
        s["reject_stats"] = {today: rs}     # keep only today (state stays small)
    _save()


def _note_considered() -> None:
    """One candidate reached the gate funnel today (denominator)."""
    today = _ist_today().isoformat()
    with _lock:
        s = _load()
        n = int(s.get("considered", {}).get(today, 0)) + 1
        s["considered"] = {today: n}          # keep only today (state stays small)
    _save()


def reject_funnel() -> dict:
    today = _ist_today().isoformat()
    s = _load()
    return {"considered": int(s.get("considered", {}).get(today, 0)),
            "rejects": dict(s.get("reject_stats", {}).get(today, {}))}

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
                _state.update({k: v for k, v in on_disk.items()
                               if k in _DEFAULTS})
        except Exception as exc:
            log.warning("autopilot_state_load_failed", error=str(exc))
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
        log.warning("autopilot_state_save_failed", error=str(exc))


def _log_activity(msg: str) -> None:
    with _lock:
        s = _load()
        s["activity"] = ([f"{datetime.now(IST).strftime('%d %b %H:%M')} · {msg}"]
                         + s.get("activity", []))[:40]
    _save()
    log.info("autopilot", msg=msg)


def _notify(msg: str) -> None:
    """Autopilot's own Telegram line — additive, existing alerts untouched."""
    try:
        from alerts.telegram_alerts import AlertEngine
        AlertEngine().send(f"🤖 <b>Autopilot</b>\n{msg}")
    except Exception:
        pass


# ── Public config / status API ────────────────────────────────────────────────

def get_status() -> dict:
    with _lock:
        s = dict(_load())
    s["open_trades"] = _open_autopilot_trades()
    s["deployed"] = round(sum(float(t["entry_price"] or 0) * int(t["qty"] or 0)
                              for t in s["open_trades"]), 0)
    s["pool"] = round(s["allocation"] + s["realized_pnl"], 0)
    s["available"] = round(
        max(0.0, s["pool"] * (1 - s["cash_reserve_pct"]) - s["deployed"]), 0)
    today = _ist_today().isoformat()
    s["trades_today_count"] = int(s.get("trades_today", {}).get(today, 0))
    # LIVE arming is migration-locked by default (C-04b). This is a TEMPORARY
    # interlock, not strategy graduation — surfaced explicitly in status.
    s["live_enabled"] = _live_enabled()
    s["live_interlock"] = ("temporary migration interlock (QT_LIVE_ENABLED) — "
                           "NOT strategy graduation")
    return s


def diagnose_silence() -> dict:
    """Plain-English diagnosis when autopilot is not taking new trades.

    Never invents — reads durable autopilot state + last scan + health.
    """
    s = get_status()
    funnel = reject_funnel()
    reasons: list[str] = []
    blockers: list[str] = []

    if not s.get("armed"):
        blockers.append(
            f"DISARMED — {s.get('disarmed_reason') or 'not armed (default OFF)'}"
        )
    if float(s.get("allocation") or 0) < 5000:
        blockers.append(
            f"Allocation ₹{float(s.get('allocation') or 0):,.0f} too low "
            "(need ≥ ₹5,000 to arm/size meaningfully)"
        )
    if not _in_window():
        end = _effective_end_time(s)
        extra = (
            " — paper stays open until 15:20 IST"
            if str(s.get("mode") or "PAPER").upper() == "PAPER"
            else ""
        )
        blockers.append(
            f"Outside entry window {s.get('start_time')}-{end} IST "
            f"(weekdays only{extra})"
        )
    if s.get("brain_gate", True):
        posture, why = _brain_posture()
        if posture == "STAND_ASIDE":
            blockers.append(f"Brain STAND_ASIDE — {why or 'survival veto'}")

    today = _ist_today().isoformat()
    trades_today = int(s.get("trades_today", {}).get(today, 0) or 0)
    if trades_today >= int(s.get("max_trades_per_day") or 0):
        blockers.append(f"Daily trade limit hit ({trades_today}/{s.get('max_trades_per_day')})")
    open_n = len(s.get("open_trades") or [])
    if open_n >= int(s.get("max_open_positions") or 0):
        blockers.append(f"Max open positions full ({open_n}/{s.get('max_open_positions')})")

    # Scan feed health — React/autonomy path must call on_setups after market scan
    buy_n = 0
    scan_as_of = ""
    try:
        from product.scan_store import load_scan

        scan = load_scan() or {}
        scan_as_of = str(scan.get("scanned_at") or "")
        buy_n = sum(
            1
            for r in (scan.get("records") or [])
            if (
                str(r.get("verdict") or "") in {"BUY", "STRONG BUY"}
                or str(r.get("status") or "") == "Ready to trade"
            )
            and not r.get("chase_risk")
        )
    except Exception:
        scan = {}
    if not scan:
        reasons.append("No durable market scan yet — autonomy MARKET_SCAN / Scan now required")
    elif buy_n == 0:
        reasons.append(
            "Latest scan has 0 BUY / STRONG BUY setups "
            "(edge demotion / weak tape — nothing for autopilot to take)"
        )
    else:
        reasons.append(f"Latest scan has {buy_n} BUY-tier setup(s) as of {scan_as_of or '—'}")

    considered = int(funnel.get("considered") or 0)
    rejects = dict(funnel.get("rejects") or {})
    if considered == 0 and s.get("armed") and buy_n > 0:
        blockers.append(
            "Armed + BUY setups exist but considered=0 today — "
            "scan→autopilot feed may be dead (fixed: market_scan_service now calls on_setups)"
        )
    top_reject = ""
    if rejects:
        top_reject = max(rejects.items(), key=lambda kv: kv[1])
        reasons.append(f"Top reject today: {top_reject[0]} ×{top_reject[1]}")

    # Adaptive source pause
    try:
        for src in ("scanner", "sniper"):
            paused = _source_paused(src)
            if paused:
                blockers.append(paused)
    except Exception:
        pass

    headline = (
        blockers[0] if blockers
        else (
            "Autopilot is armed and in window — waiting for next BUY that clears gates"
            if s.get("armed")
            else "Autopilot is not armed"
        )
    )
    return {
        "available": True,
        "headline": headline,
        "armed": bool(s.get("armed")),
        "mode": s.get("mode"),
        "disarmed_reason": s.get("disarmed_reason") or "",
        "in_window": _in_window(),
        "trades_today": trades_today,
        "open_positions": open_n,
        "considered_today": considered,
        "rejects_today": rejects,
        "buy_setups_in_last_scan": buy_n,
        "scan_as_of": scan_as_of,
        "blockers": blockers,
        "notes": reasons,
        "activity": list(s.get("activity") or [])[:8],
        "places_orders": False,
        "honesty": (
            "Diagnosis reads logs/autopilot.json + latest scan only. "
            "Missing evidence stays missing."
        ),
    }


def set_config(**kwargs) -> None:
    """UI-settable limits. Numeric fields clamped to sane ranges."""
    clamps = {
        "allocation": (0.0, 1e9),
        "cash_reserve_pct": (0.05, 0.50),
        "per_trade_cap_pct": (0.05, 0.50),
        "risk_per_trade_pct": (0.0025, 0.02),
        "max_open_positions": (1, 10),
        "max_per_sector": (1, 5),
        "max_trades_per_day": (1, 15),
        "daily_loss_limit_pct": (0.01, 0.10),
        "min_score": (40.0, 95.0),
        "sector_top_n": (1, 8),
        "target_pct": (1.0, 10.0),
        "breakeven_trigger_pct": (0.5, 8.0),
        "max_hold_days": (2, 30),
        "max_chase_pct": (0.25, 5.0),
        "runner_target_pct": (5.0, 25.0),
        "rsi_protect_pct": (2.0, 3.0),
        "profit_book_pct": (0.0, 15.0),
        "profit_book_rupees": (0.0, 100000.0),
        "profit_book_min_pct": (0.0, 10.0),
        "profit_book_min_rupees": (0.0, 100000.0),
        "profit_trail_giveback_pct": (0.05, 5.0),
        "profit_trail_giveback_rupees": (50.0, 50000.0),
    }
    bools = ("trailing_enabled", "regime_gate", "conviction_sizing",
             "adaptive_source_gate", "brain_gate", "symbol_memory_gate",
             "thesis_hold")
    with _lock:
        s = _load()
        for k, v in kwargs.items():
            if k in bools:
                s[k] = bool(v)
                continue
            if k in ("start_time", "end_time") and isinstance(v, str):
                # HH:MM hi accept — galat format gate ko silently na tode
                try:
                    datetime.strptime(v.strip(), "%H:%M")
                    s[k] = v.strip()
                except ValueError:
                    log.warning("autopilot_bad_time", field=k, value=v)
            elif k in clamps:
                lo, hi = clamps[k]
                s[k] = type(_DEFAULTS[k])(min(hi, max(lo, v)))
            elif k == "mode" and v in ("PAPER", "LIVE"):
                if s["mode"] != v:
                    s["armed"] = False      # mode change always disarms
                s["mode"] = v
    _save()


# ── Aggressiveness presets — how OFTEN it trades, never how SAFE it is ───────
# These dials widen/narrow BREADTH only: score bar, daily slots, open-position
# and sector caps, sector_top_n, chase room. The non-negotiables — exchange-side
# GTT exit, 1% risk sizing, regime gate, live-price anchor, circuit breaker —
# are NOT touched by any preset. Aggressive = more shots at good setups, not
# looser discipline on each shot.
_PRESETS = {
    "Conservative": {          # fewer, only the A+ setups
        "min_score": 70.0, "max_trades_per_day": 2, "max_open_positions": 2,
        "max_per_sector": 1, "sector_top_n": 2, "max_chase_pct": 0.5},
    "Balanced": {              # the default — evidence-first, steady
        "min_score": 60.0, "max_trades_per_day": 4, "max_open_positions": 3,
        "max_per_sector": 2, "sector_top_n": 3, "max_chase_pct": 1.0},
    "Aggressive": {            # more shots — wider net, still gated
        "min_score": 52.0, "max_trades_per_day": 10, "max_open_positions": 5,
        "max_per_sector": 3, "sector_top_n": 5, "max_chase_pct": 2.0},
}


def presets() -> list[str]:
    return list(_PRESETS)


def apply_preset(name: str) -> tuple[bool, str]:
    """Set the aggressiveness dial. Only breadth knobs move — every money-safety
    invariant stays put. Returns (ok, message)."""
    cfg = _PRESETS.get(name)
    if not cfg:
        return False, f"unknown preset '{name}'"
    set_config(**cfg)
    with _lock:
        s = _load()
        s["preset"] = name
    _save()
    _log_activity(f"PRESET → {name} (min_score {cfg['min_score']:.0f}, "
                  f"{cfg['max_trades_per_day']}/day, "
                  f"{cfg['max_open_positions']} open, chase {cfg['max_chase_pct']}%)")
    return True, f"{name} preset applied"


def arm(confirm_phrase: str = "") -> tuple[bool, str]:
    """Arm the autopilot. LIVE requires the exact confirmation phrase
    AND a working Kite session AND a sane allocation vs broker margin."""
    s = _load()
    if s["allocation"] < 5000:
        return False, "Pehle allocation set karo (min ₹5,000)"
    if s["mode"] == "LIVE":
        if not _live_enabled():
            return False, ("LIVE execution is DISABLED during the Evidence Lab "
                           "overhaul (paper only). QT_LIVE_ENABLED is only a temporary "
                           "migration interlock — lifting it does NOT graduate a "
                           "strategy; the criteria in "
                           "docs/architecture/EXECUTION_SAFETY.md are a separate gate.")
        try:
            from product.production_ladder import live_arm_allowed
            ok, why = live_arm_allowed()
        except Exception:
            ok, why = False, "LIVE locked. Production ladder unread — fail closed."
        if not ok:
            return False, why
        if confirm_phrase.strip() != ARM_PHRASE:
            return False, f"LIVE arm karne ke liye exactly '{ARM_PHRASE}' type karo"
        try:
            from execution.trade_executor import kite_ready
            if not kite_ready():
                return False, "Kite logged-in nahi — LIVE arm nahi ho sakta"
        except Exception:
            return False, "Kite check fail — LIVE arm nahi ho sakta"
        margin = _broker_cash()
        if margin is not None and s["allocation"] > margin:
            return False, (f"Allocation ₹{s['allocation']:,.0f} broker cash "
                           f"₹{margin:,.0f} se zyada hai — pehle allocation "
                           f"ghatao ya funds daalo")
    with _lock:
        s["armed"] = True
        s["disarmed_reason"] = ""
    _save()
    pool = s["allocation"] + s["realized_pnl"]
    _log_activity(f"ARMED ({s['mode']}) — pool ₹{pool:,.0f}, "
                  f"deploy limit ₹{pool * (1 - s['cash_reserve_pct']):,.0f}")
    _notify(f"ARMED in <b>{s['mode']}</b> mode.\nPool: ₹{pool:,.0f} · "
            f"max deploy: ₹{pool * (1 - s['cash_reserve_pct']):,.0f} "
            f"(reserve {s['cash_reserve_pct']*100:.0f}% untouched)\n"
            f"Entries {s['start_time']}–{s['end_time']} · "
            + ("thesis-hold (exit jab structure toote; high RSI = GTT tighten)"
               if s.get("thesis_hold", True)
               else f"target +{s['target_pct']}%")
            + f" · max {s['max_trades_per_day']} trades/day")
    try:
        start_book_monitor()               # ⚡ fast exits armed hote hi zinda
    except Exception:
        pass
    return True, f"Armed ({s['mode']})"


def disarm(reason: str = "user") -> None:
    with _lock:
        s = _load()
        s["armed"] = False
        s["disarmed_reason"] = reason
    _save()
    _log_activity(f"DISARMED ({reason})")
    if reason != "user":
        _notify(f"⛔ DISARMED — {reason}")


def _broker_cash() -> float | None:
    try:
        from data.kite_client import KiteClient
        m = KiteClient().get_margins()
        eq = (m or {}).get("equity", {})
        avail = eq.get("available", {})
        return float(avail.get("live_balance")
                     or avail.get("cash") or 0.0)
    except Exception:
        return None


# ── Journal helpers ───────────────────────────────────────────────────────────

_OPEN_STATUSES = ("PAPER_OPEN", "PLACED", "PLACED_NO_GTT", "PENDING_GTT")
_CLOSED_WIN = ("PAPER_WIN", "AUTO_WIN")
_CLOSED_LOSS = ("PAPER_LOSS", "AUTO_LOSS")


def _autopilot_trades(statuses: tuple) -> list[dict]:
    import sqlite3
    try:
        from execution.trade_executor import _DB, connect as _te_connect
        if not _DB.exists():
            return []
        conn = _te_connect()
        conn.row_factory = sqlite3.Row
        q = ("SELECT * FROM trades WHERE note LIKE ? AND status IN (%s) "
             "ORDER BY id DESC" % ",".join("?" * len(statuses)))
        rows = conn.execute(q, (f"%{TAG}%", *statuses)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.debug("autopilot_journal_read_failed", error=str(exc))
        return []


def _open_autopilot_trades() -> list[dict]:
    return _autopilot_trades(_OPEN_STATUSES)


def _set_status(trade_id: int, status: str, note_suffix: str) -> None:
    import sqlite3
    try:
        from execution.trade_executor import _DB, connect as _te_connect
        conn = _te_connect()
        conn.execute("UPDATE trades SET status=?, note=note||? WHERE id=?",
                     (status, f" | {note_suffix}", trade_id))
        conn.commit()
        conn.close()
    except Exception as exc:
        log.debug("autopilot_status_update_failed", error=str(exc))


def _ensure_exit_col() -> None:
    """Additive journal column for ACTUAL exchange exit fills. Idempotent —
    existing rows/readers are untouched (NULL = fall back to planned exit)."""
    import sqlite3
    try:
        from execution.trade_executor import _DB, connect as _te_connect
        conn = _te_connect()
        for col in ("exit_price REAL", "orig_stop REAL", "signal_price REAL"):
            try:
                conn.execute(f"ALTER TABLE trades ADD COLUMN {col}")
                conn.commit()
            except sqlite3.OperationalError:
                pass                                # column already exists
        conn.close()
    except Exception as exc:
        log.debug("autopilot_exit_col_failed", error=str(exc))


def _update_trade(trade_id: int, sets: str, params: tuple) -> None:
    import sqlite3
    try:
        from execution.trade_executor import _DB, connect as _te_connect
        conn = _te_connect()
        conn.execute(f"UPDATE trades SET {sets} WHERE id=?", (*params, trade_id))
        conn.commit()
        conn.close()
    except Exception as exc:
        log.debug("autopilot_trade_update_failed", error=str(exc))


def _apply_entry_fill(trade_id: int, avg_fill: float) -> None:
    """Actual fill journal mein — SIGNAL price pehle preserve hota hai
    (COALESCE: sirf pehli baar), warna slippage kabhi measure nahi hogi."""
    import sqlite3
    try:
        from execution.trade_executor import _DB, connect as _te_connect
        conn = _te_connect()
        conn.execute(
            "UPDATE trades SET signal_price=COALESCE(signal_price, entry_price), "
            "entry_price=?, note=note||? WHERE id=?",
            (avg_fill, f" | fill ₹{avg_fill:,.2f}", trade_id))
        conn.commit()
        conn.close()
    except Exception as exc:
        log.debug("autopilot_entry_fill_failed", error=str(exc))


def execution_quality() -> dict:
    """Slippage ledger — backtest aur reality ke beech ka gap, measured.

    Entry slip: actual fill vs signal price (positive = zyada diya).
    Exit slip: actual exchange exit vs planned GTT leg (negative = kam
    mila). Cost ₹: dono sides ka total — '3% target' asli mein kitna
    hai, yeh number batata hai. LIVE trades only; PAPER mein slippage
    hota hi nahi.
    """
    rows = [t for t in _autopilot_trades(
        _OPEN_STATUSES + _CLOSED_WIN + _CLOSED_LOSS) if t["mode"] == "LIVE"]
    entry_slips: list[float] = []
    exit_slips: list[float] = []
    cost = 0.0
    for t in rows:
        qty = int(t["qty"] or 0)
        sig = float(t.get("signal_price") or 0)
        entry = float(t["entry_price"] or 0)
        if sig > 0 and entry > 0:
            entry_slips.append((entry - sig) / sig * 100)
            cost += (entry - sig) * qty
        actual_exit = float(t.get("exit_price") or 0)
        if actual_exit > 0:
            planned = (float(t["target_price"] or 0)
                       if t["status"] in _CLOSED_WIN
                       else float(t["stop_price"] or 0))
            if planned > 0:
                exit_slips.append((actual_exit - planned) / planned * 100)
                cost += (planned - actual_exit) * qty
    return {
        "n_entry": len(entry_slips),
        "avg_entry_slip_pct": (round(sum(entry_slips) / len(entry_slips), 3)
                               if entry_slips else 0.0),
        "worst_entry_slip_pct": (round(max(entry_slips), 3)
                                 if entry_slips else 0.0),
        "n_exit": len(exit_slips),
        "avg_exit_slip_pct": (round(sum(exit_slips) / len(exit_slips), 3)
                              if exit_slips else 0.0),
        "slippage_cost": round(cost, 0),
    }


def _planned_exit(t: dict) -> float:
    """Exit price for P&L: ACTUAL exchange fill when reconciled, else the
    planned GTT leg (target on WIN, stop on LOSS)."""
    actual = float(t.get("exit_price") or 0)
    if actual > 0:
        return actual
    return (float(t["target_price"] or 0) if t["status"] in _CLOSED_WIN
            else float(t["stop_price"] or 0))


def _trade_costs(t: dict) -> dict:
    """Honest bottom line for a closed trade: gross, slippage, charges,
    net. PAPER simulates fills (slippage); LIVE uses reconciled fills.
    Single source of truth for every P&L number in the system."""
    entry = float(t.get("entry_price") or 0)
    qty = int(t.get("qty") or 0)
    exit_px = _planned_exit(t)
    if entry <= 0 or qty <= 0 or exit_px <= 0:
        return {"gross": 0.0, "slippage": 0.0, "charges": 0.0,
                "net": 0.0, "net_pct": 0.0}
    try:
        from execution.cost_model import net_result
        return net_result(entry, exit_px, qty,
                          product=(t.get("product") or "CNC"),
                          exit_is_stop=(t.get("status") in _CLOSED_LOSS),
                          paper=(t.get("mode") == "PAPER"))
    except Exception:
        gross = (exit_px - entry) * qty
        return {"gross": gross, "slippage": 0.0, "charges": 0.0,
                "net": gross, "net_pct": 0.0}


def _net_pnl(t: dict) -> float:
    """Realistic net P&L for a closed trade (gross − slippage − charges)."""
    return _trade_costs(t)["net"]


# ── Gates ─────────────────────────────────────────────────────────────────────

# Cash-session close for PAPER only. LIVE keeps the configured cut-off
# (default 14:45) so it never chases the last half-hour. Paper is the
# classroom — afternoon Ready tickets still have to be practiced.
PAPER_SESSION_END = "15:20"


def _norm_hm(value: str) -> str:
    raw = str(value or "").strip()
    try:
        parsed = datetime.strptime(raw, "%H:%M")
        return parsed.strftime("%H:%M")
    except ValueError:
        return raw


def _effective_end_time(s: dict | None = None) -> str:
    s = s or _load()
    end = _norm_hm(str(s.get("end_time") or "14:45"))
    if str(s.get("mode") or "PAPER").upper() == "PAPER":
        return end if end >= PAPER_SESSION_END else PAPER_SESSION_END
    return end


def _in_window(now: datetime | None = None) -> bool:
    s = _load()
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    hm = now.strftime("%H:%M")
    start = _norm_hm(str(s.get("start_time") or "09:30"))
    return start <= hm <= _effective_end_time(s)


def _top_sectors() -> tuple[list[str], str]:
    """Return (top positive sectors, status).

    status:
      ok — ranked list (may be empty if every sector is ≤0 today)
      unavailable — sector heat failed; callers must fail-open
    """
    try:
        from scan.sector_heat import sector_performance
        s = _load()
        perf = sector_performance()
        if not perf:
            return [], "unavailable"
        tops = [p["sector"] for p in perf[:s["sector_top_n"]] if p["chg_1d"] > 0]
        return tops, "ok"
    except Exception:
        return [], "unavailable"


def _open_in_sector(sector: str) -> int:
    """Kitni open autopilot positions isi sector ki hain (correlation cap)."""
    if not sector:
        return 0
    try:
        from scan.sector_heat import sector_of
        return sum(1 for t in _open_autopilot_trades()
                   if sector_of(t["symbol"]) == sector)
    except Exception:
        return 0


def _passes_gates(symbol: str, score: float, edge, sector: str,
                  rsi: float = 0.0, volume_ratio: float = 0.0,
                  source: str = "") -> str | None:
    """None = all gates pass; otherwise the (logged) rejection reason."""
    s = _load()
    if not s["armed"]:
        return "not armed"
    if not _in_window():
        return (
            f"time window ({s['start_time']}-{_effective_end_time(s)}) ke bahar"
        )
    # Fewer, higher-win: RSI blow-off ignored (same ceiling as sniper/scanner)
    if rsi > 70:
        return f"RSI {rsi:.0f} — blow-off-top, ignore (win-rate filter)"
    # Whole-board survival veto: if the Brain reads STAND_ASIDE (over-risked
    # book, or hostile tape + negative edge) no new entry today — regardless of
    # how good this one setup looks. Cached (~5 min); demote-only, never loosens.
    if s.get("brain_gate", True):
        posture, why = _brain_posture()
        if posture == "STAND_ASIDE":
            return f"Brain STAND_ASIDE — {why or 'survival-first, naye entries off'}"
    today = _ist_today().isoformat()
    if int(s.get("trades_today", {}).get(today, 0)) >= s["max_trades_per_day"]:
        return "daily trade limit reached"
    if len(_open_autopilot_trades()) >= s["max_open_positions"]:
        return "max open positions reached"
    if symbol in s.get("traded_symbols", {}).get(today, []):
        return "symbol already traded today"
    # 🧠 Symbol memory — is naam ne humein baar-baar kata hai (measured
    # negative expectancy on THIS stock). Serial false-breaker dobara nahi.
    if s.get("symbol_memory_gate", True) and symbol in _serial_losers_cached():
        return (f"symbol memory: {symbol} ka apna record negative hai "
                f"(serial false-breaker) — is naam se door")
    # Gate 14: sector concentration — ek sector ki N positions ek hi bet hai,
    # woh sector bika toh sab saath stop out. Survival > extra exposure.
    if sector and _open_in_sector(sector) >= s.get("max_per_sector", 2):
        return (f"sector '{sector}' mein already {s['max_per_sector']} "
                f"positions — concentration cap (ek sector = ek bet)")
    if score < s["min_score"]:
        return f"score {score:.0f} < min {s['min_score']:.0f}"
    if edge is not None and edge < 0:
        return f"negative measured edge ({edge:+.2f}R)"
    # Non-sniper path: volume < 1× is lower priority — skip so scarce daily
    # slots go to real volume breaks (sniper confirms volume at fire time).
    # Hard floor for every source — sniper confirms already require ≥1.0× avg-day;
    # scanner/Telegram paths must not sneak thin tape through.
    if volume_ratio > 0 and volume_ratio < 1.0:
        return f"volume {volume_ratio:.2f}× < 1× — hard reject"
    if volume_ratio <= 0 and source != "sniper":
        return "volume missing/zero — hard reject"
    tops, sector_status = _top_sectors()
    if sector_status == "unavailable":
        # Data outage must not freeze the whole desk — sector gate fail-open.
        pass
    elif tops and sector not in tops:
        return f"sector '{sector or '?'}' not in top {len(tops)} {tops}"
    elif not tops:
        return "no positive sector trend right now"
    if s.get("regime_gate", True):
        reg = _market_regime()
        if reg in _BAD_REGIMES:
            return f"market regime {reg} — breakout entries yahan fail hote hain"
    return None


# Regimes where fresh breakout longs have historically the worst odds.
# UNKNOWN passes (fail-open): the sector gate already demands a positive
# tape, and a data hiccup should not paralyse the system.
_BAD_REGIMES = ("DISTRIBUTION", "TRENDING_BEAR", "BEAR")

# Own 15-min cache: the ui.regime_bar cache is @st.cache_data, which is a
# NO-OP in the scan/sniper daemon threads (no Streamlit context) — without
# this, every candidate would recompute the full regime and slow the scan.
_regime_cache = {"ts": 0.0, "value": "UNKNOWN"}

# 🧠 Symbol-memory cache — serial false-breakers (DB hit once per ~15 min,
# per candidate nahi). Demote-only: naam sirf BLOCK ho sakta hai, promote kabhi.
_symmem_cache = {"ts": 0.0, "losers": set()}


def _serial_losers_cached() -> set:
    import time as _t
    now = _t.time()
    if now - _symmem_cache["ts"] < 900:
        return _symmem_cache["losers"]
    losers: set = set()
    try:
        from scan.live_edge import serial_losers
        losers = serial_losers()
    except Exception as exc:
        log.debug("symbol_memory_skip", error=str(exc))
    _symmem_cache.update(ts=now, losers=losers)
    return losers


# Brain posture cache — assess() composes many subsystems, so compute it ONCE
# per few minutes, not per candidate. The Brain is READ-ONLY; here it only ever
# adds a survival veto (STAND_ASIDE), never loosens a gate.
_brain_cache = {"ts": 0.0, "posture": "NORMAL", "reason": ""}


def _brain_posture() -> tuple[str, str]:
    import time as _t
    now = _t.time()
    if now - _brain_cache["ts"] < 300:
        return _brain_cache["posture"], _brain_cache["reason"]
    posture, reason = "NORMAL", ""
    try:
        from core.brain import assess
        a = assess(market="IN", capital=float(_load().get("allocation") or 0)
                   or 100_000.0)
        posture = a.get("posture", "NORMAL")
        reason = a.get("posture_reason", "")
    except Exception as exc:
        log.debug("brain_posture_skip", error=str(exc))
    _brain_cache.update(ts=now, posture=posture, reason=reason)
    return posture, reason


def _market_regime() -> str:
    import time as _t
    now = _t.time()
    if now - _regime_cache["ts"] < 900:
        return _regime_cache["value"]
    val = "UNKNOWN"
    try:
        from core.regime_engine import compute_regime   # streamlit-free path
        val = str(getattr(compute_regime(), "market_regime",
                          "UNKNOWN") or "UNKNOWN").upper()
    except Exception:
        try:
            from ui.regime_bar import get_regime
            val = str((get_regime() or {}).get("regime", "UNKNOWN")).upper()
        except Exception:
            val = "UNKNOWN"
    _regime_cache["ts"] = now
    _regime_cache["value"] = val
    return val


# ── Live price anchor: EOD/scan-time price pe trade NAHI hota ────────────────

def _anchor_live(symbol: str, entry: float, stop: float,
                 max_chase_pct: float) -> tuple[float | None, str]:
    """(live_entry, '') ya (None, reason). Har autopilot entry ka price
    order ke MOMENT ka live quote hota hai — signal ka entry sirf
    reference level hai:
      - live quote hi nahi (source down / off-hours) → NO TRADE, kabhi
        kal ke close pe order nahi jata
      - live stop ke neeche → setup already toota, entry bekaar
      - live entry-level se max_chase% upar → bhaagti train ka peecha
        nahi — extension pe buy karna edge kha jata hai
    """
    try:
        from data.live_quotes import get_live_quotes
        q = get_live_quotes([symbol]).get(symbol) or {}
    except Exception:
        q = {}
    px = float(q.get("price") or 0)
    if px <= 0:
        return None, "live quote nahi mila — EOD/stale price pe trade NAHI"
    if px <= stop:
        return None, (f"live ₹{px:,.1f} stop ₹{stop:,.1f} ke neeche — "
                      f"setup toota hua hai")
    if entry > 0 and px > entry * (1 + max_chase_pct / 100):
        return None, (f"live ₹{px:,.1f} entry ₹{entry:,.1f} se "
                      f"{(px / entry - 1) * 100:.1f}% upar — chase nahi karte")
    return px, ""


# ── Sizing ────────────────────────────────────────────────────────────────────

def _conviction_multiplier(score: float, edge, ev_conf=None) -> float:
    """Kelly-lite: risk scales with MEASURED conviction, never with vibes.

    Base 1.0. Only evidence moves it:
      score ≥ 80          → +0.25   (calibrated composite is strong)
      backtested edge     → +0.25 if ≥ +0.20R · −0.25 if < +0.05R
      edge is None (no backtest data, e.g. raw sniper hit) → −0.25:
      unmeasured conviction is LOW conviction by definition.
    Clamped to [0.5, 1.5] — the 1%-rule never more than 1.5× stretched,
    and a weak-but-passing signal risks half.
    """
    m = 1.0
    if score >= 80:
        m += 0.25
    if edge is None:
        m -= 0.25
    elif edge >= 0.20:
        m += 0.25
    elif edge < 0.05:
        m -= 0.25
    # EV-confidence extension: jab apne LIVE outcomes ka Wilson-shrunk EV
    # HIGH-confidence ho aur score strong, risk 2.0x tak stretch — "large
    # qty jab conviction bole", par sirf MEASURED conviction pe. Rails
    # (per-trade cap, reserve, available pool) upar se laagu rehte hain.
    if ev_conf == "HIGH" and score >= 75:
        m += 0.5
    return min(2.0 if ev_conf == "HIGH" else 1.5, max(0.5, m))


def _booking_enabled(s: dict | None = None) -> bool:
    """True when either capital-% AIM or absolute ₹ AIM is set."""
    s = s if s is not None else _load()
    return (float(s.get("profit_book_rupees") or 0) > 0
            or float(s.get("profit_book_pct") or 0) > 0)


def _book_quality_mult(score: float, grade: str = "",
                       breakout_conviction: float = 0.0,
                       fund_score=None) -> float:
    """Scale the 3% capital profit AIM by technicals + fundamentals.

    Weak setups book earlier (0.5×); A-grade + strong fund stretch toward
    1.25×. Never invents fundamentals — missing fund_score is neutral.
    """
    m = 0.70
    g = str(grade or "").upper()
    if g == "A":
        m += 0.25
    elif g == "B":
        m += 0.12
    if score >= 85:
        m += 0.15
    elif score >= 70:
        m += 0.08
    conv = float(breakout_conviction or 0)
    if conv >= 75:
        m += 0.15
    elif conv >= 55:
        m += 0.08
    if fund_score is not None:
        fs = float(fund_score)
        if fs >= 70:
            m += 0.12
        elif fs >= 50:
            m += 0.05
        elif fs < 35:
            m -= 0.10
    return max(0.50, min(1.25, round(m, 3)))


def _base_book_aim_rupees(s: dict | None = None, pool: float | None = None) -> float:
    """Unresolved AIM before quality scaling (absolute ₹ wins over %)."""
    s = s if s is not None else _load()
    rupees = float(s.get("profit_book_rupees") or 0)
    if rupees > 0:
        return rupees
    pct = float(s.get("profit_book_pct") or 0)
    if pct <= 0:
        return 0.0
    if pool is None:
        pool = float(get_status().get("pool") or 0)
    if pool <= 0:
        return 0.0
    return round(pool * pct / 100.0, 0)


def _book_plan_for_trade(score: float, grade: str = "",
                         breakout_conviction: float = 0.0,
                         fund_score=None,
                         s: dict | None = None) -> dict:
    """Per-trade AIM / floor / giveback in rupees, quality-scaled."""
    s = s if s is not None else _load()
    pool = float(get_status().get("pool") or 0)
    base = _base_book_aim_rupees(s, pool)
    if base <= 0:
        return {"aim": 0.0, "floor": 0.0, "giveback": 0.0, "quality_mult": 1.0}
    qm = _book_quality_mult(score, grade, breakout_conviction, fund_score)
    aim = round(base * qm, 0)
    abs_mode = float(s.get("profit_book_rupees") or 0) > 0
    if abs_mode:
        floor = float(s.get("profit_book_min_rupees") or 0)
        giveback = max(1.0, float(s.get("profit_trail_giveback_rupees") or 300))
    else:
        floor = round(pool * float(s.get("profit_book_min_pct") or 0) / 100.0 * qm, 0)
        giveback = max(
            1.0,
            round(pool * float(s.get("profit_trail_giveback_pct") or 0.6) / 100.0, 0),
        )
    floor = min(floor, aim) if aim > 0 else floor
    return {
        "aim": aim,
        "floor": floor,
        "giveback": giveback,
        "quality_mult": qm,
        "base_aim": base,
    }


def _size(entry: float, stop: float, mult: float = 1.0,
          cap_mult: float = 1.0) -> int:
    """Qty from 1%-of-pool risk (× conviction multiplier), capped by
    per-trade % and available cash — the caps always win over conviction.

    cap_mult > 1 sirf FAST-BOOK confirmed entries ke liye: tight structural
    stop ke saath rupee-risk waise hi 1% budget mein bandha hai, par 20%
    concentration cap ₹-booking ki speed ko clip kar deta tha. Relax bounded
    hai — user cap × cap_mult, HARD ceiling 40% of pool — aur exit exchange-
    side GTT + 60s monitor pe hai (exposure minutes-hours, overnight swing
    nahi)."""
    st = get_status()
    pool, available = st["pool"], st["available"]
    if pool <= 0 or available <= 0 or entry <= stop or entry <= 0:
        return 0
    risk_qty = int((pool * _load()["risk_per_trade_pct"] * mult) / (entry - stop))
    cap_pct = _load()["per_trade_cap_pct"]
    if cap_mult > 1.0:
        cap_pct = min(0.40, cap_pct * cap_mult)
    cap_qty = int((pool * cap_pct) / entry)
    avail_qty = int(available / entry)
    return max(0, min(risk_qty, cap_qty, avail_qty))


# ── Entry paths (hooked from scanner + sniper, additive) ─────────────────────

def consider(symbol: str, entry: float, stop: float, score: float,
             edge, sector: str, source: str, ev_pct=None, p_win=None,
             ev_conf=None, grade: str = "",
             breakout_conviction: float = 0.0,
             fund_score=None, rsi: float = 0.0,
             volume_ratio: float = 0.0,
             signal_target: float = 0.0) -> bool:
    """Full gate → size → execute path. Returns True if a trade was placed.

    Thread-safe: scanner + sniper hooks may call this concurrently; the
    lock guarantees the count-gates see every prior trade before passing.
    ev_pct/p_win/ev_conf: the prediction held at decision time — journaled
    with the decision so calibration can later audit our probabilities.
    """
    with _consider_lock:
        return _consider_locked(symbol, entry, stop, score, edge, sector,
                                source, ev_pct, p_win, ev_conf, grade,
                                breakout_conviction, fund_score, rsi,
                                volume_ratio, signal_target)


def _consider_locked(symbol: str, entry: float, stop: float, score: float,
                     edge, sector: str, source: str, ev_pct=None, p_win=None,
                     ev_conf=None, grade: str = "",
                     breakout_conviction: float = 0.0,
                     fund_score=None, rsi: float = 0.0,
                     volume_ratio: float = 0.0,
                     signal_target: float = 0.0) -> bool:
    def _jlog(decision: str, reason: str = "") -> None:
        # Evidence infrastructure: EVERY decision journaled with its
        # prediction — even rejections. "not armed" skipped (pure noise).
        if reason == "not armed":
            return
        try:
            from core.decision_journal import log_decision
            log_decision(symbol, decision, reason=_reject_category(reason)
                         if reason else "", source=source,
                         entry_ref=float(entry or 0), stop_ref=float(stop or 0),
                         score=float(score or 0), ev_pct=ev_pct, p_win=p_win,
                         confidence=ev_conf)
        except Exception:
            pass

    try:
        s = _load()
        # Scan results sirf "pack" (≥3 setups ek sector se) hone par sector
        # carry karte hain — akela strong stock bhi apne sector se judge ho,
        # isliye fallback lookup yahin karo (Gate 7 accuracy).
        if not sector:
            try:
                from scan.sector_heat import sector_of
                sector = sector_of(symbol)
            except Exception:
                sector = ""
        entry = float(entry)
        stop = float(stop)
        # This candidate reached the funnel — count it, so "kitne dekhe vs kitne
        # liye" (considered vs taken) is honest and visible on the page.
        _note_considered()
        # Invariant #3: har trade exchange-side exit ke saath. Bina real stop
        # (sniper hits from watchlist rows can carry stop=0) koi trade nahi —
        # stop invent karna fake data hai.
        if entry <= 0 or stop <= 0 or stop >= entry:
            log.debug("autopilot_reject", symbol=symbol,
                      reason=f"invalid stop {stop} for entry {entry}")
            _note_reject("invalid stop")
            _jlog("REJECTED", "invalid stop")
            return False

        reject = _passes_gates(symbol, score, edge, sector,
                               rsi=float(rsi or 0),
                               volume_ratio=float(volume_ratio or 0),
                               source=source)
        if reject:
            log.debug("autopilot_reject", symbol=symbol, reason=reject)
            _note_reject(reject)
            _jlog("REJECTED", reject)
            return False

        # Gate 12: source's OWN measured record — a feed that has proven
        # it loses money gets paused, evidence over vibes on ourselves too
        if s.get("adaptive_source_gate", True):
            paused = _source_paused(source)
            if paused:
                _log_activity(f"SKIP {symbol}: {paused}")
                _note_reject("source paused")
                _jlog("REJECTED", "source paused")
                return False

        # Gate 13: LIVE price anchor — order, target, sizing sab order ke
        # MOMENT ke price pe, kabhi kal ke close ya 15-min-purane scan pe nahi
        live_entry, why = _anchor_live(symbol, entry, stop,
                                       s.get("max_chase_pct", 1.0))
        if live_entry is None:
            _log_activity(f"SKIP {symbol}: {why}")
            _note_reject(why or "live quote")
            _jlog("REJECTED", why or "live quote")
            return False
        entry = live_entry

        # ⚡ Fast-book stop GEOMETRY — only when an explicit ₹/% book AIM is
        # armed (optional scalp). Default thesis_hold keeps the scanner stop.
        _fast_book = False
        _plan = _book_plan_for_trade(
            score, grade, breakout_conviction, fund_score, s,
        )
        if (_plan["aim"] > 0
                and (grade in ("A", "B") or source == "sniper")):
            atr_proxy = (entry - stop) / 2.0          # scanner stop = 2×ATR
            tight = entry - max(0.75 * atr_proxy, entry * 0.008)
            if tight > stop:
                stop = round(tight, 1)
                _fast_book = True

        # Thesis hold: wide runner GTT ceiling (hold while tech+fund good).
        # Scalp mode: fixed target_pct cut.
        if s.get("thesis_hold", True):
            from execution.thesis_hold import runner_target as _runner_target
            target = _runner_target(
                entry, float(signal_target or 0),
                float(s.get("runner_target_pct") or 10.0),
            )
            _tgt_bit = f"runner +{(target / entry - 1) * 100:.0f}%"
        else:
            target = round(entry * (1 + s["target_pct"] / 100), 1)
            _tgt_bit = f"+{s['target_pct']}%"
        mult = (_conviction_multiplier(score, edge, ev_conf)
                if s.get("conviction_sizing", True) else 1.0)
        qty = _size(entry, stop, mult, cap_mult=2.0 if _fast_book else 1.0)
        if qty < 1:
            _log_activity(f"SKIP {symbol}: pool/limits ke andar 1 share bhi "
                          f"nahi aata (entry ₹{entry:,.0f})")
            _note_reject("pool/1-share")
            _jlog("REJECTED", "pool/1-share")
            return False

        # LIVE: broker margin double-check — allocation par bharosa nahi,
        # asli cash dekho
        if s["mode"] == "LIVE":
            cash = _broker_cash()
            if cash is not None and qty * entry > cash:
                _log_activity(f"SKIP {symbol}: broker cash ₹{cash:,.0f} < "
                              f"required ₹{qty*entry:,.0f}")
                _note_reject("pool/1-share")
                return False

        from execution.trade_executor import place_trade
        res = place_trade(symbol=symbol, qty=qty, entry_type="MARKET",
                          entry_price=entry, stop=stop, target=target,
                          product="CNC", paper=(s["mode"] == "PAPER"),
                          note=f"{TAG}:{source}")
        if not res.get("ok"):
            _log_activity(f"FAIL {symbol}: {res.get('message', '')[:80]}")
            return False

        today = _ist_today().isoformat()
        with _lock:
            s = _load()
            s.setdefault("trades_today", {})
            s["trades_today"] = {today: int(s["trades_today"].get(today, 0)) + 1}
            s.setdefault("traded_symbols", {})
            s["traded_symbols"] = {
                today: s["traded_symbols"].get(today, []) + [symbol]}
            # Persist per-trade quality-scaled book plan (survives restarts)
            if _plan["aim"] > 0:
                opens = [t for t in _open_autopilot_trades()
                         if t.get("symbol") == symbol]
                if opens:
                    tid = str(max(int(t["id"]) for t in opens))
                    plans = dict(s.get("book_plans") or {})
                    plans[tid] = {
                        "aim": float(_plan["aim"]),
                        "floor": float(_plan["floor"]),
                        "giveback": float(_plan["giveback"]),
                        "quality_mult": float(_plan["quality_mult"]),
                    }
                    s["book_plans"] = plans
        _save()
        _book_amt = float(_plan.get("aim") or 0)
        _book_bit = ""
        if _book_amt > 0 and qty > 0:
            # NET-honest: charges ke baad ₹X bachane ke liye kitna GROSS move
            try:
                from execution.cost_model import gross_for_net
                _need_gross = gross_for_net(_book_amt, entry, qty)
            except Exception:
                _need_gross = _book_amt
            _mv = _need_gross / (qty * entry) * 100
            _floor = float(_plan.get("floor") or 0)
            _qm = float(_plan.get("quality_mult") or 1.0)
            _book_bit = (f" [NET ₹{_book_amt:,.0f} aim "
                         f"(quality {_qm:.2f}×) @ +{_mv:.1f}% "
                         f"(gross ₹{_need_gross:,.0f}), phir TRAIL, floor "
                         f"₹{_floor:,.0f}]")
        _log_activity(f"BUY {qty}×{symbol} @ ₹{entry:,.1f} "
                      f"(stop ₹{stop:,.1f} / target ₹{target:,.1f}) "
                      f"[{source}] [{s['mode']}] [conviction {mult:.2f}×]"
                      f"{_book_bit}")
        _notify(f"BUY <b>{qty} × {symbol}</b> @ ₹{entry:,.1f} ({s['mode']})\n"
                f"stop ₹{stop:,.1f} · target ₹{target:,.1f} ({_tgt_bit}) "
                f"· via {source}"
                + (" · thesis-hold" if s.get("thesis_hold", True) else ""))
        _jlog("TAKEN")
        return True
    except Exception as exc:
        log.warning("autopilot_consider_failed", symbol=symbol, error=str(exc))
        return False


def on_setups(results: list[dict]) -> None:
    """Hook: called after every scan with serialized results (read-only).

    Candidates are taken HIGHEST CONVICTION FIRST — daily trade slots and
    pool capital are limited, so the best sniper-quality setups claim them.
    High-RSI blow-offs are ignored; volume < 1× is deprioritized.
    """
    s = _load()
    if not s["armed"]:
        return
    try:
        from product.radar_workspace import (
            MIN_VOLUME_RATIO,
            RSI_BLOWOFF,
            breakout_quality_score,
            is_sniper_breakout_candidate,
            _volume_ratio,
        )
    except Exception:
        breakout_quality_score = lambda r: float(r.get("score") or 0)  # noqa: E731
        is_sniper_breakout_candidate = lambda r: False  # noqa: E731
        _volume_ratio = lambda r: float(r.get("volume_ratio") or 0)  # noqa: E731
        MIN_VOLUME_RATIO, RSI_BLOWOFF = 1.0, 70.0

    def _rank_key(r: dict):
        rsi = float(r.get("rsi") or 0)
        vol = _volume_ratio(r)
        conv = float(r.get("conviction_rank")
                     if r.get("conviction_rank") is not None
                     else float(r.get("score", 0) or 0))
        return (
            0 if rsi > RSI_BLOWOFF else 1,                 # ignore high RSI last
            1 if is_sniper_breakout_candidate(r) else 0,   # sniper first
            0 if (0 < vol < MIN_VOLUME_RATIO) else 1,      # vol≥1× before thin
            breakout_quality_score(r),
            conv,
        )

    paper = str(s.get("mode") or "PAPER").upper() == "PAPER"

    def _is_candidate(row: dict) -> bool:
        if row.get("verdict") in ("STRONG BUY", "BUY"):
            return True
        # Trade-desk Ready tickets (complete plan). PAPER classroom only —
        # LIVE still requires an explicit BUY / STRONG BUY.
        return paper and str(row.get("status") or "") == "Ready to trade"

    candidates = [r for r in results if _is_candidate(r)]
    ranked = sorted(candidates, key=_rank_key, reverse=True)[:40]
    for r in ranked:
        rsi = float(r.get("rsi") or 0)
        if rsi > RSI_BLOWOFF:
            continue  # hard ignore — don't spend a slot
        # PLANNED entry (breakout level) as the chase reference — the
        # live anchor inside consider() supplies the actual order price
        fund = r.get("fundamental_score")
        consider(symbol=r["symbol"], entry=float(r.get("entry") or r.get("price") or 0),
                 stop=float(r.get("stop") or 0), score=float(r.get("score") or 0),
                 edge=r.get("edge_r"), sector=r.get("sector") or "",
                 source="scanner", ev_pct=r.get("ev_pct"),
                 p_win=r.get("p_win"), ev_conf=r.get("ev_conf"),
                 grade=str(r.get("breakout_grade") or ""),
                 breakout_conviction=float(r.get("breakout_conviction") or 0),
                 fund_score=float(fund) if fund is not None else None,
                 rsi=rsi,
                 volume_ratio=float(r.get("volume_ratio") or 0),
                 signal_target=float(r.get("target") or 0))


def on_breakout(hit: dict) -> None:
    """Hook: called by the sniper for each fresh breakout (additive)."""
    s = _load()
    if not s["armed"]:
        return
    try:
        from scan.sector_heat import sector_of
        sector = sector_of(hit["symbol"])
    except Exception:
        sector = ""
    consider(symbol=hit["symbol"], entry=float(hit.get("ltp") or hit.get("trigger") or 0),
             stop=float(hit.get("stop") or 0), score=70.0,
             edge=None, sector=sector, source="sniper",
             signal_target=float(hit.get("target") or 0))


# ── P&L snapshot: frontend ke liye ek sach — live, realized, day ─────────────

def pnl_snapshot() -> dict:
    """Autopilot ka poora P&L ek call mein:
    - positions: har open trade LIVE price + unrealized ₹/% ke saath
    - unrealized: total on open positions (live quotes; quote na mile
      toh us position ka pnl None — stale ko zero bola nahi jata)
    - day_realized: aaj place hue closed trades ka P&L (actual exchange
      fill jahan reconciled, warna planned GTT leg)
    - realized_total: full compounded ledger (= pool - allocation)
    - day_pnl: day_realized + unrealized — aaj ka asli scoreboard
    """
    s = _load()
    open_trades = _open_autopilot_trades()
    live: dict = {}
    if open_trades:
        try:
            from data.live_quotes import get_live_quotes
            live = get_live_quotes(sorted({t["symbol"] for t in open_trades}))
        except Exception:
            live = {}
    positions: list[dict] = []
    unreal = 0.0
    for t in open_trades:
        entry = float(t["entry_price"] or 0)
        qty = int(t["qty"] or 0)
        q = live.get(t["symbol"])
        px = float(q["price"]) if q and q.get("price") else None
        pnl = round((px - entry) * qty, 0) if (px and entry) else None
        if pnl is not None:
            unreal += pnl
        positions.append({
            "symbol": t["symbol"], "mode": t["mode"], "qty": qty,
            "entry": entry, "live": px, "pnl": pnl,
            "pnl_pct": (round((px / entry - 1) * 100, 2)
                        if (px and entry) else None),
            "stop": float(t["stop_price"] or 0),
            "target": float(t["target_price"] or 0),
            "status": t["status"],
        })
    today = _ist_today().isoformat()
    day_realized = 0.0
    day_closed = 0
    for t in _autopilot_trades(_CLOSED_WIN + _CLOSED_LOSS):
        if _is_ist_today(t["placed_at"], today):    # IST-boundary safe
            day_realized += _net_pnl(t)         # net of costs
            day_closed += 1
    return {
        "positions": positions,
        "unrealized": round(unreal, 0),
        "day_realized": round(day_realized, 0),
        "day_closed": day_closed,
        "realized_total": round(float(s.get("realized_pnl", 0.0)), 0),
        "day_pnl": round(day_realized + unreal, 0),
    }


# ── Adaptive source gate: the autopilot judges its own feeds ─────────────────

_SOURCE_MIN_TRADES = 15      # evidence threshold before a source can be paused


def source_stats() -> dict[str, dict]:
    """{source: {n, wins, total_pnl, expectancy_r}} from CLOSED autopilot
    trades — scanner vs sniper, measured on our own record."""
    out: dict[str, dict] = {}
    for t in _autopilot_trades(_CLOSED_WIN + _CLOSED_LOSS):
        src = "sniper" if "sniper" in (t["note"] or "") else "scanner"
        entry = float(t["entry_price"] or 0)
        qty = int(t["qty"] or 0)
        stop = float(t.get("orig_stop") or 0) or float(t["stop_price"] or 0)
        pnl = _net_pnl(t)
        risk = (entry - stop) * qty
        d = out.setdefault(src, {"n": 0, "wins": 0, "total_pnl": 0.0, "_r": 0.0})
        d["n"] += 1
        d["wins"] += 1 if pnl > 0 else 0
        d["total_pnl"] = round(d["total_pnl"] + pnl, 0)
        d["_r"] += (pnl / risk) if risk > 0 else 0.0
    for d in out.values():
        d["expectancy_r"] = round(d.pop("_r") / d["n"], 2) if d["n"] else 0.0
    return out


def _source_paused(source: str) -> str | None:
    """Rejection reason if this source's own record (≥15 closed trades)
    shows negative expectancy — None otherwise. Fails open on any error."""
    try:
        key = "sniper" if "sniper" in (source or "") else "scanner"
        st = source_stats().get(key)
        if st and st["n"] >= _SOURCE_MIN_TRADES and st["expectancy_r"] < 0:
            return (f"source '{source}' apne hi record se paused "
                    f"({st['n']} trades, {st['expectancy_r']:+.2f}R) — "
                    f"pehle scanner/backtest sudharo")
    except Exception as exc:
        log.debug("autopilot_source_stats_failed", error=str(exc))
    return None


# ── Time-stop: dead money is a cost ──────────────────────────────────────────

def _trail_stop(peak: float, floor: float, giveback: float) -> float:
    """Pure formula (testable without DB/quotes): peak NET se `giveback`
    neeche, par `floor` se kabhi neeche nahi."""
    return max(floor, peak - giveback)


def _profit_book() -> None:
    """💰⛓️ NET-aware profit booking WITH A TRAIL.

    Teen level, saaf:
      AIM   — isse pehle book NAHI hota, chalne do
              (default = 3% of pool × quality; absolute ₹ overrides)
      FLOOR — isse KAM pe kabhi book nahi hota (armed hone ke baad worst-case)
      GIVEBACK — AIM cross hote hi trailing ARM; peak NET se itna neeche
                 aane par lock — runner ko chalne do, jo mila usse pakad lo

    trail_stop = max(FLOOR, peak_net − GIVEBACK). Per-trade plans (from
    technicals + fundamentals at entry) live in book_plans; missing plan
    falls back to the global base AIM.

    PAPER → turant close (PAPER_WIN, live price pe) — capital recycle.
    LIVE → GTT exchange pe baitha hai isliye auto-sell nahi; trail-stop hit
    hote hi Telegram nudge (ek baar per symbol per day): 'book now'.
    Trader math note: entry filter (prime/gates) hi asli edge hai, ye
    sirf uska disciplined cashier."""
    s = _load()
    if not _booking_enabled(s):
        return
    default_aim = _base_book_aim_rupees(s)
    if default_aim <= 0:
        return
    abs_mode = float(s.get("profit_book_rupees") or 0) > 0
    pool = float(get_status().get("pool") or 0)
    if abs_mode:
        default_floor = float(s.get("profit_book_min_rupees") or 0)
        default_give = max(1.0, float(s.get("profit_trail_giveback_rupees") or 300))
    else:
        default_floor = round(
            pool * float(s.get("profit_book_min_pct") or 0) / 100.0, 0)
        default_give = max(
            1.0,
            round(pool * float(s.get("profit_trail_giveback_pct") or 0.6) / 100.0, 0),
        )
    opens = _open_autopilot_trades()
    if not opens:
        return
    try:
        from data.live_quotes import get_live_quotes
        live = get_live_quotes(sorted({t["symbol"] for t in opens}))
    except Exception:
        return
    today = _ist_today().isoformat()
    open_ids = {str(t["id"]) for t in opens}
    with _lock:
        st0 = _load()
        trail = {k: v for k, v in dict(st0.get("trail_peaks") or {}).items()
                 if k in open_ids}              # stale entries prune (already closed)
        plans = {k: v for k, v in dict(st0.get("book_plans") or {}).items()
                 if k in open_ids}

    for t in opens:
        q = live.get(t["symbol"])
        if not (q and q.get("price")):
            continue                       # no quote → no claim, no action
        px = float(q["price"])
        entry = float(t["entry_price"] or 0)
        qty = int(t["qty"] or 0)
        if entry <= 0 or qty <= 0:
            continue
        pnl = (px - entry) * qty
        # 💸 NET: charges (STT/stamp/GST/DP) ke BAAD ka asli paisa. exit_price
        # RAW px hi store hota hai (report-card apne costs khud lagata hai —
        # double-count nahi).
        try:
            from execution.cost_model import zerodha_charges
            ch = float(zerodha_charges(entry, px, qty)["total"])
        except Exception:
            ch = 0.0
        net = pnl - ch
        tid = str(t["id"])
        if abs_mode:
            # Absolute ₹ override is global — ignore per-trade plan so
            # /book 1500 (and tests that set AIM after entry) stay exact.
            aim = default_aim
            floor = min(default_floor, aim)
            giveback = default_give
        else:
            plan = plans.get(tid) or {}
            aim = float(plan.get("aim") or default_aim)
            floor = float(plan.get("floor") if plan.get("floor") is not None
                          else default_floor)
            giveback = max(1.0, float(plan.get("giveback") or default_give))
            floor = min(floor, aim)                    # floor kabhi aim se upar nahi
        peak = max(trail.get(tid, 0.0), net)
        trail[tid] = peak

        if peak < aim:
            continue                       # abhi AIM tak nahi pahuncha — chalne do
        trail_stop = _trail_stop(peak, floor, giveback)
        if net > trail_stop:
            continue                       # trailing armed, abhi upar hi hai

        if t["mode"] == "PAPER":
            _ensure_exit_col()
            _update_trade(t["id"], "exit_price=?, status=?, note=note||?",
                          (px, "PAPER_WIN",
                           f" | trail-book NET ₹{net:,.0f} (peak ₹{peak:,.0f} "
                           f"se trail, gross ₹{pnl:,.0f} − charges ₹{ch:,.0f}) "
                           f"@ ₹{px:,.1f}"))
            _log_activity(f"💰 BOOKED {t['symbol']} NET ₹{net:+,.0f} "
                          f"(peak ₹{peak:,.0f} se trail-lock, charges "
                          f"₹{ch:,.0f} kat ke) @ ₹{px:,.1f} — capital recycle")
            _notify(f"💰 <b>NET ₹{net:,.0f} BOOKED</b> — {t['symbol']} "
                    f"@ ₹{px:,.1f} ({qty} sh, peak tha ₹{peak:,.0f}, "
                    f"charges ₹{ch:,.0f} minus). Trade band, capital wapas "
                    f"pool mein.")
            trail.pop(tid, None)
            plans.pop(tid, None)
        else:
            with _lock:
                st = _load()
                nudged = dict(st.get("book_nudges") or {})
                done = list(nudged.get(today, []))
                if t["symbol"] in done:
                    continue
                st["book_nudges"] = {today: done + [t["symbol"]]}
            _save()
            _notify(f"💰 <b>{t['symbol']} NET ₹{net:,.0f} book-ready</b> "
                    f"(LTP ₹{px:,.1f}, peak tha ₹{peak:,.0f}, gross "
                    f"₹{pnl:,.0f} − charges ₹{ch:,.0f}). Trail-stop hit — "
                    f"GTT cancel karke ticket se sell karo. "
                    f"(LIVE auto-sell nahi hota — tumhara click, tumhara "
                    f"paisa.)")
    with _lock:
        st = _load()
        st["trail_peaks"] = trail
        st["book_plans"] = plans
    _save()


def daily_scoreboard() -> dict:
    """🎯 MACHINE KA PURPOSE, ek number mein: aaj ka target vs aaj ki NET
    booking. 1+1=2 — koi kahani nahi, sirf hisaab:

      target  = max_trades_per_day × base_book_aim   (full-house goal)
      booked  = aaj ke CLOSED trades ka NET (charges ke baad — pnl_snapshot)
      slots   = kitne trades abhi baaki

    line: ek vaakya jo bataye machine aaj apna kaam kar rahi hai ya nahi.
    """
    s = _load()
    pnl = pnl_snapshot()
    book = _base_book_aim_rupees(s)
    max_day = int(s.get("max_trades_per_day") or 0)
    today = _ist_today().isoformat()
    taken = int(s.get("trades_today", {}).get(today, 0))
    booked_net = float(pnl.get("day_realized") or 0)
    booked_n = int(pnl.get("day_closed") or 0)
    target = round(book * max_day, 0) if book > 0 else 0.0
    slots = max(0, max_day - taken)
    pct = round(booked_net / target * 100, 1) if target > 0 else 0.0

    if not s.get("armed"):
        line = "🔴 Machine OFF — ARM karo (ya /resume), warna aaj ka target ₹0."
    elif book <= 0:
        if s.get("thesis_hold", True):
            line = ("🟢 Thesis-hold ON — jab tak technicals/fundamentals "
                    "theek hain hold; optional scalp: /book 3% ya /book 1500.")
        else:
            line = "⚪ Booking OFF — /book 3% set karo, machine ko target do."
    elif booked_net >= target > 0:
        line = f"🏆 FULL HOUSE — target ₹{target:,.0f} poora! Machine ne kaam kar diya."
    elif booked_n > 0:
        line = (f"🟢 Chal rahi hai — ₹{booked_net:,.0f} booked "
                f"({booked_n} trade), {slots} slots baaki.")
    elif taken > 0:
        line = (f"🟡 {taken} trade(s) khule/chal rahe — booking ka intezaar, "
                f"{slots} slots baaki.")
    else:
        line = f"⏳ Abhi koi trade nahi — {slots} slots ready, scan/sniper dhundh rahe."
    return {"armed": bool(s.get("armed")), "book": book, "max_day": max_day,
            "target": target, "booked_net": round(booked_net, 0),
            "booked_n": booked_n, "taken": taken, "slots": slots,
            "pct": pct, "line": line,
            "book_pct": float(s.get("profit_book_pct") or 0),
            "book_rupees": float(s.get("profit_book_rupees") or 0)}


# ── ⚡ Fast-exit monitor — booking/exits scan-cycle ka wait nahi karte ─────────
# Problem: review_cycle 15-30 min pe chalta hai — ₹1,500 ka momentum pop beech
# mein aake nikal sakta hai. Ye halka daemon sirf OPEN positions ke quotes
# har ~60s dekhta hai (5-6 symbols ka ek bulk call — sasta) aur:
#   • ₹-profit booking turant (PAPER close / LIVE nudge)
#   • paper stop/target crosses turant close (risk bhi fast)
# LIVE stop/target already exchange-side GTT hai — woh pehle se instant.

_monitor_started = False


def _market_open_ist() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= minutes <= 15 * 60 + 30


def _book_tick() -> int:
    """One fast-monitor tick. Returns seconds until the next tick.
    Active sirf jab armed + market open; warna halki neend (idle poll)."""
    try:
        s = _load()
        if not (s.get("armed") and _market_open_ist()):
            return 60                              # idle — sasta re-check
        # fast paper exits: stop/target crosses turant close hote hain
        try:
            from risk.position_manager import review_positions
            review_positions()
        except Exception:
            pass
        # ₹/%-booking turant (optional scalp — OFF by default)
        if _booking_enabled(s):
            _profit_book()
        # Extended RSI: tighten GTT, do not sell the winner
        _protect_extended_rsi()
        # Thesis hold exits when tech/fund break
        _thesis_exits()
        # 🐕 cross-watch: scan worker mar jaye toh yahan se khabar jaye
        # (watchdog ka apna 5-min throttle hai — yahan sasta hai)
        try:
            from core.watchdog import check as _wd_check
            _wd_check()
        except Exception:
            pass
        try:
            from core.eco import eco_on
            return 90 if eco_on() else 60
        except Exception:
            return 60
    except Exception as exc:
        log.debug("book_tick_failed", error=str(exc))
        return 60


def start_book_monitor() -> None:
    """Idempotent: fast-exit daemon ek hi baar. arm() + review_cycle dono
    bulate hain — restart ke baad armed state wapas aate hi khud zinda."""
    global _monitor_started
    with _lock:
        if _monitor_started:
            return
        _monitor_started = True

    def _loop() -> None:
        import time as _t
        while True:
            _t.sleep(max(15, _book_tick()))

    threading.Thread(target=_loop, name="fast-exit-monitor",
                     daemon=True).start()
    log.info("fast_exit_monitor_started")


def _lookup_thesis_context(symbol: str) -> tuple[dict | None, dict | None]:
    """Latest scan + long-term fund rows for a symbol (best-effort, no invent)."""
    scan_row = None
    fund_row = None
    try:
        from product.scan_store import load_scan
        payload = load_scan() or {}
        for row in payload.get("records") or []:
            if str(row.get("symbol") or "").upper() == symbol.upper():
                scan_row = dict(row)
                break
    except Exception:
        pass
    try:
        from product.long_term_store import load_long_term_scan
        lt = load_long_term_scan() or {}
        for row in lt.get("records") or []:
            if str(row.get("symbol") or "").upper() == symbol.upper():
                fund_row = dict(row)
                break
    except Exception:
        pass
    return scan_row, fund_row


def _thesis_status(t: dict, live_px: float) -> tuple[bool, str]:
    from execution.thesis_hold import evaluate_thesis
    scan_row, fund_row = _lookup_thesis_context(str(t.get("symbol") or ""))
    return evaluate_thesis(
        entry=float(t.get("entry_price") or 0),
        stop=float(t.get("stop_price") or 0),
        live_px=float(live_px or 0),
        scan_row=scan_row,
        fund_row=fund_row,
    )


def _orig_stop_value(t: dict, current_stop: float) -> float:
    raw = t.get("orig_stop")
    try:
        if raw not in (None, ""):
            val = float(raw)
            if val > 0:
                return val
    except (TypeError, ValueError):
        pass
    return float(current_stop or 0)


def _ensure_live_protective_gtt(t: dict, new_stop: float, ltp: float) -> bool:
    """Move or place the live GTT stop. True only when the exchange accepts."""
    gtt_id = str(t.get("gtt_id") or "").strip()
    if gtt_id and gtt_id.lower() not in ("none", "0"):
        t["gtt_id"] = gtt_id
        return _modify_gtt_stop(t, new_stop, ltp)
    try:
        from execution.trade_executor import _place_gtt, kite_ready
        if not kite_ready():
            return False
        from data.kite_client import KiteClient
        kite = KiteClient()
        placed_id, _msg = _place_gtt(
            kite, t["symbol"], int(t["qty"] or 0),
            ltp, new_stop, float(t["target_price"] or 0),
            t.get("product") or "CNC",
        )
        if not placed_id:
            return False
        t["gtt_id"] = str(placed_id)
        return True
    except Exception as exc:
        log.warning("autopilot_gtt_protect_place_failed",
                    symbol=t.get("symbol"), error=str(exc))
        return False


def _protect_extended_rsi() -> None:
    """Open trade + RSI extended → ratchet a 2–3% GTT below LTP, hold.

    High RSI is an ENTRY filter. On an already-open position it is not a
    sell: tighten protection and wait for real technical deterioration.
    Never loosens a stop that is already tighter than the 2–3% band.
    """
    from execution.thesis_hold import (
        clamp_rsi_protect_pct, evaluate_thesis, protective_stop,
        rsi_is_extended,
    )
    s = _load()
    if not s.get("thesis_hold", True):
        return
    pct = clamp_rsi_protect_pct(s.get("rsi_protect_pct"))
    opens = _open_autopilot_trades()
    if not opens:
        return
    try:
        from data.live_quotes import get_live_quotes
        live = get_live_quotes(sorted({t["symbol"] for t in opens}))
    except Exception:
        return
    _ensure_exit_col()
    for t in opens:
        q = live.get(t["symbol"])
        if not (q and q.get("price")):
            continue
        px = float(q["price"])
        scan_row, fund_row = _lookup_thesis_context(str(t.get("symbol") or ""))
        if not rsi_is_extended(scan_row):
            continue
        ok, _why = evaluate_thesis(
            entry=float(t.get("entry_price") or 0),
            stop=float(t.get("stop_price") or 0),
            live_px=px,
            scan_row=scan_row,
            fund_row=fund_row,
        )
        if not ok:
            continue  # real break — _thesis_exits handles it
        current = float(t.get("stop_price") or 0)
        new_stop = protective_stop(px, current, pct)
        if new_stop <= current + 1e-9 or new_stop >= px:
            continue
        if t.get("mode") == "LIVE" and not _ensure_live_protective_gtt(
                t, new_stop, px):
            continue
        orig = _orig_stop_value(t, current)
        gtt_set = ", gtt_id=?" if t.get("gtt_id") else ""
        gtt_param = (str(t["gtt_id"]),) if t.get("gtt_id") else ()
        note = (f" | RSI-protect GTT ₹{new_stop:,.1f} "
                f"({pct:g}% below LTP ₹{px:,.1f})")
        _update_trade(
            t["id"],
            f"orig_stop=?, stop_price=?, note=note||?{gtt_set}",
            (orig, new_stop, note, *gtt_param),
        )
        _log_activity(
            f"🛡 {t['symbol']} RSI-protect stop → ₹{new_stop:,.1f} "
            f"(RSI extended, hold — not a blow-off exit)"
        )
        _notify(
            f"🛡 <b>{t['symbol']}</b> RSI extended — holding. "
            f"Protective GTT ₹{new_stop:,.1f} "
            f"({pct:g}% below LTP ₹{px:,.1f}). "
            f"Exit only if structure breaks."
        )


def _thesis_exits() -> None:
    """Exit (paper) / nudge (live) when technicals or fundamentals break.

    Default hold policy: not a fixed ₹/% scalp — keep the trade while the
    thesis is intact, leave when it isn't.
    """
    s = _load()
    if not s.get("thesis_hold", True):
        return
    opens = _open_autopilot_trades()
    if not opens:
        return
    try:
        from data.live_quotes import get_live_quotes
        live = get_live_quotes(sorted({t["symbol"] for t in opens}))
    except Exception:
        return
    today = _ist_today().isoformat()
    for t in opens:
        q = live.get(t["symbol"])
        if not (q and q.get("price")):
            continue
        px = float(q["price"])
        ok, why = _thesis_status(t, px)
        if ok:
            continue
        entry = float(t["entry_price"] or 0)
        qty = int(t["qty"] or 0)
        pnl = (px - entry) * qty if entry and qty else 0.0
        if t["mode"] == "PAPER":
            _ensure_exit_col()
            outcome = "PAPER_WIN" if pnl >= 0 else "PAPER_LOSS"
            _update_trade(
                t["id"], "exit_price=?, status=?, note=note||?",
                (px, outcome,
                 f" | thesis-exit ({why}) NET ₹{pnl:+,.0f} @ ₹{px:,.1f}"),
            )
            _log_activity(
                f"{'💰' if pnl >= 0 else '🔻'} THESIS EXIT {t['symbol']} "
                f"₹{pnl:+,.0f} — {why}"
            )
            _notify(
                f"{'💰' if pnl >= 0 else '🔻'} <b>THESIS EXIT</b> — "
                f"{t['symbol']} @ ₹{px:,.1f}\n{why}"
            )
        else:
            with _lock:
                st = _load()
                nudged = dict(st.get("thesis_nudges") or {})
                done = list(nudged.get(today, []))
                if t["symbol"] in done:
                    continue
                st["thesis_nudges"] = {today: done + [t["symbol"]]}
            _save()
            _notify(
                f"⚠️ <b>{t['symbol']} thesis toot gaya</b> "
                f"(LTP ₹{px:,.1f}, P&amp;L ₹{pnl:+,.0f}).\n{why}\n"
                f"LIVE auto-sell nahi — GTT/ticket se exit tumhara call."
            )


def _time_stop() -> None:
    """Positions older than max_hold_days that hit neither stop nor target:
    PAPER → close at market (capital recycles, outcome recorded honestly).
    LIVE  → Telegram nudge only, once per day — autonomous selling of a
    live position is the user's call, not the machine's.

    Under thesis_hold: skip the time-force exit while technicals/fundamentals
    still look good — hold the runner.
    """
    s = _load()
    max_days = int(s.get("max_hold_days", 15))
    stale = []
    for t in _open_autopilot_trades():
        try:
            placed = datetime.fromisoformat(str(t["placed_at"]))
        except Exception:
            continue
        if (_now_ist_naive() - placed).days >= max_days:
            stale.append(t)
    if not stale:
        return
    try:
        from data.live_quotes import get_live_quotes
        live = get_live_quotes(sorted({t["symbol"] for t in stale}))
    except Exception:
        return
    today = _ist_today().isoformat()
    for t in stale:
        q = live.get(t["symbol"])
        if not (q and q.get("price")):
            continue
        px = float(q["price"])
        entry = float(t["entry_price"] or 0)
        if s.get("thesis_hold", True):
            ok, _why = _thesis_status(t, px)
            if ok:
                continue  # still healthy — keep holding
        if t["mode"] == "PAPER":
            _ensure_exit_col()
            status = "PAPER_WIN" if px >= entry else "PAPER_LOSS"
            _update_trade(t["id"], "exit_price=?, status=?, note=note||?",
                          (px, status,
                           f" | time-stop day {max_days} @ ₹{px:,.1f}"))
            _log_activity(f"⏳ TIME-STOP {t['symbol']} @ ₹{px:,.1f} "
                          f"(₹{(px-entry)*int(t['qty'] or 0):+,.0f}) — "
                          f"{max_days} din, thesis weak/flat, capital recycle")
        else:
            with _lock:
                nudged = _load().setdefault("time_nudges", {})
                done = nudged.get(today, [])
                if t["symbol"] in done:
                    continue
                nudged.clear()
                nudged[today] = done + [t["symbol"]]
            _save()
            _notify(f"⏳ <b>{t['symbol']}</b> {max_days}+ din se flat hai "
                    f"(LTP ₹{px:,.1f} vs entry ₹{entry:,.1f}). Capital "
                    f"recycle karna ho toh Kite se exit karo — LIVE position "
                    f"main khud nahi bechta.")


# ── Report Card: autopilot's own track record ────────────────────────────────

# Evidence thresholds — CLAUDE.md invariant: <30 trades = no claim.
_EVIDENCE_MIN_TRADES = 30
_READY_PROFIT_FACTOR = 1.3


def report_card() -> dict:
    """Autopilot ka apna scorecard — sirf uske trades, manual alag.

    Returns {trades, stats, verdict}: closed trades oldest-first with
    per-trade P&L / R-multiple / running equity, aggregate stats, and an
    evidence-gated verdict for the PAPER→LIVE decision. Exit prices are
    the journal's stop/target (the GTT legs), so P&L is the planned
    exit — fill-reconciliation with the Kite orderbook can tighten this
    later.
    """
    closed = _autopilot_trades(_CLOSED_WIN + _CLOSED_LOSS)
    trades: list[dict] = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in reversed(closed):                      # oldest → newest
        entry = float(t["entry_price"] or 0)
        qty = int(t["qty"] or 0)
        stop = float(t["stop_price"] or 0)
        win = t["status"] in _CLOSED_WIN
        exit_px = _planned_exit(t)
        costs = _trade_costs(t)
        pnl = round(costs["net"], 0)            # NET of slippage + charges
        # R uses the ORIGINAL stop — a breakeven-trailed stop would zero
        # the denominator and hide the risk that was actually taken
        risk_stop = float(t.get("orig_stop") or 0) or stop
        risk = (entry - risk_stop) * qty
        equity = round(equity + pnl, 0)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        trades.append({
            "date": str(t["placed_at"])[:10],
            "symbol": t["symbol"],
            "mode": t["mode"],
            "source": "sniper" if "sniper" in (t["note"] or "") else "scanner",
            "qty": qty,
            "entry": entry,
            "exit": exit_px,
            "win": pnl > 0,
            "pnl": pnl,
            "gross": round(costs["gross"], 0),
            "cost": round(costs["slippage"] + costs["charges"], 0),
            "r": round(pnl / risk, 2) if risk > 0 else 0.0,
            "equity": equity,
        })

    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    stats = {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
        "total_pnl": round(sum(t["pnl"] for t in trades), 0),
        "gross_pnl": round(sum(t["gross"] for t in trades), 0),
        "total_costs": round(sum(t["cost"] for t in trades), 0),
        "avg_win": round(gross_win / len(wins), 0) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 0) if losses else 0.0,
        "expectancy_r": round(sum(t["r"] for t in trades) / n, 2) if n else 0.0,
        "profit_factor": (round(gross_win / gross_loss, 2) if gross_loss
                          else (99.0 if gross_win else 0.0)),
        "max_drawdown": round(max_dd, 0),
        "paper_n": sum(1 for t in trades if t["mode"] == "PAPER"),
        "live_n": sum(1 for t in trades if t["mode"] == "LIVE"),
    }

    # Evidence-gated verdict — vibes se LIVE nahi jaana
    if n < _EVIDENCE_MIN_TRADES:
        verdict = ("COLLECTING_EVIDENCE",
                   f"Abhi koi claim nahi — {n}/{_EVIDENCE_MIN_TRADES} closed "
                   f"trades. PAPER mein chalne do.")
    elif stats["expectancy_r"] > 0 and stats["profit_factor"] >= _READY_PROFIT_FACTOR:
        verdict = ("READY_CANDIDATE",
                   f"{n} trades: expectancy {stats['expectancy_r']:+.2f}R, "
                   f"profit factor {stats['profit_factor']:.2f} — LIVE "
                   f"sochne layak record. Chhoti allocation se shuru karo.")
    else:
        verdict = ("NOT_READY",
                   f"{n} trades: expectancy {stats['expectancy_r']:+.2f}R, "
                   f"profit factor {stats['profit_factor']:.2f} — system "
                   f"abhi paisa deserve nahi karta. PAPER mein raho.")
    return {"trades": trades, "stats": stats, "by_source": source_stats(),
            "verdict": verdict[0], "verdict_reason": verdict[1]}


# ── Review cycle: closes, compounding, circuit breaker ───────────────────────

def review_cycle() -> None:
    """Called every scan cycle during market hours."""
    s = _load()
    try:
        from core.health import beat
        beat("autopilot", note=f"{'ARMED ' + s['mode'] if s['armed'] else 'off'}")
    except Exception:
        pass
    if s["allocation"] <= 0:
        return
    try:
        if s.get("armed"):
            start_book_monitor()     # self-heal: restart ke baad bhi zinda
    except Exception:
        pass
    try:
        _reconcile_live_fills()      # exchange truth first…
        _account_closed_trades()
        _close_crossed_live_trades() # …price-cross estimate as fallback
        _trail_stops()
        _profit_book()
        _protect_extended_rsi()
        _thesis_exits()
        _time_stop()
        _circuit_breaker()
    except Exception as exc:
        log.debug("autopilot_review_failed", error=str(exc))


# ── Fill reconciliation: journal follows the EXCHANGE, not estimates ──────────

def _reconcile_live_fills() -> None:
    """Sync open LIVE autopilot trades with the Kite orderbook:
    - entry COMPLETE → journal entry_price becomes the ACTUAL avg fill
    - entry REJECTED/CANCELLED → ENTRY_FAILED (frees the pool; position
      never existed, so it must not count as deployed)
    - a COMPLETE SELL of the same symbol+qty (a GTT leg firing) → close
      with the ACTUAL exit fill, so compounding uses exchange numbers
    Orderbook only covers today, so older exits still fall back to the
    price-cross estimate in _close_crossed_live_trades().

    The sell match is by symbol+qty — Kite's orderbook doesn't link a
    triggered order back to the GTT that spawned it. If TWO open trades
    share the same symbol+qty (re-entry after a stop-out, or a coincident
    sizing match), a single real fill can't be told apart between them —
    guessing wrong would falsely mark a still-open position as closed and
    drop it from all monitoring (trailing stop, profit-book, time-stop)
    while real capital is still on the table. So that group is skipped
    here entirely and flagged for a manual look, rather than guessed."""
    open_live = [t for t in _open_autopilot_trades() if t["mode"] == "LIVE"]
    if not open_live:
        return
    try:
        from data.kite_client import KiteClient
        kc = KiteClient()
        if not kc.is_connected():
            return
        orders = kc.get_orders() or []
    except Exception as exc:
        log.debug("autopilot_orderbook_failed", error=str(exc))
        return
    _ensure_exit_col()
    from collections import Counter
    _dupe_key = Counter((t["symbol"], int(t["qty"] or 0)) for t in open_live)
    for t in open_live:
        oid = str(t.get("entry_order_id") or "")
        od = next((o for o in orders
                   if str(o.get("order_id", "")) == oid), None) if oid else None
        if od:
            stt = str(od.get("status", "")).upper()
            if stt in ("REJECTED", "CANCELLED"):
                _set_status(t["id"], "ENTRY_FAILED", f"entry {stt} at broker")
                _log_activity(f"❌ {t['symbol']} entry {stt} — position bani "
                              f"hi nahi, pool free")
                continue
            if stt == "COMPLETE":
                avg = float(od.get("average_price") or 0)
                if avg > 0 and abs(avg - float(t["entry_price"] or 0)) >= 0.05:
                    _apply_entry_fill(t["id"], avg)
        if _dupe_key[(t["symbol"], int(t["qty"] or 0))] > 1:
            _log_activity(f"⚠️ {t['symbol']} — {_dupe_key[(t['symbol'], int(t['qty'] or 0))]} "
                          f"open LIVE trades share the same symbol+qty, exit-fill "
                          f"match ambiguous — skipping auto-reconcile, check manually")
            continue
        sells = [o for o in orders
                 if str(o.get("tradingsymbol", "")) == t["symbol"]
                 and str(o.get("transaction_type", "")).upper() == "SELL"
                 and str(o.get("status", "")).upper() == "COMPLETE"
                 and int(o.get("quantity") or 0) == int(t["qty"] or 0)
                 and str(o.get("order_id", "")) != oid]
        if sells:
            exit_px = float(sells[-1].get("average_price") or 0)
            if exit_px > 0:
                entry = float(t["entry_price"] or 0)
                status = "AUTO_WIN" if exit_px >= entry else "AUTO_LOSS"
                _update_trade(t["id"],
                              "exit_price=?, status=?, note=note||?",
                              (exit_px, status,
                               f" | exchange exit ₹{exit_px:,.2f}"))
                _log_activity(f"{'🟢' if exit_px >= entry else '🔴'} "
                              f"{t['symbol']} exit fill ₹{exit_px:,.2f} "
                              f"(exchange-reconciled)")


# ── Breakeven trailing: free-trade conversion ─────────────────────────────────

def _trail_stops() -> None:
    """Once a position is +breakeven_trigger_pct in profit, the stop moves
    to the entry price — worst case becomes a scratch, the +target_pct
    upside stays open. LIVE moves the GTT AT THE EXCHANGE first; the
    journal only follows a successful broker modify."""
    s = _load()
    if not s.get("trailing_enabled", True):
        return
    candidates = [t for t in _open_autopilot_trades()
                  if 0 < float(t["stop_price"] or 0) < float(t["entry_price"] or 0)]
    if not candidates:
        return
    try:
        from data.live_quotes import get_live_quotes
        live = get_live_quotes(sorted({t["symbol"] for t in candidates}))
    except Exception:
        return
    trigger = 1 + s.get("breakeven_trigger_pct", 2.0) / 100
    _ensure_exit_col()
    for t in candidates:
        q = live.get(t["symbol"])
        if not (q and q.get("price")):
            continue
        px = float(q["price"])
        entry = float(t["entry_price"] or 0)
        if entry <= 0 or px < entry * trigger:
            continue
        new_stop = round(entry, 1)
        if t["mode"] == "LIVE" and not _modify_gtt_stop(t, new_stop, px):
            continue                      # exchange refused — journal stays true
        _update_trade(t["id"],
                      "orig_stop=?, stop_price=?, note=note||?",
                      (float(t["stop_price"] or 0), new_stop,
                       f" | breakeven trail @ ₹{px:,.1f}"))
        _log_activity(f"🛡 {t['symbol']} stop → breakeven ₹{new_stop:,.1f} "
                      f"(LTP ₹{px:,.1f}, +{(px/entry-1)*100:.1f}%)")
        _notify(f"🛡 <b>{t['symbol']}</b> ab FREE TRADE hai — stop breakeven "
                f"₹{new_stop:,.1f} pe (LTP ₹{px:,.1f}). Worst case: scratch.")


def _modify_gtt_stop(t: dict, new_stop: float, ltp: float) -> bool:
    """Move the live GTT OCO's stop leg at the exchange. True on success."""
    try:
        from data.kite_client import KiteClient
        raw = KiteClient().raw
        qty = int(t["qty"] or 0)
        target = float(t["target_price"] or 0)
        product = t["product"] or "CNC"
        raw.modify_gtt(
            trigger_id=int(t["gtt_id"]),
            trigger_type=raw.GTT_TYPE_OCO,
            tradingsymbol=t["symbol"], exchange="NSE",
            trigger_values=[round(new_stop, 1), round(target, 1)],
            last_price=ltp,
            orders=[
                {"transaction_type": "SELL", "quantity": qty,
                 "order_type": "LIMIT", "product": product,
                 "price": round(new_stop * 0.995, 1)},
                {"transaction_type": "SELL", "quantity": qty,
                 "order_type": "LIMIT", "product": product,
                 "price": round(target, 1)},
            ])
        return True
    except Exception as exc:
        log.warning("autopilot_gtt_trail_failed", symbol=t.get("symbol"),
                    error=str(exc))
        return False


# ── EOD digest: din ka hisaab, khud aata hai ─────────────────────────────────

def eod_digest(now: datetime | None = None) -> bool:
    """Post-close daily summary on Telegram — once per trading day, only
    when there is something to report. Returns True if sent."""
    s = _load()
    if s["allocation"] <= 0:
        return False
    now = now or datetime.now(IST)
    if now.weekday() >= 5 or now.strftime("%H:%M") < "15:31":
        return False
    today = _ist_today().isoformat()
    if s.get("digest_date") == today:
        return False

    # Digest exchange truth, estimates nahi — close ke baad ke GTT fills
    # market-hours review cycle ne nahi dekhe hote
    try:
        _reconcile_live_fills()
        _account_closed_trades()
    except Exception as exc:
        log.debug("autopilot_digest_reconcile_skip", error=str(exc))

    placed = int(s.get("trades_today", {}).get(today, 0))
    open_trades = _open_autopilot_trades()
    closed_today = [t for t in _autopilot_trades(_CLOSED_WIN + _CLOSED_LOSS)
                    if _is_ist_today(t["placed_at"], today)]
    if not (placed or open_trades or closed_today or s["armed"]):
        return False                       # boring day, koi spam nahi

    wins = [t for t in closed_today if t["status"] in _CLOSED_WIN]
    day_pnl = sum(_net_pnl(t) for t in closed_today)      # net of costs
    unreal = 0.0
    if open_trades:
        try:
            from data.live_quotes import get_live_quotes
            live = get_live_quotes(sorted({t["symbol"] for t in open_trades}))
            for t in open_trades:
                q = live.get(t["symbol"])
                if q and q.get("price"):
                    unreal += ((float(q["price"]) - float(t["entry_price"] or 0))
                               * int(t["qty"] or 0))
        except Exception:
            pass
    st = get_status()
    rc = report_card()
    lines = [f"📒 <b>EOD Digest</b> — {now.strftime('%d %b %Y')} "
             f"({s['mode']}{', ARMED' if s['armed'] else ', off'})",
             f"Aaj: {placed} entries · {len(wins)}W/"
             f"{len(closed_today) - len(wins)}L closed (₹{day_pnl:+,.0f})",
             f"Open: {len(open_trades)} positions "
             f"(unrealized ₹{unreal:+,.0f})",
             f"Pool: ₹{st['pool']:,.0f} · available ₹{st['available']:,.0f} · "
             f"total realized ₹{s['realized_pnl']:+,.0f}",
             f"Report Card: {rc['verdict_reason']}"]
    _notify("\n".join(lines))
    with _lock:
        _load()["digest_date"] = today
    _save()
    _log_activity(f"EOD digest bheja ({placed} entries, ₹{day_pnl:+,.0f} closed)")
    return True


def _account_closed_trades() -> None:
    """Fold closed autopilot trades into realized_pnl (compounding).

    Tracked via a high-water mark (max trade id already folded in), not a
    bounded "seen ids" list — trade ids are monotonic (SQLite autoincrement)
    so a watermark can never lose track, however many trades accumulate.
    (A 500-id-capped list used to do this: once total closed trades passed
    500, the oldest ids fell out of the truncated window and got re-folded
    into realized_pnl on every subsequent cycle, forever — silently
    corrupting the compounding pool. Migrates any existing capped list to
    a watermark once, below.)"""
    closed = _autopilot_trades(_CLOSED_WIN + _CLOSED_LOSS)
    with _lock:
        s = _load()
        watermark = int(s.get("max_accounted_id", 0) or 0)
        old_ids = s.get("accounted_ids")
        if not watermark and old_ids:
            watermark = max(int(i) for i in old_ids)   # one-time migration
        changed = bool(old_ids)
        max_seen = watermark
        for t in closed:                      # ORDER BY id DESC — monotonic
            tid = int(t["id"])
            if tid <= watermark:
                break
            qty = int(t["qty"] or 0)
            pnl = round(_net_pnl(t), 0)         # net of slippage + charges
            s["realized_pnl"] = round(s.get("realized_pnl", 0.0) + pnl, 0)
            max_seen = max(max_seen, tid)
            changed = True
            _notify(f"CLOSED <b>{t['symbol']}</b> {'🟢 WIN' if pnl >= 0 else '🔴 LOSS'} "
                    f"₹{pnl:+,.0f} (net)\nPool ab: ₹{s['allocation'] + s['realized_pnl']:,.0f} "
                    f"(compounded)")
        s["max_accounted_id"] = max_seen
        s.pop("accounted_ids", None)
    if changed:
        _save()


def _close_crossed_live_trades() -> None:
    """LIVE autopilot trades: GTT executes at the exchange; we mark the
    journal when price crosses so the ledger compounds. (PAPER trades are
    closed by the position manager already.)"""
    open_live = [t for t in _open_autopilot_trades() if t["mode"] == "LIVE"]
    if not open_live:
        return
    try:
        from data.live_quotes import get_live_quotes
        live = get_live_quotes(sorted({t["symbol"] for t in open_live}))
    except Exception:
        return
    for t in open_live:
        q = live.get(t["symbol"])
        if not (q and q.get("price")):
            continue
        px = float(q["price"])
        if px >= float(t["target_price"] or 9e18):
            _set_status(t["id"], "AUTO_WIN", f"target cross @ ₹{px:,.1f}")
        elif px <= float(t["stop_price"] or 0):
            _set_status(t["id"], "AUTO_LOSS", f"stop cross @ ₹{px:,.1f}")


def _circuit_breaker() -> None:
    """Day P&L (realized-today + unrealized-open) breaches the limit → disarm."""
    s = _load()
    if not s["armed"]:
        return
    pool = s["allocation"] + s["realized_pnl"]
    if pool <= 0:
        return
    today = _ist_today().isoformat()

    day_realized = 0.0
    for t in _autopilot_trades(_CLOSED_WIN + _CLOSED_LOSS):
        if _is_ist_today(t["placed_at"], today):    # IST-boundary safe breaker
            day_realized += _net_pnl(t)         # net of costs — breaker on truth

    unrealized = 0.0
    open_trades = _open_autopilot_trades()
    if open_trades:
        try:
            from data.live_quotes import get_live_quotes
            live = get_live_quotes(sorted({t["symbol"] for t in open_trades}))
            for t in open_trades:
                q = live.get(t["symbol"])
                if q and q.get("price"):
                    unrealized += ((float(q["price"]) - float(t["entry_price"] or 0))
                                   * int(t["qty"] or 0))
        except Exception:
            pass

    day_pnl = day_realized + unrealized
    if day_pnl <= -pool * s["daily_loss_limit_pct"]:
        disarm(f"circuit breaker: day P&L ₹{day_pnl:,.0f} "
               f"({day_pnl/pool*100:.1f}% of pool)")
