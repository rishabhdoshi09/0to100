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
import math
import os
import sqlite3
import threading
from datetime import datetime, date

from logger import get_logger
from core.runtime_paths import logs_path

log = get_logger(__name__)

_STATE_FILE = logs_path("us_autopilot.json")
_lock = threading.RLock()
_consider_lock = threading.Lock()

TAG = "US_AUTOPILOT"


def _us_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()


def _us_session_date() -> str:
    return _us_now().date().isoformat()


def _trade_session_date(trade: dict) -> str:
    note = str(trade.get("note") or "")
    marker = "session="
    if marker in note:
        value = note.split(marker, 1)[1].split("|", 1)[0].strip()
        if len(value) >= 10:
            return value[:10]
    return str(trade.get("placed_at") or "")[:10]
_OPEN = ("PAPER_OPEN",)
_WIN = ("PAPER_WIN",)
_LOSS = ("PAPER_LOSS",)

_DEFAULTS = {
    "armed": True,
    "allocation": float(os.getenv("QT_US_PAPER_CAPITAL", "100000") or 100000),  # virtual USD paper pool
    "realized_pnl": 0.0,
    "cash_reserve_pct": 0.10,
    "per_trade_cap_pct": 0.10,
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
    "max_accounted_id": 0,         # legacy checkpoint; never infer closure order from IDs
    "accounted_ids": [],            # legacy capped list; accepted only during migration
    "accounted_close_ids": [],      # exact settled IDs; trades can close out of order
    "accounting_ids_migrated": False,
    "accounting_reconciliation_required": False,
    "state_recovery_required": False,  # runtime-only: this load failed closed
    "state_recovery_error": "",        # runtime-only: why (never read from disk)
    "activity": [],
    "disarmed_reason": "",
    "reject_stats": {},            # {YYYY-MM-DD: {category: count}} — the funnel
    "considered": {},              # {YYYY-MM-DD: candidates seen}
    "preset": "Balanced",          # aggressiveness dial (frequency, not safety)
}

_state: dict = {}

# The single recovery authority is the in-memory state itself:
# ``state_recovery_required`` / ``state_recovery_error`` describe the outcome of
# THIS process's load. They are never trusted from disk (see ``_load``), and a
# state carrying them is never written back (see ``_save``), so a damaged or
# lost recovery source can only be replaced by an operator, never by defaults.
_RECOVERY_REASON = "US_PAPER_STATE_RECOVERY_REQUIRED"
_RUNTIME_ONLY_KEYS = ("state_recovery_required", "state_recovery_error")


# ── Persistence ───────────────────────────────────────────────────────────────

def _require_runtime_storage() -> None:
    """Raise when the configured persistent runtime root is not present.

    ``core.runtime_paths`` is the canonical authority: on an installed host
    (``QT_RUNTIME_ROOT_REQUIRE_EXISTING``) it raises instead of falling back,
    so a detached external volume is never mistaken for a fresh install and its
    path is never recreated on the internal disk. In a dev checkout it is a
    no-op that returns the repository root.
    """
    from core.runtime_paths import runtime_root
    runtime_root()


def _has_us_trade_history() -> bool:
    """Check the journal *read-only* before treating a missing state as a new install.

    An inaccessible or malformed journal raises. The caller then fails closed;
    it must not infer zero paper positions from an unreadable database. A
    journal that exists but has never had a ``trades`` table holds no history.
    """
    from urllib.parse import quote
    from execution.trade_executor import _DB

    if not _DB.exists():
        return False
    uri = "file:" + quote(str(_DB.absolute()), safe="/") + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    try:
        conn.execute("PRAGMA query_only=ON")
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trades'"
        ).fetchone()
        if has_table is None:
            return False
        row = conn.execute(
            "SELECT 1 FROM trades WHERE note LIKE ? LIMIT 1",
            (f"%{TAG}%",),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _is_int_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_saved_state(payload: object) -> dict:
    """Do not convert damaged persisted consent/accounting into defaults."""
    if not isinstance(payload, dict):
        raise ValueError("US paper state is not a JSON object")
    if not isinstance(payload.get("armed"), bool):
        raise ValueError("US paper arm/disarm consent is absent or invalid")
    # Older *disarmed* state files legitimately lacked the daily dictionaries.
    # They remain disarmed. An armed state cannot silently lose its limits.
    for key in ("trades_today", "traded_symbols"):
        if key in payload and not isinstance(payload[key], dict):
            raise ValueError(f"US paper {key} is invalid")
        if payload["armed"] and key not in payload:
            raise ValueError(f"armed US paper state is missing {key}")
    if not isinstance(payload.get("max_accounted_id"), int) or isinstance(
        payload.get("max_accounted_id"), bool
    ):
        raise ValueError("US paper P&L watermark is absent or invalid")
    for key in ("allocation", "realized_pnl"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"US paper {key} is absent or invalid")
        if not math.isfinite(float(value)):
            raise ValueError(f"US paper {key} is not finite")
    # Exact-ID settlement ledger. A damaged list would either crash settlement
    # or, worse, re-book every closed trade's P&L into the pool.
    for key in ("accounted_ids", "accounted_close_ids"):
        if key in payload:
            ids = payload[key]
            if not isinstance(ids, list) or not all(_is_int_id(i) for i in ids):
                raise ValueError(f"US paper {key} is invalid")
    for key in ("accounting_ids_migrated", "accounting_reconciliation_required"):
        if key in payload and not isinstance(payload[key], bool):
            raise ValueError(f"US paper {key} is invalid")
    if payload.get("accounting_ids_migrated") is True and "accounted_close_ids" not in payload:
        raise ValueError("migrated US paper state lost its settled close IDs")
    return payload


def _latch_recovery(reason: str) -> None:
    """Fail closed in memory: no arming, no entries, no accounting, no saves."""
    _state["armed"] = False
    _state["allocation"] = 0.0          # unknown pool is not a $100k default
    _state["disarmed_reason"] = _RECOVERY_REASON
    _state["state_recovery_required"] = True
    _state["state_recovery_error"] = reason
    _state["accounting_reconciliation_required"] = True
    log.warning("us_autopilot_state_recovery_required", error=reason)


def _load() -> dict:
    global _state
    with _lock:
        if _state:
            return _state
        _state = dict(_DEFAULTS)
        migrated = False
        try:
            if _STATE_FILE.exists():
                on_disk = _validate_saved_state(
                    json.loads(_STATE_FILE.read_text(encoding="utf-8"))
                )
                _state.update({k: v for k, v in on_disk.items()
                               if k in _DEFAULTS and k not in _RUNTIME_ONLY_KEYS})
                # An untouched old zero-allocation default may be migrated.
                # Existing explicit disarm and accounting history must survive.
                if (
                    float(_state.get("allocation") or 0.0) <= 0.0
                    and not str(_state.get("disarmed_reason") or "").strip()
                    and not list(_state.get("activity") or [])
                    and int(_state.get("max_accounted_id") or 0) == 0
                    and not list(_state.get("accounted_close_ids") or [])
                ):
                    _state["allocation"] = float(
                        os.getenv("QT_US_PAPER_CAPITAL", "100000") or 100000
                    )
                    _state["armed"] = os.getenv(
                        "QT_US_PAPER_AUTO", "1"
                    ).strip().lower() not in {"0", "false", "off", "no"}
                    migrated = True
            else:
                # A missing state file is normal only on a truly fresh install.
                # A detached runtime volume also has no file: ask the runtime
                # authority first. Never reset virtual cash/P&L while an older
                # US trade journal survives; restoring the state is mandatory.
                _require_runtime_storage()
                if _has_us_trade_history():
                    raise ValueError("US paper state file missing with existing trade history")
        except Exception as exc:
            _state = dict(_DEFAULTS)
            _latch_recovery(type(exc).__name__ + ":" + str(exc)[:160])
            return _state
        _state["state_recovery_required"] = False
        _state["state_recovery_error"] = ""
        if migrated:
            _save()
            _log_activity("legacy zero-state migrated to autonomous virtual PAPER mode")
        return _state


def _save() -> bool:
    """Atomically persist the paper state; report failure to accounting callers.

    Serialize the full write under the process lock so a concurrent status or
    accounting update cannot overwrite a more recent checkpoint with stale JSON.
    An unrestored state is never written: the file on disk is the recovery source.
    """
    try:
        with _lock:
            data = dict(_load())
            if data.get("state_recovery_required"):
                log.warning("us_autopilot_save_blocked", reason=_RECOVERY_REASON)
                return False
            for key in _RUNTIME_ONLY_KEYS:
                data.pop(key, None)
            _require_runtime_storage()   # never recreate a vanished volume's path
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_FILE.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=1)
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(_STATE_FILE)
            try:
                dir_fd = os.open(str(_STATE_FILE.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass  # directory fsync is unsupported on some filesystems
        return True
    except Exception as exc:
        log.warning("us_autopilot_save_failed", error=str(exc))
        return False


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


# ── Rejection funnel — WHY so few US trades (transparency) ─────────────────────

def _reject_category(reason: str) -> str:
    r = (reason or "").lower()
    if "open risk" in r:          return "total open-risk cap (5%)"
    if "window" in r:             return "time window (US band)"
    if "daily trade limit" in r:  return "daily limit hit (max/day)"
    if "max open positions" in r: return "position limit full"
    if "already holding" in r:    return "symbol already held (10% per name)"
    if "already traded" in r:     return "symbol already traded today"
    if "conviction" in r:         return "conviction kam"
    if "score" in r:              return "score kam"
    if "quote" in r:              return "live quote nahi mila"
    if "neeche" in r:             return "setup toot gaya (live < stop)"
    if "chase" in r:              return "chase (live entry se upar)"
    if "share" in r or "pool" in r:   return "capital/1-share nahi aata"
    if "stop" in r:               return "invalid stop"
    return "other"


def _note_reject(reason: str) -> None:
    today = _us_session_date()
    with _lock:
        s = _load()
        rs = dict(s.get("reject_stats", {}).get(today, {}))   # copy, not in-place
        cat = _reject_category(reason)
        rs[cat] = rs.get(cat, 0) + 1
        s["reject_stats"] = {today: rs}
    _save()


def _note_considered() -> None:
    today = _us_session_date()
    with _lock:
        s = _load()
        n = int(s.get("considered", {}).get(today, 0)) + 1
        s["considered"] = {today: n}
    _save()


def reject_funnel() -> dict:
    today = _us_session_date()
    s = _load()
    return {"considered": int(s.get("considered", {}).get(today, 0)),
            "rejects": dict(s.get("reject_stats", {}).get(today, {}))}


# ── Journal (shared DB, US_AUTOPILOT tag) ─────────────────────────────────────

def _trades(statuses: tuple, *, strict: bool = False) -> list[dict]:
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
        if strict:
            raise
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
        log.warning("us_autopilot_update_failed", trade_id=trade_id, error=str(exc))
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
    s["state_recovery_error"] = s.get("state_recovery_error") or None
    opens = _open_trades()
    s["open_trades"] = opens
    s["deployed"] = round(sum(float(t["entry_price"] or 0) * int(t["qty"] or 0)
                              for t in opens), 2)
    s["pool"] = round(s["allocation"] + s["realized_pnl"], 2)
    s["available"] = round(
        max(0.0, s["pool"] * (1 - s["cash_reserve_pct"]) - s["deployed"]), 2)
    today = _us_session_date()
    s["trades_today_count"] = int(s.get("trades_today", {}).get(today, 0))
    return s


def set_config(**kwargs) -> bool:
    """Apply clamped config; False when nothing could be durably recorded."""
    if _load().get("state_recovery_required"):
        return False  # An operator must restore the persisted state before editing it.
    clamps = {
        "allocation": (0.0, 1e9), "cash_reserve_pct": (0.05, 0.50),
        "per_trade_cap_pct": (0.05, MAX_PER_NAME_PCT),
        "risk_per_trade_pct": (0.0025, MAX_RISK_PER_TRADE_PCT),
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
    return _save()


# ── Aggressiveness presets — breadth only, never the safety rails ─────────────
_PRESETS = {
    "Conservative": {
        "min_score": 70.0, "min_conviction": 60.0, "max_trades_per_day": 2,
        "max_open_positions": 2, "max_chase_pct": 0.75},
    "Balanced": {
        "min_score": 60.0, "min_conviction": 50.0, "max_trades_per_day": 4,
        "max_open_positions": 3, "max_chase_pct": 1.5},
    "Aggressive": {
        "min_score": 52.0, "min_conviction": 40.0, "max_trades_per_day": 8,
        "max_open_positions": 5, "max_chase_pct": 2.5},
}


def presets() -> list[str]:
    return list(_PRESETS)


def apply_preset(name: str) -> tuple[bool, str]:
    cfg = _PRESETS.get(name)
    if not cfg:
        return False, f"unknown preset '{name}'"
    with _lock:
        if _load().get("state_recovery_required"):
            return False, "US PAPER state recovery required; preset not applied"
        _load()["preset"] = name
        # One durable write carries the preset name and its thresholds.
        if not set_config(**cfg):
            return False, "US PAPER state could not be saved; preset not applied"
    _log_activity(f"PRESET → {name} (min_score {cfg['min_score']:.0f}, "
                  f"{cfg['max_trades_per_day']}/day, "
                  f"{cfg['max_open_positions']} open)")
    return True, f"{name} preset applied"


def arm() -> tuple[bool, str]:
    s = _load()
    if s.get("state_recovery_required"):
        return False, ("US PAPER state recovery required; restore the saved "
                       "state and restart before arming")
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
    if s.get("state_recovery_required"):
        return "US paper state recovery required"
    if not s["armed"]:
        return "not armed"
    if not _in_window():
        return f"US window ({s['start_time']}-{s['end_time']} ET) ke bahar"
    today = _us_session_date()
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


MAX_RISK_PER_TRADE_PCT = 0.01
MAX_PER_NAME_PCT = 0.10
MAX_TOTAL_OPEN_RISK_PCT = 0.05


def _size(entry: float, stop: float, mult: float) -> int:
    """1% of pool at risk (conviction may only scale it DOWN), 10% of pool
    per name. Persisted config above those ceilings is clamped here."""
    st = get_status()
    pool, available = st["pool"], st["available"]
    if pool <= 0 or available <= 0 or entry <= stop or entry <= 0:
        return 0
    s = _load()
    risk_pct = min(float(s["risk_per_trade_pct"]), MAX_RISK_PER_TRADE_PCT) * min(float(mult), 1.0)
    cap_pct = min(float(s["per_trade_cap_pct"]), MAX_PER_NAME_PCT)
    risk_qty = int((pool * risk_pct) / (entry - stop))
    cap_qty = int((pool * cap_pct) / entry)
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

def _record_learning_decision(
    *,
    symbol: str,
    action: str,
    reason: str,
    entry: float,
    stop: float,
    target: float | None,
    score: float,
    conviction: float,
    meta: dict | None,
) -> None:
    try:
        from product.us_learning import record_decision
        record_decision(
            symbol=symbol,
            action=action,
            reason=reason,
            entry=entry,
            stop=stop,
            target=target,
            score=score,
            conviction=conviction,
            meta=meta or {},
        )
    except Exception as exc:
        log.debug("us_learning_record_skip", symbol=symbol, error=str(exc)[:120])


def consider(
    symbol: str,
    entry: float,
    stop: float,
    score: float,
    conviction: float,
    source: str = "us_scanner",
    meta: dict | None = None,
) -> bool:
    with _consider_lock:
        return _consider_locked(symbol, entry, stop, score, conviction, source, meta)


def _consider_locked(symbol, entry, stop, score, conviction, source, meta=None) -> bool:
    original_entry = float(entry or 0.0)
    original_stop = float(stop or 0.0)
    target_hint = None
    try:
        if isinstance(meta, dict) and meta.get("target") not in (None, ""):
            target_hint = float(meta.get("target"))
    except Exception:
        target_hint = None

    def reject(reason: str) -> bool:
        _record_learning_decision(
            symbol=symbol,
            action="REJECT",
            reason=reason,
            entry=original_entry,
            stop=original_stop,
            target=target_hint,
            score=float(score or 0.0),
            conviction=float(conviction or 0.0),
            meta=meta,
        )
        return False

    try:
        s = _load()
        entry, stop = float(entry), float(stop)
        _note_considered()
        if entry <= 0 or stop <= 0 or stop >= entry:
            _note_reject("invalid stop")
            return reject("invalid stop")
        gate_reject = _passes_gates(symbol, score, conviction)
        if gate_reject:
            log.debug("us_autopilot_reject", symbol=symbol, reason=gate_reject)
            _note_reject(gate_reject)
            return reject(gate_reject)
        # The open book is the authority for per-name and total exposure. Read
        # it once, strictly: an unreadable book is unknown exposure, not zero.
        try:
            open_book = _trades(_OPEN, strict=True)
        except Exception as exc:
            reason = f"open-risk unknown (US paper book unreadable: {type(exc).__name__})"
            _log_activity(f"SKIP {symbol}: {reason}")
            _note_reject("open risk cap")
            return reject(reason)
        # traded_symbols only covers today's session. A name still held from an
        # earlier session already uses its 10% allocation; a second position
        # would double it.
        if any(str(t.get("symbol") or "") == symbol for t in open_book):
            reason = f"already holding an open US paper position in {symbol}"
            _note_reject(reason)
            return reject(reason)
        live_entry, why = _anchor_live(symbol, entry, stop, s["max_chase_pct"])
        if live_entry is None:
            _log_activity(f"SKIP {symbol}: {why}")
            _note_reject(why or "live quote")
            return reject(why or "live quote")
        entry = live_entry
        target = round(entry * (1 + s["target_pct"] / 100), 2)
        qty = _size(entry, stop, _conviction_mult(score, conviction))
        if qty < 1:
            reason = "pool/limits mein 1 share nahi aata"
            _log_activity(f"SKIP {symbol}: {reason}")
            _note_reject("pool/1-share")
            return reject(reason)
        open_risk = sum(
            int(t["qty"] or 0) * max(0.0, float(t["entry_price"] or 0) - float(t["stop_price"] or 0))
            for t in open_book
        )
        pool = float(get_status()["pool"])
        if open_risk + qty * (entry - stop) > pool * MAX_TOTAL_OPEN_RISK_PCT + 1e-6:
            reason = (f"total open risk ${open_risk + qty * (entry - stop):,.0f} would exceed "
                      f"{MAX_TOTAL_OPEN_RISK_PCT:.0%} of pool")
            _log_activity(f"SKIP {symbol}: {reason}")
            _note_reject("open risk cap")
            return reject(reason)

        from execution.trade_executor import place_trade
        res = place_trade(symbol=symbol, qty=qty, entry_type="MARKET",
                          entry_price=entry, stop=round(stop, 2), target=target,
                          product="CNC", paper=True, note=f"{TAG}:{source}|session={_us_session_date()}")
        if not res.get("ok"):
            reason = f"paper order failed: {res.get('message','')[:80]}"
            _log_activity(f"FAIL {symbol}: {res.get('message','')[:80]}")
            return reject(reason)
        today = _us_session_date()
        with _lock:
            s = _load()
            s["trades_today"] = {today: int(s.get("trades_today", {}).get(today, 0)) + 1}
            s["traded_symbols"] = {
                today: s.get("traded_symbols", {}).get(today, []) + [symbol]}
        _save()
        _record_learning_decision(
            symbol=symbol,
            action="TAKE",
            reason="ELIGIBLE",
            entry=entry,
            stop=stop,
            target=target,
            score=float(score or 0.0),
            conviction=float(conviction or 0.0),
            meta=meta,
        )
        _log_activity(f"BUY {qty}×{symbol} @ ${entry:,.2f} "
                      f"(stop ${stop:,.2f} / target ${target:,.2f}) [PAPER]")
        _notify(f"BUY <b>{qty} × {symbol}</b> @ ${entry:,.2f}\n"
                f"stop ${stop:,.2f} · target ${target:,.2f} (+{s['target_pct']}%)")
        return True
    except Exception as exc:
        log.warning("us_autopilot_consider_failed", symbol=symbol, error=str(exc))
        return reject(f"exception:{type(exc).__name__}")


def on_setups(results: list[dict]) -> None:
    s = _load()
    if not s["armed"]:
        return
    ranked = sorted(
        [r for r in results if r.get("verdict") in ("STRONG BUY", "BUY")],
        key=lambda r: float(
            r.get("learned_rank_score")
            or r.get("conviction_rank")
            or r.get("score", 0)
            or 0
        ),
        reverse=True)
    for r in ranked[:15]:
        consider(symbol=r["symbol"],
                 entry=float(r.get("entry") or r.get("price") or 0),
                 stop=float(r.get("stop") or 0),
                 score=float(r.get("score") or 0),
                 conviction=float(r.get("breakout_conviction") or 0),
                 source="us_scanner",
                 meta=dict(r))


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
    if s.get("state_recovery_required"):
        raise RuntimeError("US_PAPER_STATE_RECOVERY_REQUIRED")
    if s["allocation"] <= 0:
        return
    try:
        _close_positions()
        _account_closed()
        _circuit_breaker()
        from product.us_learning import settle_pending
        settle_pending()
    except Exception as exc:
        # US_MARKET_SCAN is responsible for marking the paper feed BLOCKED.
        # Swallowing this error previously reported a successful scan even when
        # settlement/accounting/learning had not completed.
        log.warning("us_autopilot_review_failed", error=str(exc)[:160])
        raise


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
    conn = None
    try:
        from execution.trade_executor import connect as _te_connect, ensure_schema
        conn = _te_connect()
        ensure_schema(conn)
    except Exception as exc:
        log.warning("us_autopilot_schema_upgrade_failed", error=str(exc))
        return
    finally:
        if conn is not None:
            conn.close()
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
            # A resting target fills AT the target; a later, higher poll is not the fill.
            _update(t["id"], "status=?, exit_price=?, note=note||?",
                    ("PAPER_WIN", target, f" | target ${target:,.2f} (seen ${px:,.2f})"))
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
    """Account each closed US paper trade exactly once, regardless of close order.

    A highest-ID watermark is unsafe: trade 12 may close before trade 11.
    During a one-time legacy migration preserve recorded realized P&L and seed
    exact IDs from trades already closed at or below the historical watermark.
    Existing open trades are deliberately NOT seeded, even with lower IDs.
    Older historical accounting cannot be reconstructed reliably from the
    watermark alone; expose that uncertainty for an explicit reconciliation.
    """
    notifications = []
    with _lock:
        s = _load()
        if s.get("state_recovery_required"):
            # Settlement against an unrestored ledger would re-book P&L that the
            # lost state already counted. Deferred is a failure, not a success.
            raise RuntimeError(_RECOVERY_REASON)
        closed = _trades(_WIN + _LOSS, strict=True)
        was_migrated = bool(s.get("accounting_ids_migrated"))
        accounted = {int(i) for i in (s.get("accounted_close_ids") or [])}
        previous_ids = {int(i) for i in (s.get("accounted_ids") or [])}
        old_watermark = int(s.get("max_accounted_id") or 0)
        changed = not was_migrated or bool(previous_ids)
        if not was_migrated:
            # Prior code used a high-water mark, or derived one from a capped
            # accounted_ids list. Preserve its recorded P&L without replaying
            # historical closures. Open positions remain eligible when closed.
            legacy_cutoff = max(old_watermark, max(previous_ids, default=0))
            accounted.update(previous_ids)
            accounted.update(int(t["id"]) for t in closed
                             if int(t["id"]) <= legacy_cutoff)
            needs_review = bool(legacy_cutoff)
        else:
            needs_review = bool(s.get("accounting_reconciliation_required"))

        realized = float(s.get("realized_pnl") or 0.0)
        highest = old_watermark
        for t in reversed(closed):
            tid = int(t["id"])
            if tid in accounted:
                continue
            pnl = round(_net_pnl(t), 2)
            realized = round(realized + pnl, 2)
            accounted.add(tid)
            highest = max(highest, tid)
            changed = True
            notifications.append(
                f"CLOSED <b>{t['symbol']}</b> {'🟢' if pnl >= 0 else '🔴'} "
                f"${pnl:+,.2f} (net)\nPool: ${s['allocation'] + realized:,.0f}"
            )

        if changed:
            before = dict(s)
            s["realized_pnl"] = realized
            s["accounted_close_ids"] = sorted(accounted)
            s["accounting_ids_migrated"] = True
            s["accounting_reconciliation_required"] = needs_review
            s["max_accounted_id"] = highest  # for backward visibility only
            s.pop("accounted_ids", None)
            if _save() is False:
                s.clear()
                s.update(before)
                raise OSError("US paper accounting checkpoint was not persisted")

    # Notify only after the checkpoint is durable; never announce phantom P&L.
    for message in notifications:
        _notify(message)


def _circuit_breaker() -> None:
    s = _load()
    if not s["armed"]:
        return
    pool = s["allocation"] + s["realized_pnl"]
    if pool <= 0:
        return
    today = _us_session_date()
    day = sum(_net_pnl(t) for t in _trades(_WIN + _LOSS)
              if _trade_session_date(t) == today)
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
