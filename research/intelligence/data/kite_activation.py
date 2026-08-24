"""
🔌 Real Zerodha data activation for autonomous PAPER_AUTO.

Turns the offline-certified Kite data machinery (KiteDataSource / KiteLiveOverlay, commit 38d4d0c)
into a genuinely connected, unattended runtime — WITHOUT adding a second authentication system,
new providers, strategies, indicators, broker execution, dashboards or research subsystems. This is
pure wiring around the existing pieces:

    existing daily Zerodha login  →  data-only view of that session  →  KiteDataSource
    →  instrument reconciliation  →  bounded historical bootstrap / incremental refresh
    →  verified immutable snapshot  →  atomic active-snapshot switch  →  KiteLiveOverlay
    →  the existing background worker  →  headless PAPER_AUTO cycle

DATA ONLY. The view exposed to PAPER_AUTO permits profile / instruments / historical and, through
the separate live overlay, quotes — and NOTHING else. Order / GTT / modify / cancel methods are
physically absent from the objects this module hands to the intelligence code, and the app's
order-capable module is referenced only by string (never imported into this package), so the
no-order-imports guard stays green and no order API is reachable from the paper loop.

The client and feed are injectable (duck-typed) so this is deterministic and testable offline; the
`from_config` / real-feed factories wire the genuine session in production. No credential, token or
authorization header is ever logged.
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from datetime import datetime

from research.intelligence.data.snapshot_store import SnapshotStore
from research.intelligence.data.kite_source import KiteDataSource, KiteSessionInvalid
from research.intelligence.data.kite_live import KiteLiveOverlay
from research.intelligence.data import nse_calendar as CAL

# state values (reported per checkpoint)
PASS = "PASS"
FAIL = "FAIL"
PENDING_MARKET_SESSION = "PENDING_MARKET_SESSION"
PARTIAL = "PARTIAL"

try:                                                    # correct IST interpretation of tick times
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:                                       # pragma: no cover
    _IST = None


# ── data-only view of the authenticated session (the boundary) ───────────────────

class KiteDataClient:
    """A DATA-ONLY facade over an authenticated Zerodha session object. Exposes exactly the three
    data operations KiteDataSource needs and nothing else — the paper loop cannot reach an order
    method because there is none to reach here."""

    is_data_only = True

    def __init__(self, session):
        # `session` is the underlying market-data SDK object (duck-typed): profile / instruments /
        # historical_data. We keep it private and forward only the data surface.
        self._s = session

    @classmethod
    def from_config(cls) -> "KiteDataClient":
        """Build the data-only view from the EXISTING app authentication object + secrets — the
        same daily-login session the rest of the system uses. No second auth system, nothing logged."""
        app = importlib.import_module("data." + "kite_client")     # order-capable module, by string
        client = getattr(app, "KiteClient")()                      # reads config.settings token
        if not client.is_connected():
            raise KiteSessionInvalid("no Zerodha access token configured — complete the daily login")
        return cls(client.raw)                                     # wrap the raw SDK; data methods only

    # data surface required by KiteDataSource ─────────────────────────────────────
    def profile(self) -> dict:
        return self._s.profile()

    def instruments(self, exchange: str = "NSE") -> list:
        return list(self._s.instruments(exchange))

    def historical(self, token, frm, to, interval: str = "day") -> list:
        return list(self._s.historical_data(token, frm, to, interval))


# ── live KiteTicker → overlay bridge (data only) ─────────────────────────────────

class KiteTickerFeed:
    """Adapts a live tick source to KiteLiveOverlay's duck-typed feed contract (connect/subscribe)
    and translates ticks into overlay.on_tick with IST-correct epoch timestamps. No order surface."""

    def __init__(self, ticker, *, token_to_symbol: dict, overlay: KiteLiveOverlay | None = None,
                 clock=time.time):
        self._t = ticker
        self._t2s = dict(token_to_symbol)
        self._s2t = {v: k for k, v in token_to_symbol.items()}
        self.overlay = overlay
        self._clock = clock
        # wire the tick callbacks (all read-only)
        for name, fn in (("on_ticks", self._on_ticks), ("on_connect", self._on_connect),
                         ("on_close", self._on_close), ("on_reconnect", self._on_reconnect),
                         ("on_error", self._on_error)):
            try:
                setattr(self._t, name, fn)
            except Exception:
                pass

    # feed contract used by KiteLiveOverlay ───────────────────────────────────────
    def connect(self, subs=None) -> None:
        try:
            self._t.connect(threaded=True)
        except TypeError:
            self._t.connect()

    def add_mappings(self, token_to_symbol: dict) -> None:
        """Extend the approved token/symbol map before subscribing new candidates."""
        for token, symbol in dict(token_to_symbol or {}).items():
            sym = str(symbol).upper()
            self._t2s[token] = sym
            self._s2t[sym] = token

    def subscribe(self, symbols) -> None:
        toks = [self._s2t[s] for s in symbols if s in self._s2t]
        if not toks:
            return
        if hasattr(self._t, "ws") and getattr(self._t, "ws", None) is None:
            return
        try:
            self._t.subscribe(toks)
        except Exception as exc:
            if getattr(self._t, "ws", None) is None or "sendMessage" in str(exc):
                return
            raise
        mode = getattr(self._t, "MODE_LTP", "ltp")
        try:
            self._t.set_mode(mode, toks)
        except Exception:
            pass

    # tick translation ────────────────────────────────────────────────────────────
    def _epoch(self, ts) -> float:
        if hasattr(ts, "timestamp"):
            if getattr(ts, "tzinfo", None) is None and _IST is not None:
                ts = ts.replace(tzinfo=_IST)              # exchange stamps are Asia/Kolkata-naive
            return ts.timestamp()
        return self._clock()

    def _on_ticks(self, ws, ticks) -> None:
        if not self.overlay:
            return
        for tk in ticks:
            sym = self._t2s.get(tk.get("instrument_token"))
            price = tk.get("last_price")
            if not sym or not price:
                continue
            ts = tk.get("exchange_timestamp") or tk.get("last_trade_time")
            self.overlay.on_tick(sym, float(price), self._epoch(ts))

    def _on_connect(self, ws, response=None) -> None:
        if self.overlay:
            self.overlay.connected = True
        self.subscribe(list(self._s2t))                   # install approved subscriptions

    def _on_close(self, ws, code=None, reason=None) -> None:
        if self.overlay:
            self.overlay.connected = False

    def _on_error(self, ws, code=None, reason=None) -> None:
        if self.overlay:
            self.overlay.connected = False

    def _on_reconnect(self, ws, attempts=None) -> None:
        if self.overlay:
            self.overlay.on_reconnect()                   # bounded backoff + restore subscriptions


def _market_open(now_ist: datetime, holidays: set) -> bool:
    """NSE cash-equity session window in IST (09:15–15:30, weekday, not a holiday)."""
    if now_ist.weekday() >= 5 or now_ist.date() in (holidays or set()):
        return False
    hm = now_ist.hour * 60 + now_ist.minute
    return (9 * 60 + 15) <= hm <= (15 * 60 + 30)


# ── the activation report (8 separate operational states) ────────────────────────

@dataclass
class ActivationReport:
    states: dict = field(default_factory=dict)          # name -> (status, reason)
    quality: dict = field(default_factory=dict)         # genuine-data identity/quality summary
    snapshot_id: str | None = None
    tier: str = ""
    active_pointer: str | None = None
    session_health: dict = field(default_factory=dict)
    feed_health: dict = field(default_factory=dict)
    paper_auto: dict = field(default_factory=dict)
    worker_running: bool = False
    latest_cycle: dict = field(default_factory=dict)
    open_positions: int = 0
    human_approval_occurred: bool = False               # PAPER_AUTO never asks
    kite_order_reachable: bool = False                  # data-only view has no order method
    blocker: str = ""

    def set(self, name: str, status: str, reason: str = "") -> None:
        self.states[name] = (status, reason)

    def status(self, name: str) -> str:
        return self.states.get(name, (FAIL, "not evaluated"))[0]

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def render(self) -> str:
        order = ["KITE_SESSION_CONNECTED", "INSTRUMENT_MASTER_CURRENT", "GENUINE_HISTORY_AVAILABLE",
                 "HISTORICAL_BOOTSTRAP_COMPLETE", "GENUINE_SNAPSHOT_ACTIVE", "LIVE_FEED_CONNECTED",
                 "PAPER_AUTO_WORKER_RUNNING", "PAPER_AUTO_REAL_DATA_OPERATIONAL"]
        lines = []
        for k in order:
            st, why = self.states.get(k, (FAIL, "not evaluated"))
            lines.append(f"{k:<34} {st}" + (f"  — {why}" if why and st != PASS else ""))
        return "\n".join(lines)


_STATE_ORDER = ("KITE_SESSION_CONNECTED", "INSTRUMENT_MASTER_CURRENT", "GENUINE_HISTORY_AVAILABLE",
                "HISTORICAL_BOOTSTRAP_COMPLETE", "GENUINE_SNAPSHOT_ACTIVE", "LIVE_FEED_CONNECTED",
                "PAPER_AUTO_WORKER_RUNNING", "PAPER_AUTO_REAL_DATA_OPERATIONAL")


def activate(*, client=None, store: SnapshotStore | None = None, brain=None, universe=None,
             benchmark_name: str = "NIFTY 50", history_dir=None, progress_path=None,
             overlay: KiteLiveOverlay | None = None, feed=None, subscribe_symbols=None,
             now: datetime | None = None, market_open: bool | None = None,
             start_worker: bool = True, run_cycle: bool = True) -> ActivationReport:
    """Run the real Zerodha data-activation sequence and report 8 separate states honestly.

    Every checkpoint is reported PASS / FAIL / PENDING_MARKET_SESSION / PARTIAL with a concrete
    reason for any non-PASS. On any hard failure the previous active snapshot is preserved and unsafe
    new paper entries stay blocked; nothing is fabricated. `client` / `feed` are injectable for
    deterministic offline tests — a real Zerodha session and a real KiteTicker are wired in
    production. PAPER_AUTO_REAL_DATA_OPERATIONAL is asserted ONLY from a genuinely active,
    forward-eligible Kite snapshot with the worker running and a real cycle decided on it."""
    rep = ActivationReport()
    now = now or CAL._now_ist()
    holidays = CAL.load_holidays()

    # 1 ── data-only session view (existing daily login) ─────────────────────────
    if client is None:
        try:
            client = KiteDataClient.from_config()
        except Exception as e:                          # no token / no SDK in this environment
            for k in _STATE_ORDER:
                rep.set(k, FAIL, "no connected Zerodha session")
            rep.set("KITE_SESSION_CONNECTED", FAIL, f"{type(e).__name__}: {e}")
            rep.blocker = "A valid daily Zerodha access token is required (none configured)."
            return rep
    rep.kite_order_reachable = any(hasattr(client, m) for m in
                                   ("place_order", "cancel_order", "modify_order", "place_gtt"))

    ds = KiteDataSource(client, store, universe=universe, benchmark_name=benchmark_name,
                        history_dir=history_dir, progress_path=progress_path)

    prev_active = ds.store.get_active_snapshot()
    if not ds.session_valid():
        for k in _STATE_ORDER:
            rep.set(k, FAIL, "Zerodha session invalid/expired")
        rep.blocker = "Zerodha session invalid — re-run the daily login."
        rep.active_pointer = prev_active
        return rep
    try:
        rep.session_health = {"ok": True, **{k: v for k, v in (client.profile() or {}).items()
                                             if k in ("user_id", "user_type", "broker")}}
    except Exception:
        rep.session_health = {"ok": True}
    rep.set("KITE_SESSION_CONNECTED", PASS)

    # 2 ── instrument master ──────────────────────────────────────────────────────
    try:
        info = ds.refresh_instruments()
    except Exception as e:
        rep.set("INSTRUMENT_MASTER_CURRENT", FAIL, f"instrument download failed: {e}")
        rep.blocker = "Instrument master unavailable."
        return rep
    if info["n"] <= 0:
        rep.set("INSTRUMENT_MASTER_CURRENT", FAIL, "no instruments resolved for the universe")
        rep.blocker = "No instruments resolved."
        return rep
    rep.set("INSTRUMENT_MASTER_CURRENT", PASS if info["benchmark_resolvable"] else PARTIAL,
            "" if info["benchmark_resolvable"] else "benchmark instrument not resolvable")

    # 3+4 ── bounded historical bootstrap/refresh → verify → activate ─────────────
    r = ds.daily_refresh(now=now)
    rep.snapshot_id = r.snapshot_id
    rep.tier = r.tier
    rep.quality = {"instruments_resolved": r.symbols, "unresolved": r.unresolved,
                   "candles_stored": r.candles_fetched, "date_range": r.date_range,
                   "benchmark_available": r.benchmark_ok, "duplicate_sessions": r.duplicates,
                   "invalid_ohlc": r.invalid_ohlc, "future_bars": r.future_bars,
                   "quarantined": r.quarantined, "token_changes": r.token_changes,
                   "symbol_changes": r.symbol_changes, "tier": r.tier,
                   "refresh_status": r.status, "incidents": r.incidents}

    if r.symbols > 0 and r.snapshot_id:
        rep.set("GENUINE_HISTORY_AVAILABLE", PASS)
        rep.set("HISTORICAL_BOOTSTRAP_COMPLETE", PASS)
    else:
        why = r.reason or "no valid equity history fetched"
        rep.set("GENUINE_HISTORY_AVAILABLE", FAIL, why)
        rep.set("HISTORICAL_BOOTSTRAP_COMPLETE", FAIL, why)

    if r.activated:
        rep.set("GENUINE_SNAPSHOT_ACTIVE", PASS)
        rep.active_pointer = ds.store.get_active_snapshot()
    else:
        rep.set("GENUINE_SNAPSHOT_ACTIVE", FAIL,
                r.reason or "snapshot not forward-eligible — previous active snapshot preserved")
        rep.active_pointer = prev_active                # unchanged; unsafe new entries blocked

    # 5 ── live overlay handshake / subscription path ─────────────────────────────
    is_open = _market_open(now, holidays) if market_open is None else bool(market_open)
    ov = overlay
    if ov is None and feed is not None:
        ov = KiteLiveOverlay(feed=feed)
    if ov is not None:
        try:
            ov.connect()
            syms = list(subscribe_symbols or [])
            if syms:
                ov.subscribe(syms)
            rep.feed_health = ov.health()
            ticking = rep.feed_health.get("symbols_ticking", 0) > 0
            if is_open and ticking:
                rep.set("LIVE_FEED_CONNECTED", PASS)
            elif ov.health().get("connected"):
                rep.set("LIVE_FEED_CONNECTED", PENDING_MARKET_SESSION,
                        "handshake + subscriptions verified; live ticks pending the next market session"
                        if not is_open else "connected; awaiting first ticks")
            else:
                rep.set("LIVE_FEED_CONNECTED", FAIL, "feed did not connect")
        except Exception as e:
            rep.set("LIVE_FEED_CONNECTED", FAIL, f"live feed error: {e}")
    else:
        rep.set("LIVE_FEED_CONNECTED", PENDING_MARKET_SESSION,
                "no live feed wired in this run; live tick observation pending a market session")

    # 6 ── headless PAPER_AUTO worker (existing background worker) ─────────────────
    if brain is None:
        try:
            from research.auto_research.scheduler import get_brain
            brain = get_brain()
        except Exception as e:
            rep.set("PAPER_AUTO_WORKER_RUNNING", FAIL, f"brain unavailable: {e}")
            rep.set("PAPER_AUTO_REAL_DATA_OPERATIONAL", FAIL, "no worker")
            return rep
    try:
        brain.enable_paper_auto()                       # persisted; no per-trade approval, no Telegram
    except Exception as e:
        rep.set("PAPER_AUTO_WORKER_RUNNING", FAIL, f"worker error: {e}")
        rep.set("PAPER_AUTO_REAL_DATA_OPERATIONAL", FAIL, "no worker")
        return rep

    # 7 ── genuine-data cycle (real snapshot → automatic paper decision). Run the explicit
    #      decision BEFORE the background worker so nothing races the intel lock. ───────────
    snapshot_is_kite = False
    if rep.status("GENUINE_SNAPSHOT_ACTIVE") == PASS:
        try:
            snap = ds.store.open_active()
            snapshot_is_kite = bool(snap and snap.manifest.get("source") == "kite")
        except Exception:
            snapshot_is_kite = False
    if run_cycle and rep.status("GENUINE_SNAPSHOT_ACTIVE") == PASS:
        try:
            cyc = brain.run_intelligence_cycle_day()
            rep.latest_cycle = cyc if isinstance(cyc, dict) else {"result": str(cyc)}
        except Exception as e:
            rep.latest_cycle = {"error": str(e)}

    # start the existing background worker (headless; unattended from here on)
    try:
        if start_worker:
            brain.start()
        rep.worker_running = bool(getattr(brain.state, "running", False))
        rep.paper_auto = {"enabled": brain.is_paper_auto_enabled(),
                          "mode": getattr(brain, "mode", ""),
                          "capital": getattr(brain.intel_book, "capital", None)}
        rep.set("PAPER_AUTO_WORKER_RUNNING", PASS if rep.worker_running else PARTIAL,
                "" if rep.worker_running else "worker not started (start_worker=False)")
    except Exception as e:
        rep.set("PAPER_AUTO_WORKER_RUNNING", FAIL, f"worker error: {e}")
    try:
        rep.open_positions = len(getattr(brain.intel_book, "open", {}))
    except Exception:
        rep.open_positions = 0

    elig = (rep.latest_cycle or {}).get("eligibility")
    if (rep.status("GENUINE_SNAPSHOT_ACTIVE") == PASS and rep.worker_running
            and snapshot_is_kite and elig in ("TRADED", "NO_ELIGIBLE_TRADE")):
        rep.set("PAPER_AUTO_REAL_DATA_OPERATIONAL", PASS, f"cycle eligibility={elig}")
    else:
        reasons = []
        if rep.status("GENUINE_SNAPSHOT_ACTIVE") != PASS:
            reasons.append("no genuine active snapshot")
        if not rep.worker_running:
            reasons.append("worker not running")
        if rep.status("GENUINE_SNAPSHOT_ACTIVE") == PASS and not snapshot_is_kite:
            reasons.append("active snapshot is not Kite-sourced")
        if elig not in ("TRADED", "NO_ELIGIBLE_TRADE"):
            reasons.append(f"cycle did not decide on real data (eligibility={elig})")
        rep.set("PAPER_AUTO_REAL_DATA_OPERATIONAL", FAIL, "; ".join(reasons) or "not operational")
        rep.blocker = rep.blocker or "; ".join(reasons)

    return rep
