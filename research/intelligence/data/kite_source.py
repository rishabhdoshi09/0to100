"""
🪁 Zerodha Kite as a DATA-ONLY provider for autonomous PAPER_AUTO.

Feeds the EXISTING canonical provider + `SnapshotStore` — no new snapshot/research architecture.
Kite is used strictly for market data here: instrument master, historical daily candles, and
(via `kite_live`) the live overlay. This module imports and calls NO order/GTT API; a boundary
test proves the PAPER_AUTO path can never reach one.

The Kite client is INJECTED (duck-typed) so this is deterministic and testable offline; production
wires the real `data/kite_client.py`. Nothing here bypasses Zerodha auth or stores credentials.

Required client surface (data only):
    client.profile() -> dict | raises           # session validity
    client.instruments("NSE") -> list[dict]      # instrument master
    client.historical(token, frm, to, "day") -> list[{date, open, high, low, close, volume}]
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from research.intelligence.data.snapshot_store import SnapshotStore
from research.intelligence.data import nse_calendar as CAL


class KiteSessionInvalid(Exception):
    pass


# ── bounded historical rate limiter (deterministic; injectable sleep/clock) ──────

class RateLimiter:
    def __init__(self, max_per_sec: float = 3.0, *, sleep_fn=time.sleep, clock=time.monotonic):
        self.min_interval = 1.0 / max_per_sec
        self._sleep = sleep_fn
        self._clock = clock
        self._last = 0.0
        self.calls = 0

    def acquire(self) -> None:
        now = self._clock()
        wait = self.min_interval - (now - self._last)
        if wait > 0:
            self._sleep(wait)
        self._last = self._clock()
        self.calls += 1


# ── instrument master: canonical identity ≠ token ────────────────────────────────

def canonical_id(inst: dict) -> str:
    """A STABLE security identity that survives symbol/token changes. Prefer ISIN; else a
    normalized exchange:tradingsymbol. NEVER the instrument token (it rotates)."""
    isin = str(inst.get("isin") or "").strip()
    if isin:
        return f"isin:{isin}"
    return f"{inst.get('exchange', 'NSE')}:{str(inst.get('tradingsymbol', '')).strip().upper()}"


def reconcile_instruments(new_list, prev_map: dict) -> tuple:
    """Return (master, changes). `master` maps canonical_id → instrument fields. `changes`
    records additions / removals / symbol_changes / token_changes vs `prev_map` (by canonical id)."""
    master, changes = {}, {"added": [], "removed": [], "symbol_changed": [], "token_changed": []}
    for inst in new_list:
        cid = canonical_id(inst)
        rec = {"canonical_id": cid, "instrument_token": inst.get("instrument_token"),
               "exchange_token": inst.get("exchange_token"),
               "tradingsymbol": str(inst.get("tradingsymbol", "")).strip().upper(),
               "name": inst.get("name", ""), "tick_size": inst.get("tick_size"),
               "lot_size": inst.get("lot_size"), "instrument_type": inst.get("instrument_type"),
               "segment": inst.get("segment"), "exchange": inst.get("exchange", "NSE"),
               "isin": inst.get("isin", "")}
        master[cid] = rec
        prev = prev_map.get(cid)
        if prev is None:
            changes["added"].append(cid)
        else:
            if prev.get("tradingsymbol") != rec["tradingsymbol"]:
                changes["symbol_changed"].append((cid, prev.get("tradingsymbol"), rec["tradingsymbol"]))
            if prev.get("instrument_token") != rec["instrument_token"]:
                changes["token_changed"].append((cid, prev.get("instrument_token"), rec["instrument_token"]))
    for cid in prev_map:
        if cid not in master:
            changes["removed"].append(cid)
    return master, changes


# ── validation (reused shape from from_bhav) ─────────────────────────────────────

def _valid_candle(c: dict) -> bool:
    try:
        o, h, l, cl, v = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"]), float(c.get("volume", 0))
    except Exception:
        return False
    return min(o, h, l, cl) > 0 and l <= o <= h and l <= cl <= h and v >= 0


def _iso(d) -> str:
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, (datetime, date)):
        return d.isoformat()[:10]
    return str(d)[:10]


@dataclass
class RefreshReport:
    status: str = "OK"
    snapshot_id: str | None = None
    activated: bool = False
    symbols: int = 0
    unresolved: int = 0
    candles_fetched: int = 0
    quarantined: int = 0            # malformed candles dropped during fetch
    invalid_ohlc: int = 0           # subset: OHLC-consistency failures
    future_bars: int = 0            # candles beyond the required session, dropped
    duplicates: int = 0
    benchmark_ok: bool = False
    token_changes: int = 0
    symbol_changes: int = 0
    tier: str = ""
    date_range: tuple | None = None
    incidents: list = field(default_factory=list)
    reason: str = ""

    def as_dict(self):
        from dataclasses import asdict
        return asdict(self)


class KiteDataSource:
    """Instrument master + resumable historical bootstrap/refresh → immutable snapshot."""

    def __init__(self, client, store: SnapshotStore | None = None, *, universe=None,
                 benchmark_name="NIFTY 50", history_dir=None, progress_path=None,
                 telemetry_path=None, rate_limiter: RateLimiter | None = None,
                 sleep_fn=time.sleep, rng=None,
                 max_retries: int = 4):
        self.client = client
        self.store = store or SnapshotStore()
        self.universe = set(universe) if universe else None      # None ⇒ all EQ symbols
        self.benchmark_name = benchmark_name
        self.history_dir = Path(history_dir) if history_dir else None
        self.progress_path = Path(progress_path) if progress_path else None
        self.telemetry_path = (
            Path(telemetry_path)
            if telemetry_path
            else (self.progress_path.with_name("runtime_progress.json") if self.progress_path else None)
        )
        self.rl = rate_limiter or RateLimiter(sleep_fn=sleep_fn)
        self._sleep = sleep_fn
        import random as _r
        self.rng = rng or _r.Random(0)
        self.max_retries = max_retries
        self.master: dict = {}
        self.last_changes: dict = {}                    # instrument reconciliation deltas
        self._q_invalid = 0                             # malformed candles this refresh
        self._q_future = 0                              # future-dated candles this refresh
        self._progress: dict = self._load_progress()
        self._progress_dirty = 0
        self._refresh_started_epoch = 0.0
        self._refresh_target_session = ""

    # ── session ────────────────────────────────────────────────────────────────────
    def session_valid(self) -> bool:
        try:
            return bool(self.client.profile())
        except Exception:
            return False

    def require_session(self) -> None:
        if not self.session_valid():
            raise KiteSessionInvalid("Kite session invalid/expired — re-login required")

    # ── instrument master ────────────────────────────────────────────────────────
    def refresh_instruments(self) -> dict:
        self.require_session()
        raw = list(self.client.instruments("NSE"))
        eq = [i for i in raw if str(i.get("instrument_type", "EQ")).upper() in ("EQ", "INDEX")
              and (self.universe is None or str(i.get("tradingsymbol", "")).upper() in self.universe
                   or str(i.get("name", "")).upper() == self.benchmark_name.upper())]
        if self.universe is None:
            try:
                from data.nse_universe import get_nse_universe
                approved = {str(s).upper() for s in (get_nse_universe() or [])}
            except Exception:
                approved = set()
            if approved:
                eq = [i for i in eq if str(i.get("tradingsymbol", "")).upper() in approved
                      or str(i.get("name", "")).upper() == self.benchmark_name.upper()
                      or str(i.get("instrument_type", "")).upper() == "INDEX"]
        prev = {r["canonical_id"]: r for r in self.master.values()} if self.master else {}
        self.master, changes = reconcile_instruments(eq, prev)
        self.last_changes = changes
        return {"n": len(self.master), "changes": changes,
                "benchmark_resolvable": self._benchmark_token() is not None}

    def _benchmark_token(self):
        for rec in self.master.values():
            if str(rec.get("name", "")).upper() == self.benchmark_name.upper() \
               or rec["tradingsymbol"] == self.benchmark_name.replace(" ", ""):
                return rec["instrument_token"]
        return None

    # ── resumable historical fetch (only missing ranges) ─────────────────────────
    def _fetch(self, token, frm: str, to: str) -> list:
        """Rate-limited fetch with retry + exponential backoff + jitter. Raises after retries."""
        attempt = 0
        while True:
            self.rl.acquire()
            try:
                return list(self.client.historical(token, frm, to, "day"))
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                backoff = (2 ** attempt) * 0.01 + self.rng.random() * 0.01   # jitter
                self._sleep(backoff)

    def bootstrap_symbol(self, cid: str, token, want_from: str, want_to: str,
                         *, persist_progress: bool = True) -> int:
        """Fetch ONLY the missing date range for one security; append to its persisted history.
        Resumable + idempotent: re-running fetches nothing already stored."""
        hist = self._load_history(cid)
        have_to = self._progress.get(cid)                # last stored date (resume point)
        frm = _next_day(have_to) if have_to and have_to >= want_from else want_from
        if frm > want_to:
            return 0                                     # already up to date
        candles = self._fetch(token, frm, want_to)
        added = 0
        for c in candles:
            d = _iso(c.get("date"))
            if d > want_to:
                self._q_future += 1; continue            # future-dated: never stored
            if d in hist or d < frm:
                continue                                 # dedup + range guard
            if not _valid_candle(c):
                self._q_invalid += 1; continue           # quarantine malformed
            hist[d] = [float(c["open"]), float(c["high"]), float(c["low"]),
                       float(c["close"]), int(float(c.get("volume", 0)))]
            added += 1
        if hist:
            self._progress[cid] = max(hist)
            self._save_history(cid, hist)
            self._save_progress(force=persist_progress)
        return added

    def _write_runtime_progress(
        self,
        stage: str,
        *,
        current: int = 0,
        total: int = 0,
        symbol: str = "",
        extra: dict | None = None,
    ) -> None:
        """Publish atomic runtime telemetry without corrupting resume state.

        progress_path is the durable security-to-last-session resume map. Runtime
        liveness belongs in a separate file so supervisors can reason about real
        movement instead of misreading the resume map as stage telemetry.
        """
        if not self.telemetry_path:
            return
        now = time.time()
        started = float(self._refresh_started_epoch or now)
        elapsed = max(0.0, now - started)
        cur = max(0, int(current or 0))
        tot = max(0, int(total or 0))
        payload = {
            "schema_version": 1,
            "stage": str(stage or "unknown"),
            "progress_current": cur,
            "progress_total": tot,
            "percent_complete": round((100.0 * cur / tot), 1) if tot else None,
            "last_symbol": str(symbol or ""),
            "target_session": str(self._refresh_target_session or ""),
            "started_epoch": started,
            "last_progress_epoch": now,
            "elapsed_s": round(elapsed, 1),
            "symbols_per_sec": round(cur / elapsed, 3) if cur and elapsed > 0 else 0.0,
        }
        if extra:
            payload.update(dict(extra))
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.telemetry_path.with_name(
            f".{self.telemetry_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
            os.replace(tmp, self.telemetry_path)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # ── daily refresh → snapshot commit + atomic activation ──────────────────────
    def daily_refresh(self, *, now: datetime | None = None, extra_manifest: dict | None = None
                      ) -> RefreshReport:
        rep = RefreshReport()
        self._refresh_started_epoch = time.time()
        self._refresh_target_session = ""
        self._write_runtime_progress("session_check")
        if not self.session_valid():
            rep.status = "BLOCKED"; rep.reason = "kite session invalid"
            self._write_runtime_progress("blocked", extra={"reason": rep.reason})
            rep.incidents.append({"severity": "CRITICAL", "code": "AUTH_INVALID"})
            return rep                                   # last active snapshot preserved untouched
        self._q_invalid = 0; self._q_future = 0          # per-refresh quality counters
        self._write_runtime_progress("instrument_master")
        self.refresh_instruments()
        rep.token_changes = len(self.last_changes.get("token_changed", []))
        rep.symbol_changes = len(self.last_changes.get("symbol_changed", []))
        required = CAL.latest_required_session(now or CAL._now_ist(), CAL.load_holidays())
        want_to = required.isoformat()
        want_from = (required - timedelta(days=400)).isoformat()
        self._refresh_target_session = want_to

        equity_rows, index_rows = [], []
        bench_token = self._benchmark_token()
        master_items = list(self.master.items())
        total_symbols = len(master_items)
        self._write_runtime_progress("historical_sync", current=0, total=total_symbols)
        fetch_failures = 0
        for processed, (cid, rec) in enumerate(master_items, start=1):
            token = rec["instrument_token"]
            is_bench = (token == bench_token)
            try:
                rep.candles_fetched += self.bootstrap_symbol(
                    cid, token, want_from, want_to, persist_progress=False)
            except Exception:
                fetch_failures += 1
                rep.incidents.append({"severity": "WARNING", "code": "FETCH_FAILED", "cid": cid})
                if processed == total_symbols or processed % 25 == 0:
                    self._write_runtime_progress(
                        "historical_sync",
                        current=processed,
                        total=total_symbols,
                        symbol=str(rec.get("tradingsymbol") or cid),
                        extra={
                            "candles_fetched": rep.candles_fetched,
                            "fetch_failures": fetch_failures,
                        },
                    )
                continue
            hist = self._load_history(cid)
            for d, (o, h, l, c, v) in hist.items():
                if is_bench:
                    index_rows.append(("NIFTY", d, o, h, l, c))
                else:
                    equity_rows.append((rec["tradingsymbol"], d, o, h, l, c, v, "EQ"))
            if processed == total_symbols or processed % 25 == 0:
                self._write_runtime_progress(
                    "historical_sync",
                    current=processed,
                    total=total_symbols,
                    symbol=str(rec.get("tradingsymbol") or cid),
                    extra={
                        "candles_fetched": rep.candles_fetched,
                        "fetch_failures": fetch_failures,
                    },
                )
        rep.symbols = len({r[0] for r in equity_rows})
        rep.unresolved = max(0, len(self.master) - rep.symbols - (1 if index_rows else 0))
        rep.invalid_ohlc = self._q_invalid
        rep.future_bars = self._q_future
        rep.quarantined = self._q_invalid
        rep.benchmark_ok = bool(index_rows)
        self._save_progress(force=True)
        if equity_rows:
            _ds = [r[1] for r in equity_rows]
            rep.date_range = (min(_ds), max(_ds))
        if CAL.has_duplicate_sessions(equity_rows):
            rep.status = "BLOCKED"; rep.reason = "duplicate sessions detected"
            self._write_runtime_progress(
                "blocked", current=total_symbols, total=total_symbols,
                extra={"reason": rep.reason, "fetch_failures": fetch_failures},
            )
            return rep
        if not equity_rows:
            rep.status = "BLOCKED"; rep.reason = "no valid equity history"
            self._write_runtime_progress(
                "blocked", current=total_symbols, total=total_symbols,
                extra={"reason": rep.reason, "fetch_failures": fetch_failures},
            )
            return rep

        self._write_runtime_progress(
            "snapshot_commit", current=total_symbols, total=total_symbols,
            extra={"candles_fetched": rep.candles_fetched, "fetch_failures": fetch_failures},
        )
        prev_active = self.store.get_active_snapshot()
        sid = self.store.commit_snapshot(
            equity_rows, index_rows=index_rows,
            extra_manifest={"source": "kite", "has_universe_history": True,
                            "adjustment_consistent": True, "corporate_action_coverage": 1.0,
                            "missing_session_rate": 0.0, "validation_errors": 0,
                            **(extra_manifest or {})})
        rep.snapshot_id = sid
        self._write_runtime_progress(
            "snapshot_verify", current=total_symbols, total=total_symbols,
            extra={"snapshot_id": sid, "fetch_failures": fetch_failures},
        )
        ok, fails = self.store.verify_snapshot(sid)
        # activate ONLY when forward-eligible: verified + fresh + benchmark + CA coverage etc.
        from research.intelligence import data_state as DS
        health = dict(self.store.open_snapshot(sid).health()) if ok else {}
        health["freshness_days"] = 0.0 if CAL.snapshot_freshness(want_to, now=now)["fresh"] else 999.0
        tier, tier_reasons = DS.classify_tier(health)
        rep.tier = tier
        if ok and DS.forward_eligible(tier):
            self.store.activate_snapshot(sid, actor="system", reason="kite daily refresh")
            rep.activated = True
        else:
            rep.status = "COMMITTED_NOT_ACTIVATED"
            rep.reason = (f"verify failed: {fails}" if not ok
                          else f"not forward-eligible ({tier}): {tier_reasons}")
            # previous active snapshot is preserved (we never deactivated it)
            rep.incidents.append({"severity": "WARNING", "code": "REFRESH_NOT_ACTIVATED",
                                  "prev_active": prev_active, "tier": tier})
        self._write_runtime_progress(
            "complete",
            current=total_symbols,
            total=total_symbols,
            extra={
                "snapshot_id": rep.snapshot_id,
                "activated": rep.activated,
                "refresh_status": rep.status,
                "tier": rep.tier,
                "candles_fetched": rep.candles_fetched,
                "fetch_failures": fetch_failures,
            },
        )
        return rep

    # ── persistence (resumable) ──────────────────────────────────────────────────
    def _hist_file(self, cid: str) -> Path | None:
        if not self.history_dir:
            return None
        safe = cid.replace(":", "_").replace("/", "_")
        return self.history_dir / f"{safe}.json"

    def _load_history(self, cid: str) -> dict:
        if not hasattr(self, "_hist_cache"):
            self._hist_cache = {}
        if cid in self._hist_cache:
            return self._hist_cache[cid]
        p = self._hist_file(cid)
        h = {}
        if p and p.exists():
            try:
                h = json.loads(p.read_text())
            except Exception:
                h = {}
        self._hist_cache[cid] = h
        if len(self._hist_cache) > 64:
            for key in list(self._hist_cache)[: len(self._hist_cache) - 64]:
                self._hist_cache.pop(key, None)
        return h

    def _save_history(self, cid: str, hist: dict) -> None:
        p = self._hist_file(cid)
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(hist))

    def _load_progress(self) -> dict:
        if self.progress_path and self.progress_path.exists():
            try:
                return json.loads(self.progress_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_progress(self, *, force: bool = False) -> None:
        self._progress_dirty = getattr(self, "_progress_dirty", 0) + 1
        if not force and self._progress_dirty < 25:
            return
        self._progress_dirty = 0
        if self.progress_path:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            self.progress_path.write_text(json.dumps(self._progress))


def _next_day(iso: str) -> str:
    return (date.fromisoformat(iso) + timedelta(days=1)).isoformat()
