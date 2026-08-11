"""
⏱️ Headless scheduler — the brain runs itself.

A tiny background daemon that fires `run_cycle` on an interval, threads the ResearchThread
and LearningLedger across cycles (so the system remembers and improves), and stays alive
across errors. No human has to press anything: start it once and it keeps thinking.

It is deliberately dumb about *safety*: it can only call the autonomous loop, which parks
proposals at the human gate and never approves, activates, or trades. The scheduler adds no
new powers — it just repeats the honest loop.

`AutoResearchBrain.run_once()` is fully synchronous and deterministic (used by tests and by
the UI's "think once now" button). `start()` spawns a daemon thread for real 24/7 use.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from research.auto_research.loop import run_cycle, CycleReport, canonical_readiness
from research.auto_research.learning import LearningLedger
from research.auto_research.knowledge import Knowledge
from research.auto_research.growth import calibrate, FORWARD_PENDING
from research.auto_research.paper_autonomy import PaperAutonomyManager
from research.auto_research.thread import ResearchThread
from research.strategy_studio import discovery as DISC
from research.strategy_studio import spec as S


@dataclass
class BrainState:
    cycles_run: int = 0
    days_grown: int = 0
    last_grow_date: str = ""
    last_report: dict | None = None
    running: bool = False
    last_error: str = ""
    total_proposals: int = 0
    paper_autonomy: bool = False
    paper_deployed: int = 0
    paper_retired: int = 0
    last_intel_cycle: dict | None = None


class AutoResearchBrain:
    """Owns the long-lived research memory and drives cycles."""

    def __init__(self, thread_path=None, *, evaluate_fn=None, budget=None,
                 interval_s: float = 3600.0, clock=None,
                 dataset_status_fn=canonical_readiness,
                 paper=None, signal_fn=None, bars_fn=None, knowledge=None,
                 regime_fn=None, paper_state_path=None,
                 intel_registry_fn=None, event_store_path=None,
                 runtime_state_path=None, mode="PAPER_AUTO",
                 intel_book_path=None, paper_config_path=None):
        self.thread = (ResearchThread(thread_path, clock=clock) if clock
                       else ResearchThread(thread_path))
        self.ledger = LearningLedger()
        self.knowledge = knowledge if knowledge is not None else Knowledge()
        self.registry = DISC.AttemptRegistry()
        self.evaluate_fn = evaluate_fn
        self.budget = budget
        self.interval_s = interval_s
        self.dataset_status_fn = dataset_status_fn
        # paper autonomy (opt-in): the brain trades its own strategies in PAPER, hands-off.
        self.paper = paper if paper is not None else PaperAutonomyManager()
        self.signal_fn = signal_fn        # (PaperStrategy, date) -> list[signal dict]
        self.bars_fn = bars_fn            # (date) -> {symbol: (high, low, close)}
        self.regime_fn = regime_fn        # () -> "RISK_ON"/"RISK_OFF" (deployment gate)
        self.paper_state_path = paper_state_path
        if paper_state_path:              # resume the book + journal across restarts
            try:
                self.paper.load(paper_state_path)
            except Exception:
                pass
        # ── two-brain intelligence runtime (Phase O) — its OWN book + event store so it
        # never fights the legacy growth path. In-memory by default (tests); production
        # wires disk paths via get_brain(). Honest no-op until a real registry is provided.
        import threading as _th
        from research.intelligence.event_store import EventStore as _ES
        from research.intelligence.runtime.runtime_state import RuntimeState as _RS
        from research.auto_research.paper_book import PaperBook as _PB
        from research.auto_research.costs import india_cash_costs as _costs
        self.mode = mode
        self.intel_registry_fn = intel_registry_fn   # (as_of) -> ctx-building dict, or None
        self.snapshot_store = None                   # set by get_active_snapshot wiring / prod
        self.strategy_registry = None                # populated from frozen specs in production
        self.event_store = _ES(event_store_path)
        self.runtime_state = _RS(runtime_state_path)
        self.intel_book = _PB(slippage_bps=3.0, cost_model=_costs)
        self.intel_book_path = intel_book_path
        self.paper_config_path = paper_config_path
        # persistent PAPER_AUTO enablement: survives restart, so ordinary restart never
        # reverts it or asks for a new click. Paper config is NOT real-money authorization.
        self.paper_auto_enabled = self._load_paper_config()
        # RECOVERY-FIRST: restore the paper book (open positions) then reconcile, so a restart
        # resumes management/exits of existing positions before opening any new ones.
        if intel_book_path:
            try:
                import json as _json
                from pathlib import Path as _P
                p = _P(intel_book_path)
                if p.exists():
                    self.intel_book.restore(_json.loads(p.read_text()))
                    self.runtime_state.reconcile(self.intel_book)
            except Exception:
                pass
        self._intel_lock = _th.Lock()                # prevents overlapping mutation jobs
        self._insample_cache: dict = {}              # (snapshot_id, strategy_id) -> (R, n)
        self.state = BrainState()
        self._specs_by_family: dict = {}
        self._thread_obj: threading.Thread | None = None
        self._stop = threading.Event()

    # ── paper autonomy switches (a human turns this on deliberately) ─────────────
    def engage_paper_autonomy(self) -> None:
        self.paper.engage(); self.state.paper_autonomy = True
        self.thread.decide(self.state.cycles_run,
                           "Full PAPER autonomy ENGAGED by the user. I will now auto-deploy "
                           "survivors to paper, trade them, and retire proven losers on my "
                           "own. Live stays locked; only simulated money is at risk.",
                           {"paper_autonomy": True})

    def disengage_paper_autonomy(self) -> None:
        self.paper.disengage(); self.state.paper_autonomy = False

    # ── persistent PAPER_AUTO enable/disable (survives restart, no re-click) ──────
    def enable_paper_auto(self) -> None:
        self.paper_auto_enabled = True
        self._save_paper_config()

    def disable_paper_auto(self) -> None:
        """The user's explicit opt-out. Persisted, so a restart honours it."""
        self.paper_auto_enabled = False
        self._save_paper_config()

    def is_paper_auto_enabled(self) -> bool:
        return bool(self.paper_auto_enabled) and self.mode in ("PAPER_AUTO", "PAPER_FORWARD_EVIDENCE")

    def _load_paper_config(self) -> bool:
        if not self.paper_config_path:
            return True                              # default ON (paper only; not real money)
        try:
            import json as _json
            from pathlib import Path as _P
            p = _P(self.paper_config_path)
            if p.exists():
                return bool(_json.loads(p.read_text()).get("enabled", True))
        except Exception:
            pass
        return True

    def _save_paper_config(self) -> None:
        if not self.paper_config_path:
            return
        try:
            import json as _json, os as _os
            from pathlib import Path as _P
            p = _P(self.paper_config_path); p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(_json.dumps({"enabled": self.paper_auto_enabled,
                                        "starting_capital": self.intel_book.capital}))
            _os.replace(tmp, p)                       # atomic
        except Exception:
            pass

    def _save_intel_book(self) -> None:
        """Persist the paper book (open positions + trade history) so it survives restart."""
        if not self.intel_book_path:
            return
        try:
            import json as _json, os as _os
            from pathlib import Path as _P
            p = _P(self.intel_book_path); p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(_json.dumps(self.intel_book.snapshot(), default=str))
            _os.replace(tmp, p)                       # atomic (crash-safe)
        except Exception:
            pass

    # ── one synchronous cycle (deterministic; safe for tests + UI button) ────────
    def run_once(self, dataset_status=None, date=None, family_weights=None,
                 adapt=True, allow_deploy=True) -> CycleReport:
        self.state.cycles_run += 1
        cyc = self.state.cycles_run
        status = dataset_status if dataset_status is not None else self.dataset_status_fn()
        report = run_cycle(cyc, self.thread, dataset_status=status,
                           evaluate_fn=self.evaluate_fn, budget=self.budget,
                           registry=self.registry, family_weights=family_weights)
        # feed the learning memory (market-evidence proposals only) and record decay/gain
        events = self.ledger.observe_cycle(cyc, report.proposals)
        for e in events:
            if e.kind in ("IMPROVED", "DECAYED"):
                self.thread.reason(cyc, e.note, e.as_dict())
        for prop in report.proposals:
            self._specs_by_family.setdefault(prop.family, None)

        # ── full PAPER autonomy: deploy → trade → learn (only if the user engaged) ──
        if self.paper.engaged:
            if allow_deploy:
                for spec, ev in report.survivors_for_paper:
                    self.paper.deploy(spec, ev, status, cycle=cyc, thread=self.thread)
            self._run_paper_day(cyc, date)
            if adapt:                                  # calibration is the authority in growth
                self.paper.review_and_adapt(cycle=cyc, thread=self.thread)
            self.state.paper_deployed = len(self.paper.strategies)
            self.state.paper_retired = len(self.paper.retired)

        self.state.last_report = report.as_dict()
        self.state.total_proposals += len(report.proposals)
        return report

    # ── one DAY of growing up: backtest → forward test → calibrate → remember ────
    def grow_one_day(self, date=None, dataset_status=None) -> dict:
        """The daily learning step that makes the system smarter. It:
          1. biases discovery by what has forward-tested well (learned search weights),
          2. runs a research cycle (BACKTEST) + auto-deploys survivors to PAPER,
          3. trades one paper day (FORWARD TEST on unseen data),
          4. CALIBRATES each strategy's forward edge against its backtest edge,
          5. retires overfits/decayers, folds every verdict into persistent Knowledge.
        Paper-only, live-locked. Honest no-op shape when data isn't ready."""
        day = date or _today_str()
        weights = self.knowledge.search_weights(list(self.budget.families) if self.budget
                                                else list(DISC.DiscoveryBudget().families))
        # regime gate (connectivity): don't open NEW paper bets in RISK_OFF tape; existing
        # positions are still managed and calibrated. A conservative, demote-only guard.
        regime = "RISK_ON"
        if self.regime_fn:
            try:
                regime = self.regime_fn() or "RISK_ON"
            except Exception:
                regime = "RISK_ON"
        allow_deploy = regime != "RISK_OFF"
        if not allow_deploy:
            self.thread.observe(self.state.cycles_run + 1,
                                f"Regime is {regime}: standing down NEW paper deployments "
                                "today; existing strategies keep trading and being judged.",
                                {"regime": regime})
        report = self.run_once(dataset_status=dataset_status, date=day,
                               family_weights=weights, adapt=False, allow_deploy=allow_deploy)

        # remember today's BACKTEST edges (in-sample)
        for spec, ev in report.survivors_for_paper:
            self.knowledge.remember_backtest(spec.family, ev.net_expectancy_R)

        # CALIBRATE every actively-trading strategy: forward (paper) vs backtest
        calibrations = []
        if self.paper.engaged:
            for ps in list(self.paper.strategies.values()):
                if ps.state != S.PAPER_EVALUATION:
                    continue
                fR, n = self.paper.forward_R(ps.spec.strategy_id)
                rs = self.paper.book.r_stats(ps.spec.strategy_id)
                cal = calibrate(ps.spec.strategy_id, ps.spec.family, ps.backtest_R, fR, n,
                                forward_lower_R=rs["lower_R"])   # noise-aware
                calibrations.append(cal.as_dict())
                if cal.verdict != FORWARD_PENDING:
                    self.knowledge.remember_forward(ps.spec.family, fR, cal.verdict)
                    self.thread.reason(self.state.cycles_run, cal.note, cal.as_dict())
                if not cal.keep:
                    self.paper.retire(ps.spec.strategy_id, cal.note,
                                      cycle=self.state.cycles_run, thread=self.thread)
            self.state.paper_retired = len(self.paper.retired)

        self.knowledge.save()
        if self.paper_state_path:                       # the book + journal survive restarts
            try:
                self.paper.save(self.paper_state_path)
            except Exception:
                pass
        self.state.days_grown += 1
        self.state.last_grow_date = day
        self.thread.conclude(self.state.cycles_run,
                             f"Grew one day ({day}): {len(report.survivors_for_paper)} "
                             f"backtested survivor(s), {len(calibrations)} forward-calibrated, "
                             f"{self.state.paper_retired} retired. Search now favours what "
                             "forward-tests best. Live stays locked.",
                             {"day": day, "calibrations": calibrations,
                              "knowledge": self.knowledge.summary()})
        return {"day": day, "report": report.as_dict(), "calibrations": calibrations,
                "knowledge": self.knowledge.summary()}

    def maybe_grow_today(self, date=None) -> dict | None:
        """Grow at most once per calendar day (idempotent for the daemon loop)."""
        day = date or _today_str()
        if self.state.last_grow_date == day:
            return None
        return self.grow_one_day(date=day)

    # ── the authoritative two-brain paper cycle (Phase O job) ────────────────────
    def run_intelligence_cycle_day(self, date=None, *, ctx=None,
                                   new_entries_allowed: bool = True,
                                   entry_block_reason: str = "",
                                   session_phase: str | None = None,
                                   capability_failures=(), fresh_live_symbols=(),
                                   live_confirmation_required: bool = False) -> dict:
        """Run ONE end-to-end intelligence cycle.

        Operational entry permission is applied to the CycleContext *before* the authoritative
        runtime executes.  A blocked cycle may manage existing positions and update evidence, but it
        is structurally unable to reach ``PaperBook.open_position``.  This method remains paper-only
        and broker-free.
        """
        from research.intelligence.runtime import run_intelligence_cycle
        if not self._intel_lock.acquire(blocking=False):
            return {"status": "SKIPPED_LOCKED", "eligibility": "ALREADY_DONE"}
        try:
            day = date or _today_str()
            provider = None
            if ctx is None:
                ctx, provider = self._build_intel_ctx(day)
            # Apply the supervisor-owned safety controls before computing cycle_id / idempotency.
            ctx.new_entries_allowed = bool(new_entries_allowed)
            ctx.entry_block_reason = str(entry_block_reason or "")
            ctx.capability_failures = tuple(sorted(set(capability_failures or ())))
            ctx.fresh_live_symbols = frozenset(str(s).upper() for s in (fresh_live_symbols or ()))
            ctx.live_confirmation_required = bool(live_confirmation_required)
            if session_phase:
                ctx.session_phase = str(session_phase)

            # establish IN-SAMPLE evidence per strategy from its OWN rules on the snapshot
            bt_R, bt_n = {}, {}
            if provider is not None:
                for spec in ctx.strategies:
                    r, n = self._insample_evidence(spec, provider, ctx.as_of_date)
                    bt_R[spec.strategy_id] = r
                    bt_n[spec.strategy_id] = n
            res = run_intelligence_cycle(ctx, store=self.event_store, book=self.intel_book,
                                         runtime_state=self.runtime_state,
                                         knowledge=self.knowledge,
                                         backtest_R=bt_R, backtest_trades=bt_n)
            self.state.last_intel_cycle = res.as_dict()
            self._save_intel_book()
            out = res.as_dict()
            if not res.data_ok:
                out["eligibility"] = "NO_DATA"
            elif res.positions_opened or res.positions_closed:
                out["eligibility"] = "TRADED"
            elif not ctx.new_entries_allowed:
                out["eligibility"] = "BLOCKED_SAFETY"
            else:
                out["eligibility"] = "NO_ELIGIBLE_TRADE"
            out["new_entries_allowed"] = bool(ctx.new_entries_allowed)
            out["entry_block_reason"] = ctx.entry_block_reason
            out["session_phase"] = ctx.session_phase
            return out
        finally:
            self._intel_lock.release()

    def _build_intel_ctx(self, day):
        """Build a CycleContext for the cycle. Preference order:
          1. an ACTIVE, VERIFIED snapshot (real NSE data) → snapshot-pinned production context;
          2. an injected `intel_registry_fn` (tests / custom wiring);
          3. an honest empty context (data_ok=False) → safe no-op.
        """
        from research.intelligence.runtime.cycle_context import CycleContext
        regime = "RISK_ON"
        if self.regime_fn:
            try:
                regime = self.regime_fn() or "RISK_ON"
            except Exception:
                regime = "RISK_ON"
        # 1 — real data: pin the active snapshot and build the production context
        if self.snapshot_store is not None and self.strategy_registry is not None:
            try:
                snap = self.snapshot_store.open_active()   # verifies before opening
                if snap is not None:
                    from research.intelligence.data.provider import SnapshotBarProvider
                    from research.intelligence.runtime.context_builder import (
                        build_context_from_snapshot)
                    prov = SnapshotBarProvider(snap)
                    as_of = day if day <= (prov.latest_available_date() or day) \
                        else prov.latest_available_date()
                    ctx, _readiness = build_context_from_snapshot(
                        prov, self.strategy_registry, as_of=as_of, mode=self.mode,
                        market_regime=regime)
                    return ctx, prov
            except Exception as _se:
                self.state.last_error = f"snapshot ctx: {_se}"
        # 2 — injected registry provider
        if self.intel_registry_fn:
            try:
                built = self.intel_registry_fn(day) or {}
            except Exception:
                built = {}
        else:
            built = {}
        ctx = CycleContext(
            as_of_date=day, cycle_type="paper_session", mode=self.mode,
            data_ok=bool(built.get("data_ok", False)),
            data_snapshot_id=str(built.get("data_snapshot_id", "")),
            market_regime=built.get("regime", regime),
            config_hash=str(built.get("config_hash", "cfg0")),
            registry_version=str(built.get("registry_version", "reg0")),
            strategies=list(built.get("strategies", [])),
            data=dict(built.get("data", {})), clusters=dict(built.get("clusters", {})))
        return ctx, None

    def _insample_evidence(self, spec, provider, as_of: str):
        """Real in-sample backtest of a strategy's OWN frozen rules over the snapshot history
        up to (but excluding) `as_of`, simulated in a scratch realistic PaperBook. Returns
        (expectancy_R, n_trades). Cached per (snapshot, strategy). Reuses the runtime + book —
        no new research system, no fabricated evidence."""
        key = (provider.snapshot_id, spec.strategy_id)
        if key in self._insample_cache:
            return self._insample_cache[key]
        from research.intelligence import strategy_runtime as RT
        from research.auto_research.paper_book import PaperBook
        from research.auto_research.costs import india_cash_costs
        syms = provider.snapshot.symbols()
        all_dates = sorted({b.date for s in syms for b in provider.universe_history(s, as_of)})
        all_dates = [d for d in all_dates if d < as_of]      # strictly in-sample (no look-ahead)
        warm = 121 if RT.is_cross_sectional(spec.family) else 51
        book = PaperBook(slippage_bps=3.0, cost_model=india_cash_costs)
        for d in all_dates[warm:]:
            bars = {}
            for (_sid, sym) in list(book.open.keys()):
                h = provider.bars(sym, d)
                if h and h[-1].date == d:
                    b = h[-1]; bars[sym] = (b.open, b.high, b.low, b.close)
            if bars:
                book.mark(bars, d)
            uni = {sym: provider.bars(sym, d) for sym in syms}
            try:
                sigs = RT.signals(spec, d, uni, benchmark=provider.benchmark(d))
            except Exception:
                sigs = []
            for sig in sigs[:1]:                              # one entry/day in-sample
                book.open_position(spec.strategy_id, sig["symbol"], sig["entry"], sig["stop"],
                                   sig["target"], d, int(sig.get("max_hold", 10)))
        st = book.stats()
        res = (float(st["expectancy_R"]), int(st["n_trades"]))
        self._insample_cache[key] = res
        return res

    def _run_paper_day(self, cyc: int, date=None) -> None:
        """Advance one paper trading day using the injected signal/price providers. Without
        them the brain can deploy but has no live bars to trade — it says so honestly and
        never fabricates prices."""
        if not (self.signal_fn and self.bars_fn):
            return
        day = date or _today_str()
        bars = self.bars_fn(day) or {}
        signals: list = []
        for ps in self.paper.active():
            for sig in (self.signal_fn(ps, day) or []):
                sig = dict(sig); sig["strategy_id"] = ps.spec.strategy_id
                signals.append(sig)
        self.paper.trade_day(signals, bars, day, cycle=cyc, thread=self.thread)

    # ── daemon lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread_obj and self._thread_obj.is_alive():
            return
        self._stop.clear()
        self.state.running = True
        self._thread_obj = threading.Thread(target=self._worker, name="auto-research",
                                            daemon=True)
        self._thread_obj.start()

    def stop(self) -> None:
        self._stop.set()
        self.state.running = False

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                # when paper autonomy is engaged, grow up once per day (backtest → forward
                # test → calibrate → remember); otherwise just keep researching each cycle.
                if self.paper.engaged:
                    self.maybe_grow_today()
                # the authoritative two-brain paper cycle runs each tick in PAPER_AUTO when
                # persistently enabled; honest no-op until a real registry + data are wired.
                if self.is_paper_auto_enabled():
                    try:
                        self.run_intelligence_cycle_day()
                    except Exception as _ie:
                        self.state.last_error = f"intel cycle: {_ie}"
                if not self.paper.engaged and self.mode != "PAPER_AUTO":
                    self.run_once()
                self.state.last_error = self.state.last_error or ""
            except Exception as e:                     # a bad cycle never kills the brain
                self.state.last_error = str(e)
            # interruptible sleep so stop() is prompt
            self._stop.wait(self.interval_s)
        self.state.running = False


def _today_str() -> str:
    try:
        from core.market_clock import now_ist_naive
        return now_ist_naive().date().isoformat()
    except Exception:
        return time.strftime("%Y-%m-%d")


# module-level singleton for the app to share one brain across pages/daemons
_BRAIN: AutoResearchBrain | None = None


def get_brain(**kwargs) -> AutoResearchBrain:
    """Shared production brain. Wires the REAL data providers (backtest / bars / signals)
    by default so, once engaged, it backtests and forward-tests on live market data with no
    human in the loop. Tests construct AutoResearchBrain directly with injected providers."""
    global _BRAIN
    if _BRAIN is None:
        try:
            from pathlib import Path
            from research.auto_research import providers as P
            kwargs.setdefault("evaluate_fn", P.backtest_evaluator)
            kwargs.setdefault("signal_fn", P.signals_for)
            kwargs.setdefault("bars_fn", P.daily_bars)
            kwargs.setdefault("regime_fn", P.current_regime)
            _logs = Path(__file__).resolve().parent.parent.parent / "logs"
            kwargs.setdefault("paper_state_path", _logs / "auto_research" / "paper_book.json")
            # the two-brain runtime persists to logs/intelligence/ (what Brain Observatory reads)
            kwargs.setdefault("event_store_path", _logs / "intelligence" / "events.jsonl")
            kwargs.setdefault("runtime_state_path", _logs / "intelligence" / "runtime_state.json")
            # PAPER_AUTO operational persistence: the paper book + the enable flag survive restart
            kwargs.setdefault("intel_book_path", _logs / "intelligence" / "intel_book.json")
            kwargs.setdefault("paper_config_path", _logs / "intelligence" / "paper_config.json")
        except Exception:
            pass
        _BRAIN = AutoResearchBrain(**kwargs)
        # wire the immutable snapshot store + a validated registry so the loop flips from
        # no-op to real-data-driven the moment a snapshot is activated (none in this sandbox).
        try:
            from pathlib import Path as _P
            from research.intelligence.data.snapshot_store import SnapshotStore
            from research.intelligence.registry import StrategyRegistry
            from research.strategy_studio import discovery as _D
            _BRAIN.snapshot_store = SnapshotStore(
                _P(__file__).resolve().parent.parent.parent / "logs" / "snapshots")
            _BRAIN.strategy_registry = StrategyRegistry().build(
                _D.generate(_D.DiscoveryBudget()))
        except Exception:
            pass
    return _BRAIN
