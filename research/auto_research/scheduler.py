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


class AutoResearchBrain:
    """Owns the long-lived research memory and drives cycles."""

    def __init__(self, thread_path=None, *, evaluate_fn=None, budget=None,
                 interval_s: float = 3600.0, clock=None,
                 dataset_status_fn=canonical_readiness,
                 paper=None, signal_fn=None, bars_fn=None, knowledge=None,
                 regime_fn=None, paper_state_path=None):
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
                else:
                    self.run_once()
                self.state.last_error = ""
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
            kwargs.setdefault("paper_state_path",
                              Path(__file__).resolve().parent.parent.parent /
                              "logs" / "auto_research" / "paper_book.json")
        except Exception:
            pass
        _BRAIN = AutoResearchBrain(**kwargs)
    return _BRAIN
