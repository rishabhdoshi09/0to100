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
from research.auto_research.thread import ResearchThread
from research.strategy_studio import discovery as DISC


@dataclass
class BrainState:
    cycles_run: int = 0
    last_report: dict | None = None
    running: bool = False
    last_error: str = ""
    total_proposals: int = 0


class AutoResearchBrain:
    """Owns the long-lived research memory and drives cycles."""

    def __init__(self, thread_path=None, *, evaluate_fn=None, budget=None,
                 interval_s: float = 3600.0, clock=None,
                 dataset_status_fn=canonical_readiness):
        self.thread = (ResearchThread(thread_path, clock=clock) if clock
                       else ResearchThread(thread_path))
        self.ledger = LearningLedger()
        self.registry = DISC.AttemptRegistry()
        self.evaluate_fn = evaluate_fn
        self.budget = budget
        self.interval_s = interval_s
        self.dataset_status_fn = dataset_status_fn
        self.state = BrainState()
        self._specs_by_family: dict = {}
        self._thread_obj: threading.Thread | None = None
        self._stop = threading.Event()

    # ── one synchronous cycle (deterministic; safe for tests + UI button) ────────
    def run_once(self, dataset_status=None) -> CycleReport:
        self.state.cycles_run += 1
        cyc = self.state.cycles_run
        status = dataset_status if dataset_status is not None else self.dataset_status_fn()
        report = run_cycle(cyc, self.thread, dataset_status=status,
                           evaluate_fn=self.evaluate_fn, budget=self.budget,
                           registry=self.registry)
        # feed the learning memory (market-evidence proposals only) and record decay/gain
        events = self.ledger.observe_cycle(cyc, report.proposals)
        for e in events:
            if e.kind in ("IMPROVED", "DECAYED"):
                self.thread.reason(cyc, e.note, e.as_dict())
        # propose fresh re-tested versions for decayed families (advice only)
        for prop in report.proposals:
            self._specs_by_family.setdefault(prop.family, None)
        self.state.last_report = report.as_dict()
        self.state.total_proposals += len(report.proposals)
        return report

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
                self.run_once()
                self.state.last_error = ""
            except Exception as e:                     # a bad cycle never kills the brain
                self.state.last_error = str(e)
            # interruptible sleep so stop() is prompt
            self._stop.wait(self.interval_s)
        self.state.running = False


# module-level singleton for the app to share one brain across pages/daemons
_BRAIN: AutoResearchBrain | None = None


def get_brain(**kwargs) -> AutoResearchBrain:
    global _BRAIN
    if _BRAIN is None:
        _BRAIN = AutoResearchBrain(**kwargs)
    return _BRAIN
