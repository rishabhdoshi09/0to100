"""
🤖 Paper Autonomy — the brain running its own strategies in PAPER, hands-off.

This is the module the user deliberately switched on: "in paper money I want to give full
autonomy… blow up paper money, I don't care, but it needs to learn and get smarter by the
day." So here the system, with no human in the loop:

    auto-approve a survivor for PAPER  →  activate it  →  place SIMULATED trades
        →  mark them against real bars  →  book real outcomes  →  learn
        →  autonomously RETIRE proven losers and keep what works.

The seatbelt that stays bolted on: this is PAPER-ONLY and cannot reach live. Every approval
uses the `paper_autopilot` actor, which the lifecycle forbids from the only transition that
leads toward live review. Nothing here imports any real-order execution path — the worst it
can do is lose imaginary money, which is exactly the point.

It still refuses SYNTHETIC evidence: full autonomy means trading real-data strategies on its
own, not fabricating results. With no research-grade data, it honestly deploys nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from research.strategy_studio import approval as A
from research.strategy_studio import spec as S
from research.strategy_studio.discovery import EvidenceReport
from research.auto_research.paper_book import PaperBook

# a strategy needs at least this many closed paper trades before autonomy judges it
MIN_TRADES_TO_JUDGE = 20


@dataclass
class PaperStrategy:
    spec: S.StrategySpec
    approval: A.ApprovalRecord
    activation: A.PaperActivation
    state: str
    deployed_cycle: int
    backtest_R: float = 0.0        # in-sample edge at deploy — compared to forward later
    trades_today: int = 0

    def as_dict(self):
        return {"strategy_id": self.spec.strategy_id, "version": self.spec.version,
                "family": self.spec.family, "state": self.state,
                "config_hash": self.spec.config_hash(),
                "deployed_cycle": self.deployed_cycle,
                "backtest_R": self.backtest_R}


class PaperAutonomyManager:
    """Owns the paper book + the set of autonomously-deployed strategies."""

    def __init__(self, book: PaperBook | None = None, *, engaged: bool = False,
                 max_allocation: float = 100_000.0, max_open_risk_pct: float = 5.0,
                 max_trades_per_day: int = 3):
        self.book = book or PaperBook()
        self.engaged = engaged
        self.max_allocation = max_allocation
        self.max_open_risk_pct = max_open_risk_pct
        self.max_trades_per_day = max_trades_per_day
        self.strategies: dict[str, PaperStrategy] = {}
        self.retired: list[str] = []

    def engage(self):  self.engaged = True
    def disengage(self): self.engaged = False

    # ── autonomous deployment (paper-only, live-locked) ──────────────────────────
    def deploy(self, spec: S.StrategySpec, ev: EvidenceReport, readiness: dict, *,
               cycle: int, review_date: str = "", thread=None) -> PaperStrategy | None:
        """Auto-approve + activate a survivor for PAPER. Returns the PaperStrategy, or None
        (with a thread note) if autonomy is off or the evidence gate refuses it."""
        if not self.engaged:
            return None
        if spec.strategy_id in self.strategies:
            return self.strategies[spec.strategy_id]
        try:
            rec, state = A.autonomous_paper_approve(
                spec, ev, readiness, current_state=S.AWAITING_USER_APPROVAL,
                max_allocation=self.max_allocation, max_open_risk_pct=self.max_open_risk_pct,
                max_trades_per_day=self.max_trades_per_day,
                review_date=review_date or "auto")
            act = A.activate_paper(rec, spec, actor=S.PAPER_AUTOPILOT, confirmed=True)
            # move APPROVED_FOR_PAPER -> PAPER_EVALUATION autonomously (paper autopilot)
            S.require_transition(state, S.PAPER_EVALUATION, S.PAPER_AUTOPILOT)
            ps = PaperStrategy(spec=spec, approval=rec, activation=act,
                               state=S.PAPER_EVALUATION, deployed_cycle=cycle,
                               backtest_R=round(float(ev.net_expectancy_R), 4))
            self.strategies[spec.strategy_id] = ps
            if thread is not None:
                thread.decide(cycle, f"PAPER-DEPLOY {spec.strategy_id} ({spec.family}) "
                              "autonomously — trading it in paper now. Live stays locked.",
                              ps.as_dict())
            return ps
        except A.ApprovalRefused as e:
            if thread is not None:
                thread.reason(cycle, f"Did not paper-deploy {spec.strategy_id}: {e}",
                              {"strategy_id": spec.strategy_id})
            return None

    # ── one trading day: open signals, mark the book, book outcomes ──────────────
    def trade_day(self, signals: list, bars: dict, date: str, *, cycle: int = 0,
                  thread=None) -> dict:
        """`signals` is a list of dicts {strategy_id, symbol, entry, stop, target, max_hold}
        for strategies we have deployed. `bars` maps symbol -> (high, low, close). Opens the
        new signals (respecting per-strategy daily caps + book risk caps), then advances the
        book one day. Returns a summary."""
        for ps in self.strategies.values():
            ps.trades_today = 0
        opened = 0
        for sig in signals:
            sid = sig.get("strategy_id")
            ps = self.strategies.get(sid)
            if ps is None or ps.state != S.PAPER_EVALUATION:
                continue
            if ps.trades_today >= self.max_trades_per_day:
                continue
            pos = self.book.open_position(sid, sig["symbol"], float(sig["entry"]),
                                          float(sig["stop"]), float(sig["target"]), date,
                                          int(sig.get("max_hold", ps.spec.max_holding_days)))
            if pos is not None:
                ps.trades_today += 1
                opened += 1
        closed = self.book.mark(bars, date)
        if thread is not None and (opened or closed):
            thread.observe(cycle, f"Paper day {date}: opened {opened}, closed "
                           f"{len(closed)}. Equity ₹{self.book.equity():,.0f}.",
                           {"opened": opened, "closed": len(closed),
                            "equity": round(self.book.equity(), 2)})
        return {"date": date, "opened": opened, "closed": len(closed),
                "equity": round(self.book.equity(), 2)}

    # ── learn: retire proven losers on their own (get smarter by the day) ────────
    def review_and_adapt(self, *, cycle: int, thread=None,
                         min_trades: int = MIN_TRADES_TO_JUDGE) -> list[str]:
        """For each deployed strategy with enough closed paper trades, RETIRE the ones whose
        real paper expectancy has proven negative — autonomously. Returns retired ids. This
        is the daily 'get smarter': stop trading what loses, keep what earns."""
        retired_now: list[str] = []
        for sid, ps in list(self.strategies.items()):
            if ps.state != S.PAPER_EVALUATION:      # already retired/decayed — skip
                continue
            st = self.book.stats(sid)
            if st["n_trades"] < min_trades:
                continue
            if st["expectancy_R"] < 0:
                S.require_transition(ps.state, S.DECAYED, S.PAPER_AUTOPILOT)
                ps.state = S.DECAYED
                self.retired.append(sid); retired_now.append(sid)
                if thread is not None:
                    thread.decide(cycle, f"RETIRE {sid}: proven paper loser "
                                  f"({st['expectancy_R']:+.2f}R over {st['n_trades']} "
                                  "trades). Autonomously stopping it.", st)
        return retired_now

    def retire(self, strategy_id: str, reason: str, *, cycle: int = 0, thread=None) -> bool:
        """Stand a paper strategy down (e.g. because forward-test calibration says OVERFIT).
        Idempotent; returns True if it was active and is now retired."""
        ps = self.strategies.get(strategy_id)
        if ps is None or ps.state != S.PAPER_EVALUATION:
            return False
        S.require_transition(ps.state, S.DECAYED, S.PAPER_AUTOPILOT)
        ps.state = S.DECAYED
        self.retired.append(strategy_id)
        if thread is not None:
            thread.decide(cycle, f"RETIRE {strategy_id}: {reason}",
                          {"strategy_id": strategy_id, "reason": reason})
        return True

    def forward_R(self, strategy_id: str) -> tuple[float, int]:
        """(expectancy_R, n_trades) of a strategy's REAL paper (out-of-sample) trades."""
        st = self.book.stats(strategy_id)
        return st["expectancy_R"], st["n_trades"]

    # ── reporting ────────────────────────────────────────────────────────────────
    def active(self) -> list[PaperStrategy]:
        return [p for p in self.strategies.values() if p.state == S.PAPER_EVALUATION]

    def performance_report(self) -> dict:
        return {"engaged": self.engaged, "book": self.book.as_dict(),
                "deployed": len(self.strategies), "active": len(self.active()),
                "retired": list(self.retired),
                "per_strategy": {sid: self.book.stats(sid)
                                 for sid in self.strategies}}
