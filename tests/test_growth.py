"""
Deterministic, network-free tests for GROWTH — how the system gets smarter each day.

The mechanisms under test are the answer to "how will it get smarter?":
  • calibrate(): tell a real edge from an overfit backtest by comparing forward vs backtest.
  • Knowledge: remember verdicts, move a family's trust, persist across restarts.
  • adaptive search: bias tomorrow's discovery toward families that forward-test well.
  • grow_one_day(): the full daily loop, backtest → forward test → calibrate → remember,
    proven end-to-end on injected deterministic data — and still PAPER-only / live-locked.
"""
from __future__ import annotations

import pytest

from research.auto_research import growth as GR
from research.auto_research.knowledge import Knowledge
from research.auto_research.scheduler import AutoResearchBrain
from research.strategy_studio import discovery as DISC
from research.strategy_studio import spec as S


GREEN = {"color": "green", "can_run": True}


# ── calibration: the core "is this edge real?" test ──────────────────────────────

class TestCalibrate:
    def test_pending_until_enough_forward_trades(self):
        c = GR.calibrate("s", "momentum", backtest_R=0.3, forward_R=0.3, n_forward=5)
        assert c.verdict == GR.FORWARD_PENDING and c.keep is True

    def test_overfit_when_forward_edge_vanishes(self):
        c = GR.calibrate("s", "momentum", backtest_R=0.4, forward_R=-0.1, n_forward=40)
        assert c.verdict == GR.OVERFIT and c.keep is False

    def test_confirmed_when_forward_holds(self):
        c = GR.calibrate("s", "momentum", backtest_R=0.30, forward_R=0.28, n_forward=40)
        assert c.verdict == GR.CONFIRMED and c.keep is True

    def test_decayed_when_forward_much_weaker(self):
        c = GR.calibrate("s", "momentum", backtest_R=0.40, forward_R=0.05, n_forward=40)
        assert c.verdict == GR.DECAYED and c.keep is False

    def test_weaker_but_positive_is_kept(self):
        c = GR.calibrate("s", "momentum", backtest_R=0.40, forward_R=0.20, n_forward=40)
        assert c.verdict == GR.WEAKER_POSITIVE and c.keep is True


# ── knowledge: memory + trust + persistence ──────────────────────────────────────

class TestKnowledge:
    def test_trust_rises_on_confirm_falls_on_overfit(self, tmp_path):
        k = Knowledge(tmp_path / "k.json")
        k.remember_forward("momentum", 0.3, GR.CONFIRMED)
        assert k.family_trust("momentum") > 0.5
        k.remember_forward("meanrev", -0.1, GR.OVERFIT)
        assert k.family_trust("meanrev") < 0.5

    def test_search_weights_never_starve_a_family(self, tmp_path):
        k = Knowledge(tmp_path / "k.json")
        for _ in range(20):
            k.remember_forward("loser", -0.2, GR.OVERFIT)
        w = k.search_weights(["loser", "winner", "unseen"])
        assert w["loser"] >= 0.1                     # floored — keeps exploring
        assert w["unseen"] == 0.5                     # never-seen defaults neutral

    def test_persists_across_restart(self, tmp_path):
        p = tmp_path / "k.json"
        k1 = Knowledge(p)
        k1.remember_forward("momentum", 0.3, GR.CONFIRMED)
        k1.save()
        k2 = Knowledge(p)                             # the child remembers after a restart
        assert k2.family_trust("momentum") == k1.family_trust("momentum")

    def test_corrupt_memory_starts_neutral(self, tmp_path):
        p = tmp_path / "k.json"; p.write_text("{ not json")
        k = Knowledge(p)
        assert k.family_trust("anything") == 0.5


# ── adaptive search: the search distribution shifts toward what works ────────────

class TestAdaptiveSearch:
    def test_higher_weight_gets_more_attempts(self):
        fams = ["a", "b", "c"]
        import random
        seq = DISC._family_sequence(fams, {"a": 0.9, "b": 0.2, "c": 0.2}, 30,
                                    random.Random(1))
        assert seq.count("a") > seq.count("b")
        assert seq.count("b") >= 1 and seq.count("c") >= 1   # never starved

    def test_no_weights_is_round_robin(self):
        import random
        seq = DISC._family_sequence(["a", "b", "c"], None, 6, random.Random(1))
        assert seq == ["a", "b", "c", "a", "b", "c"]

    def test_generate_accepts_weights_deterministically(self):
        b = DISC.DiscoveryBudget(seed=3)
        w = {f: 1.0 for f in b.families}
        a1 = DISC.generate(b, family_weights=w)
        a2 = DISC.generate(b, family_weights=w)
        assert [s.as_dict() for s in a1] == [s.as_dict() for s in a2]


# ── grow_one_day: the whole daily loop, end-to-end ───────────────────────────────

def _market_eval(spec, split):
    return DISC.EvidenceReport(n_trades=120, n_symbols=40, net_expectancy_R=0.25,
                               gross_expectancy_R=0.35, cost_drag_R=0.1, p_value=0.01,
                               max_drawdown=0.15, is_synthetic=False, verdict="PASS")


def _winning_bars(day):                              # every position hits target (+2R)
    return {f"{day}_{i}": (121, 101, 120) for i in range(4)}


def _losing_bars(day):                               # every position hits stop (-1R)
    return {f"{day}_{i}": (101, 89, 90) for i in range(4)}


def _signals(ps, day):                               # 3 fresh symbols per day (daily cap)
    return [{"symbol": f"{day}_{i}", "entry": 100, "stop": 90, "target": 120,
             "max_hold": 5} for i in range(3)]


def _brain(bars_fn, tmp_path):
    return AutoResearchBrain(evaluate_fn=_market_eval, dataset_status_fn=lambda: GREEN,
                             signal_fn=_signals, bars_fn=bars_fn,
                             knowledge=Knowledge(tmp_path / "k.json"))


class TestGrowOneDay:
    def test_backtest_then_forward_confirms_and_trust_rises(self, tmp_path):
        brain = _brain(_winning_bars, tmp_path)
        brain.engage_paper_autonomy()
        fam = None
        for d in range(9):                            # ~9 days → >20 forward trades
            out = brain.grow_one_day(date=f"d{d}")
            if brain.paper.strategies:
                fam = next(iter(brain.paper.strategies.values())).spec.family
        assert brain.state.days_grown == 9
        assert brain.paper.book.stats()["n_trades"] >= 20
        # forward confirmed the backtest → family trust climbed above the neutral prior
        assert brain.knowledge.family_trust(fam) > 0.5
        assert brain.state.paper_retired == 0         # a real edge is kept

    def test_overfit_backtest_is_caught_and_retired(self, tmp_path):
        brain = _brain(_losing_bars, tmp_path)
        brain.engage_paper_autonomy()
        fam = None
        for d in range(9):
            brain.grow_one_day(date=f"d{d}")
            if brain.paper.strategies and fam is None:
                fam = next(iter(brain.paper.strategies.values())).spec.family
        # forward test proved the backtest overfit → retired, trust fell
        assert brain.state.paper_retired >= 1
        assert brain.knowledge.family_trust(fam) < 0.5

    def test_grow_is_paper_only_and_live_locked(self, tmp_path):
        brain = _brain(_winning_bars, tmp_path)
        brain.engage_paper_autonomy()
        brain.grow_one_day(date="d0")
        # the only path toward live is still user-only, even mid-growth
        assert not S.can_transition(S.PAPER_EVALUATION, S.ELIGIBLE_FOR_LIVE_REVIEW,
                                    S.PAPER_AUTOPILOT)

    def test_maybe_grow_today_is_idempotent_per_day(self, tmp_path):
        brain = _brain(_winning_bars, tmp_path)
        brain.engage_paper_autonomy()
        assert brain.maybe_grow_today(date="d0") is not None
        assert brain.maybe_grow_today(date="d0") is None      # same day → no double-grow
        assert brain.maybe_grow_today(date="d1") is not None

    def test_knowledge_persists_between_brains(self, tmp_path):
        b1 = _brain(_winning_bars, tmp_path)
        b1.engage_paper_autonomy()
        for d in range(9):
            b1.grow_one_day(date=f"d{d}")
        fam = next(iter(b1.paper.strategies.values())).spec.family
        # a fresh brain loading the same knowledge file inherits what was learned
        b2 = AutoResearchBrain(evaluate_fn=_market_eval, dataset_status_fn=lambda: GREEN,
                               knowledge=Knowledge(tmp_path / "k.json"))
        assert b2.knowledge.family_trust(fam) == b1.knowledge.family_trust(fam)
