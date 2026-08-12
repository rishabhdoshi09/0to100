"""
Deterministic, network-free tests for FULL PAPER AUTONOMY.

They prove the two things that matter: the brain can deploy, trade, and learn from its own
strategies in PAPER with no human in the loop — and that this can NEVER reach live money.
"""
from __future__ import annotations

import pytest

from research.auto_research.paper_book import PaperBook
from research.auto_research.paper_autonomy import PaperAutonomyManager
from research.auto_research.scheduler import AutoResearchBrain
from research.strategy_studio import approval as A
from research.strategy_studio import discovery as DISC
from research.strategy_studio import spec as S


GREEN = {"color": "green", "can_run": True}
RED = {"color": "red", "can_run": False, "reasons": ["no data"]}


def _spec(sid="STR-0001"):
    cands = DISC.generate(DISC.DiscoveryBudget())
    base = cands[0]
    import dataclasses
    return dataclasses.replace(base, strategy_id=sid)


def _ev(synthetic=False, verdict="PASS"):
    return DISC.EvidenceReport(n_trades=120, n_symbols=40, net_expectancy_R=0.25,
                               gross_expectancy_R=0.35, cost_drag_R=0.1, p_value=0.01,
                               max_drawdown=0.15, is_synthetic=synthetic, verdict=verdict)


def _market_eval(spec, split):
    return _ev(synthetic=False)


# ── the live boundary is structural ──────────────────────────────────────────────

class TestLiveBoundaryHolds:
    def test_paper_autopilot_can_cross_paper_gate(self):
        assert S.can_transition(S.AWAITING_USER_APPROVAL, S.APPROVED_FOR_PAPER,
                                S.PAPER_AUTOPILOT)
        assert S.can_transition(S.APPROVED_FOR_PAPER, S.PAPER_EVALUATION, S.PAPER_AUTOPILOT)

    def test_paper_autopilot_can_NEVER_reach_live_review(self):
        # the only transition toward live must remain user-only
        assert not S.can_transition(S.PAPER_EVALUATION, S.ELIGIBLE_FOR_LIVE_REVIEW,
                                    S.PAPER_AUTOPILOT)
        assert not S.can_transition(S.PAPER_EVALUATION, S.ELIGIBLE_FOR_LIVE_REVIEW, "system")
        assert S.can_transition(S.PAPER_EVALUATION, S.ELIGIBLE_FOR_LIVE_REVIEW, "user")
        with pytest.raises(S.LifecycleError):
            S.require_transition(S.PAPER_EVALUATION, S.ELIGIBLE_FOR_LIVE_REVIEW,
                                 S.PAPER_AUTOPILOT)

    def test_system_still_cannot_self_approve_studio_path(self):
        # the human studio guarantee is unchanged: research 'system' can't approve
        assert not S.can_transition(S.AWAITING_USER_APPROVAL, S.APPROVED_FOR_PAPER, "system")

    def test_live_enabled_flag_unchanged(self, monkeypatch):
        import execution.autopilot as ap
        monkeypatch.delenv("QT_LIVE_ENABLED", raising=False)
        assert ap._live_enabled() is False

    def test_autonomous_approval_is_paper_only(self):
        rec, state = A.autonomous_paper_approve(
            _spec(), _ev(), GREEN, current_state=S.AWAITING_USER_APPROVAL,
            max_allocation=100000, max_open_risk_pct=5, max_trades_per_day=3,
            review_date="auto")
        assert rec.allowed_mode == "PAPER"
        assert state == S.APPROVED_FOR_PAPER
        with pytest.raises(Exception):
            rec.allowed_mode = "LIVE"                    # frozen

    def test_no_order_or_broker_import_in_autonomy_package(self):
        import inspect
        from research.auto_research import paper_autonomy, paper_book, scheduler
        for mod in (paper_autonomy, paper_book, scheduler):
            src = inspect.getsource(mod)
            for banned in ("trade_executor", "kite_client", "KiteConnect", "zerodha",
                           "place_trade", "telegram"):
                assert banned not in src, f"{banned} leaked into {mod.__name__}"


# ── autonomous deployment gate (still refuses fake evidence) ─────────────────────

class TestDeploymentGate:
    def test_not_engaged_deploys_nothing(self):
        m = PaperAutonomyManager(engaged=False)
        assert m.deploy(_spec(), _ev(), GREEN, cycle=1) is None
        assert m.strategies == {}

    def test_engaged_deploys_real_survivor(self):
        m = PaperAutonomyManager(engaged=True)
        ps = m.deploy(_spec(), _ev(), GREEN, cycle=1)
        assert ps is not None and ps.state == S.PAPER_EVALUATION

    def test_synthetic_is_refused_even_engaged(self):
        m = PaperAutonomyManager(engaged=True)
        assert m.deploy(_spec(), _ev(synthetic=True), GREEN, cycle=1) is None

    def test_red_gate_refused_even_engaged(self):
        m = PaperAutonomyManager(engaged=True)
        assert m.deploy(_spec(), _ev(), RED, cycle=1) is None


# ── the paper book: risk caps + honest outcomes ──────────────────────────────────

class TestPaperBook:
    def test_one_percent_risk_sizing(self):
        b = PaperBook(capital=100_000)
        # entry 100 / stop 90 → unit risk 10; 1% of 100k = 1000 → qty 100 (notional 10k =
        # exactly the 10% cap, so the concentration cap does not reduce it further)
        pos = b.open_position("s", "AAA", entry=100, stop=90, target=130, date="d1",
                              max_holding_days=10)
        assert pos.qty == 100
        assert abs(pos.qty * pos.r_unit - 100_000 * 0.01) < 1e-6   # ~1% risk

    def test_ten_percent_concentration_cap(self):
        b = PaperBook(capital=100_000)
        pos = b.open_position("s", "AAA", entry=100, stop=99.9, target=120, date="d1",
                              max_holding_days=10)
        assert pos.qty * pos.entry_price <= 100_000 * 0.10 + 1e-6   # capped

    def test_total_open_risk_cap_blocks_excess(self):
        b = PaperBook(capital=100_000, max_positions=99)
        # entry 100 / stop 90 → each position risks a full 1% (₹1000); the 6th breaches the
        # 5% total-open-risk cap
        opened = 0
        for i in range(10):
            p = b.open_position("s", f"SYM{i}", entry=100, stop=90, target=130,
                                date="d1", max_holding_days=10)
            opened += int(p is not None)
        assert opened == 5
        assert any("total open risk" in why for _, why in b.refusals)

    def test_max_positions_cap(self):
        b = PaperBook(capital=1_000_000, max_positions=3)
        opened = sum(b.open_position("s", f"S{i}", 100, 95, 115, "d1", 10) is not None
                     for i in range(6))
        assert opened == 3

    def test_stop_and_target_and_maxhold_close(self):
        b = PaperBook(capital=100_000)
        b.open_position("s", "WIN", 100, 90, 120, "d1", 10)
        b.open_position("s", "LOSE", 100, 90, 120, "d1", 10)
        b.open_position("s", "HOLD", 100, 90, 120, "d1", 2)
        c1 = b.mark({"WIN": (121, 101, 120), "LOSE": (100, 89, 90), "HOLD": (105, 99, 104)}, "d2")
        reasons = {t.symbol: t.exit_reason for t in c1}
        assert reasons["WIN"] == "TARGET" and reasons["LOSE"] == "STOP"
        c2 = b.mark({"HOLD": (106, 100, 105)}, "d3")           # 2nd bar → max hold
        assert c2[0].exit_reason == "MAX_HOLD"
        # realized R: win = +2R, loss = -1R
        rr = {t.symbol: round(t.realized_R, 2) for t in b.closed}
        assert rr["WIN"] == 2.0 and rr["LOSE"] == -1.0

    def test_blow_up_is_possible_but_bounded_by_sizing(self):
        b = PaperBook(capital=100_000)
        # a strategy that loses every trade bleeds equity down, honestly
        for i in range(20):
            b.open_position("bad", f"S{i%5}", 100, 95, 130, f"d{i}", 10)
            b.mark({f"S{i%5}": (101, 94, 95)}, f"d{i}")        # each hits stop
        assert b.equity() < b.capital                          # blew up (a bit)
        assert b.stats("bad")["expectancy_R"] < 0


# ── trade + learn end-to-end, hands-off ──────────────────────────────────────────

class TestTradeAndLearn:
    def test_deploy_trade_and_autonomously_retire_a_loser(self):
        m = PaperAutonomyManager(engaged=True)
        m.deploy(_spec("STR-LOSER"), _ev(), GREEN, cycle=1)
        # 25 losing paper trades → autonomy should retire it on its own
        for i in range(25):
            m.trade_day([{"strategy_id": "STR-LOSER", "symbol": f"S{i%5}",
                          "entry": 100, "stop": 95, "target": 130, "max_hold": 5}],
                        {f"S{i%5}": (101, 94, 95)}, f"d{i}")
        retired = m.review_and_adapt(cycle=2)
        assert "STR-LOSER" in retired
        assert m.strategies["STR-LOSER"].state == S.DECAYED
        assert "STR-LOSER" not in [p.spec.strategy_id for p in m.active()]

    def test_winner_is_kept(self):
        m = PaperAutonomyManager(engaged=True)
        m.deploy(_spec("STR-WIN"), _ev(), GREEN, cycle=1)
        for i in range(25):
            m.trade_day([{"strategy_id": "STR-WIN", "symbol": f"S{i%5}",
                          "entry": 100, "stop": 90, "target": 120, "max_hold": 5}],
                        {f"S{i%5}": (121, 101, 120)}, f"d{i}")
        assert m.review_and_adapt(cycle=2) == []
        assert m.strategies["STR-WIN"].state == S.PAPER_EVALUATION
        assert m.book.stats("STR-WIN")["expectancy_R"] > 0

    def test_per_strategy_daily_trade_cap(self):
        m = PaperAutonomyManager(engaged=True, max_trades_per_day=2)
        m.deploy(_spec("STR-CAP"), _ev(), GREEN, cycle=1)
        sigs = [{"strategy_id": "STR-CAP", "symbol": f"S{i}", "entry": 100, "stop": 95,
                 "target": 115, "max_hold": 10} for i in range(5)]
        out = m.trade_day(sigs, {f"S{i}": (100, 99, 100) for i in range(5)}, "d1")
        assert out["opened"] == 2


# ── the brain: engage → self-drives paper end to end ─────────────────────────────

class TestBrainPaperAutonomy:
    def _bars(self):
        # every deployed symbol wins (target), deterministically
        return lambda date: {"SYM": (121, 101, 120)}

    def _signals(self):
        return lambda ps, date: [{"symbol": "SYM", "entry": 100, "stop": 90, "target": 120,
                                  "max_hold": 5}]

    def test_engaged_brain_deploys_and_trades_without_human(self):
        brain = AutoResearchBrain(evaluate_fn=_market_eval, dataset_status_fn=lambda: GREEN,
                                  signal_fn=self._signals(), bars_fn=self._bars())
        brain.engage_paper_autonomy()
        brain.run_once(date="d1")
        assert brain.state.paper_autonomy is True
        assert brain.state.paper_deployed >= 1
        assert len(brain.paper.book.closed) >= 1          # it actually traded, hands-off

    def test_not_engaged_brain_never_deploys(self):
        brain = AutoResearchBrain(evaluate_fn=_market_eval, dataset_status_fn=lambda: GREEN,
                                  signal_fn=self._signals(), bars_fn=self._bars())
        brain.run_once(date="d1")
        assert brain.state.paper_autonomy is False
        assert brain.state.paper_deployed == 0
        assert brain.paper.strategies == {}

    def test_red_data_engaged_still_deploys_nothing(self):
        brain = AutoResearchBrain(evaluate_fn=_market_eval, dataset_status_fn=lambda: RED,
                                  signal_fn=self._signals(), bars_fn=self._bars())
        brain.engage_paper_autonomy()
        brain.run_once(date="d1")
        assert brain.state.paper_deployed == 0

    def test_gets_smarter_retires_loser_over_days(self):
        losing_bars = lambda date: {"SYM": (101, 89, 90)}    # always stops out
        brain = AutoResearchBrain(evaluate_fn=_market_eval, dataset_status_fn=lambda: GREEN,
                                  signal_fn=self._signals(), bars_fn=losing_bars)
        brain.engage_paper_autonomy()
        for i in range(30):
            brain.run_once(date=f"d{i}")
        assert brain.state.paper_retired >= 1                # autonomously dropped the loser
