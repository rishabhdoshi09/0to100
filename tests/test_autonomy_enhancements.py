"""
Deterministic, network-free tests for the trustworthiness enhancements:
realistic frictions, gap-through-stop, noise-aware calibration, the regime deployment gate,
and book/journal persistence. All paper-only; the live boundary is unaffected.
"""
from __future__ import annotations

from research.auto_research import costs as C
from research.auto_research import growth as GR
from research.auto_research.paper_book import PaperBook
from research.auto_research.paper_autonomy import PaperAutonomyManager
from research.auto_research.scheduler import AutoResearchBrain
from research.auto_research.knowledge import Knowledge
from research.strategy_studio import discovery as DISC
from research.strategy_studio import spec as S


GREEN = {"color": "green", "can_run": True}


# ── costs ────────────────────────────────────────────────────────────────────────

class TestCosts:
    def test_round_trip_cost_positive_and_scales(self):
        c1 = C.india_cash_costs(100, 110, 100)
        c2 = C.india_cash_costs(100, 110, 200)
        assert c1 > 0 and c2 > c1                       # more qty → more cost
        # sanity: total drag is a small fraction of a ~₹21k turnover, not absurd
        assert c1 < 100

    def test_cost_in_R(self):
        r = C.cost_in_R(100, 120, 100, r_unit=10)       # 1R = ₹1000 here
        assert 0 < r < 1
        assert C.cost_in_R(100, 120, 0, 10) == 0.0


# ── realistic fills: costs, slippage, gap-through-stop ───────────────────────────

class TestRealisticFills:
    def test_frictionless_default_is_exact(self):
        b = PaperBook()                                  # no frictions
        b.open_position("s", "WIN", 100, 90, 120, "d1", 10)
        b.mark({"WIN": (121, 101, 120)}, "d2")
        assert round(b.closed[0].realized_R, 2) == 2.0

    def test_costs_and_slippage_reduce_net_R(self):
        b = PaperBook(slippage_bps=5.0, cost_model=C.india_cash_costs)
        b.open_position("s", "WIN", 100, 90, 120, "d1", 10)
        b.mark({"WIN": (121, 101, 120)}, "d2")
        # still a winner, but strictly less than the frictionless +2R
        assert 0 < b.closed[0].realized_R < 2.0

    def test_gap_through_stop_fills_worse(self):
        b = PaperBook()
        b.open_position("s", "GAP", 100, 95, 120, "d1", 10)
        # 4-tuple with an open BELOW the stop → gap-down fill at the open (worse than −1R)
        c = b.mark({"GAP": (92, 96, 91, 93)}, "d2")      # (open, high, low, close)
        assert c[0].exit_reason == "GAP_STOP"
        assert c[0].realized_R < -1.0                    # worse than a clean stop

    def test_gap_through_target_fills_better(self):
        b = PaperBook()
        b.open_position("s", "GU", 100, 95, 110, "d1", 10)
        c = b.mark({"GU": (115, 116, 112, 114)}, "d2")   # gaps up through target
        assert c[0].exit_reason == "GAP_TARGET"
        assert c[0].realized_R > 2.0                     # (115-100)/5 = +3R


# ── noise-aware calibration ──────────────────────────────────────────────────────

class TestNoiseAwareCalibration:
    def test_lower_bound_can_flip_to_overfit(self):
        # point mean is positive, but the conservative lower estimate is ≤ 0 → OVERFIT
        c = GR.calibrate("s", "fam", backtest_R=0.4, forward_R=0.05, n_forward=40,
                         forward_lower_R=-0.02)
        assert c.verdict == GR.OVERFIT and c.keep is False

    def test_r_stats_lower_below_mean(self):
        b = PaperBook()
        for i in range(6):
            b.open_position("s", f"S{i}", 100, 90, 120, f"d{i}", 10)
            # alternate big win / small loss → positive mean, real dispersion
            bar = (121, 101, 120) if i % 2 == 0 else (100, 89, 90)
            b.mark({f"S{i}": bar}, f"d{i}")
        rs = b.r_stats()
        assert rs["n"] == 6 and rs["stderr_R"] > 0
        assert rs["lower_R"] < rs["mean_R"]


# ── regime deployment gate ───────────────────────────────────────────────────────

def _eval(spec, split):
    return DISC.EvidenceReport(n_trades=120, n_symbols=40, net_expectancy_R=0.25,
                               gross_expectancy_R=0.35, cost_drag_R=0.1, p_value=0.01,
                               max_drawdown=0.15, is_synthetic=False, verdict="PASS")


class TestRegimeGate:
    def test_risk_off_blocks_new_deployment(self, tmp_path):
        brain = AutoResearchBrain(evaluate_fn=_eval, dataset_status_fn=lambda: GREEN,
                                  signal_fn=lambda ps, d: [], bars_fn=lambda d: {},
                                  knowledge=Knowledge(tmp_path / "k.json"),
                                  regime_fn=lambda: "RISK_OFF")
        brain.engage_paper_autonomy()
        brain.grow_one_day(date="d0")
        assert brain.paper.strategies == {}              # stood down, nothing deployed

    def test_risk_on_allows_deployment(self, tmp_path):
        brain = AutoResearchBrain(evaluate_fn=_eval, dataset_status_fn=lambda: GREEN,
                                  signal_fn=lambda ps, d: [], bars_fn=lambda d: {},
                                  knowledge=Knowledge(tmp_path / "k.json"),
                                  regime_fn=lambda: "RISK_ON")
        brain.engage_paper_autonomy()
        brain.grow_one_day(date="d0")
        assert len(brain.paper.strategies) >= 1


# ── persistence: the book + journal survive a restart ────────────────────────────

class TestPersistence:
    def test_book_snapshot_restore(self):
        b = PaperBook()
        b.open_position("s", "A", 100, 90, 120, "d1", 10)
        b.mark({"A": (121, 101, 120)}, "d2")
        snap = b.snapshot()
        b2 = PaperBook()
        b2.restore(snap)
        assert b2.stats()["n_trades"] == 1
        assert round(b2.realized_pnl, 2) == round(b.realized_pnl, 2)

    def test_manager_save_load_journal_and_book(self, tmp_path):
        m = PaperAutonomyManager(engaged=True, realistic=False)
        m.deploy(_spec("STR-P"), _eval(None, None), GREEN, cycle=1)
        m.trade_day([{"strategy_id": "STR-P", "symbol": "A", "entry": 100, "stop": 90,
                      "target": 120, "max_hold": 5}], {"A": (121, 101, 120)}, "d1")
        p = tmp_path / "paper.json"
        m.save(p)
        m2 = PaperAutonomyManager(realistic=False)
        assert m2.load(p) is True
        assert m2.book.stats()["n_trades"] == 1
        assert any(j["action"] == "DEPLOY" for j in m2.journal)

    def test_brain_resumes_book_from_path(self, tmp_path):
        p = tmp_path / "paper.json"
        b1 = AutoResearchBrain(evaluate_fn=_eval, dataset_status_fn=lambda: GREEN,
                               signal_fn=lambda ps, d: [{"symbol": "A", "entry": 100,
                                   "stop": 90, "target": 120, "max_hold": 5}],
                               bars_fn=lambda d: {"A": (121, 101, 120)},
                               knowledge=Knowledge(tmp_path / "k.json"), paper_state_path=p)
        b1.engage_paper_autonomy()
        b1.grow_one_day(date="d0")
        n1 = b1.paper.book.stats()["n_trades"]
        assert n1 >= 1
        # a fresh brain pointed at the same file inherits the trade history
        b2 = AutoResearchBrain(evaluate_fn=_eval, dataset_status_fn=lambda: GREEN,
                               knowledge=Knowledge(tmp_path / "k.json"), paper_state_path=p)
        assert b2.paper.book.stats()["n_trades"] == n1


# ── the live boundary is still bolted down ───────────────────────────────────────

class TestLiveBoundaryStillHolds:
    def test_paper_autopilot_cannot_reach_live_review(self):
        assert not S.can_transition(S.PAPER_EVALUATION, S.ELIGIBLE_FOR_LIVE_REVIEW,
                                    S.PAPER_AUTOPILOT)


def _spec(sid):
    import dataclasses
    return dataclasses.replace(DISC.generate(DISC.DiscoveryBudget())[0], strategy_id=sid)
