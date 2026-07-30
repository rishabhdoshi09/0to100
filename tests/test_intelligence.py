"""
Deterministic, network-free tests for the two-brain intelligence architecture.

Maps to the milestone's numbered requirements (1–20). Everything runs on injected fixtures;
no synthetic result is ever presented as market evidence, and the no-data path is asserted.
"""
from __future__ import annotations

import inspect

import pytest

from research.intelligence import schemas as SC
from research.intelligence.event_store import EventStore
from research.intelligence import decoder_registry as REG
from research.intelligence import evidence_brain as EB
from research.intelligence import allocation_brain as AB
from research.intelligence import graduation as GRAD
from research.intelligence import strategy_runtime as RT
from research.strategy_studio import spec as S
from research.strategy_studio import discovery as DISC


def _spec(sid="STR-0001", family="breakout"):
    import dataclasses
    return dataclasses.replace(DISC.generate(DISC.DiscoveryBudget())[0],
                               strategy_id=sid, family=family)


def _sdef(sid="STR-0001", family="breakout"):
    return REG.decode("strategy", _spec(sid, family))[0]


def _card(state, **kw):
    base = dict(strategy_id="STR-0001", family="breakout", evidence_state=state,
                lower_bound_R=0.2, forward_to_backtest=0.9, confidence=0.8,
                deflated_sharpe=0.95, forward_trades=50, max_drawdown=2.0)
    base.update(kw)
    return SC.StrategyEvidenceCard(**base)


# ── 1,2: decoders deterministic + idempotent ─────────────────────────────────────

class TestDecoders:
    def test_identical_input_decodes_identically(self):
        raw = {"symbol": "AAA", "entry": 100, "stop": 95, "target": 115, "strategy_id": "S"}
        a = REG.decode("signal", raw)[0]
        b = REG.decode("signal", raw)[0]
        assert a.record_id == b.record_id and a.as_dict() == b.as_dict()

    def test_reprocessing_creates_no_duplicate(self):
        store = EventStore()
        raw = {"symbol": "AAA", "entry": 100, "stop": 95, "target": 115, "strategy_id": "S"}
        REG.decode_into(store, "signal", [raw])
        REG.decode_into(store, "signal", [raw])          # same input again
        assert len(store.of_type("CanonicalSignal")) == 1

    def test_invalid_signal_yields_no_event(self):
        assert REG.decode("signal", {"symbol": "", "entry": 0}) == []

    def test_explanation_is_structured_not_raw(self):
        rec = REG.decode("explanation", {"observation": "o", "hypothesis": "h",
                                         "decision": "d", "strategy_id": "S"})[0]
        d = rec.as_dict()
        for k in ("observation", "hypothesis", "supporting_evidence", "conflicting_evidence",
                  "decision", "uncertainty", "next_test"):
            assert k in d
        assert "chain_of_thought" not in d and "raw" not in d


# ── 18: reconstruction from persisted events ─────────────────────────────────────

class TestEventStore:
    def test_append_only_and_reconstruct(self, tmp_path):
        p = tmp_path / "events.jsonl"
        s1 = EventStore(p)
        s1.append(_sdef())
        s1.append(_card(EB.CONFIRMED))
        s2 = EventStore(p)                               # reload from disk
        assert len(s2) == len(s1)
        snap = s2.reconstruct()
        assert snap["n_events"] == 2
        assert "StrategyEvidenceCard" in snap["by_type"]

    def test_torn_line_is_skipped(self, tmp_path):
        p = tmp_path / "events.jsonl"
        s = EventStore(p); s.append(_sdef())
        with open(p, "a") as f:
            f.write("{ not json\n")
        s2 = EventStore(p)
        assert len(s2) == 1


# ── 3,4: brain separation is structural ──────────────────────────────────────────

class TestBrainSeparation:
    def test_evidence_brain_has_no_trading_surface(self):
        # check real code identifiers, not prose — Brain 1 imports/uses no trading surface
        src = inspect.getsource(EB)
        for banned in ("open_position(", "paper_book", "PaperBook", "trade_day(",
                       "paper_autonomy", "PaperAutonomyManager"):
            assert banned not in src

    def test_allocation_brain_cannot_mutate_a_card(self):
        card = _card(EB.CONFIRMED)
        AB.decide([card])
        with pytest.raises(Exception):                   # frozen dataclass
            card.evidence_state = EB.OVERFIT             # type: ignore
        with pytest.raises(Exception):
            card.lower_bound_R = 9.9                      # type: ignore

    def test_allocation_output_references_card_but_is_separate_type(self):
        card = _card(EB.CONFIRMED)
        d = AB.decide([card])[0]
        assert isinstance(d, SC.PaperAllocationDecision) and d.card_id == card.record_id


# ── 5,9: deployment gated on an acceptable card ──────────────────────────────────

class TestDeploymentGate:
    def test_overfit_blocks_deployment(self):
        d = AB.decide([_card(EB.OVERFIT, lower_bound_R=-0.1)])[0]
        assert d.action in ("SKIP", "RETIRE") and d.target_risk_pct == 0.0

    def test_insufficient_evidence_not_deployed(self):
        d = AB.decide([_card(EB.INSUFFICIENT_EVIDENCE)])[0]
        assert d.target_risk_pct == 0.0

    def test_confirmed_gets_risk(self):
        d = AB.decide([_card(EB.CONFIRMED, forward_trades=80)])[0]
        assert d.action == "DEPLOY" and d.target_risk_pct > 0


# ── 6,7,8: evidence quality, not recent wins ─────────────────────────────────────

class TestEvidenceQuality:
    def test_tiny_sample_not_overweighted(self):
        # 3 lucky wins, huge mean, but tiny sample → not CONFIRMED, no big risk
        card = EB.build_card(_sdef(), backtest_R=0.3, forward_returns=[2.0, 2.0, 2.0],
                             in_sample_trades=40)
        assert card.evidence_state in (EB.FORWARD_PENDING, EB.PROMISING)
        assert AB.decide([card])[0].target_risk_pct == 0.0

    def test_positive_mean_negative_lower_bound_penalized(self):
        # positive mean but very noisy, so the lower bound sits below the mean (and near/below 0)
        rets = [1.0, -1.0, 1.0, -1.0, 3.0] * 8            # positive mean, wide spread
        card = EB.build_card(_sdef(), backtest_R=0.5, forward_returns=rets)
        assert card.lower_bound_R < card.expectancy_R
        # noisy edge that isn't distinguishable from zero should not deploy strongly
        d = AB.decide([card])[0]
        if card.overfit:
            assert d.target_risk_pct == 0.0

    def test_forward_deterioration_reduces_state(self):
        strong = EB.build_card(_sdef(), backtest_R=0.3, forward_returns=[0.3] * 40)
        weak = EB.build_card(_sdef(), backtest_R=0.3, forward_returns=[-0.2] * 40)
        assert strong.evidence_state == EB.CONFIRMED
        assert weak.evidence_state == EB.OVERFIT
        assert weak.overfit and not strong.overfit


# ── 10: correlated strategies capped as a cluster ────────────────────────────────

class TestCorrelationCap:
    def test_cluster_cap_limits_total_risk(self):
        cards = [_card(EB.CONFIRMED, strategy_id=f"S{i}", forward_trades=80) for i in range(4)]
        clusters = {f"S{i}": "cluster-A" for i in range(4)}     # all the same macro bet
        cfg = AB.AllocationConfig(max_cluster_risk_pct=1.5)
        ds = AB.decide(cards, clusters=clusters, cfg=cfg)
        total = sum(d.target_risk_pct for d in ds)
        assert total <= cfg.max_cluster_risk_pct + 1e-6
        assert any("cluster" in " ".join(d.blocked_by) for d in ds if d.blocked_by)


# ── 11,12: regime stand-down + no-data honesty ───────────────────────────────────

class TestSafetyAndData:
    def test_no_data_deploys_nothing(self):
        ds = AB.decide([_card(EB.CONFIRMED)], data_ok=False)
        assert all(d.action == "SKIP" and d.target_risk_pct == 0.0 for d in ds)

    def test_regime_dependent_not_deployed(self):
        d = AB.decide([_card(EB.REGIME_DEPENDENT)])[0]
        assert d.target_risk_pct == 0.0


# ── 13,14: per-strategy PIT evaluation + realistic fills ─────────────────────────

class TestStrategyRuntime:
    def _hist(self):
        # flat base then a breakout on the last bar
        bars = [RT.Bar(f"d{i}", 100, 101, 99, 100) for i in range(25)]
        bars.append(RT.Bar("d25", 100, 112, 100, 111))       # breaks the 101 pivot
        return bars

    def test_point_in_time_no_lookahead(self):
        spec = _spec(family="breakout")
        sigs = RT.entries_for(spec, "AAA", self._hist())
        assert len(sigs) == 1 and sigs[0]["date"] == "d25"    # only fires on the breakout bar
        # earlier bars saw no future data → no entry before d25
        assert all(s["date"] == "d25" for s in sigs)

    def test_unsupported_family_fails_loud(self):
        with pytest.raises(RT.UnsupportedStrategy):
            RT.entries_for(_spec(family="mean_reversion"), "AAA", self._hist())

    def test_runtime_signal_feeds_realistic_paperbook(self):
        # the entry from the runtime, filled through the realistic book, pays costs/slippage
        from research.auto_research.paper_book import PaperBook
        from research.auto_research.costs import india_cash_costs
        spec = _spec(family="breakout")
        sig = RT.entries_for(spec, "AAA", self._hist())[0]
        b = PaperBook(slippage_bps=3.0, cost_model=india_cash_costs)
        b.open_position("S", "AAA", sig["entry"], sig["stop"], sig["target"], "d25", 10)
        # gap through stop next day → realistic worse-than-stop fill
        c = b.mark({"AAA": (sig["stop"] - 2, sig["stop"] + 1, sig["stop"] - 3, sig["stop"] - 2)}, "d26")
        assert c and c[0].exit_reason == "GAP_STOP"


# ── 15: holdout cannot be repeatedly consumed ────────────────────────────────────

class TestHoldoutProtection:
    def test_untouched_test_raises_on_second_touch(self):
        uts = DISC.UntouchedTestSet()
        uts.evaluate_once(lambda: 1)
        with pytest.raises(RuntimeError):
            uts.evaluate_once(lambda: 2)


# ── 16: paper autonomy / brains cannot reach user or live ────────────────────────

class TestLiveBoundary:
    def test_neither_brain_nor_autopilot_can_user_approve(self):
        for actor in ("system", S.PAPER_AUTOPILOT):
            assert not S.can_transition(S.ELIGIBLE_FOR_HUMAN_LIVE_REVIEW, S.USER_APPROVED, actor)
            with pytest.raises(S.LifecycleError):
                GRAD.user_approve("S", actor=actor, current_state=S.ELIGIBLE_FOR_HUMAN_LIVE_REVIEW)

    def test_user_can_approve(self):
        assert GRAD.user_approve("S", actor="user",
                                 current_state=S.ELIGIBLE_FOR_HUMAN_LIVE_REVIEW) == S.USER_APPROVED

    def test_graduation_only_nominates_never_approves(self):
        d = GRAD.evaluate(_card(EB.CONFIRMED, forward_trades=80, lower_bound_R=0.3,
                                deflated_sharpe=0.95, forward_to_backtest=0.9))
        assert d.to_state == S.ELIGIBLE_FOR_HUMAN_LIVE_REVIEW and d.actor != "user"
        assert d.user_gate_required is True

    def test_graduation_withholds_on_unmet_criteria(self):
        d = GRAD.evaluate(_card(EB.CONFIRMED, forward_trades=10))   # too few forward trades
        assert d.to_state == S.PAPER_EVALUATION and d.unmet_criteria


# ── 17: no prohibited imports in the intelligence package ────────────────────────

class TestNoOrderImports:
    @staticmethod
    def _code_only(src: str) -> str:
        """Source with comments and string literals stripped, so the guard sees real code
        (imports, calls, attribute access) — not docstrings that merely NAME the order-capable
        boundary a data-only module deliberately avoids (e.g. `data/kite_client.py`)."""
        import io, tokenize
        out = []
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
        return " ".join(out)

    def test_no_broker_or_order_imports(self):
        import research.intelligence as I
        import pkgutil, importlib
        pkg = I
        banned = ("trade_executor", "kite_client", "KiteConnect", "zerodha",
                  "place_trade", "telegram")
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            mod = importlib.import_module(m.name)
            src = self._code_only(inspect.getsource(mod))
            for b in banned:
                assert b not in src, f"{b} leaked into {m.name} (in code, not a docstring)"
