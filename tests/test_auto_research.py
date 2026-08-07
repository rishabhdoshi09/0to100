"""
Deterministic, network-free tests for the Autonomous Research Brain.

They prove the two things that matter: the brain reasons and improves ON ITS OWN, and it
NEVER crosses the human-approval gate (no approve/activate/trade), stays honest with no
data, and refuses to present synthetic numbers as market evidence.
"""
from __future__ import annotations

import dataclasses

import pytest

from research.auto_research import loop as L
from research.auto_research.thread import ResearchThread, OBSERVE, PROPOSE, CONCLUDE
from research.auto_research.learning import LearningLedger
from research.auto_research.scheduler import AutoResearchBrain
from research.strategy_studio import discovery as DISC
from research.strategy_studio import spec as S


GREEN = {"color": "green", "can_run": True}
AMBER = {"color": "amber", "can_run": True}
RED = {"color": "red", "can_run": False, "reasons": ["no data"]}


def _fixed_clock():
    return lambda: "2026-07-30T09:15:00"


# ── evaluators (all clearly SYNTHETIC or REAL, deterministically) ────────────────

def _strong_market_evaluator(spec, split):
    """A survivor with clean, market-labelled evidence — deterministic per strategy_id."""
    bump = (int(spec.strategy_id.split("-")[1]) % 5) * 0.02
    return DISC.EvidenceReport(
        n_trades=120, n_symbols=40, gross_expectancy_R=0.35, net_expectancy_R=0.25 + bump,
        cost_drag_R=0.10, p_value=0.01, max_drawdown=0.15, turnover=1.2,
        max_symbol_share=0.08, regime_consistency=0.8, sector_consistency=0.8,
        is_synthetic=False, verdict="PASS")


def _weak_evaluator(spec, split):
    """Fails the evidence gate: too few trades + costs eat the edge."""
    return DISC.EvidenceReport(
        n_trades=5, n_symbols=1, gross_expectancy_R=0.02, net_expectancy_R=-0.05,
        cost_drag_R=0.07, p_value=0.6, max_drawdown=0.5, turnover=5.0,
        max_symbol_share=0.9, is_synthetic=False, verdict="INCONCLUSIVE")


def _synthetic_evaluator(spec, split):
    """Looks great but is SYNTHETIC — must never become a proposal."""
    return DISC.EvidenceReport(
        n_trades=200, n_symbols=50, gross_expectancy_R=0.4, net_expectancy_R=0.3,
        cost_drag_R=0.1, p_value=0.001, max_drawdown=0.1, turnover=1.0,
        max_symbol_share=0.05, regime_consistency=0.9, sector_consistency=0.9,
        is_synthetic=True, verdict="PASS")


# ── thread: append-only + deterministic ──────────────────────────────────────────

class TestResearchThread:
    def test_append_only_and_kinds(self):
        t = ResearchThread(clock=_fixed_clock())
        t.observe(1, "saw data")
        t.propose(1, "try X")
        assert len(t) == 2
        assert [e.kind for e in t.all()] == [OBSERVE, PROPOSE]
        assert [e.seq for e in t.all()] == [1, 2]

    def test_rejects_unknown_kind(self):
        t = ResearchThread(clock=_fixed_clock())
        with pytest.raises(ValueError):
            t.add(1, "MUTATE", "nope")

    def test_persists_and_reloads_jsonl(self, tmp_path):
        p = tmp_path / "thread.jsonl"
        t = ResearchThread(p, clock=_fixed_clock())
        t.observe(1, "a"); t.propose(1, "b"); t.conclude(1, "c")
        t2 = ResearchThread(p, clock=_fixed_clock())
        assert len(t2) == 3
        assert t2.all()[-1].kind == CONCLUDE
        # a torn final line is skipped, not fatal
        with open(p, "a") as f:
            f.write("{ not json\n")
        t3 = ResearchThread(p, clock=_fixed_clock())
        assert len(t3) == 3

    def test_stamp_not_identity(self):
        # different stamps, identical reasoning content
        a = ResearchThread(clock=lambda: "t1"); a.observe(1, "same")
        b = ResearchThread(clock=lambda: "t2"); b.observe(1, "same")
        assert a.all()[0].text == b.all()[0].text
        assert a.all()[0].stamp != b.all()[0].stamp


# ── loop: honesty when no data ───────────────────────────────────────────────────

class TestNoDataHonesty:
    def test_red_data_concludes_unavailable_and_proposes_nothing(self):
        t = ResearchThread(clock=_fixed_clock())
        r = L.run_cycle(1, t, dataset_status=RED, evaluate_fn=_strong_market_evaluator)
        assert r.data_ready is False
        assert r.generated == 0
        assert r.proposals == []
        assert DISC.DISCOVERY_UNAVAILABLE_MSG in r.conclusion
        assert t.all()[-1].kind == CONCLUDE

    def test_none_status_falls_back_to_canonical_and_stays_safe(self, monkeypatch):
        # force canonical readiness to red so the cycle is honest without touching disk
        monkeypatch.setattr(L, "canonical_readiness", lambda: RED)
        t = ResearchThread(clock=_fixed_clock())
        r = L.run_cycle(1, t, dataset_status=None, evaluate_fn=_strong_market_evaluator)
        assert r.data_ready is False
        assert r.proposals == []


# ── loop: reasons, rejects, proposes — autonomously ──────────────────────────────

class TestAutonomousReasoning:
    def test_generates_reasons_and_proposes_on_strong_evidence(self):
        t = ResearchThread(clock=_fixed_clock())
        r = L.run_cycle(1, t, dataset_status=GREEN, evaluate_fn=_strong_market_evaluator)
        assert r.data_ready and r.generated > 0
        assert r.survivors > 0
        assert len(r.proposals) > 0
        # every proposal parks at the ONE gate and is market evidence
        for p in r.proposals:
            assert p.lifecycle_state == S.AWAITING_USER_APPROVAL
            assert p.is_market_evidence is True
        # the thread actually contains reasoning + a proposal + a conclusion
        kinds = {e.kind for e in t.all()}
        assert PROPOSE in kinds and OBSERVE in kinds and CONCLUDE in kinds

    def test_weak_ideas_are_rejected_with_reasons(self):
        t = ResearchThread(clock=_fixed_clock())
        r = L.run_cycle(1, t, dataset_status=GREEN, evaluate_fn=_weak_evaluator)
        assert r.rejected_evidence > 0
        assert r.proposals == []
        # a DECIDE entry records WHY at least one was rejected
        assert any(e.kind == "DECIDE" and "Reject" in e.text for e in t.all())

    def test_deterministic_same_seed_same_thread(self):
        t1 = ResearchThread(clock=_fixed_clock())
        t2 = ResearchThread(clock=_fixed_clock())
        r1 = L.run_cycle(1, t1, dataset_status=GREEN, evaluate_fn=_strong_market_evaluator)
        r2 = L.run_cycle(1, t2, dataset_status=GREEN, evaluate_fn=_strong_market_evaluator)
        assert [e.text for e in t1.all()] == [e.text for e in t2.all()]
        assert r1.as_dict()["proposals"] == r2.as_dict()["proposals"]

    def test_no_evaluator_produces_no_proposal_but_stays_alive(self):
        t = ResearchThread(clock=_fixed_clock())
        r = L.run_cycle(1, t, dataset_status=GREEN, evaluate_fn=None)
        assert r.generated > 0
        assert r.proposals == []
        assert "Nothing was approved or traded" in r.conclusion


# ── loop: the safety boundary is structural ──────────────────────────────────────

class TestSafetyBoundary:
    def test_never_acts_or_approves(self):
        t = ResearchThread(clock=_fixed_clock())
        r = L.run_cycle(1, t, dataset_status=GREEN, evaluate_fn=_strong_market_evaluator)
        assert r.acted_on_market is False
        assert r.approved_anything is False

    def test_synthetic_is_never_market_evidence_or_proposal(self):
        t = ResearchThread(clock=_fixed_clock())
        r = L.run_cycle(1, t, dataset_status=GREEN, evaluate_fn=_synthetic_evaluator)
        assert r.proposals == []              # synthetic never reaches the human gate
        assert any("SYNTHETIC" in e.text for e in t.all())

    def test_advance_to_gate_uses_only_system_transitions(self):
        spec = _one_spec()
        assert L._advance_to_gate(spec) == S.AWAITING_USER_APPROVAL

    def test_gate_step_is_user_only(self):
        # the step BEYOND the gate must be user-actored — proving the brain can't take it
        with pytest.raises(S.LifecycleError):
            S.require_transition(S.AWAITING_USER_APPROVAL, S.APPROVED_FOR_PAPER,
                                 actor="system")


# ── learning: decay + improvement across cycles ──────────────────────────────────

class TestLearning:
    def _prop(self, family, r):
        return L.Proposal(strategy_id="STR-0001", name="x", family=family,
                          config_hash="h", net_expectancy_R=r, n_trades=100,
                          recommendation="ok", lifecycle_state=S.AWAITING_USER_APPROVAL,
                          is_market_evidence=True)

    def test_detects_improvement_then_decay(self):
        led = LearningLedger()
        e1 = led.observe_cycle(1, [self._prop("momentum", 0.20)])
        assert e1[0].kind == "NEW"
        e2 = led.observe_cycle(2, [self._prop("momentum", 0.30)])
        assert e2[0].kind == "IMPROVED"
        e3 = led.observe_cycle(3, [self._prop("momentum", 0.10)])
        assert e3[0].kind == "DECAYED"
        assert "momentum" in led.decayed_families(3)

    def test_ignores_synthetic_proposals(self):
        led = LearningLedger()
        p = dataclasses.replace(self._prop("momentum", 0.9), is_market_evidence=False)
        assert led.observe_cycle(1, [p]) == []

    def test_decay_proposes_retested_child_never_mutates_parent(self):
        led = LearningLedger()
        parent = _one_spec()
        led.observe_cycle(1, [self._prop(parent.family, 0.30)])
        led.observe_cycle(2, [self._prop(parent.family, 0.10)])   # decay
        kids = led.improvement_proposals(2, {parent.family: parent})
        assert len(kids) == 1
        child = kids[0]
        assert child.version == parent.version + 1
        assert child.parent_id == parent.strategy_id
        assert child.max_holding_days <= parent.max_holding_days
        assert child.turnover_cap < parent.turnover_cap
        # parent object is untouched (frozen dataclass, new object returned) and the
        # child is a MATERIAL change → new evidence identity, old evidence can't transfer
        assert child.config_hash() != parent.config_hash()


# ── scheduler: threads memory across cycles, stays honest ────────────────────────

class TestScheduler:
    def test_run_once_accumulates_and_is_safe(self):
        brain = AutoResearchBrain(evaluate_fn=_strong_market_evaluator,
                                  dataset_status_fn=lambda: GREEN)
        r1 = brain.run_once()
        r2 = brain.run_once()
        assert brain.state.cycles_run == 2
        assert brain.state.total_proposals == len(r1.proposals) + len(r2.proposals)
        assert r1.acted_on_market is False and r2.approved_anything is False

    def test_red_data_scheduler_proposes_nothing(self):
        brain = AutoResearchBrain(evaluate_fn=_strong_market_evaluator,
                                  dataset_status_fn=lambda: RED)
        r = brain.run_once()
        assert r.proposals == []
        assert brain.state.total_proposals == 0

    def test_learning_reasoning_written_to_thread_over_cycles(self):
        # feed improving then decaying evidence via a stateful evaluator
        state = {"r": 0.20}

        def evolving(spec, split):
            ev = _strong_market_evaluator(spec, split)
            return dataclasses.replace(ev, net_expectancy_R=state["r"],
                                       max_symbol_share=0.05)

        brain = AutoResearchBrain(evaluate_fn=evolving, dataset_status_fn=lambda: GREEN)
        brain.run_once()
        state["r"] = 0.02
        brain.run_once()
        # a DECAYED reasoning note reached the thread
        assert any(e.kind == "REASON" and "decay" in e.text.lower()
                   for e in brain.thread.all())


# ── canonical readiness fails closed ─────────────────────────────────────────────

class TestCanonicalReadiness:
    def test_missing_folder_fails_closed_to_red(self, tmp_path):
        r = L.canonical_readiness(logs_root=tmp_path / "does_not_exist")
        assert r["color"] == "red"
        assert r["can_run"] is False


# ── helpers ──────────────────────────────────────────────────────────────────────

def _one_spec():
    budget = DISC.DiscoveryBudget()
    cands = DISC.generate(budget)
    assert cands
    return cands[0]
