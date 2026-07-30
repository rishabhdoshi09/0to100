"""
Deterministic, network-free certification of fully autonomous PAPER_AUTO.

Proves QuantTerm takes, manages, exits and learns from paper trades on snapshot-backed data
with NO per-trade click, NO live envelope, NO broker, NO Streamlit, NO Telegram — and that the
whole thing survives restart without duplication. The user is an optional supervisor.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest

from research.auto_research.scheduler import AutoResearchBrain
from research.intelligence.data.snapshot_store import SnapshotStore
from research.intelligence.registry import StrategyRegistry
from research.strategy_studio import discovery as DISC

_FWD = {"adjustment_consistent": True, "has_universe_history": True,
        "corporate_action_coverage": 1.0, "missing_session_rate": 0.0, "validation_errors": 0}


# enough history for a genuine in-sample backtest (≈35 trades) to promote the strategy
_N = 300
_OPEN = "d298"        # open cycle (in-sample uses d121..d297 → PROMISING)
_EXIT = "d299"        # next session; the exit variant gaps WIN through its stop here


def _spec(sid, family):
    return dataclasses.replace(DISC.generate(DISC.DiscoveryBudget())[0],
                               strategy_id=sid, family=family, max_holding_days=5)


def _eq_rows(symbol, closes):
    rows, prev = [], closes[0]
    for i, c in enumerate(closes):
        rows.append((symbol, f"d{i:03d}", prev, c + 1.5, c - 1.5, c, 1000, "EQ"))
        prev = c
    return rows


def _universe(n=_N, exit_gap=False):
    win = [100 + i * 0.5 for i in range(n)]
    if exit_gap:                                    # final bar gaps far below → stop-out
        win[-1] = 60
    rows = _eq_rows("WIN", win)
    rows += _eq_rows("FLAT", [100 for _ in range(n)])
    rows += _eq_rows("WEAK", [100 - i * 0.2 for i in range(n)])
    return rows


def _registry():
    return StrategyRegistry().build([_spec("MOM", "cross_sectional_momentum")])


def _brain(tmp_path, *, regime="RISK_ON", exit_gap=False, activate=True):
    s = SnapshotStore(tmp_path / "snaps")
    sid = s.commit_snapshot(_universe(_N, exit_gap),
                            index_rows=[("NIFTY", f"d{i:03d}", 100, 101, 99, 100) for i in range(_N)],
                            extra_manifest=_FWD)
    if activate:
        s.activate_snapshot(sid, actor="user")
    brain = _mk_brain(tmp_path)
    brain.regime_fn = lambda: regime                 # honour the requested regime
    brain.snapshot_store = s
    brain.strategy_registry = _registry()
    return brain, s, sid


def _mk_brain(tmp_path):
    return AutoResearchBrain(
        event_store_path=tmp_path / "events.jsonl",
        runtime_state_path=tmp_path / "state.json",
        intel_book_path=tmp_path / "book.json",
        paper_config_path=tmp_path / "paper.json",
        regime_fn=lambda: "RISK_ON")


# ── autonomy: opens without a click, no approval, no envelope, no creds ──────────

class TestNoHumanInLoop:
    def test_opens_paper_position_without_a_click(self, tmp_path):
        brain, *_ = _brain(tmp_path)
        out = brain.run_intelligence_cycle_day(date=_OPEN)
        assert out["positions_opened"]                  # traded, unattended
        assert len(brain.intel_book.open) == 1

    def test_no_operating_envelope_or_broker_required(self, tmp_path):
        # submit path takes no envelope/broker/credential argument at all
        sig = inspect.signature(brain_cls().run_intelligence_cycle_day)
        assert "envelope" not in sig.parameters and "broker" not in sig.parameters

    def test_paper_auto_never_imports_a_broker_or_ems(self):
        import research.intelligence.runtime.autonomous_loop as L
        import research.auto_research.scheduler as S
        for mod in (L, S):
            src = inspect.getsource(mod)
            for banned in ("from ems", "import ems", "kite_client", "place_trade", "BrokerAdapter"):
                assert banned not in src, f"{banned} leaked into {mod.__name__}"

    def test_paper_risk_rejection_is_automatic_no_approval(self, tmp_path):
        # RISK_OFF regime → the loop blocks new entries with no approval request, no exception
        brain, *_ = _brain(tmp_path, regime="RISK_OFF")
        out = brain.run_intelligence_cycle_day(date=_OPEN)
        assert out["positions_opened"] == [] and len(brain.intel_book.open) == 0


# ── headless: scheduler, no Streamlit, telegram-independent, locked, isolated ────

class TestHeadless:
    def test_loop_has_no_streamlit_dependency(self):
        import research.intelligence.runtime.autonomous_loop as L
        assert "streamlit" not in inspect.getsource(L)

    def test_scheduler_lock_prevents_overlap(self, tmp_path):
        brain, *_ = _brain(tmp_path)
        brain._intel_lock.acquire()                     # simulate a cycle already running
        try:
            out = brain.run_intelligence_cycle_day(date=_OPEN)
            assert out["status"] == "SKIPPED_LOCKED"    # competing cycle refused
        finally:
            brain._intel_lock.release()

    def test_duplicate_cycle_creates_no_duplicate_position(self, tmp_path):
        brain, *_ = _brain(tmp_path)
        brain.run_intelligence_cycle_day(date=_OPEN)
        n = len(brain.intel_book.open)
        out2 = brain.run_intelligence_cycle_day(date=_OPEN)   # identical cycle id
        assert out2["status"] == "ALREADY_DONE"
        assert len(brain.intel_book.open) == n

    def test_missing_snapshot_blocks_entries_safely(self, tmp_path):
        brain, *_ = _brain(tmp_path, activate=False)    # snapshot committed but NOT active
        out = brain.run_intelligence_cycle_day(date=_OPEN)
        assert out["positions_opened"] == [] and len(brain.intel_book.open) == 0


# ── persistent activation + restart recovery ─────────────────────────────────────

class TestPersistenceAndRecovery:
    def test_paper_auto_enabled_survives_restart(self, tmp_path):
        b1, *_ = _brain(tmp_path)
        assert b1.is_paper_auto_enabled()
        b1.disable_paper_auto()                          # explicit user opt-out (persisted)
        b2 = _reopen(tmp_path)
        assert b2.paper_auto_enabled is False and not b2.is_paper_auto_enabled()

    def test_open_positions_survive_restart_with_stops(self, tmp_path):
        b1, s, sid = _brain(tmp_path)
        b1.run_intelligence_cycle_day(date=_OPEN)
        assert len(b1.intel_book.open) == 1
        pos = next(iter(b1.intel_book.open.values()))
        b2 = _reopen(tmp_path)                            # process restart
        assert len(b2.intel_book.open) == 1              # position recovered
        rpos = next(iter(b2.intel_book.open.values()))
        assert rpos.stop_price == pos.stop_price and rpos.target_price == pos.target_price
        # a completed cycle is remembered → not replayed (no duplicate)
        assert b2.runtime_state.last_completed_cycle

    def test_management_resumes_and_exits_after_restart(self, tmp_path):
        # open on d129, restart, then a later cycle on d130 gaps through the stop → auto exit
        b1, s, sid = _brain(tmp_path, exit_gap=True)
        b1.run_intelligence_cycle_day(date=_OPEN)
        assert len(b1.intel_book.open) == 1
        b2 = _reopen(tmp_path)
        out = b2.run_intelligence_cycle_day(date=_EXIT)  # manages first → exits
        assert out["positions_closed"]                    # exited automatically, no click
        assert b2.event_store.of_type("OutcomeObservation")   # outcome flowed back


# ── learning loop + honest labelling ─────────────────────────────────────────────

class TestLearningLoop:
    def test_outcome_feeds_brain1_and_is_paper_labelled(self, tmp_path):
        b1, s, sid = _brain(tmp_path, exit_gap=True)
        b1.run_intelligence_cycle_day(date=_OPEN)        # open
        b1.run_intelligence_cycle_day(date=_EXIT)        # exit → outcome
        outcomes = b1.event_store.of_type("OutcomeObservation")
        assert outcomes and all(o.split == "forward" for o in outcomes)   # paper, never "live"
        assert b1.event_store.of_type("StrategyEvidenceCard")             # Brain 1 updated
        # every event carries the snapshot provenance (never fixture-labelled as live evidence)
        cev = b1.event_store.of_type("CanonicalEvent")
        assert all(e.data_snapshot_id in ("", sid) for e in cev)

    def test_manual_disable_is_an_available_override(self, tmp_path):
        b1, *_ = _brain(tmp_path)
        b1.disable_paper_auto()
        assert not b1.is_paper_auto_enabled()             # supervisor override, no live needed


# ── one-shot end-to-end certification ────────────────────────────────────────────

class TestCertification:
    def test_full_paper_auto_chain_then_restart(self, tmp_path):
        # PAPER_AUTO already enabled; verified snapshot active; NO envelope/broker/click anywhere
        brain, s, sid = _brain(tmp_path, exit_gap=True)
        assert brain.is_paper_auto_enabled()

        opened = brain.run_intelligence_cycle_day(date=_OPEN)
        assert opened["positions_opened"] and opened["cards_created"]      # signal→Brain1→Brain2→entry
        assert opened["allocation_decisions"]

        # restart, then automatic management → exit → outcome → evidence update
        brain2 = _reopen(tmp_path)
        assert len(brain2.intel_book.open) == 1                           # recovered, no duplication
        closed = brain2.run_intelligence_cycle_day(date=_EXIT)
        assert closed["positions_closed"] and brain2.event_store.of_type("OutcomeObservation")
        assert brain2.event_store.of_type("StrategyEvidenceCard")

        # no confirmation object / approval click / envelope / broker anywhere in the chain
        import research.intelligence.runtime.autonomous_loop as L
        src = inspect.getsource(L)
        for banned in ("input(", "OperatingEnvelope", "BrokerAdapter", "st.button", "telegram"):
            assert banned not in src


def _reopen(tmp_path):
    brain = _mk_brain(tmp_path)
    brain.snapshot_store = SnapshotStore(tmp_path / "snaps")
    brain.strategy_registry = _registry()
    return brain


def brain_cls():
    return AutoResearchBrain
