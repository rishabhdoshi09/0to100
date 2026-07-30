"""
Deterministic, network-free tests for the Autonomous Intelligence Runtime (end-to-end loop).

Proves the operational milestone: the two brains run a real paper feedback loop that is
deterministic, idempotent, point-in-time safe, restartable, portfolio-gated, honest with no
data, and structurally incapable of live.
"""
from __future__ import annotations

import dataclasses

import pytest

from research.intelligence.event_store import EventStore
from research.intelligence.runtime import run_intelligence_cycle
from research.intelligence.runtime.cycle_context import CycleContext
from research.intelligence.runtime import modes as MODES
from research.intelligence.runtime import controls as CTRL
from research.intelligence.runtime.runtime_state import RuntimeState
from research.intelligence.runtime.cycle_result import (
    STATUS_ALREADY_DONE, STATUS_NO_ACTION)
from research.intelligence import strategy_runtime as RT
from research.intelligence import allocation_brain as AB
from research.auto_research.paper_book import PaperBook
from research.strategy_studio import discovery as DISC


def _breakout_spec(sid="STR-BO"):
    return dataclasses.replace(DISC.generate(DISC.DiscoveryBudget())[0],
                               strategy_id=sid, family="breakout")


def _breakout_history(entry_day="d25"):
    bars = [RT.Bar(f"d{i}", 100, 101, 99, 100) for i in range(25)]
    bars.append(RT.Bar(entry_day, 100, 112, 100, 111))     # breaks the 101 pivot on entry_day
    return bars


def _ctx(as_of="d25", *, data_ok=True, mode=MODES.PAPER_AUTO, sid="STR-BO",
         family="breakout", history=None):
    spec = dataclasses.replace(_breakout_spec(sid), family=family)
    hist = history if history is not None else _breakout_history(as_of)
    return CycleContext(as_of_date=as_of, mode=mode, data_ok=data_ok,
                        data_snapshot_id="snap1", strategies=[spec],
                        data={sid: {"AAA": hist}})


def _book():
    return PaperBook()


def _run(ctx, store=None, book=None, state=None, **kw):
    store = EventStore() if store is None else store      # empty store is falsy — use `is None`
    book = _book() if book is None else book
    state = RuntimeState() if state is None else state
    # strong backtest so the fresh strategy is PROMISING → earns exploratory bootstrap risk
    kw.setdefault("backtest_R", {ctx.strategies[0].strategy_id: 0.3})
    kw.setdefault("backtest_trades", {ctx.strategies[0].strategy_id: 40})
    res = run_intelligence_cycle(ctx, store=store, book=book, runtime_state=state, **kw)
    return res, store, book, state


# ── end-to-end: signal → brains → intent → gate → paper position ─────────────────

class TestEndToEnd:
    def test_promising_strategy_opens_a_paper_position(self):
        res, store, book, state = _run(_ctx())
        assert res.signals_generated == [("STR-BO", "AAA")]
        assert res.cards_created == ["STR-BO"]
        assert any(a == "DEPLOY" for _, a in res.allocation_decisions)
        assert res.positions_opened == [("STR-BO", "AAA")]
        assert len(book.open) == 1
        # full audit trail exists
        types = {e.event_type for e in store.of_type("CanonicalEvent")}
        for t in ("CYCLE_STARTED", "SIGNAL_GENERATED", "EVIDENCE_CARD_CREATED",
                  "ALLOCATION_DECISION_CREATED", "TRADE_INTENT_CREATED",
                  "PAPER_POSITION_OPENED", "CYCLE_COMPLETED"):
            assert t in types

    def test_position_exits_and_outcome_feeds_back(self):
        store, book, state = EventStore(), _book(), RuntimeState()
        _run(_ctx("d25"), store, book, state)
        assert len(book.open) == 1
        # next day the stop is hit → managed exit + outcome decoded
        spec = _breakout_spec()
        hist2 = _breakout_history("d25") + [RT.Bar("d26", 100, 100, 80, 82)]  # gaps down
        ctx2 = CycleContext(as_of_date="d26", mode=MODES.PAPER_AUTO, data_ok=True,
                            data_snapshot_id="snap1", strategies=[spec],
                            data={"STR-BO": {"AAA": hist2}})
        res2, *_ = _run(ctx2, store, book, state)
        assert res2.positions_closed and res2.positions_closed[0][0] == "STR-BO"
        assert len(book.open) == 0
        assert store.of_type("OutcomeObservation")           # outcome flowed back


# ── idempotency (Phase C) ────────────────────────────────────────────────────────

class TestIdempotency:
    def test_same_cycle_id_is_no_op_second_time(self):
        store, book, state = EventStore(), _book(), RuntimeState()
        ctx = _ctx()
        _run(ctx, store, book, state)
        n_events, n_open = len(store), len(book.open)
        res2, *_ = _run(ctx, store, book, state)              # identical cycle again
        assert res2.status == STATUS_ALREADY_DONE
        assert len(book.open) == n_open                       # no duplicate position
        assert res2.positions_opened == []

    def test_deterministic_cycle_id(self):
        assert _ctx().cycle_id() == _ctx().cycle_id()


# ── no-data / mode safety (Phase Q, P) ───────────────────────────────────────────

class TestSafety:
    def test_no_data_is_honest_no_action(self):
        res, store, book, _ = _run(_ctx(data_ok=False))
        assert res.status == STATUS_NO_ACTION and not book.open
        assert "data gate failed" in " ".join(res.no_action_reasons)

    def test_live_mode_is_refused(self):
        with pytest.raises(MODES.LiveModeDisabled):
            _run(_ctx(mode="LIMITED_LIVE"))

    def test_paper_paused_manages_but_opens_nothing(self):
        res, store, book, _ = _run(_ctx(mode=MODES.PAPER_PAUSED))
        assert not book.open                                  # no new entries when paused

    def test_unsupported_family_emits_event_and_no_position(self):
        res, store, book, _ = _run(_ctx(family="mean_reversion"))
        assert "STR-BO" in res.unsupported and not book.open
        assert any(e.event_type == "STRATEGY_RUNTIME_UNSUPPORTED"
                   for e in store.of_type("CanonicalEvent"))


# ── portfolio gate (Phase J) ─────────────────────────────────────────────────────

class TestPortfolioGate:
    def test_duplicate_symbol_is_blocked(self):
        store, book, state = EventStore(), _book(), RuntimeState()
        _run(_ctx("d25"), store, book, state)                 # opens AAA
        # a second, different strategy signalling the SAME symbol on a later cycle is blocked
        spec2 = _breakout_spec("STR-BO2")
        ctx2 = CycleContext(as_of_date="d99", mode=MODES.PAPER_AUTO, data_ok=True,
                            data_snapshot_id="snap1", strategies=[spec2],
                            data={"STR-BO2": {"AAA": _breakout_history("d99")}})
        res2, *_ = _run(ctx2, store, book, state)
        assert ("STR-BO2", "DUPLICATE_SYMBOL") in res2.intents_blocked


# ── restart / persistence (Phase N) ──────────────────────────────────────────────

class TestRestart:
    def test_state_and_book_survive_restart(self, tmp_path):
        ev_p = tmp_path / "events.jsonl"
        rs_p = tmp_path / "state.json"
        store, book, state = EventStore(ev_p), _book(), RuntimeState(rs_p)
        ctx = _ctx()
        _run(ctx, store, book, state)
        state.save()
        assert len(book.open) == 1
        # fresh objects loaded from disk
        store2 = EventStore(ev_p)
        state2 = RuntimeState(rs_p)
        assert len(store2) == len(store)
        assert state2.is_cycle_done(ctx.cycle_id())           # completed cycle remembered
        # re-running the same cycle post-restart is a no-op
        res2, *_ = _run(ctx, store2, book, state2)
        assert res2.status == STATUS_ALREADY_DONE

    def test_unreconciled_state_refuses_new_risk(self):
        # a book with an open position whose strategy the state doesn't know about
        book = _book()
        book.open_position("GHOST", "ZZZ", 100, 90, 120, "d0", 10)
        state = RuntimeState()
        res, store, book, _ = _run(_ctx(), book=book, state=state)
        assert not any(p.strategy_id == "STR-BO" for p in book.open.values())  # no new entry
        assert any("reconcil" in w.lower() for w in res.warnings)


# ── owner controls (Phase P) produce canonical events ────────────────────────────

class TestControls:
    def test_close_all_and_audit_event(self):
        store, book, state = EventStore(), _book(), RuntimeState()
        _run(_ctx(), store, book, state)
        assert len(book.open) == 1
        n = CTRL.close_all(store, book, "d26")
        assert n == 1 and len(book.open) == 0
        assert any(e.event_type == "MANUAL_CLOSE_ALL" and e.actor == "user"
                   for e in store.of_type("CanonicalEvent"))

    def test_set_mode_rejects_live(self):
        store = EventStore()
        with pytest.raises(MODES.LiveModeDisabled):
            CTRL.set_mode(store, RuntimeState(), "FULL_AUTO")


# ── scheduler integration (Phase O) — honest no-op with no registry ──────────────

class TestSchedulerIntegration:
    def test_brain_runs_intel_cycle_noop_without_registry(self):
        from research.auto_research.scheduler import AutoResearchBrain
        brain = AutoResearchBrain()                            # no intel_registry_fn, no data
        out = brain.run_intelligence_cycle_day(date="d1")
        assert out["status"] in ("NO_ACTION", "OK", "ALREADY_DONE")
        assert out["positions_opened"] == []                  # nothing traded with no data

    def test_intel_book_is_separate_from_legacy_paper(self):
        from research.auto_research.scheduler import AutoResearchBrain
        brain = AutoResearchBrain()
        assert brain.intel_book is not brain.paper.book       # no shared-book conflict
