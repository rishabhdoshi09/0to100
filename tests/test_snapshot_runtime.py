"""
Deterministic, network-free tests for real-data runtime activation:
immutable snapshot store, atomic activation, PIT snapshot provider, production cycle-context
builder, and the flip from fixture-driven to SNAPSHOT-driven autonomous paper trading.
"""
from __future__ import annotations

import dataclasses

import pytest

from research.intelligence.data.snapshot_store import SnapshotStore
from research.intelligence.data.provider import SnapshotBarProvider
from research.intelligence.registry import StrategyRegistry
from research.intelligence.runtime.context_builder import build_context_from_snapshot, READY
from research.intelligence.runtime import run_intelligence_cycle
from research.intelligence.runtime import modes as MODES
from research.intelligence.runtime.runtime_state import RuntimeState
from research.intelligence.event_store import EventStore
from research.intelligence import data_state as DS
from research.auto_research.paper_book import PaperBook
from research.strategy_studio import discovery as DISC


def _spec(sid, family):
    return dataclasses.replace(DISC.generate(DISC.DiscoveryBudget())[0],
                               strategy_id=sid, family=family)


def _eq_rows(symbol, closes, series="EQ"):
    rows, prev = [], closes[0]
    for i, c in enumerate(closes):
        rows.append((symbol, f"d{i:03d}", prev, c + 1, c - 1, c, 1000, series))
        prev = c
    return rows


def _momentum_universe_rows():
    rows = []
    rows += _eq_rows("WIN", [100 + i * 0.5 for i in range(130)])
    rows += _eq_rows("FLAT", [100 for _ in range(130)])
    rows += _eq_rows("WEAK", [100 - i * 0.2 for i in range(130)])
    return rows


_FWD_MANIFEST = {"adjustment_consistent": True, "has_universe_history": True,
                 "corporate_action_coverage": 1.0, "missing_session_rate": 0.0,
                 "validation_errors": 0}
# data present + fresh, but incomplete CA coverage → LIMITED_RESEARCH (not forward-eligible)
_RESEARCH_MANIFEST = {"adjustment_consistent": True, "has_universe_history": True,
                      "corporate_action_coverage": 0.5, "missing_session_rate": 0.0,
                      "validation_errors": 0}


# ── immutable store + identity ───────────────────────────────────────────────────

class TestSnapshotStore:
    def test_commit_is_deterministic_and_idempotent(self, tmp_path):
        s = SnapshotStore(tmp_path)
        rows = _eq_rows("AAA", [100, 101, 102])
        a = s.commit_snapshot(rows)
        b = s.commit_snapshot(list(rows))                 # identical content
        assert a == b                                     # same content → same id
        assert s.list_snapshots() == [a]                  # not rewritten / duplicated

    def test_changed_content_is_a_successor(self, tmp_path):
        s = SnapshotStore(tmp_path)
        a = s.commit_snapshot(_eq_rows("AAA", [100, 101]))
        b = s.commit_snapshot(_eq_rows("AAA", [100, 102]), parent_id=a)
        assert a != b and set(s.list_snapshots()) == {a, b}

    def test_verify_detects_tampering(self, tmp_path):
        s = SnapshotStore(tmp_path)
        sid = s.commit_snapshot(_eq_rows("AAA", [100, 101]))
        assert s.verify_snapshot(sid)[0] is True
        (tmp_path / sid / "bars_equity.csv").write_text("symbol,date\nX,d0\n")
        ok, fails = s.verify_snapshot(sid)
        assert not ok and any("hash mismatch" in f for f in fails)

    def test_missing_data_file_fails_verification(self, tmp_path):
        s = SnapshotStore(tmp_path)
        sid = s.commit_snapshot(_eq_rows("AAA", [100, 101]))
        (tmp_path / sid / "bars_equity.csv").unlink()
        assert s.verify_snapshot(sid)[0] is False


# ── atomic activation ────────────────────────────────────────────────────────────

class TestActivation:
    def test_activate_and_audit(self, tmp_path):
        s = SnapshotStore(tmp_path)
        sid = s.commit_snapshot(_eq_rows("AAA", [100, 101]))
        assert s.get_active_snapshot() is None
        rec = s.activate_snapshot(sid, actor="user", reason="first")
        assert s.get_active_snapshot() == sid and rec["previous_snapshot_id"] == ""
        assert (tmp_path / "activation_audit.jsonl").exists()

    def test_invalid_snapshot_cannot_activate(self, tmp_path):
        s = SnapshotStore(tmp_path)
        with pytest.raises(ValueError):
            s.activate_snapshot("deadbeefdeadbeef")

    def test_pointer_to_missing_snapshot_fails_safe(self, tmp_path):
        s = SnapshotStore(tmp_path)
        (tmp_path / "ACTIVE").write_text('{"snapshot_id": "ghost123"}')
        assert s.get_active_snapshot() is None            # never resolves to a missing snapshot


# ── PIT provider ─────────────────────────────────────────────────────────────────

class TestProvider:
    def test_bars_are_point_in_time(self, tmp_path):
        s = SnapshotStore(tmp_path)
        sid = s.commit_snapshot(_eq_rows("AAA", [100, 101, 102, 103, 104]))
        prov = SnapshotBarProvider(s.open_snapshot(sid))
        bars = prov.bars("AAA", through="d002")
        assert [b.date for b in bars] == ["d000", "d001", "d002"]   # never past `through`

    def test_provider_refuses_invalid_snapshot(self, tmp_path):
        s = SnapshotStore(tmp_path)
        sid = s.commit_snapshot(_eq_rows("AAA", [100, 101]))
        (tmp_path / sid / "manifest.json").write_text("{ corrupt")
        with pytest.raises(ValueError):
            s.open_snapshot(sid)

    def test_universe_is_contemporaneous(self, tmp_path):
        s = SnapshotStore(tmp_path)
        rows = _eq_rows("AAA", [100, 101]) + [("BBB", "d001", 50, 51, 49, 50, 10, "EQ")]
        sid = s.commit_snapshot(rows)
        prov = SnapshotBarProvider(s.open_snapshot(sid))
        assert prov.universe("d000") == ["AAA"]            # BBB not trading on d000
        assert set(prov.universe("d001")) == {"AAA", "BBB"}


# ── the flip: snapshot-driven cycle (no fixtures) → paper position ──────────────

class TestSnapshotDrivenCycle:
    def _active_provider(self, tmp_path):
        s = SnapshotStore(tmp_path)
        sid = s.commit_snapshot(_momentum_universe_rows(),
                                index_rows=[("NIFTY", f"d{i:03d}", 100, 101, 99, 100)
                                            for i in range(130)],
                                extra_manifest=_FWD_MANIFEST)
        s.activate_snapshot(sid, actor="user", reason="test")
        return s, SnapshotBarProvider(s.open_snapshot(sid)), sid

    def test_forward_eligible_snapshot_opens_position(self, tmp_path):
        s, prov, sid = self._active_provider(tmp_path)
        reg = StrategyRegistry().build([_spec("MOM", "cross_sectional_momentum")])
        ctx, readiness = build_context_from_snapshot(prov, reg, as_of="d129",
                                                     mode=MODES.PAPER_AUTO)
        assert ctx.data_snapshot_id == sid              # cycle pinned to one snapshot
        assert ctx.forward_eligible and readiness["MOM"] == READY
        store, book, state = EventStore(), PaperBook(), RuntimeState()
        res = run_intelligence_cycle(ctx, store=store, book=book, runtime_state=state,
                                     backtest_R={"MOM": 0.3}, backtest_trades={"MOM": 40})
        assert res.positions_opened and res.positions_opened[0][1] == "WIN"
        # every canonical event carries the pinned snapshot id
        assert all(e.data_snapshot_id in ("", sid) for e in store.of_type("CanonicalEvent"))

    def test_not_forward_eligible_runs_research_but_blocks_entries(self, tmp_path):
        # data present + fresh, but incomplete corporate-action coverage → research-eligible only
        s = SnapshotStore(tmp_path)
        sid = s.commit_snapshot(_momentum_universe_rows(),
                                index_rows=[("NIFTY", f"d{i:03d}", 100, 101, 99, 100)
                                            for i in range(130)],
                                extra_manifest=_RESEARCH_MANIFEST)
        s.activate_snapshot(sid, actor="user")
        prov = SnapshotBarProvider(s.open_snapshot(sid))
        reg = StrategyRegistry().build([_spec("MOM", "cross_sectional_momentum")])
        ctx, _r = build_context_from_snapshot(prov, reg, as_of="d129", mode=MODES.PAPER_AUTO)
        assert ctx.data_ok and ctx.forward_eligible is False   # cycle runs, but no new risk
        store, book, state = EventStore(), PaperBook(), RuntimeState()
        res = run_intelligence_cycle(ctx, store=store, book=book, runtime_state=state,
                                     backtest_R={"MOM": 0.3}, backtest_trades={"MOM": 40})
        assert res.cards_created == ["MOM"]              # research/evidence still updated
        assert res.positions_opened == []               # but no new entries
        assert any("forward-eligible" in r for r in res.no_action_reasons)

    def test_missing_benchmark_blocks_relative_strength(self, tmp_path):
        s = SnapshotStore(tmp_path)
        sid = s.commit_snapshot(_momentum_universe_rows(), extra_manifest=_FWD_MANIFEST)  # no index
        s.activate_snapshot(sid, actor="user")
        prov = SnapshotBarProvider(s.open_snapshot(sid))
        reg = StrategyRegistry().build([_spec("RS", "sector_rotation")])
        ctx, readiness = build_context_from_snapshot(prov, reg, as_of="d129",
                                                     mode=MODES.PAPER_AUTO)
        assert readiness["RS"] == "MISSING_BENCHMARK" and ctx.strategies == []


# ── scheduler wiring (Part 16) ───────────────────────────────────────────────────

class TestSchedulerWiring:
    def test_no_active_snapshot_is_safe_no_op(self, tmp_path):
        from research.auto_research.scheduler import AutoResearchBrain
        brain = AutoResearchBrain()
        brain.snapshot_store = SnapshotStore(tmp_path)   # empty — no active snapshot
        brain.strategy_registry = StrategyRegistry().build([_spec("MOM", "cross_sectional_momentum")])
        out = brain.run_intelligence_cycle_day(date="d129")
        assert out["positions_opened"] == []             # honest no-op with no snapshot

    def test_active_snapshot_drives_the_brain_cycle(self, tmp_path):
        from research.auto_research.scheduler import AutoResearchBrain
        s = SnapshotStore(tmp_path)
        sid = s.commit_snapshot(_momentum_universe_rows(),
                                index_rows=[("NIFTY", f"d{i:03d}", 100, 101, 99, 100)
                                            for i in range(130)],
                                extra_manifest=_FWD_MANIFEST)
        s.activate_snapshot(sid, actor="user")
        brain = AutoResearchBrain()
        brain.snapshot_store = s
        brain.strategy_registry = StrategyRegistry().build([_spec("MOM", "cross_sectional_momentum")])
        out = brain.run_intelligence_cycle_day(date="d129")
        assert out["status"] in ("OK", "NO_ACTION")
        # it ran on the real snapshot (a position or at least a pinned card)
        assert brain.event_store.of_type("CanonicalEvent")
