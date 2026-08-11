"""
Phase A / A1 — Unified Point-in-Time Access Facade tests.

Network-free. Uses the existing SnapshotStore fixtures pattern.
Asserts: no future leakage, no silent contamination, explicit unsafe states,
Snapshot behaviour unchanged, no fake fallbacks, no live network path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.intelligence import data_state as DS
from research.intelligence.data.snapshot_store import SnapshotStore
from research.intelligence.data.provider import SnapshotBarProvider
from research.intelligence.data.pit_contract import (
    PitContract,
    PitReadResult,
    DOMAIN_BARS,
    DOMAIN_UNIVERSE,
    DOMAIN_FUNDAMENTALS,
    DOMAIN_SECTORS,
    DOMAIN_VALUATIONS,
)


def _eq_rows(symbol, closes, series="EQ"):
    rows, prev = [], closes[0]
    for i, c in enumerate(closes):
        rows.append((symbol, f"d{i:03d}", prev, c + 1, c - 1, c, 1000, series))
        prev = c
    return rows


_FWD_MANIFEST = {
    "adjustment_consistent": True,
    "has_universe_history": True,
    "corporate_action_coverage": 1.0,
    "missing_session_rate": 0.0,
    "validation_errors": 0,
    "freshness_days": 0,
}


@pytest.fixture
def store_and_sid(tmp_path):
    store = SnapshotStore(tmp_path)
    rows = _eq_rows("AAA", [100, 101, 102, 103, 104])
    rows += [("BBB", "d002", 50, 51, 49, 50, 10, "EQ"),
             ("BBB", "d003", 50, 52, 49, 51, 10, "EQ"),
             ("BBB", "d004", 51, 53, 50, 52, 10, "EQ")]
    index = [("NIFTY", f"d{i:03d}", 100, 101, 99, 100) for i in range(5)]
    sid = store.commit_snapshot(rows, index_rows=index, extra_manifest=_FWD_MANIFEST)
    return store, sid


# ── PIT read status constants ────────────────────────────────────────────────

def test_pit_read_states_extend_data_state_without_breaking_automation():
    assert DS.READY in DS.DATA_STATES
    assert DS.DEGRADED in DS.DATA_STATES
    assert DS.STALE in DS.DATA_STATES
    assert DS.INCOMPLETE in DS.PIT_READ_STATES
    assert DS.NOT_PIT_SAFE in DS.PIT_READ_STATES
    assert DS.BLOCKED in DS.PIT_READ_STATES
    # New statuses must NOT unlock automation entries.
    assert not DS.allows_new_entries(DS.INCOMPLETE)
    assert not DS.allows_new_entries(DS.NOT_PIT_SAFE)
    assert not DS.allows_new_entries(DS.BLOCKED)


# ── Future leakage ───────────────────────────────────────────────────────────

class TestNoFutureLeakage:
    def test_bars_as_of_never_past_through(self, store_and_sid):
        store, sid = store_and_sid
        pit = PitContract.from_store(store, sid)
        result = pit.as_of(DOMAIN_BARS, when="d002", symbol="AAA")
        assert result.status == DS.READY
        assert [b.date for b in result.data] == ["d000", "d001", "d002"]
        assert all(b.date <= "d002" for b in result.data)

    def test_history_alias_matches_as_of(self, store_and_sid):
        store, sid = store_and_sid
        pit = PitContract.from_store(store, sid)
        a = pit.as_of(DOMAIN_BARS, when="d001", symbol="AAA")
        h = pit.history(DOMAIN_BARS, symbol="AAA", through="d001")
        assert [b.date for b in a.data] == [b.date for b in h.data]

    def test_as_of_beyond_snapshot_is_blocked(self, store_and_sid):
        store, sid = store_and_sid
        pit = PitContract.from_store(store, sid)
        result = pit.as_of(DOMAIN_BARS, when="d999", symbol="AAA")
        assert result.status == DS.BLOCKED
        assert result.data is None
        assert any("beyond" in r or "future" in r for r in result.reasons)

    def test_latest_returns_only_bar_at_or_before_as_of(self, store_and_sid):
        store, sid = store_and_sid
        pit = PitContract.from_store(store, sid)
        result = pit.latest(DOMAIN_BARS, symbol="AAA", as_of="d002")
        assert result.status == DS.READY
        assert result.data.date == "d002"


# ── No silent contamination / Snapshot unchanged ─────────────────────────────

class TestSnapshotCompatibility:
    def test_direct_snapshot_behaviour_unchanged(self, store_and_sid):
        store, sid = store_and_sid
        snap = store.open_snapshot(sid)
        # Existing Snapshot API still works identically.
        assert [b.date for b in snap.bars("AAA", through="d002")] == [
            "d000", "d001", "d002"
        ]
        assert snap.universe("d000") == ["AAA"]
        assert set(snap.universe("d002")) == {"AAA", "BBB"}

    def test_provider_still_works(self, store_and_sid):
        store, sid = store_and_sid
        prov = SnapshotBarProvider(store.open_snapshot(sid))
        assert [b.date for b in prov.bars("AAA", through="d001")] == ["d000", "d001"]

    def test_facade_does_not_mutate_snapshot_files(self, store_and_sid, tmp_path):
        store, sid = store_and_sid
        sdir = Path(tmp_path) / sid
        before = (sdir / "bars_equity.csv").read_text()
        pit = PitContract.from_store(store, sid)
        pit.as_of(DOMAIN_BARS, when="d003", symbol="AAA")
        pit.coverage(as_of="d003")
        assert (sdir / "bars_equity.csv").read_text() == before
        assert store.verify_snapshot(sid)[0] is True


# ── Explicit unsafe / incomplete states ──────────────────────────────────────

class TestExplicitUnsafeStates:
    def test_fundamentals_are_not_pit_safe(self, store_and_sid):
        store, sid = store_and_sid
        pit = PitContract.from_store(store, sid)
        result = pit.as_of(DOMAIN_FUNDAMENTALS, when="d002", symbol="AAA")
        assert result.status == DS.NOT_PIT_SAFE
        assert result.data is None
        assert not result.usable

    def test_sectors_are_not_pit_safe(self, store_and_sid):
        store, sid = store_and_sid
        pit = PitContract.from_store(store, sid)
        result = pit.as_of(DOMAIN_SECTORS, when="d002")
        assert result.status == DS.NOT_PIT_SAFE
        assert result.data is None

    def test_missing_universe_ledger_refuses_biased_fallback(self, store_and_sid, tmp_path):
        store, sid = store_and_sid
        missing = tmp_path / "no_universe_history.json"
        pit = PitContract.from_store(store, sid, universe_history_path=missing)
        result = pit.as_of(DOMAIN_UNIVERSE, when="d002", universe_source="ledger")
        assert result.status == DS.NOT_PIT_SAFE
        assert result.data is None  # no fake today's survivors

    def test_missing_valuation_ledger_is_incomplete(self, store_and_sid, tmp_path):
        store, sid = store_and_sid
        pit = PitContract.from_store(
            store, sid, valuations_path=tmp_path / "no_vals.json"
        )
        result = pit.as_of(DOMAIN_VALUATIONS, when="d002", symbol="AAA")
        assert result.status == DS.INCOMPLETE
        assert result.data is None

    def test_valuation_respects_available_ts(self, store_and_sid, tmp_path):
        store, sid = store_and_sid
        path = tmp_path / "vals.json"
        # ISO dates required — pit_valuations parses available_ts via pandas.
        path.write_text(json.dumps({
            "schema_version": 1,
            "source": "operator",
            "rows": [
                {"symbol": "AAA", "available_ts": "2024-01-03", "pe": 20.0},
                {"symbol": "AAA", "available_ts": "2024-01-01", "pe": 18.0},
            ],
        }), encoding="utf-8")
        pit = PitContract.from_store(store, sid, valuations_path=path)
        early = pit.as_of(DOMAIN_VALUATIONS, when="2024-01-02", symbol="AAA")
        assert early.status == DS.READY
        assert early.data["pe"] == 18.0
        assert early.data["available_ts"] == "2024-01-01"
        # Later revision must not contaminate earlier as_of
        assert early.data["pe"] != 20.0
        late = pit.as_of(DOMAIN_VALUATIONS, when="2024-01-03", symbol="AAA")
        assert late.data["pe"] == 20.0

    def test_unknown_domain_blocked(self, store_and_sid):
        store, sid = store_and_sid
        pit = PitContract.from_store(store, sid)
        result = pit.as_of("order_book", when="d001")
        assert result.status == DS.BLOCKED


# ── No network / no fake data ────────────────────────────────────────────────

class TestNoNetworkNoFake:
    def test_allow_network_raises(self, store_and_sid):
        store, sid = store_and_sid
        snap = store.open_snapshot(sid)
        with pytest.raises(ValueError, match="allow_network"):
            PitContract(snap, allow_network=True)

    def test_no_snapshot_blocks_bars(self):
        pit = PitContract(None)
        result = pit.as_of(DOMAIN_BARS, when="d001", symbol="AAA")
        assert result.status == DS.BLOCKED
        assert result.data is None


# ── Coverage + universe snapshot path ────────────────────────────────────────

class TestCoverageAndUniverse:
    def test_snapshot_universe_is_contemporaneous(self, store_and_sid):
        store, sid = store_and_sid
        pit = PitContract.from_store(store, sid)
        early = pit.as_of(DOMAIN_UNIVERSE, when="d000")
        assert early.status in (DS.READY, DS.DEGRADED)
        assert early.data == ["AAA"]
        later = pit.as_of(DOMAIN_UNIVERSE, when="d002")
        assert set(later.data) == {"AAA", "BBB"}

    def test_coverage_reports_domain_states(self, store_and_sid, tmp_path):
        store, sid = store_and_sid
        pit = PitContract.from_store(
            store,
            sid,
            universe_history_path=tmp_path / "missing_uh.json",
            ca_events_path=tmp_path / "missing_ca.json",
            valuations_path=tmp_path / "missing_val.json",
        )
        cov = pit.coverage(as_of="d004")
        assert isinstance(cov, PitReadResult)
        assert cov.data["domains"][DOMAIN_FUNDAMENTALS] == DS.NOT_PIT_SAFE
        assert cov.data["domains"][DOMAIN_SECTORS] == DS.NOT_PIT_SAFE
        assert cov.data["domains"][DOMAIN_BARS] == DS.READY
        assert "tier" in cov.data

    def test_coverage_blocks_future_as_of(self, store_and_sid):
        store, sid = store_and_sid
        pit = PitContract.from_store(store, sid)
        cov = pit.coverage(as_of="d999")
        assert cov.status == DS.BLOCKED


# ── Research-grade universe ledger happy path ────────────────────────────────

def test_universe_ledger_research_grade(store_and_sid, tmp_path):
    store, sid = store_and_sid
    path = tmp_path / "universe_history.json"
    # Membership ledger uses real calendar dates (pandas Timestamp).
    path.write_text(json.dumps({
        "schema_version": 1,
        "source": "nse_archives",
        "rows": [
            {"symbol": "AAA", "listed": "2024-01-01", "delisted": None},
            {"symbol": "BBB", "listed": "2024-01-03", "delisted": None},
            {"symbol": "CCC", "listed": "2024-01-01", "delisted": "2024-01-02"},
        ],
    }), encoding="utf-8")
    pit = PitContract.from_store(store, sid, universe_history_path=path)
    result = pit.as_of(
        DOMAIN_UNIVERSE, when="2024-01-03", universe_source="ledger"
    )
    # CCC delisted before 2024-01-03; BBB listed that day; AAA continuous.
    assert result.status in (DS.READY, DS.DEGRADED)
    assert result.data is not None
    assert "CCC" not in result.data
    assert "AAA" in result.data
    assert "BBB" in result.data
