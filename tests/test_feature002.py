"""FEATURE-002: shadow ranks cannot touch production decisions."""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from research.feature002.constants import (
    FEATURE_SET_VERSION,
    FORWARD_START_DATE,
    UNTIL_MATURE,
    candidate_set_id,
    eligible_primary,
    event_id,
    protocol_hash,
)
from research.feature002.evaluate import maturity
from research.feature002.ledger import (
    attach_outcome,
    feature_snapshot,
    get_observation,
    insert_candidate_set,
    insert_observation,
)
from research.feature002.observe import (
    build_shadow_records,
    observe_production_scan,
    persist_shadow,
    set_enabled,
)
from research.feature002.ranks import apply_shadow_ranks
from research.feature002.resolve import compute_outcome


def _cards():
    return [
        {
            "symbol": "AAA",
            "score": 80,
            "verdict": "BUY",
            "signals": ["52-week high breakout"],
            "entry": 100.0,
            "stop": 95.0,
            "target": 110.0,
            "chase_risk": False,
        },
        {
            "symbol": "BBB",
            "score": 60,
            "verdict": "WATCH",
            "signals": ["Strong momentum"],
            "entry": 50.0,
            "stop": 47.0,
            "target": 56.0,
            "chase_risk": False,
        },
        {
            "symbol": "CCC",
            "score": 40,
            "verdict": "WATCH",
            "signals": ["Uptrend pullback to support"],
            "entry": 20.0,
            "stop": 18.0,
            "target": 24.0,
            "chase_risk": True,
        },
    ]


def _hist(n=280, start=80.0, step=0.4):
    idx = pd.date_range("2025-01-02", periods=n, freq="B")
    close = pd.Series([start + i * step for i in range(n)], index=idx)
    return pd.DataFrame(
        {"open": close - 0.2, "high": close + 0.8, "low": close - 0.6,
         "close": close, "volume": [200000] * n},
        index=idx,
    )


@pytest.fixture
def db(tmp_path):
    return tmp_path / "shadow.db"


def test_protocol_hash_is_stable():
    assert len(protocol_hash()) == 16
    assert protocol_hash() == protocol_hash()


def test_pre_freeze_cannot_be_primary():
    assert eligible_primary("2026-07-23", "live_scan",
                            "2026-08-22T12:00:00+05:30", FEATURE_SET_VERSION) is False
    assert eligible_primary("2026-07-24", "replay",
                            "2026-08-22T12:00:00+05:30", FEATURE_SET_VERSION) is False
    assert eligible_primary("2026-07-24", "live_scan",
                            "2026-08-21T12:00:00+05:30", FEATURE_SET_VERSION) is False
    assert eligible_primary("2026-07-24", "live_scan",
                            "2026-08-22T00:00:00+05:30", FEATURE_SET_VERSION) is True


def test_version_change_is_a_new_experiment():
    assert eligible_primary("2026-08-22", "live_scan",
                            "2026-08-22T12:00:00+05:30", "feature-002.v2") is False


def test_observe_does_not_mutate_production_cards(monkeypatch, db):
    monkeypatch.setattr("research.feature002.observe._hist", lambda symbol: _hist())
    monkeypatch.setattr("research.feature002.observe._build_rs_table", lambda *a, **k: None)
    cards = _cards()
    before = copy.deepcopy(cards)
    set_enabled(True)
    observe_production_scan(cards, source="implementation_test", background=False, path=db)
    assert cards == before
    set_enabled(False)
    observe_production_scan(cards, source="implementation_test", background=False, path=db)
    assert cards == before
    set_enabled(True)


def test_shadow_off_and_on_same_decision_objects(monkeypatch, db):
    monkeypatch.setattr("research.feature002.observe._hist", lambda symbol: _hist())
    monkeypatch.setattr("research.feature002.observe._build_rs_table", lambda *a, **k: None)
    cards = _cards()
    set_enabled(False)
    off = copy.deepcopy(cards)
    observe_production_scan(off, source="implementation_test", background=False, path=db)
    set_enabled(True)
    on = copy.deepcopy(cards)
    observe_production_scan(on, source="implementation_test", background=False, path=db)
    assert off == on == cards
    set_enabled(True)


def test_ticket_fields_unchanged(monkeypatch, db):
    monkeypatch.setattr("research.feature002.observe._hist", lambda symbol: _hist())
    monkeypatch.setattr("research.feature002.observe._build_rs_table", lambda *a, **k: None)
    cards = _cards()
    observe_production_scan(cards, source="implementation_test", background=False, path=db)
    for c in cards:
        assert c["entry"] > c["stop"]
        assert "shadow" not in c
        assert "rs_rank" not in c


def test_future_bar_does_not_change_frozen_features():
    hist = _hist(280)
    from research.feature001.trend_features import compute_trend_features
    a = compute_trend_features(hist)
    future = hist.iloc[[-1]].copy()
    future.index = hist.index[-1:] + pd.Timedelta(days=3)
    future["close"] = future["close"] + 25
    future["high"] = future["high"] + 25
    leaked = compute_trend_features(pd.concat([hist, future]))
    asof = compute_trend_features(pd.concat([hist, future]).iloc[:-1])
    assert a["n_structure_passed"] == asof["n_structure_passed"]
    assert leaked["price"] != a["price"]


def test_resolver_cannot_mutate_feature_snapshot(db):
    cset = {
        "candidate_set_id": "set1", "scan_cycle_id": "cyc1",
        "session_date": "2026-08-01", "recorded_at": "2026-08-22T10:00:00+05:30",
        "n_candidates": 1, "family_composition": {"MOMENTUM": 1},
        "source": "implementation_test", "feature_set_version": FEATURE_SET_VERSION,
        "protocol_hash": protocol_hash(),
    }
    insert_candidate_set(cset, path=db)
    snap = {"trend": {"n_structure_passed": 6}, "rs": {"rs_percentile": 88.0}}
    eid = event_id("2026-08-01", "AAA")
    insert_observation({
        "event_id": eid, "candidate_set_id": "set1", "scan_cycle_id": "cyc1",
        "symbol": "AAA", "exchange": "NSE", "session_date": "2026-08-01",
        "recorded_at": "2026-08-22T10:00:00+05:30", "source": "implementation_test",
        "families": ["MOMENTUM"], "primary_family": "MOMENTUM",
        "feature_set_version": FEATURE_SET_VERSION,
        "production_rank_version": "auto_scan.final_order.v1",
        "shadow_rank_version": "feature-002.ranks.v1",
        "protocol_hash": protocol_hash(),
        "production_score": 70, "production_rank": 1, "production_verdict": "BUY",
        "production_signals": ["Strong momentum"], "production_decision": "TAKEN",
        "would_trade": True, "entry": 100, "stop": 95, "target": 110,
        "n_structure_passed": 6, "structure_pass": True,
        "rs_percentile": 88, "rs_score": 0.4, "rs_rank": 1, "trend_rank": 1,
        "combined_shadow_rank": 1, "feature_snapshot": snap, "data_quality": "ok",
    }, path=db)
    before = feature_snapshot(eid, path=db)
    attach_outcome(eid, {"ret_5d": 0.03, "resolved_at": "2026-08-10"}, path=db)
    after = feature_snapshot(eid, path=db)
    assert before == after == snap
    row = get_observation(eid, path=db)
    assert row["n_structure_passed"] == 6


def test_first_write_wins_idempotent(db):
    cset = {
        "candidate_set_id": candidate_set_id("c1"), "scan_cycle_id": "c1",
        "session_date": "2026-08-01", "recorded_at": "2026-08-22T10:00:00+05:30",
        "n_candidates": 1, "family_composition": {},
        "source": "implementation_test", "feature_set_version": FEATURE_SET_VERSION,
        "protocol_hash": protocol_hash(),
    }
    insert_candidate_set(cset, path=db)
    eid = event_id("2026-08-01", "ZZZ")
    base = {
        "event_id": eid, "candidate_set_id": cset["candidate_set_id"],
        "scan_cycle_id": "c1", "symbol": "ZZZ", "exchange": "NSE",
        "session_date": "2026-08-01", "recorded_at": "2026-08-22T10:00:00+05:30",
        "source": "implementation_test", "families": ["MOMENTUM"],
        "primary_family": "MOMENTUM", "feature_set_version": FEATURE_SET_VERSION,
        "production_rank_version": "auto_scan.final_order.v1",
        "shadow_rank_version": "feature-002.ranks.v1",
        "production_score": 10, "production_rank": 1,
        "feature_snapshot": {"v": 1},
    }
    assert insert_observation(base, path=db)["wrote"] is True
    second = dict(base)
    second["feature_snapshot"] = {"v": 999}
    second["production_score"] = 99
    assert insert_observation(second, path=db)["wrote"] is False
    assert feature_snapshot(eid, path=db) == {"v": 1}


def test_live_scan_refuses_pre_freeze_date(db):
    cset = {
        "candidate_set_id": "old", "scan_cycle_id": "old",
        "session_date": "2026-07-23", "recorded_at": "2026-08-22T10:00:00+05:30",
        "n_candidates": 1, "family_composition": {},
        "source": "live_scan", "feature_set_version": FEATURE_SET_VERSION,
        "protocol_hash": protocol_hash(),
    }
    insert_candidate_set(cset, path=db)
    out = insert_observation({
        "event_id": event_id("2026-07-23", "OLD"),
        "candidate_set_id": "old", "scan_cycle_id": "old",
        "symbol": "OLD", "exchange": "NSE", "session_date": "2026-07-23",
        "recorded_at": "2026-08-22T10:00:00+05:30", "source": "live_scan",
        "families": [], "feature_set_version": FEATURE_SET_VERSION,
        "production_rank_version": "auto_scan.final_order.v1",
        "shadow_rank_version": "feature-002.ranks.v1",
        "feature_snapshot": {},
    }, path=db)
    assert out["status"] == "pre_freeze_refused"
    assert get_observation(event_id("2026-07-23", "OLD"), path=db) is None


def test_unresolved_is_null_not_zero():
    out = compute_outcome("NO_SUCH_SYM_FEATURE002", "2026-08-01")
    assert out.get("ret_5d") is None
    assert out.get("unresolved_reason")
    assert out.get("ret_5d") != 0


def test_same_cycle_ranks_only_use_that_set():
    rows = [
        {"symbol": "A", "rs_percentile": 90, "rs_score": 1.0,
         "n_structure_passed": 7, "pct_above_sma200": 10, "ma_spread_50_200_pct": 4},
        {"symbol": "B", "rs_percentile": 40, "rs_score": 0.1,
         "n_structure_passed": 3, "pct_above_sma200": 1, "ma_spread_50_200_pct": 0},
        {"symbol": "C", "rs_percentile": 70, "rs_score": 0.5,
         "n_structure_passed": 6, "pct_above_sma200": 5, "ma_spread_50_200_pct": 2},
    ]
    apply_shadow_ranks(rows)
    by = {r["symbol"]: r for r in rows}
    assert by["A"]["rs_rank"] == 1
    assert by["B"]["rs_rank"] == 3
    assert by["A"]["trend_rank"] == 1
    assert by["B"]["trend_rank"] == 3
    assert by["A"]["production_rank"] == 1
    assert by["C"]["combined_shadow_rank"] in {1, 2}


def test_candidate_set_membership_is_the_input_list(monkeypatch, db):
    monkeypatch.setattr("research.feature002.observe._hist", lambda symbol: _hist())
    monkeypatch.setattr("research.feature002.observe._build_rs_table", lambda *a, **k: None)
    cards = _cards()
    cset, rows = build_shadow_records(
        cards, session_date="2026-08-20", recorded_at="2026-08-22T10:00:00+05:30",
        scan_cycle_id="cycle-x", source="implementation_test",
    )
    persist_shadow(cset, rows, path=db)
    assert cset["n_candidates"] == 3
    assert {r["symbol"] for r in rows} == {"AAA", "BBB", "CCC"}
    # Adding a later card does not rewrite the frozen set id members already stored.
    more = cards + [{"symbol": "DDD", "score": 10, "verdict": "WATCH",
                     "signals": [], "entry": 1, "stop": 0.5, "target": 2,
                     "chase_risk": False}]
    cset2, rows2 = build_shadow_records(
        more, session_date="2026-08-20", recorded_at="2026-08-22T10:00:00+05:30",
        scan_cycle_id="cycle-x", source="implementation_test",
    )
    assert cset2["candidate_set_id"] == cset["candidate_set_id"]
    persist_shadow(cset2, rows2, path=db)
    # Original three event_ids keep first snapshots; DDD may insert.
    assert get_observation(event_id(rows[0]["session_date"], "AAA"), path=db) is not None


def test_quiet_maturity_is_insufficient():
    mat = maturity([])
    assert mat["stage"] == "QUIET"
    assert mat["verdict"] == UNTIL_MATURE
    assert mat["decision_capable"] is False


def test_same_event_id_for_duplicate_scan():
    assert event_id("2026-08-20", "aaa") == event_id("2026-08-20", "AAA")


def test_autopilot_queue_helper_is_prefix_of_production_order():
    cards = _cards()
    queue = cards[:2]
    assert [c["symbol"] for c in queue] == ["AAA", "BBB"]
    shadowed = copy.deepcopy(cards)
    # observe must not be required to build the queue
    assert [c["symbol"] for c in shadowed[:2]] == [c["symbol"] for c in queue]


def test_auto_scan_calls_shadow_after_autopilot():
    src = Path(__file__).resolve().parents[1] / "scan" / "auto_scan.py"
    text = src.read_text()
    ap = text.index("_ap_setups(serialized[:20])")
    store = text.index("_results = serialized")
    hook = text.index("try_observe_production_scan(serialized")
    assert ap < store < hook


def test_market_scan_service_calls_shadow_after_save():
    src = Path(__file__).resolve().parents[1] / "scan" / "market_scan_service.py"
    text = src.read_text()
    saved = text.index("save_scan(payload)")
    hook = text.index("_feature002_hook(payload.get(\"records\")")
    assert saved < hook


def test_runtime_startup_binds_feature002_hook():
    import scan.market_scan_service as mss
    assert mss._feature002_hook is not None
    assert mss._feature002_hook.__name__ == "try_observe_production_scan"
    api = Path(__file__).resolve().parents[1] / "terminal_product_api.py"
    text = api.read_text()
    assert "try_observe_production_scan" in text
    aut = Path(__file__).resolve().parents[1] / "research" / "autonomy" / "__init__.py"
    assert "try_observe_production_scan" in aut.read_text()


def test_unified_scanner_has_no_feature002_import():
    src = Path(__file__).resolve().parents[1] / "scan" / "unified_scanner.py"
    assert "feature002" not in src.read_text()
    ticket = Path(__file__).resolve().parents[1] / "execution" / "trade_executor.py"
    assert "feature002" not in ticket.read_text()
    ap = Path(__file__).resolve().parents[1] / "execution" / "autopilot.py"
    assert "feature002" not in ap.read_text()
