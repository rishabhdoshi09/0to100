"""End-to-end FEATURE-002 logging through the production scan path.

Broker / paper / production ranks must not move. Network-free.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from research.feature002.constants import (
    FEATURE_SET_VERSION,
    FORWARD_START_TS_IST,
    UNTIL_MATURE,
    eligible_primary,
    event_id,
    protocol_hash,
)
from research.feature002.ledger import feature_snapshot, get_observation
from research.feature002.observe import (
    build_shadow_records,
    observe_production_scan,
    set_enabled,
)
from research.feature002.resolve import resolve_event


def _hist(n=280, start=80.0, step=0.4, last="2026-08-21"):
    end = pd.Timestamp(last)
    idx = pd.bdate_range(end=end, periods=n)
    close = pd.Series([start + i * step for i in range(n)], index=idx)
    return pd.DataFrame(
        {"open": close - 0.2, "high": close + 0.8, "low": close - 0.6,
         "close": close, "volume": [200000] * n},
        index=idx,
    )


def _result(symbol, verdict="BUY", score=80.0, signals=None):
    return SimpleNamespace(
        symbol=symbol,
        price=100.0,
        change_pct=1.2,
        momentum_5d=3.0,
        volume_ratio=1.5,
        rsi=58.0,
        signal_labels=signals or ["52-week high breakout"],
        categories={"MOMENTUM"},
        reasons=["test"],
        score=score,
        verdict=verdict,
        entry=100.0,
        stop=95.0,
        target=110.0,
        risk_reward=2.0,
        pivot_distance_pct=0.4,
        breakout_grade="A",
        breakout_conviction=0.7,
        avg_vol20=200000,
        above_sma50=True,
        above_sma200=True,
        chase_risk=False,
    )


@pytest.fixture
def db(tmp_path):
    return tmp_path / "shadow.db"


def _patch_observe_io(monkeypatch, db):
    monkeypatch.setattr("research.feature002.observe._hist", lambda symbol: _hist())
    monkeypatch.setattr("research.feature002.observe._build_rs_table", lambda *a, **k: None)
    monkeypatch.setattr("research.feature002.observe._regime", lambda: "TEST")
    monkeypatch.setattr(
        "research.feature002.observe._ist_now_iso",
        lambda: "2026-08-24T10:30:00+05:30",
    )
    monkeypatch.setattr(
        "research.feature002.observe._session_date",
        lambda cards: "2026-08-24",
    )
    # Hook log must not write into the production logs/ tree during tests.
    monkeypatch.setattr(
        "research.feature002.observe.HOOK_LOG",
        db.parent / "hook_log.jsonl",
    )
    monkeypatch.setattr(
        "research.feature002.observe.LEDGER_DIR",
        db.parent,
    )
    set_enabled(True)


def test_session_date_is_ist_scan_not_hist_as_of(monkeypatch, db):
    """Weekend/Monday scan must not inherit Friday's hist date as event_id."""
    _patch_observe_io(monkeypatch, db)
    cards = [{
        "symbol": "AAA", "score": 80, "verdict": "BUY",
        "signals": ["52-week high breakout"], "entry": 100, "stop": 95,
        "target": 110, "chase_risk": False,
    }]
    cset, rows = build_shadow_records(
        cards, session_date="2026-08-24",
        recorded_at="2026-08-24T10:30:00+05:30",
        scan_cycle_id="e2e-session", source="live_scan",
    )
    assert cset["session_date"] == "2026-08-24"
    assert rows[0]["session_date"] == "2026-08-24"
    assert rows[0]["feature_snapshot"]["hist_as_of"] == "2026-08-21"
    assert rows[0]["event_id"] == event_id("2026-08-24", "AAA")
    assert rows[0]["event_id"] != event_id("2026-08-21", "AAA")


def test_production_scan_path_to_ledger_and_resolver(monkeypatch, tmp_path, db):
    """Production _scan_once_locked → shadow row → reload → resolve.

    No order, no paper trade, production rank fields unchanged.
    """
    _patch_observe_io(monkeypatch, db)
    order_calls: list = []

    class FakeScanner:
        def __init__(self, *a, **k):
            pass

        def scan(self, universe, progress=None):
            return [
                _result("AAA", "BUY", 82.0),
                _result("BBB", "WATCH", 55.0, ["Strong momentum"]),
            ]

    real_observe = observe_production_scan

    def fake_observe(serialized, source="live_scan"):
        before = copy.deepcopy(serialized)
        out = real_observe(
            serialized, source=source, background=False, path=db,
        )
        assert serialized == before
        return out

    def boom_order(*a, **k):
        order_calls.append((a, k))
        raise AssertionError("order side effect")

    import scan.unified_scanner as us
    monkeypatch.setattr(us, "UnifiedScanner", FakeScanner)

    import scan.auto_scan as auto

    monkeypatch.setattr(auto, "_STATE_FILE", tmp_path / "scan_store.json")
    monkeypatch.setattr(auto, "_log_buys_for_tracking", lambda *a, **k: None)
    monkeypatch.setattr(auto, "_log_non_events_for_learning", lambda *a, **k: None)
    monkeypatch.setattr(auto, "_log_decisions_for_calibration", lambda *a, **k: None)
    monkeypatch.setattr(auto, "_push_new_setups", lambda *a, **k: None)
    if hasattr(auto, "_push_breakout_confirmed"):
        monkeypatch.setattr(auto, "_push_breakout_confirmed", lambda *a, **k: None)

    import scan.sector_heat as sh
    monkeypatch.setattr(sh, "apply_sector_heat", lambda rows: None)
    import scan.conviction as conv
    monkeypatch.setattr(conv, "build_conviction", lambda rows: rows)
    import data.live_quotes as lq
    monkeypatch.setattr(lq, "get_live_quotes", lambda syms: {})
    import scan.signal_backtest as sb
    monkeypatch.setattr(sb, "combo_edge", lambda keys: None)
    import scan.ev_engine as ev
    monkeypatch.setattr(ev, "tag_ev", lambda rows: None)
    import scan.breadth as br
    monkeypatch.setattr(br, "breadth_from_cache", lambda: {})
    import scan.prime_filter as pf
    monkeypatch.setattr(pf, "tag_prime", lambda *a, **k: 0)
    import data.institutional_flows as flows
    monkeypatch.setattr(flows, "get_flows", lambda: {})
    import execution.autopilot as ap
    monkeypatch.setattr(ap, "on_setups", lambda rows: None)
    import execution.trade_executor as te
    if hasattr(te, "place_trade"):
        monkeypatch.setattr(te, "place_trade", boom_order)

    import research.feature002.observe as obs
    monkeypatch.setattr(obs, "observe_production_scan", fake_observe)

    import research.feature002.watchdog as wd
    monkeypatch.setattr(wd, "HOOK_LOG", tmp_path / "hook_log.jsonl")
    monkeypatch.setattr(wd, "note_production_scan", lambda **k: None)

    auto._scan_once_locked(["AAA", "BBB"])
    results, count, last_ts, st = auto.get_results()
    assert st in {"ready", "error"}
    assert len(results) == 2
    assert [r["symbol"] for r in results] == ["AAA", "BBB"]
    assert results[0]["verdict"] == "BUY"
    assert results[1]["verdict"] == "WATCH"
    assert "shadow" not in results[0]
    assert "combined_shadow_rank" not in results[0]
    assert results[0]["score"] == 82.0
    assert order_calls == []

    # Immutable shadow record persisted.
    eid = event_id("2026-08-24", "AAA")
    row = get_observation(eid, path=db)
    assert row is not None
    assert row["source"] == "live_scan"
    assert row["session_date"] == "2026-08-24"
    assert row["feature_set_version"] == FEATURE_SET_VERSION
    assert row["protocol_hash"] == protocol_hash()
    assert eligible_primary(
        row["session_date"], row["source"], row["recorded_at"], row["feature_set_version"]
    )
    snap = json.loads(row["feature_snapshot"])
    assert snap.get("hist_as_of") == "2026-08-21"
    assert "trend" in snap

    # Reload + unresolved outcome, then resolver; snapshot frozen.
    before = feature_snapshot(eid, path=db)
    res = resolve_event(eid, path=db)
    assert res["status"] in {"resolved", "unresolved"}
    after = feature_snapshot(eid, path=db)
    assert before == after
    row2 = get_observation(eid, path=db)
    assert row2["n_structure_passed"] == row["n_structure_passed"]
    assert row2["rs_percentile"] == row["rs_percentile"]
    assert row2["combined_shadow_rank"] == row["combined_shadow_rank"]


def test_observe_failure_cannot_affect_production(monkeypatch, db):
    _patch_observe_io(monkeypatch, db)

    def explode(*a, **k):
        raise RuntimeError("shadow exploded")

    monkeypatch.setattr("research.feature002.observe.build_shadow_records", explode)
    cards = [{
        "symbol": "AAA", "score": 80, "verdict": "BUY",
        "signals": ["52-week high breakout"], "entry": 100, "stop": 95,
        "target": 110, "chase_risk": False,
    }]
    before = copy.deepcopy(cards)
    out = observe_production_scan(cards, source="live_scan", background=False, path=db)
    assert cards == before
    assert out == {"skipped": "observe_failed"}
    assert get_observation(event_id("2026-08-24", "AAA"), path=db) is None


def test_protocol_invalid_row_not_primary(db):
    from research.feature002.ledger import insert_candidate_set, insert_observation, list_primary_observations

    insert_candidate_set({
        "candidate_set_id": "bad", "scan_cycle_id": "bad",
        "session_date": "2026-08-24", "recorded_at": "2026-08-21T10:00:00+05:30",
        "n_candidates": 1, "family_composition": {},
        "source": "live_scan", "feature_set_version": FEATURE_SET_VERSION,
        "protocol_hash": protocol_hash(),
    }, path=db)
    insert_observation({
        "event_id": event_id("2026-08-24", "BAD"),
        "candidate_set_id": "bad", "scan_cycle_id": "bad",
        "symbol": "BAD", "exchange": "NSE", "session_date": "2026-08-24",
        "recorded_at": "2026-08-21T10:00:00+05:30", "source": "live_scan",
        "families": [], "feature_set_version": FEATURE_SET_VERSION,
        "production_rank_version": "auto_scan.final_order.v1",
        "shadow_rank_version": "feature-002.ranks.v1",
        "feature_snapshot": {"v": 1},
    }, path=db)
    prim = list_primary_observations(path=db)
    assert prim == []
    row = get_observation(event_id("2026-08-24", "BAD"), path=db)
    assert row is not None
    assert row["eligible_primary"] == 0


def test_health_zero_primary_is_expected_without_scan(tmp_path, monkeypatch):
    from research.feature002.health import build_health, classify_empty_primary

    monkeypatch.setattr(
        "research.feature002.health.SCAN_STATE", tmp_path / "missing_scan_store.json",
    )
    empty = classify_empty_primary(
        {"scan_store_exists": False, "latest_production_scan_ts_ist": None, "n_results": 0},
        {"primary": 0},
    )
    assert empty["is_bug"] is False
    assert empty["reason"] == "no_post_activation_production_scan"

    bug = classify_empty_primary(
        {
            "scan_store_exists": True,
            "latest_production_scan_ts_ist": "2026-08-24T10:30:00+05:30",
            "n_results": 12,
        },
        {"primary": 0},
    )
    assert bug["is_bug"] is True

    h = build_health(ledger_path=tmp_path / "empty.db")
    assert h["feature_version"] == FEATURE_SET_VERSION
    assert h["status"] == UNTIL_MATURE
    assert "latest_production_scan_timestamp" in h
    assert "rejection_reasons" in h
