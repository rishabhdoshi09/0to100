from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo


def test_us_zero_state_migrates_to_autonomous_virtual_paper(tmp_path, monkeypatch):
    import execution.us_autopilot as us

    path = tmp_path / "us_autopilot.json"
    path.write_text(json.dumps({
        "armed": False,
        "allocation": 0.0,
        "realized_pnl": 0.0,
        "activity": [],
        "disarmed_reason": "",
        "max_accounted_id": 0,
    }), encoding="utf-8")
    monkeypatch.setattr(us, "_STATE_FILE", path)
    monkeypatch.setattr(us, "_state", {})
    monkeypatch.setenv("QT_US_PAPER_CAPITAL", "100000")
    monkeypatch.setenv("QT_US_PAPER_AUTO", "1")

    state = us._load()

    assert state["armed"] is True
    assert state["allocation"] == 100000.0
    assert any("legacy zero-state migrated" in row for row in state["activity"])


def test_explicit_us_disarm_is_preserved(tmp_path, monkeypatch):
    import execution.us_autopilot as us

    path = tmp_path / "us_autopilot.json"
    path.write_text(json.dumps({
        "armed": False,
        "allocation": 0.0,
        "realized_pnl": 0.0,
        "activity": [],
        "disarmed_reason": "user",
        "max_accounted_id": 0,
    }), encoding="utf-8")
    monkeypatch.setattr(us, "_STATE_FILE", path)
    monkeypatch.setattr(us, "_state", {})

    state = us._load()

    assert state["armed"] is False
    assert state["allocation"] == 0.0


def test_us_scan_store_survives_process_memory_reset(tmp_path, monkeypatch):
    import scan.us_scanner as scanner

    path = tmp_path / "us_scan.json"
    monkeypatch.setattr(scanner, "_STORE", path)
    scanner._persist_results(
        [{"symbol": "TEST", "score": 77.0, "verdict": "BUY"}],
        scope="S&P 500",
    )
    monkeypatch.setattr(scanner, "_results", [])
    monkeypatch.setattr(scanner, "_last_ts", 0.0)
    monkeypatch.setattr(scanner, "_status", "idle")

    rows, stamp, status = scanner.get_us_results()

    assert rows[0]["symbol"] == "TEST"
    assert stamp > 0
    assert status == "ready"
    assert scanner.persisted_us_scan()["scope"] == "S&P 500"


def test_canonical_supervisor_schedules_us_market_ops_slot(monkeypatch):
    import data.us_data as us_data
    import operations.store as op_store
    from operations.market_ops import US_MARKET_SCAN
    from research.autonomy.supervisor import Supervisor

    captured = []

    class FakeStore:
        def __init__(self, *_a, **_k):
            pass

        def latest(self, _kind):
            return {}

        def enqueue(self, kind, **kwargs):
            captured.append((kind, kwargs))
            return ({"status": "PENDING"}, True)

    monkeypatch.setattr(op_store, "OperationStore", FakeStore)
    monkeypatch.setattr(us_data, "us_market_open", lambda _now=None: True)

    dummy = SimpleNamespace(_incident=lambda *_a, **_k: None)
    now = datetime(2026, 9, 25, 20, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    Supervisor._ensure_us_market_pipeline(dummy, now)

    assert captured
    kind, kwargs = captured[0]
    assert kind == US_MARKET_SCAN
    assert kwargs["requested_by"] == "autonomy"
    assert kwargs["payload"]["paper_only"] is True
    assert kwargs["payload"]["live_locked"] is True
    assert kwargs["payload"]["slot"]


def test_market_ops_exposes_dedicated_us_lane():
    from operations.market_ops import LANES, US_MARKET_SCAN

    assert LANES[US_MARKET_SCAN] == "us_market"



def test_us_learning_requires_forward_sample_before_rank_change(tmp_path, monkeypatch):
    import product.us_learning as learning

    monkeypatch.setattr(learning, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(learning, "MODEL", tmp_path / "model.json")
    rows = []
    for i in range(10):
        rows.append({
            "decision_id": f"d{i}",
            "settled": True,
            "action": "TAKE",
            "outcome_R": 1.0,
            "score_bucket": "80+",
            "conviction_bucket": "70+",
            "categories": ["Breakout"],
        })
    model = learning.rebuild_model(rows=rows)
    assert model["status"] == "COLLECTING_FORWARD_EVIDENCE"
    out = learning.adjustment({
        "score": 85,
        "breakout_conviction": 75,
        "categories": ["Breakout"],
    })
    assert out["adjustment"] == 0.0
    assert out["affects_selection"] is False


def test_us_forward_learning_can_boundedly_reorder_paper_ranking(tmp_path, monkeypatch):
    import product.us_learning as learning

    monkeypatch.setattr(learning, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(learning, "MODEL", tmp_path / "model.json")
    rows = []
    for i in range(35):
        rows.append({
            "decision_id": f"good-{i}",
            "settled": True,
            "action": "TAKE",
            "outcome_R": 1.0,
            "score_bucket": "80+",
            "conviction_bucket": "70+",
            "categories": ["Breakout"],
        })
    model = learning.rebuild_model(rows=rows)
    assert model["status"] == "ACTIVE_PAPER_RANKING"
    out = learning.adjustment({
        "score": 85,
        "breakout_conviction": 75,
        "categories": ["Breakout"],
    })
    assert 0.0 < out["adjustment"] <= 3.0
    assert out["affects_selection"] is True
    assert out["live_locked"] is True


def test_us_negative_forward_evidence_demotes_but_never_unlocks_live(tmp_path, monkeypatch):
    import product.us_learning as learning

    monkeypatch.setattr(learning, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(learning, "MODEL", tmp_path / "model.json")
    rows = []
    for i in range(25):
        rows.append({
            "decision_id": f"bad-{i}",
            "settled": True,
            "action": "TAKE",
            "outcome_R": -1.0,
            "score_bucket": "70-79",
            "conviction_bucket": "55-69",
            "categories": ["Momentum"],
        })
    learning.rebuild_model(rows=rows)
    out = learning.adjustment({
        "score": 75,
        "breakout_conviction": 60,
        "categories": ["Momentum"],
    })
    assert -5.0 <= out["adjustment"] < 0.0
    assert out["live_locked"] is True


def test_us_session_date_uses_new_york_calendar(monkeypatch):
    import execution.us_autopilot as us

    class FakeNow:
        def date(self):
            from datetime import date
            return date(2026, 9, 25)

    monkeypatch.setattr(us, "_us_now", lambda: FakeNow())
    assert us._us_session_date() == "2026-09-25"


def test_us_status_truthfully_reports_forward_parity_without_fake_historical_pit(monkeypatch):
    import data.us_data as us_data
    import execution.us_autopilot as us
    import product.us_learning as learning
    import scan.us_scanner as scanner
    from product.us_market_status import status

    monkeypatch.setattr(
        scanner,
        "persisted_us_scan",
        lambda: {
            "status": "ready",
            "scope": "S&P 500",
            "scanned_at": "2026-09-25T20:00:00+00:00",
            "count": 1,
            "records": [{"symbol": "AAPL", "score": 80, "learned_rank_score": 81}],
        },
    )
    monkeypatch.setattr(
        us,
        "get_status",
        lambda: {"armed": True, "open_trades": [], "trades_today_count": 0},
    )
    monkeypatch.setattr(us, "report_card", lambda: {})
    monkeypatch.setattr(us, "reject_funnel", lambda: {})
    monkeypatch.setattr(
        learning,
        "dashboard",
        lambda: {"selection_learning_active": False, "decisions": 0, "settled": 0, "pending": 0},
    )
    monkeypatch.setattr(us_data, "us_market_open", lambda: False)

    payload = status()
    parity = payload["parity"]

    assert parity["state"] == "FORWARD_PAPER_PARITY"
    assert parity["autonomous_scan"] is True
    assert parity["paper_auto_execution"] is True
    assert parity["forward_outcome_settlement"] is True
    assert parity["forward_counterfactuals"] is True
    assert parity["forward_learning"] is True
    assert parity["historical_pit_replay"] is False
    assert parity["survivorship_biased_backtest_allowed"] is False
    assert parity["live_money_parity"] is False
    assert payload["paper_only"] is True
    assert payload["live_locked"] is True
