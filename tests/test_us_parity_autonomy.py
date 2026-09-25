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
