from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")


def _capture_store(monkeypatch, *, latest=None):
    import operations.store as op_store

    captured = []

    class FakeStore:
        def __init__(self, *_a, **_k):
            pass

        def latest(self, _kind):
            return dict(latest or {})

        def enqueue(self, kind, **kwargs):
            captured.append((kind, kwargs))
            return ({"status": "PENDING"}, True)

    monkeypatch.setattr(op_store, "OperationStore", FakeStore)
    return captured


def test_fo_scheduler_uses_canonical_intraday_slot_and_paper_lane(monkeypatch):
    from operations.market_ops import FNO_REFRESH
    from research.autonomy.supervisor import Supervisor

    captured = _capture_store(monkeypatch)
    dummy = SimpleNamespace(_incident=lambda *_a, **_k: None)
    now = datetime(2026, 9, 25, 10, 0, tzinfo=IST)

    Supervisor._ensure_fo_market_pipeline(dummy, now, set())

    assert len(captured) == 1
    kind, kwargs = captured[0]
    assert kind == FNO_REFRESH
    assert kwargs["lane"] == "fno"
    assert kwargs["requested_by"] == "autonomy"
    assert kwargs["payload"]["slot"] == "intraday-1000"
    assert kwargs["payload"]["session_date"] == "2026-09-25"
    assert kwargs["payload"]["paper_only"] is True
    assert kwargs["payload"]["live_locked"] is True


def test_fo_scheduler_does_not_scan_opening_noise(monkeypatch):
    from research.autonomy.supervisor import Supervisor

    captured = _capture_store(monkeypatch)
    dummy = SimpleNamespace(_incident=lambda *_a, **_k: None)
    now = datetime(2026, 9, 25, 9, 20, tzinfo=IST)

    Supervisor._ensure_fo_market_pipeline(dummy, now, set())

    assert captured == []


def test_fo_scheduler_adds_one_closing_management_slot(monkeypatch):
    from operations.market_ops import FNO_REFRESH
    from research.autonomy.supervisor import Supervisor

    captured = _capture_store(monkeypatch)
    dummy = SimpleNamespace(_incident=lambda *_a, **_k: None)
    now = datetime(2026, 9, 25, 15, 33, tzinfo=IST)

    Supervisor._ensure_fo_market_pipeline(dummy, now, set())

    assert len(captured) == 1
    kind, kwargs = captured[0]
    assert kind == FNO_REFRESH
    assert kwargs["payload"]["slot"] == "closing-1530"
    assert kwargs["payload"]["paper_only"] is True
    assert kwargs["payload"]["live_locked"] is True


def test_fo_scheduler_never_runs_on_weekend(monkeypatch):
    from research.autonomy.supervisor import Supervisor

    captured = _capture_store(monkeypatch)
    dummy = SimpleNamespace(_incident=lambda *_a, **_k: None)
    now = datetime(2026, 9, 26, 10, 0, tzinfo=IST)

    Supervisor._ensure_fo_market_pipeline(dummy, now, set())

    assert captured == []


def test_fo_scheduler_deduplicates_same_slot_even_after_terminal_result(monkeypatch):
    from operations.market_ops import FNO_REFRESH
    from research.autonomy.supervisor import Supervisor

    latest = {
        "status": "SUCCEEDED",
        "payload": {
            "slot": "intraday-1000",
            "session_date": "2026-09-25",
            "paper_only": True,
            "live_locked": True,
        },
    }
    captured = _capture_store(monkeypatch, latest=latest)
    dummy = SimpleNamespace(_incident=lambda *_a, **_k: None)
    now = datetime(2026, 9, 25, 10, 7, tzinfo=IST)

    Supervisor._ensure_fo_market_pipeline(dummy, now, set())

    assert captured == []


def test_market_ops_exposes_dedicated_fno_lane():
    from operations.market_ops import FNO_REFRESH, LANES

    assert LANES[FNO_REFRESH] == "fno"
