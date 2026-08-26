"""Desk pipeline queues one stale download at a time in Home → Recos → Reports order."""
from __future__ import annotations

from pathlib import Path

from operations.market_ops import DATA_PREPARE, DUE_DILIGENCE_ACQUIRE, FNO_REFRESH, LONG_TERM_REFRESH, MARKET_SCAN, NEWS_REFRESH
from operations.store import FAILED, SUCCEEDED, OperationStore
from product.desk_pipeline import advance_desk_pipeline, describe_desk_pipeline


def _store(tmp_path: Path) -> OperationStore:
    return OperationStore(tmp_path / "jobs.db")


def _fresh_except(monkeypatch, **flags):
    monkeypatch.setattr("product.desk_pipeline.prices_kind_due", lambda: flags.get("prices", None))
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: flags.get("scan", True))
    monkeypatch.setattr("product.desk_pipeline.long_term_is_fresh", lambda: flags.get("long_term", True))
    monkeypatch.setattr("product.desk_pipeline.news_is_fresh", lambda: flags.get("news", True))
    monkeypatch.setattr("product.desk_pipeline.acquire_is_fresh", lambda: flags.get("investigate", True))


def test_advance_queues_only_the_first_due_step(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, prices=DATA_PREPARE, scan=False, long_term=False, news=False, investigate=False)

    store = _store(tmp_path)
    first = advance_desk_pipeline(store, requested_by="test")
    assert first["queued_kind"] == DATA_PREPARE
    assert first["queued_created"] is True
    assert [item["kind"] for item in first["operations"]] == [DATA_PREPARE]
    assert {row["id"]: row["state"] for row in first["steps"]}["prices"] in {"queued", "running"}
    assert {row["id"]: row["state"] for row in first["steps"]}["scan"] == "waiting"

    second = advance_desk_pipeline(store, requested_by="test")
    assert second["queued_kind"] is None
    assert second["active_kind"] == DATA_PREPARE
    assert len(store.active()) == 1


def test_advance_skips_fresh_prices_and_queues_scan_next(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, prices=None, scan=False, long_term=False, news=False, investigate=False)

    store = _store(tmp_path)
    payload = advance_desk_pipeline(store, requested_by="test")
    assert payload["queued_kind"] == MARKET_SCAN
    assert payload["scan_reused"] is False
    assert {row["id"]: row["state"] for row in payload["steps"]}["prices"] == "ready"
    assert {row["id"]: row["state"] for row in payload["steps"]}["news"] == "waiting"


def test_advance_after_scan_file_is_fresh_goes_to_long_term(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, prices=None, scan=True, long_term=False, news=False, investigate=False)

    store = _store(tmp_path)
    payload = advance_desk_pipeline(store, requested_by="test")
    assert payload["queued_kind"] == LONG_TERM_REFRESH
    assert payload["scan_reused"] is True


def test_all_fresh_enqueues_nothing(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch)

    store = _store(tmp_path)
    payload = advance_desk_pipeline(store, requested_by="test")
    assert payload["queued_kind"] is None
    assert payload["operations"] == []
    assert all(row["state"] == "ready" for row in payload["steps"])
    idle = describe_desk_pipeline(store)
    assert idle["sequential"] is True
    assert "current" in idle or idle["current"] is None


def test_recent_price_failure_halts_the_chain(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, prices=DATA_PREPARE, scan=False, long_term=False, news=False, investigate=False)

    store = _store(tmp_path)
    item, _ = store.enqueue(DATA_PREPARE, lane="data", requested_by="test")
    store.finish(item["operation_id"], status=FAILED, message="no history", error_code="X", error_message="no")
    payload = advance_desk_pipeline(store, requested_by="test")
    assert payload["queued_kind"] is None
    assert "paused" in payload["message"].lower() or "failed" in payload["message"].lower()
    assert payload["steps"][0]["state"] == "failed"


def test_fno_is_the_prices_step_when_history_is_ready(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, prices=FNO_REFRESH, scan=False)

    store = _store(tmp_path)
    payload = advance_desk_pipeline(store, requested_by="test")
    assert payload["queued_kind"] == FNO_REFRESH
    assert payload["current"]["id"] == "prices"


def test_recent_success_does_not_requeue_the_same_step(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, news=False)

    store = _store(tmp_path)
    item, _ = store.enqueue(NEWS_REFRESH, lane="news", requested_by="test")
    store.finish(item["operation_id"], status=SUCCEEDED, message="done", result={"articles": 3})
    payload = advance_desk_pipeline(store, requested_by="test")
    assert payload["queued_kind"] is None
    assert {row["id"]: row["state"] for row in payload["steps"]}["news"] == "ready"
    assert "current" in payload


def test_after_news_queues_investigate_acquire(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, investigate=False)
    store = _store(tmp_path)
    payload = advance_desk_pipeline(store, requested_by="test")
    assert payload["queued_kind"] == DUE_DILIGENCE_ACQUIRE
    assert payload["current"]["id"] == "investigate"
    assert {row["id"]: row["state"] for row in payload["steps"]}["news"] == "ready"

