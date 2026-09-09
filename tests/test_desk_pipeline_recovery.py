from __future__ import annotations

from pathlib import Path

from operations.market_ops import DATA_PREPARE, LANES, LONG_TERM_REFRESH, MARKET_SCAN, NEWS_REFRESH
from operations.store import OperationStore
from product.desk_pipeline import advance_desk_pipeline


def _store(tmp_path: Path) -> OperationStore:
    return OperationStore(tmp_path / "jobs.db")


def _fresh_except(monkeypatch, **flags):
    monkeypatch.setattr("product.desk_pipeline.prices_kind_due", lambda: flags.get("prices", None))
    monkeypatch.setattr("product.desk_pipeline.scan_is_fresh", lambda: flags.get("scan", True))
    monkeypatch.setattr("product.desk_pipeline.long_term_is_fresh", lambda: flags.get("long_term", True))
    monkeypatch.setattr("product.desk_pipeline.news_is_fresh", lambda: flags.get("news", True))
    monkeypatch.setattr(
        "product.desk_pipeline.acquire_freshness",
        lambda: {
            "fresh": True,
            "retry_due": False,
            "state": "FRESH",
            "unresolved_symbols": [],
        },
    )


def test_market_scan_bypasses_active_news_lane(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, scan=False)
    store = _store(tmp_path)
    store.enqueue(NEWS_REFRESH, lane=LANES[NEWS_REFRESH], requested_by="test")

    payload = advance_desk_pipeline(store, requested_by="recovery-test")

    assert payload["queued_kind"] == MARKET_SCAN
    assert payload["queued_created"] is True
    assert set(payload["active_kinds"]) == {NEWS_REFRESH, MARKET_SCAN}
    assert {item["kind"] for item in store.active()} == {NEWS_REFRESH, MARKET_SCAN}


def test_data_prepare_bypasses_secondary_long_term_lane(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, prices=DATA_PREPARE, scan=False)
    store = _store(tmp_path)
    store.enqueue(LONG_TERM_REFRESH, lane=LANES[LONG_TERM_REFRESH], requested_by="test")

    payload = advance_desk_pipeline(store, requested_by="recovery-test")

    assert payload["queued_kind"] == DATA_PREPARE
    assert set(payload["active_kinds"]) == {LONG_TERM_REFRESH, DATA_PREPARE}


def test_critical_recovery_steps_do_not_overlap_each_other(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, prices=DATA_PREPARE, scan=False)
    store = _store(tmp_path)
    store.enqueue(DATA_PREPARE, lane=LANES[DATA_PREPARE], requested_by="test")

    payload = advance_desk_pipeline(store, requested_by="recovery-test")

    assert payload["queued_kind"] is None
    assert payload["active_kind"] == DATA_PREPARE
    assert [item["kind"] for item in store.active()] == [DATA_PREPARE]


def test_secondary_steps_remain_serialized(tmp_path: Path, monkeypatch):
    _fresh_except(monkeypatch, scan=True, long_term=False)
    store = _store(tmp_path)
    store.enqueue(NEWS_REFRESH, lane=LANES[NEWS_REFRESH], requested_by="test")

    payload = advance_desk_pipeline(store, requested_by="recovery-test")

    assert payload["queued_kind"] is None
    assert {item["kind"] for item in store.active()} == {NEWS_REFRESH}
