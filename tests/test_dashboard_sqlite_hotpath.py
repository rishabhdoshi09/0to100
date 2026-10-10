"""Regression tests for read-only dashboard polling under large operation ledgers.

The status endpoints must never fetch/decode historical result_json blobs.
Explicit /api/operations/{id} remains the full-detail path.
"""
from __future__ import annotations

import terminal_api


def test_operations_dashboard_reads_metadata_only(monkeypatch):
    import operations.store as operations_store

    seen = []

    class MetadataOnlyStore:
        def __init__(self, _path):
            pass

        def recent_summary(self, limit):
            seen.append(("recent_summary", limit))
            return [{"operation_id": "op-1", "status": "SUCCEEDED"}]

        def latest_summary(self, kind):
            seen.append(("latest_summary", kind))
            return {"kind": kind, "status": "SUCCEEDED"}

        def active_summary(self):
            seen.append(("active_summary",))
            return [{"operation_id": "op-2", "status": "RUNNING"}]

        def counts(self):
            return {"RUNNING": 1}

        # Full-record methods must never be used on the polling path.
        def recent(self, *_a, **_kw):
            raise AssertionError("dashboard read full recent operation results")

        def latest(self, *_a, **_kw):
            raise AssertionError("dashboard read full latest operation result")

        def active(self, *_a, **_kw):
            raise AssertionError("dashboard read full active operation results")

    monkeypatch.setattr(operations_store, "OperationStore", MetadataOnlyStore)
    monkeypatch.setattr(
        terminal_api,
        "_ops_runtime_payload",
        lambda: {"running": True, "worker_pid": 100, "heartbeat": "ok", "active": {}},
    )

    payload = terminal_api._operations_payload()
    assert payload["available"] is True
    assert payload["running"] is True
    assert payload["latest"]["MARKET_SCAN"]["status"] == "SUCCEEDED"
    assert payload["active"] == [{"operation_id": "op-2", "status": "RUNNING"}]
    assert ("recent_summary", 100) in seen
    assert ("active_summary",) in seen
    assert not payload.get("error")


def test_news_dashboard_queries_only_news_refresh_metadata(monkeypatch):
    import operations.store as operations_store
    import news.curator_store as curator_store

    queried = []

    class MetadataOnlyStore:
        def __init__(self, _path):
            pass

        def latest_summary(self, kind):
            queried.append(kind)
            return {"kind": kind, "status": "SUCCEEDED"}

        def latest(self, *_a, **_kw):
            raise AssertionError("news read full operation result")

    class FakeNewsStore:
        def __init__(self, _path):
            pass

        def recent(self, **_kw):
            return []

        def source_health(self):
            return []

        def stats(self, **_kw):
            return {"total": 0, "important": 0, "macro": 0, "sources": 0}

        def close(self):
            pass

    monkeypatch.setattr(operations_store, "OperationStore", MetadataOnlyStore)
    monkeypatch.setattr(curator_store, "NewsCuratorStore", FakeNewsStore)
    monkeypatch.setattr(
        terminal_api,
        "_operations_payload",
        lambda: (_ for _ in ()).throw(
            AssertionError("news rebuilt complete operations dashboard")
        ),
    )

    payload = terminal_api._news_payload()
    assert payload["latest_refresh"] == {"kind": "NEWS_REFRESH", "status": "SUCCEEDED"}
    assert queried == ["NEWS_REFRESH"]
    assert "error" not in payload
