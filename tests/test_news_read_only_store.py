"""News dashboard polls read committed WAL data without initializing SQLite schema."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from news.curator_models import SourceHealth
from news.curator_store import NewsCuratorStore


def test_read_only_news_store_reads_committed_wal_without_write_access(tmp_path):
    path = tmp_path / "news.sqlite3"
    writer = NewsCuratorStore(path)
    event = SourceHealth(
        source_key="test",
        source_name="News Test",
        status="ok",
        fetched_at=datetime.now(timezone.utc).isoformat(),
        article_count=3,
    )
    reader = None
    try:
        writer.upsert_source_health([event])
        # Writer stays open: reader must see the committed WAL transaction,
        # without immutable=1 and without schema-initialization writes.
        reader = NewsCuratorStore(path, read_only=True)
        rows = reader.source_health()
        assert len(rows) == 1
        assert rows[0].source_key == "test"
        assert rows[0].article_count == 3
        assert reader.stats(hours=24)["total"] == 0
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            reader.upsert_source_health([event])
    finally:
        if reader is not None:
            reader.close()
        writer.close()


def test_read_only_news_store_does_not_create_missing_database_or_directory(tmp_path):
    parent = tmp_path / "uncreated"
    missing = parent / "news.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        NewsCuratorStore(missing, read_only=True)
    assert not parent.exists()
    assert not missing.exists()
