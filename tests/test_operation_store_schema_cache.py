"""OperationStore schema setup is one-time per real SQLite DB, not per GET."""
from __future__ import annotations

from operations import store as operations_store


def test_schema_initialization_is_not_repeated_for_each_status_poll(tmp_path, monkeypatch):
    db = tmp_path / "ops.db"
    monkeypatch.setattr(operations_store, "_SCHEMA_INITIALIZED", {})
    calls = []
    original = operations_store.OperationStore._init_schema

    def counted(self):
        calls.append(str(self.path))
        return original(self)

    monkeypatch.setattr(operations_store.OperationStore, "_init_schema", counted)
    first = operations_store.OperationStore(db)
    second = operations_store.OperationStore(db)
    third = operations_store.OperationStore(db)
    assert len(calls) == 1
    assert first.get_summary("unknown") is None
    assert second.get_summary("unknown") is None
    assert third.get_summary("unknown") is None


def test_schema_reinitializes_after_database_replacement(tmp_path, monkeypatch):
    db = tmp_path / "ops.db"
    monkeypatch.setattr(operations_store, "_SCHEMA_INITIALIZED", {})
    first = operations_store.OperationStore(db)
    first._drop_cached()
    original_id = operations_store._database_identity(db)
    assert original_id is not None

    # On an external volume a recreated DB must have its own schema applied.
    old_db = tmp_path / "previous.db"
    db.rename(old_db)
    replacement = operations_store.OperationStore(db)
    assert replacement.get_summary("unknown") is None
    assert operations_store._database_identity(db) is not None
    assert db.exists()
    assert db != old_db
    replacement._drop_cached()


def test_short_lived_wrappers_share_one_same_thread_connection(tmp_path, monkeypatch):
    db = tmp_path / "shared.db"
    monkeypatch.setattr(operations_store, "_SCHEMA_INITIALIZED", {})
    one = operations_store.OperationStore(db)
    two = operations_store.OperationStore(db)
    try:
        with one._connect() as first:
            with two._connect() as second:
                assert first is second
        assert two.get_summary("unknown") is None
    finally:
        two._drop_cached()


def test_replacement_invalidates_a_cached_read_connection(tmp_path, monkeypatch):
    import sqlite3

    db = tmp_path / "replace.db"
    monkeypatch.setattr(operations_store, "_SCHEMA_INITIALIZED", {})
    original = operations_store.OperationStore(db)
    first_identity = operations_store._database_identity(db)
    with original._connect() as first:
        assert first is not None
    db.rename(tmp_path / "old.db")

    # Create a completely different SQLite DB at the same filename.
    with sqlite3.connect(str(db)) as new_connection:
        new_connection.execute("CREATE TABLE marker (name TEXT)")

    replacement = operations_store.OperationStore(db)
    try:
        assert replacement.get_summary("missing") is None
        with replacement._connect() as new_connection:
            assert "marker" in {
                row[0] for row in new_connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert first_identity != operations_store._database_identity(db)
    finally:
        replacement._drop_cached()
