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
