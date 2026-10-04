"""Desk reads stay independent of active paper writes and cannot mutate state."""
import json
import sqlite3

import pytest

from product.fo_paper_store import FoPaperStore


def test_read_only_projection_reads_committed_state_during_active_writer(tmp_path):
    path = tmp_path / "fo # paper.sqlite3"
    with FoPaperStore(path) as writer:
        writer.conn.execute("BEGIN IMMEDIATE")
        writer.conn.execute(
            "UPDATE fo_paper_meta SET value='uncommitted' WHERE key='schema_version'"
        )
        # No timer releases the writer: a reader that bootstraps the schema
        # cannot succeed here. WAL readers should see the committed snapshot.
        with FoPaperStore(path, read_only=True) as reader:
            assert reader.status()["closed_trades"] == 0
            assert reader.load_positions() == []
            assert reader.load_trades() == []
            assert reader.conn.execute(
                "SELECT value FROM fo_paper_meta WHERE key='schema_version'"
            ).fetchone()[0] == "2"
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                reader.replace_positions([])
        writer.conn.rollback()


def test_missing_read_only_ledger_does_not_create_state(tmp_path):
    path = tmp_path / "missing" / "fo.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        FoPaperStore(path, read_only=True)
    assert not path.parent.exists()


def test_read_only_open_does_not_rewrite_schema_metadata(tmp_path):
    path = tmp_path / "fo.sqlite3"
    with FoPaperStore(path) as writer:
        writer.conn.execute("INSERT INTO fo_paper_meta VALUES('operator_note', 'keep')")
        writer.conn.commit()
        before = writer.conn.execute("SELECT * FROM fo_paper_meta ORDER BY key").fetchall()
        with FoPaperStore(path, read_only=True) as reader:
            assert reader.conn.total_changes == 0
            reader.status()
        after = writer.conn.execute("SELECT * FROM fo_paper_meta ORDER BY key").fetchall()
        assert [tuple(row) for row in after] == [tuple(row) for row in before]


def test_dashboard_and_learning_projections_use_read_only_ledger(monkeypatch, tmp_path):
    import terminal_api
    from product import fo_paper_store, conditional_evidence
    from product.fno_learning_impact import _forward_summary

    path = tmp_path / "fo.sqlite3"
    product_dir = tmp_path / "product"
    product_dir.mkdir()
    (product_dir / "fo_directional.json").write_text(json.dumps({
        "available": True, "candidates": [], "candidate_count": 0,
    }))
    monkeypatch.setattr(terminal_api, "logs_dir", lambda: tmp_path)
    monkeypatch.setattr(fo_paper_store, "logs_path", lambda *parts: path)
    monkeypatch.setattr(conditional_evidence, "load", lambda: {})

    with FoPaperStore(path) as writer:
        writer.conn.execute("BEGIN IMMEDIATE")
        # Each production projection must complete while the lock is held;
        # accidentally opening in write mode instead returns unavailable.
        assert terminal_api._fo_directional_payload()["candidate_evidence_status"] == "AVAILABLE"
        assert terminal_api._fo_paper_payload()["available"] is True
        assert _forward_summary()["available"] is True
        writer.conn.rollback()
