"""Autonomy status must be a WAL-aware read-only projection, not a scheduler."""
from __future__ import annotations

import sqlite3

from research.autonomy import health


def test_autonomy_health_counts_latest_500_committed_jobs_without_migration(tmp_path):
    db = tmp_path / "jobs.db"
    writer = sqlite3.connect(db)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        # Deliberately minimal historical schema: a read should not migrate it.
        writer.execute("CREATE TABLE jobs (status TEXT NOT NULL, created_at REAL NOT NULL)")
        rows = [
            ("SUCCEEDED" if i < 2 else "RUNNING", float(i))
            for i in range(501)
        ]
        writer.executemany("INSERT INTO jobs(status, created_at) VALUES (?, ?)", rows)
        writer.commit()

        before = writer.execute("PRAGMA table_info(jobs)").fetchall()
        result = health.read_status(state_path=tmp_path / "status.json", jobs_db=db)
        after = writer.execute("PRAGMA table_info(jobs)").fetchall()

        assert result["jobs"] == {"RUNNING": 499, "SUCCEEDED": 1}
        assert before == after
    finally:
        writer.close()


def test_autonomy_health_missing_database_does_not_create_path(tmp_path):
    db = tmp_path / "uncreated" / "jobs.db"
    result = health.read_status(state_path=tmp_path / "status.json", jobs_db=db)
    assert result["jobs"] == {}
    assert not db.parent.exists()
    assert not db.exists()
