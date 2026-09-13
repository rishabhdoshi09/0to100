from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from product.sqlite_audit import audit_database, classify, discover_databases, summary


def test_classify_readonly_success():
    kind, note = classify(readonly_ok=True, immutable_ok=True, wal_present=False)
    assert kind == "READONLY_OK"
    assert "succeeded" in note


def test_classify_wal_readonly_constraint_is_not_called_corruption():
    kind, note = classify(readonly_ok=False, immutable_ok=True, wal_present=True)
    assert kind == "WAL_READONLY_SIDECAR_CONSTRAINT"
    assert "not proven corruption" in note


def test_classify_base_failure_requires_investigation():
    kind, note = classify(readonly_ok=False, immutable_ok=False, wal_present=False)
    assert kind == "BASE_FILE_UNREADABLE_OR_INTEGRITY_FAILURE"
    assert "targeted investigation" in note


def test_audit_clean_database_is_readonly_ok(tmp_path: Path):
    path = tmp_path / "clean.db"
    con = sqlite3.connect(path)
    con.execute("create table t (id integer primary key, value text)")
    con.execute("insert into t(value) values ('ok')")
    con.commit()
    con.close()

    row = audit_database(path)
    assert row.readonly.ok is True
    assert row.immutable_base.ok is True
    assert row.classification == "READONLY_OK"


def test_discovery_ignores_wal_and_non_sqlite_files(tmp_path: Path):
    (tmp_path / "a.db").write_bytes(b"")
    (tmp_path / "b.sqlite").write_bytes(b"")
    (tmp_path / "c.sqlite3").write_bytes(b"")
    (tmp_path / "a.db-wal").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")

    names = [p.name for p in discover_databases(tmp_path)]
    assert names == ["a.db", "b.sqlite", "c.sqlite3"]


def test_summary_only_escalates_base_integrity_failures(tmp_path: Path):
    good = tmp_path / "good.db"
    con = sqlite3.connect(good)
    con.execute("create table t (id integer)")
    con.commit()
    con.close()

    row = audit_database(good)
    payload = summary([row])
    assert payload["databases"] == 1
    assert payload["needs_investigation"] == 0
    assert payload["immutable_base_failures"] == 0


def test_audit_cli_runs_directly_from_repo_root(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    runtime_root = tmp_path / "runtime"
    env = os.environ.copy()
    env["QT_RUNTIME_ROOT"] = str(runtime_root)

    result = subprocess.run(
        [sys.executable, "scripts/audit_sqlite_runtime.py"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"]["databases"] == 0
    assert payload["summary"]["needs_investigation"] == 0
