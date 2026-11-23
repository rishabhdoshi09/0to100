"""GPT cycle-8 review of 8f9dc5e0..dfe062e3: the reviewer's sketches (F1-01..F1-10).

Adapted only where the v2 design or a fix changed the API, each marked ADAPTED:
- the fixture puts the database in its own directory (F1-04's fix refuses a
  manifest beside the database);
- F1-02 crashes after the REINDEX commit through ``_live_content`` (v2 has no
  ``_verify_live``);
- F1-06 passes the rowid alias to ``_row_digest``;
- F1-10 runs the runbook's whole stage-3 F1 block, since it is no longer two
  lines;
- F1-01 also runs a writer in ANOTHER PROCESS at the point of publication:
  v2 has no rename, so the original hook point no longer exists.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from product import sqlite_index_repair as repair
from tests.sqlite_damage import corrupt_index, healthy_freeze_db


@pytest.fixture
def f1(tmp_path, monkeypatch):
    (tmp_path / "live").mkdir()                       # ADAPTED: the database in its own directory
    db = corrupt_index(tmp_path / "live" / "freeze.db")
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(repair, "_held_open", lambda _: [])
    monkeypatch.setattr(repair, "_check_desk", lambda *_: None)
    return db, manifest


def plan_f1(db, manifest):
    return repair.plan(db, manifest.parent / "stage", manifest, expect_stale=4)


def append_freeze(conn):
    conn.execute(
        "INSERT INTO freezes VALUES ('dec_new', 'fp', 'NEW',"
        " '2026-11-23', 'WATCH', 't', '{}')"
    )
    conn.commit()


@pytest.mark.parametrize("operation", ["apply", "rollback"])
def test_swap_preserves_acknowledged_freeze(f1, monkeypatch, operation):
    db, manifest = f1
    plan_f1(db, manifest)
    if operation == "rollback":
        repair.apply(manifest, "test", target_is_copy=True)

    real_replace = repair.os.replace
    committed = []

    def replace(src, dst, *args, **kwargs):
        if Path(dst) == db:
            try:
                with closing(sqlite3.connect(db, timeout=0.05)) as conn:
                    append_freeze(conn)
                committed.append(True)
            except sqlite3.OperationalError:
                pass
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(repair.os, "replace", replace)
    try:
        getattr(repair, operation)(manifest, "test", target_is_copy=True)
    except SystemExit:
        pass
    if committed:
        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute(
                "SELECT 1 FROM freezes WHERE freeze_id='dec_new'"
            ).fetchone() == (1,)


def test_interrupted_rollback_refuses_newer_evidence(f1, monkeypatch):
    db, manifest = f1
    plan_f1(db, manifest)

    def crash(*_):
        raise KeyboardInterrupt

    real_live = repair._live_content
    calls = {"n": 0}

    def crash_after_commit(*a, **k):                   # ADAPTED: v2 verifies through _live_content
        calls["n"] += 1
        if calls["n"] == 2:
            crash()
        return real_live(*a, **k)

    with monkeypatch.context() as patch:
        patch.setattr(repair, "_live_content", crash_after_commit)
        with pytest.raises(KeyboardInterrupt):
            repair.apply(manifest, "test", target_is_copy=True)

    assert json.loads(manifest.read_text())["state"] == "applying"
    with closing(sqlite3.connect(db)) as conn:
        append_freeze(conn)
    newer = db.read_bytes()

    with pytest.raises(SystemExit):
        repair.rollback(manifest, target_is_copy=True)
    assert db.read_bytes() == newer


def test_manifest_cannot_substitute_valid_unrelated_database(f1):
    db, manifest = f1
    m = plan_f1(db, manifest)
    original = db.read_bytes()
    replacement = healthy_freeze_db(db.parent / "replacement.db", n=1)
    m.update(repaired=str(replacement), repaired_sha256=repair._sha256(replacement))
    manifest.write_text(json.dumps(m))

    # ADAPTED: v2 has no candidate to install; apply repairs THE TARGET in place,
    # so the 1-row replacement can never be installed over the 200 rows.
    repair.apply(manifest, "test", target_is_copy=True)
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT count(*) FROM freezes NOT INDEXED").fetchone()[0] == 200
    assert repair.diagnose(db)["integrity_ok"]


def test_plan_cannot_publish_manifest_over_source(f1):
    db, _ = f1
    original = db.read_bytes()
    try:
        repair.plan(db, db.parent / "stage", db, expect_stale=4)
    except SystemExit:
        pass
    assert db.read_bytes() == original


def test_apply_to_symlink_cannot_report_false_repair(f1):
    db, manifest = f1
    alias = db.with_name("alias.db")
    alias.symlink_to(db)
    try:
        plan_f1(alias, manifest)
        repair.apply(manifest, "test", target_is_copy=True)
    except SystemExit:
        return
    assert repair.diagnose(db)["integrity_ok"]


def test_plan_refuses_wal_that_appears_during_snapshot(f1, monkeypatch):
    db, manifest = f1
    real_sidecars = repair._sidecars
    writers = []

    def sidecars(path):
        observed = real_sidecars(path)
        if path == db and not writers:
            conn = sqlite3.connect(db)
            writers.append(conn)
            conn.execute("PRAGMA wal_autocheckpoint=0")
            append_freeze(conn)
            assert real_sidecars(db)["-wal"] > 0
        return observed

    monkeypatch.setattr(repair, "_sidecars", sidecars)
    try:
        try:
            m = plan_f1(db, manifest)
        except SystemExit:
            return
        assert not m["ready"]
    finally:
        for conn in writers:
            conn.close()


def test_check_cannot_pass_on_base_only_with_live_wal(tmp_path):
    db = healthy_freeze_db(tmp_path / "freeze.db")
    with closing(sqlite3.connect(db)) as writer:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        append_freeze(writer)
        assert repair.main([
            "check", "--target", str(db), "--staging", str(tmp_path / "stage")
        ]) != 0


def test_plan_detects_unsupported_sqlite_prefix_user_table(f1):
    db, manifest = f1
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE sqliteX(k TEXT PRIMARY KEY) WITHOUT ROWID")
        conn.execute("INSERT INTO sqliteX VALUES ('evidence')")
        conn.commit()
    try:
        m = plan_f1(db, manifest)
    except SystemExit:
        return
    assert not m["ready"]


def test_row_digest_includes_physical_rowid():
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(
            "CREATE TABLE t(rowid INTEGER, evidence BLOB);"
            "INSERT INTO t(_rowid_, rowid, evidence) VALUES (1, 7, X'AB');"
        )
        alias = repair._rowid_alias(conn, "t")          # ADAPTED: v2 takes the unshadowed alias
        before = repair._row_digest(conn, "t", alias)
        conn.execute("UPDATE t SET _rowid_=2")
        assert repair._row_digest(conn, "t", alias) != before


def test_audit_cli_fails_on_integrity_error_in_wal(tmp_path, monkeypatch):
    import scripts.audit_sqlite_runtime as cli
    from product.sqlite_audit import audit_database

    db = tmp_path / "audit.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t(n INTEGER CHECK(n > 0))")
        conn.commit()

    with closing(sqlite3.connect(db)) as writer:
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA ignore_check_constraints=ON")
        writer.execute("INSERT INTO t VALUES (-1)")
        writer.commit()

        row = audit_database(db)
        assert row.wal_present and row.immutable_base.ok
        assert not row.readonly.error
        # ADAPTED: on SQLite 3.45 a READ-ONLY integrity_check does not verify CHECK
        # constraints (measured; a read-write one does), so this live view reads "ok".
        # The classification defect is pinned with the probe results it would see:
        from product.sqlite_audit import Probe, classify, summary
        assert row.readonly.result == "ok"
        kind, _note = classify(readonly_ok=False, immutable_ok=True, wal_present=True,
                               readonly_diagnostic=True)
        assert kind == "LIVE_VIEW_INTEGRITY_FAILURE"
        failing = row.__class__(**{**row.__dict__, "readonly": Probe(False, "CHECK constraint failed in t", ""),
                                   "classification": kind})
        assert summary([failing])["needs_investigation"] == 1
        monkeypatch.setattr(cli, "audit_runtime", lambda **_: [failing])
        assert cli.main([]) == 2


def test_existing_wal_checkpoint_connection_uses_barrier(tmp_path, monkeypatch):
    from core import atomic_io
    from product import host_install

    db = healthy_freeze_db(tmp_path / "existing.db")
    monkeypatch.setattr(atomic_io.sys, "platform", "darwin")
    monkeypatch.setattr(host_install, "_sqlite_source_paths", lambda _: [db])
    real_connect = sqlite3.connect
    statements = []

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(lambda sql: statements.append(sql.lower()))
        return conn

    monkeypatch.setattr(host_install.sqlite3, "connect", traced_connect)
    host_install._checkpoint_sqlite_sources(tmp_path)
    checkpoint = next(i for i, s in enumerate(statements) if "wal_checkpoint" in s)
    assert any("checkpoint_fullfsync=on" in s for s in statements[:checkpoint])


def test_unique_probe_uses_index_collation(f1):
    db, manifest = f1
    with closing(sqlite3.connect(db)) as conn:
        conn.executescript(
            "CREATE TABLE labels(k TEXT COLLATE NOCASE);"
            "CREATE UNIQUE INDEX labels_binary ON labels(k COLLATE BINARY);"
            "INSERT INTO labels VALUES ('a'), ('A');"
        )
    m = plan_f1(db, manifest)
    assert m["after"]["indexes"]["labels_binary"]["lookup_mismatch_count"] == 0


def test_runbook_stage3_rejects_mismatched_f1_backup(tmp_path):
    import os
    import subprocess

    runbook = Path(
        "docs/runbooks/2026-11-22-paper-upgrade-and-store-repair.md"
    ).read_text()
    stage = runbook.split("## 3. Maintenance window", 1)[1].split("## 3b.", 1)[0]
    # ADAPTED: the F1 copy is now its own block with an enforced comparison
    commands = stage.split("**F1 only, in the stage 3 SSD tier", 1)[1].split("```bash\n", 1)[1].split("```", 1)[0]
    commands = commands.replace("/Users/apple/0to100/venv/bin/python", __import__("sys").executable)
    m, r = tmp_path / "maintenance", tmp_path / "source"
    (m / "backup").mkdir(parents=True)
    (r / "product").mkdir(parents=True)
    plan_f1.__globals__  # (unchanged helper)
    db = corrupt_index(r / "product" / "counterfactuals.jsonl.freeze.db")
    plan_f1(db, m / "f1_manifest.json")
    healthy_freeze_db(db.with_name("replacement.db"), n=1).replace(db)

    result = subprocess.run(
        ["bash", "-eu", "-c", commands],
        env={**os.environ, "M": str(m), "R": str(r)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0



# ADAPTED (F1-01): v2 publishes in place; a writer in another process at each
# publication step must be held off by SQLite's lock, never silently lost.
def test_v2_a_writer_in_another_process_is_never_lost(f1, monkeypatch):
    import subprocess, sys
    db, manifest = f1
    plan_f1(db, manifest)
    code = ("import sqlite3,sys\n"
            "c=sqlite3.connect(sys.argv[1],timeout=0.2)\n"
            "try:\n c.execute(\"INSERT INTO freezes VALUES ('dec_new','fp','NEW','2026-11-23','WATCH','t','{}')\")\n"
            " c.commit(); print('WROTE')\n"
            "except sqlite3.OperationalError: print('BLOCKED')\n")
    real = repair._reindex
    outcome = []

    def writer_then_reindex(con):
        outcome.append(subprocess.run([sys.executable, "-c", code, str(db)], capture_output=True, text=True).stdout.strip())
        real(con)

    monkeypatch.setattr(repair, "_reindex", writer_then_reindex)
    repair.apply(manifest, "test", target_is_copy=True)
    with closing(sqlite3.connect(db)) as conn:
        present = conn.execute("SELECT 1 FROM freezes WHERE freeze_id='dec_new'").fetchone() == (1,)
    assert outcome == ["BLOCKED"] and not present
