"""GPT cycle-8 VERIFICATION pass of dfe062e3..6ebb95bf: the reviewer's sketches (V2-01..V2-08)."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from product import sqlite_index_repair as repair
from tests.sqlite_damage import corrupt_index, healthy_freeze_db
from tests.test_cycle8_review import f1, plan_f1  # noqa: F401  (the shared fixture)


@pytest.mark.parametrize("mode", ["replaced_after_hash", "edited_hash"])
def test_rollback_preserves_rows_if_backup_is_wrong(f1, monkeypatch, mode):
    db, manifest = f1
    plan_f1(db, manifest)
    m = repair.apply(manifest, "test", target_is_copy=True)
    backup = Path(m["backup"])
    replacement = healthy_freeze_db(manifest.parent / "other.db", n=1)

    if mode == "edited_hash":
        replacement.replace(backup)
        m["backup_sha256"] = repair._sha256(backup)
        manifest.write_text(json.dumps(m))
    else:
        write_manifest = repair._write_manifest

        def replace_after_hash(path, state):
            write_manifest(path, state)
            if state["state"] == "rolling_back":
                replacement.replace(backup)

        monkeypatch.setattr(repair, "_write_manifest", replace_after_hash)

    try:
        repair.rollback(manifest, target_is_copy=True)
    except SystemExit:
        pass
    with closing(sqlite3.connect(db)) as con:
        assert con.execute("SELECT count(*) FROM freezes NOT INDEXED").fetchone() == (200,)


def test_apply_does_not_accept_an_out_of_order_freeze_index(f1):
    db, manifest = f1
    plan_f1(db, manifest)
    repair.apply(manifest, "test", target_is_copy=True)
    with closing(sqlite3.connect(db)) as con:
        size = con.execute("PRAGMA page_size").fetchone()[0]
        root = con.execute(
            "SELECT rootpage FROM sqlite_master "
            "WHERE name='sqlite_autoindex_freezes_1'"
        ).fetchone()[0]

    raw = bytearray(db.read_bytes())
    page = (root - 1) * size
    while raw[page] == 0x02:  # interior index page, follow its leftmost child
        cell = page + int.from_bytes(raw[page + 12:page + 14], "big")
        page = (int.from_bytes(raw[cell:cell + 4], "big") - 1) * size
    assert raw[page] == 0x0A
    assert int.from_bytes(raw[page + 3:page + 5], "big") >= 2
    raw[page + 8:page + 12] = raw[page + 10:page + 12] + raw[page + 8:page + 10]
    db.write_bytes(raw)
    damaged = repair.diagnose(db)
    assert damaged["lookup_mismatch_total"] > 0
    with pytest.raises(SystemExit):
        repair.apply(manifest, "test", target_is_copy=True)


@pytest.mark.parametrize("link", ["hard", "symbolic"])
def test_apply_never_reuses_live_inode_as_backup(f1, monkeypatch, link):
    """TIGHTENED (cycle-8 review part 4): it used to pass on ANY refusal. A hard link makes
    the target multiply linked, refused for exactly that reason with nothing touched; a
    symlink leaves it singly linked, and must never be taken as the backup."""
    import os
    import subprocess
    import sys

    db, manifest = f1
    m = plan_f1(db, manifest)
    alias = db.parent.parent / "backup.db"
    (os.link if link == "hard" else os.symlink)(db, alias)
    manifest.write_text(json.dumps({**m, "backup": str(alias), "backup_pending": str(alias)}))
    reindex = repair._reindex
    observed = []

    def probe_then_reindex(con, **kw):                # ADAPTED: _reindex now takes verify=
        child = subprocess.run(
            [sys.executable, "-c",
             "import sqlite3, sys\n"
             "c = sqlite3.connect(sys.argv[1], timeout=.1)\n"
             "try:\n"
             "    c.execute('SELECT count(*) FROM freezes NOT INDEXED').fetchone()\n"
             "except sqlite3.OperationalError as exc:\n"
             "    assert 'locked' in str(exc), str(exc)\n"
             "    print('locked')\n"
             "else:\n"
             "    print('read')\n",
             str(db)],
            capture_output=True, text=True, check=True, timeout=5,
        )
        observed.append(child.stdout.strip())
        reindex(con, **kw)

    monkeypatch.setattr(repair, "_reindex", probe_then_reindex)
    if link == "hard":
        with pytest.raises(SystemExit, match="2 hard links"):
            repair.apply(manifest, "test", target_is_copy=True)
        assert repair._sha256(db) == m["target_sha256"] and observed == []
        return
    result = repair.apply(manifest, "test", target_is_copy=True)
    assert observed == ["locked"]
    assert not db.samefile(result["backup"]) and Path(result["backup"]) != alias
    assert alias.is_symlink() and os.readlink(alias) == str(db)        # never followed, never removed


def test_plan_manifest_temp_cannot_truncate_source(f1):
    import os

    db, manifest = f1
    original = db.read_bytes()
    manifest.with_name(f"{manifest.name}.tmp-{os.getpid()}").symlink_to(db)
    try:
        plan_f1(db, manifest)
    except (SystemExit, OSError):
        pass
    assert db.read_bytes() == original


def test_plan_refuses_staging_symlink_into_source_directory(f1):
    db, manifest = f1
    alias = manifest.parent / "stage-alias"
    alias.symlink_to(db.parent, target_is_directory=True)
    with pytest.raises(SystemExit):
        repair.plan(db, alias, manifest, expect_stale=4)


def test_volume_guard_includes_mount_root():
    assert repair._on_live_volume("/Volumes")


def test_audit_escalates_malformed_schema_in_wal(tmp_path, monkeypatch):
    from product import sqlite_audit
    import scripts.audit_sqlite_runtime as cli

    db = tmp_path / "audit.db"
    with closing(sqlite3.connect(db)) as con:
        con.executescript("PRAGMA journal_mode=WAL; CREATE TABLE t(n INTEGER);")

    with closing(sqlite3.connect(db)) as writer:
        writer.executescript("""
            PRAGMA wal_autocheckpoint=0;
            PRAGMA writable_schema=ON;
            BEGIN;
            UPDATE sqlite_master SET sql='CREATE TABLE t(' WHERE name='t';
            PRAGMA schema_version=99;
            COMMIT;
        """)
        row = sqlite_audit.audit_database(db)
        assert row.wal_present and row.immutable_base.ok and not row.readonly.ok
        monkeypatch.setattr(
            cli, "audit_runtime",
            lambda **kw: sqlite_audit.audit_runtime(tmp_path, **kw),
        )
        assert cli.main([]) == 2


@pytest.mark.parametrize("edit", ["expected_stale", "expected_content"])
def test_apply_checks_plan_before_committing(f1, edit):
    db, manifest = f1
    m = repair.plan(
        db, manifest.parent / "stage", manifest,
        expect_stale=3 if edit == "expected_stale" else 4,
    )
    if edit == "expected_stale":
        assert m["gates"]["expected_stale"] is False
        m["ready"] = True
    else:
        m["expected_content"]["index_digests"] = {}
    manifest.write_text(json.dumps(m))
    with pytest.raises(SystemExit):
        repair.apply(manifest, "test", target_is_copy=True)
    assert repair.diagnose(db)["stale_total"] == 4


def test_feature_reader_is_read_only_or_has_a_checkpoint_barrier(tmp_path, monkeypatch):
    from core import atomic_io
    from product import reco_methods
    from research.feature002 import constants

    db = tmp_path / "feature.db"
    with closing(sqlite3.connect(db)) as con:
        con.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE observations(
                symbol, rs_percentile, rs_score, n_structure_passed,
                combined_shadow_rank, r3_score, session_date
            );
            INSERT INTO observations VALUES ('X', 1, 1, 1, 1, 1, '2026-11-23');
        """)
    monkeypatch.setattr(constants, "DB_PATH", db)
    monkeypatch.setattr(atomic_io.sys, "platform", "darwin")
    real_connect = sqlite3.connect
    covered = []

    class TracedConnection(sqlite3.Connection):
        def close(self):
            flag = self.execute("PRAGMA checkpoint_fullfsync").fetchone()[0]
            covered.append(self.readonly_uri or bool(flag))
            super().close()

    def traced(path, *args, **kwargs):
        con = real_connect(path, *args, factory=TracedConnection, **kwargs)
        con.readonly_uri = kwargs.get("uri", False) and "mode=ro" in str(path)
        return con

    monkeypatch.setattr(sqlite3, "connect", traced)
    assert "X" in reco_methods._shadow_by_symbol(["X"])
    assert covered == [True]


def test_stage3_copy_fails_when_source_is_missing_and_old_backup_matches(f1):
    import os
    import shutil
    import subprocess
    from tests.test_sqlite_index_repair import _as_written, _block_after, _runbook_text

    db, manifest = f1
    plan_f1(db, manifest)
    m = manifest.parent
    (m / "backup").mkdir()
    shutil.copyfile(db, m / "backup" / "f1_freeze_pre_repair.db")
    shutil.copyfile(manifest, m / "f1_manifest.json")
    source = m / "source"
    (source / "product").mkdir(parents=True)
    block = _block_after(_runbook_text(), "**F1 only, in the stage 3 SSD tier")
    kept = (m / "backup" / "f1_freeze_pre_repair.db").read_bytes()
    result = subprocess.run(
        ["bash", "-c", _as_written(block, source)],
        env={**os.environ, "M": str(m), "R": str(source)},
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    # TIGHTENED (cycle-8 review part 4): the failed copy never deletes the verified one
    assert (m / "backup" / "f1_freeze_pre_repair.db").read_bytes() == kept
    assert sorted(p.name for p in (m / "backup").iterdir()) == ["f1_freeze_pre_repair.db"]
