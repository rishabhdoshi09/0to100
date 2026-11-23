"""Cycle 8, part 5: GPT's review of the part-4 fixes (62750347..2787a690), C8.5-01..10.

Each sketch is adapted to the real API and asserts the specific outcome; C8.5-10
(the audit window) lives in tests/test_cycle8_part4.py with the audit test itself.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import threading
from contextlib import closing
from pathlib import Path

import pytest

from product import sqlite_index_repair as repair
from tests.sqlite_damage import corrupt_index
from tests.test_cycle8_review import f1, plan_f1  # noqa: F401  (the shared fixture)


def freeze_count(db: Path) -> int:
    with closing(sqlite3.connect(db)) as con:
        return con.execute("SELECT count(*) FROM freezes NOT INDEXED").fetchone()[0]


# ── C8.5-01 / C8.5-06: backups are never deleted and never reused ──

@pytest.mark.parametrize("complete", [False, True])
def test_a_retry_never_deletes_a_file_the_manifest_names(f1, monkeypatch, complete):
    db, manifest = f1
    m = plan_f1(db, manifest)
    pending = (repair._fresh_backup_path(db, m["target_sha256"]) if complete
               else manifest.parent / "other-evidence.db")
    pending.write_bytes(db.read_bytes() if complete else b"unrelated evidence")
    kept = pending.read_bytes()
    if complete:
        raw = bytearray(db.read_bytes())
        raw[43] ^= 1                                   # the schema cookie: content-equal, other bytes
        db.write_bytes(raw)
    manifest.write_text(json.dumps({**m, "backup_pending": str(pending)}))

    def fail(*_a):
        raise OSError("replacement backup failed")

    monkeypatch.setattr(repair, "_durable_backup", fail)
    with pytest.raises(OSError):
        repair.apply(manifest, "test", target_is_copy=True)
    assert pending.read_bytes() == kept
    monkeypatch.undo()
    done = repair.apply(manifest, "test", target_is_copy=True)
    assert done["state"] == "applied" and pending.read_bytes() == kept
    assert str(pending) in done["earlier_backups"] and done["backup"] != str(pending)


def test_every_attempt_writes_and_syncs_its_own_backup(f1, monkeypatch):
    db, manifest = f1
    m = plan_f1(db, manifest)
    pending = repair._fresh_backup_path(db, m["target_sha256"])
    pending.write_bytes(db.read_bytes())               # all bytes visible, never synced
    manifest.write_text(json.dumps({**m, "backup_pending": str(pending)}))
    synced = set()
    real_sync, real_reindex = repair._full_fsync, repair._reindex

    def sync(fd):
        st = os.fstat(fd)
        real_sync(fd)
        synced.add((st.st_dev, st.st_ino))

    def reindex(con, **kw):
        st = Path(json.loads(manifest.read_text())["backup"]).stat()
        assert (st.st_dev, st.st_ino) in synced       # the recorded backup was synced before the repair
        return real_reindex(con, **kw)

    monkeypatch.setattr(repair, "_full_fsync", sync)
    monkeypatch.setattr(repair, "_reindex", reindex)
    done = repair.apply(manifest, "test", target_is_copy=True)
    assert done["backup"] != str(pending) and pending.exists()


# ── C8.5-02: rows the proof never reads ──

def test_rollback_refuses_rows_outside_the_row_proof(f1):
    db, manifest = f1
    plan_f1(db, manifest)
    m = repair.apply(manifest, "test", target_is_copy=True)
    backup = Path(m["backup"])
    for path, value in ((db, "keep"), (backup, "replace")):
        with closing(sqlite3.connect(path)) as con:
            con.execute("CREATE TABLE evidence(k TEXT PRIMARY KEY, v) WITHOUT ROWID")
            con.execute("INSERT INTO evidence VALUES ('x', ?)", (value,))
            con.commit()
    m["expected_content"] = repair._content(repair.diagnose(db))
    m["original_content"] = repair._content(repair.diagnose(backup))
    m["backup_sha256"] = repair._sha256(backup)
    manifest.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="outside the row proof"):
        repair.rollback(manifest, target_is_copy=True)
    with closing(sqlite3.connect(db)) as con:
        assert con.execute("SELECT v FROM evidence").fetchone() == ("keep",)


# ── C8.5-03: the directory sync can never open the database ──

def test_a_directory_rebound_to_the_database_is_never_opened(f1, monkeypatch):
    db, original = f1
    directory = original.parent / "manifests"
    directory.mkdir()
    manifest = directory / "m.json"
    plan_f1(db, manifest)
    parked = directory.with_name("parked")
    target = db.stat()
    real_sync, real_dir_sync = repair._full_fsync, repair._fsync_dir
    opened = []

    def inspect(fd):
        st = os.fstat(fd)
        if (st.st_dev, st.st_ino) == (target.st_dev, target.st_ino):
            opened.append(fd)
        real_sync(fd)

    def rebind(path):
        if Path(path) != directory:
            return real_dir_sync(path)
        directory.rename(parked)
        directory.symlink_to(db)                        # the name now leads to the database
        try:
            real_dir_sync(path)
        finally:
            directory.unlink()
            parked.rename(directory)

    monkeypatch.setattr(repair, "_full_fsync", inspect)
    monkeypatch.setattr(repair, "_fsync_dir", rebind)
    with pytest.raises(OSError):
        repair.apply(manifest, "test", target_is_copy=True)
    assert opened == []
    assert repair._sha256(db) == json.loads(manifest.read_text())["target_sha256"] or \
        repair.diagnose(db)["integrity_ok"]


# ── C8.5-04: kernel order, not text order ──

def test_reconcile_refuses_a_symlink_dotdot_output_into_the_live_tree(tmp_path, monkeypatch):
    from core import live_volume
    from research import counterfactual_reconcile as cr

    live = tmp_path / "live"
    (live / "sub").mkdir(parents=True)
    (tmp_path / "link").symlink_to(live / "sub", target_is_directory=True)
    evidence = live / "evidence.jsonl"
    evidence.write_text('{"decision_id":"keep"}\n')
    monkeypatch.setattr(live_volume, "ANCHOR", str(live))
    monkeypatch.setattr(cr, "store_outcome_fn", lambda _: lambda *args: None)
    with pytest.raises(SystemExit, match="never the live runtime"):
        cr.main(["--ledger", str(tmp_path / "absent.jsonl"), "--store-root", str(tmp_path),
                 "--out", str(tmp_path / "link" / ".." / evidence.name)])
    assert evidence.read_text() == '{"decision_id":"keep"}\n'


# ── C8.5-05: a later header write is never rolled over ──

@pytest.mark.parametrize("pragma", ["PRAGMA default_cache_size=12345", "PRAGMA journal_mode=DELETE"])
def test_rollback_refuses_header_writes_made_after_the_repair(f1, pragma):
    db, manifest = f1
    plan_f1(db, manifest)
    assert repair.apply(manifest, "test", target_is_copy=True)["applied_ok"]
    with closing(sqlite3.connect(db)) as con:
        con.execute(pragma).fetchall()
        con.commit()
    before = repair.diagnose(db)
    with pytest.raises(SystemExit, match="writes were made after the repair|not in WAL mode"):
        repair.rollback(manifest, target_is_copy=True)
    assert repair.diagnose(db)["header"] == before["header"] and repair.diagnose(db)["integrity_ok"]
    assert sorted(p.name for p in db.parent.iterdir() if not p.name.startswith(db.name + ".pre-reindex-")) \
        == [db.name]                                    # nothing left behind, no -journal either
    assert json.loads(manifest.read_text())["state"] == "applied"


def test_the_trial_flag_of_a_finished_apply_admits_nothing_later(f1):
    """The trial proof belongs to ONE attempt: after a completed apply, a later header
    change is refused by apply too, never recorded as applied_unverified."""
    db, manifest = f1
    plan_f1(db, manifest)
    repair.apply(manifest, "test", target_is_copy=True)
    with closing(sqlite3.connect(db)) as con:
        con.execute("PRAGMA default_cache_size=12345")
    with pytest.raises(SystemExit, match="neither the planned original nor the repaired"):
        repair.apply(manifest, "test", target_is_copy=True)
    m = json.loads(manifest.read_text())
    assert m["state"] == "applied" and "applied_header" not in m


def test_sqlite_never_opens_a_target_that_is_no_longer_wal(f1):
    """Checked through the pin BEFORE SQLite opens the file (GPT C8.5-05): a rollback-journal
    file is never opened under the exclusive lock at all."""
    import sys

    db, manifest = f1
    plan_f1(db, manifest)
    repair.apply(manifest, "test", target_is_copy=True)
    with closing(sqlite3.connect(db)) as con:
        con.execute("PRAGMA journal_mode=DELETE").fetchall()
    connects = []
    armed = {"on": True}

    def hook(event, args):
        if armed["on"] and event == "sqlite3.connect" and str(args[0]) == str(db):
            connects.append(args[0])

    sys.addaudithook(hook)
    try:
        for call in (lambda: repair.rollback(manifest, target_is_copy=True),
                     lambda: repair.apply(manifest, "test", target_is_copy=True)):
            with pytest.raises(SystemExit, match="not in WAL mode"):
                call()
    finally:
        armed["on"] = False
    assert connects == []


def test_the_identity_walk_follows_the_kernel_not_the_text(tmp_path, monkeypatch):
    """Where realpath keeps an alias spelling (macOS keeps the firmlink one), only the
    walk over the spelling AS GIVEN, in kernel order, can see that link/.. is the anchor."""
    from core import live_volume

    live = tmp_path / "live"
    (live / "sub").mkdir(parents=True)
    (tmp_path / "link").symlink_to(live / "sub", target_is_directory=True)
    monkeypatch.setattr(live_volume, "ANCHOR", str(live))
    monkeypatch.setattr(live_volume.os.path, "realpath", lambda p: p)
    assert live_volume.on_live_volume(tmp_path / "link" / ".." / "evidence.jsonl")
    assert not live_volume.on_live_volume(tmp_path / "link2" / ".." / "evidence.jsonl")


# ── C8.5-07: the manifest is resolved once ──

def test_apply_never_redirects_its_manifest_write(f1, monkeypatch):
    db, manifest = f1
    plan_f1(db, manifest)
    other = manifest.with_name("other.json")
    other.write_text('{"keep":true}')
    real_guards = repair._guards

    def guards(*args):
        result = real_guards(*args)
        manifest.unlink()
        manifest.symlink_to(other)                      # the manifest's name now leads elsewhere
        return result

    monkeypatch.setattr(repair, "_guards", guards)
    repair.apply(manifest, "test", target_is_copy=True)
    assert other.read_text() == '{"keep":true}'
    assert not manifest.is_symlink() and json.loads(manifest.read_text())["state"] == "applied"


def test_a_manifest_spelled_with_dotdot_is_refused_by_every_command(f1, tmp_path):
    db, manifest = f1
    plan_f1(db, manifest)
    (tmp_path / "x").mkdir()
    spelled = tmp_path / "x" / ".." / manifest.name
    for call in (lambda: repair.apply(spelled, "test", target_is_copy=True),
                 lambda: repair.rollback(spelled, target_is_copy=True)):
        with pytest.raises(SystemExit, match=r"'\.\.'"):
            call()


# ── C8.5-08: the stage-3 block ──

@pytest.mark.parametrize("case", ["symlink", "inherited_temp", "hardlink"])
def test_the_stage3_copy_certifies_only_a_separate_regular_copy(tmp_path, case):
    from tests.test_sqlite_index_repair import _as_written, _block_after, _runbook_text

    root, mdir = tmp_path / "source", tmp_path / "maintenance"
    (root / "product").mkdir(parents=True)
    (mdir / "backup").mkdir(parents=True)
    db = corrupt_index(root / "product" / "counterfactuals.jsonl.freeze.db")
    manifest = mdir / "f1_manifest.json"
    repair.plan(db, mdir / "stage", manifest, expect_stale=4)
    dest = mdir / "backup" / "f1_freeze_pre_repair.db"
    env = {**os.environ, "M": str(mdir), "R": str(root)}
    env.pop("T", None)
    if case == "symlink":
        dest.symlink_to(db)
    elif case == "hardlink":
        os.link(db, dest)
    else:
        dest.write_bytes(db.read_bytes())
        manifest.unlink()                               # the block fails before mktemp
        env["T"] = str(dest)
    block = _block_after(_runbook_text(), "**F1 only, in the stage 3 SSD tier")
    result = subprocess.run(["bash", "-eu", "-c", _as_written(block, root)], env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0 and "F1_BACKUP_OK" not in result.stdout, result.stdout + result.stderr
    assert dest.exists() or dest.is_symlink()           # never removed
    if case == "inherited_temp":
        assert dest.read_bytes() == db.read_bytes()
    assert not list((mdir / "backup").glob(".f1_copy.*"))


# ── C8.5-09: a slow alert never delays the flush ──

def test_a_stalled_sender_does_not_delay_the_battery_flush():
    from product.host_power_watch import PowerWatch

    sending, release, flushed = threading.Event(), threading.Event(), threading.Event()

    def send(_):
        sending.set()
        release.wait(5)

    watch = PowerWatch(send, flush=flushed.set)
    caller = threading.Thread(target=watch.observe, args=({"source": "BATTERY", "percent": 12},))
    caller.start()
    try:
        assert sending.wait(2)
        assert flushed.wait(0.5)
    finally:
        release.set()
        caller.join(2)
