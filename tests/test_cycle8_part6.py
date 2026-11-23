"""Cycle 8, part 6: GPT's review of the part-5 fixes (2787a690..2cf44fc0), C8.6-01..09.

C8.6-03 (a header first seen on restart never authorizes rollback) is pinned in
tests/test_sqlite_index_repair_crash.py; C8.6-09 (the audit window) here, beside its
siblings in tests/test_cycle8_part4.py.
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from product import sqlite_index_repair as repair
from product import sqlite_runtime
from tests.sqlite_damage import corrupt_index
from tests.test_cycle8_review import f1, plan_f1  # noqa: F401  (the shared fixture)


# ── C8.6-01: a freeze is durable when freeze() returns ──

def test_a_freeze_commit_is_a_full_sync(tmp_path):
    from product import decision_freeze

    con = decision_freeze._connect(tmp_path / "decision_freeze.db")
    try:
        assert con.execute("PRAGMA synchronous").fetchone()[0] == 2           # FULL: every commit syncs
        if sys.platform == "darwin":
            assert con.execute("PRAGMA fullfsync").fetchone()[0] == 1         # ... with F_FULLFSYNC
            assert con.execute("PRAGMA checkpoint_fullfsync").fetchone()[0] == 1
    finally:
        con.close()
        sqlite_runtime.close_keepers()


# ── C8.6-02: an anchor behind an alias of an alias ──

def test_reconcile_refuses_an_output_behind_an_indirect_alias(tmp_path, monkeypatch):
    from core import live_volume as lv
    from research import counterfactual_reconcile as cr

    live = tmp_path / "live"
    (live / "runtime").mkdir(parents=True)
    evidence = live / "runtime" / "evidence.jsonl"
    evidence.write_text('{"decision_id":"keep"}\n')
    alias = tmp_path / "firmlink"
    alias.symlink_to(live, target_is_directory=True)
    portal = tmp_path / "portal"
    portal.symlink_to(alias / "runtime", target_is_directory=True)
    out = portal / evidence.name
    realpath = os.path.realpath

    def preserve_alias(path, *args, **kwargs):          # as macOS keeps the firmlink spelling
        if os.fspath(path) == str(out):
            return str(alias / "runtime" / evidence.name)
        return realpath(path, *args, **kwargs)

    monkeypatch.setattr(lv, "ANCHOR", str(live))
    monkeypatch.setattr(lv.os.path, "realpath", preserve_alias)
    monkeypatch.setattr(cr, "store_outcome_fn", lambda _: lambda *_: None)
    with pytest.raises(SystemExit, match="never the live runtime"):
        cr.main(["--ledger", str(tmp_path / "missing.jsonl"), "--store-root", str(tmp_path), "--out", str(out)])
    assert evidence.read_text() == '{"decision_id":"keep"}\n'


def test_a_symlink_loop_is_refused_not_followed_forever(tmp_path, monkeypatch):
    from core import live_volume as lv

    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    monkeypatch.setattr(lv, "ANCHOR", str(tmp_path / "anchor"))
    (tmp_path / "anchor").mkdir()
    assert lv.on_live_volume(tmp_path / "loop" / "x")


# ── C8.6-03: a later header write after an interrupted apply ──

def test_a_retry_never_authorizes_rollback_over_a_later_header_write(f1, monkeypatch):
    db, manifest = f1
    plan_f1(db, manifest)
    reindex = repair._reindex

    def stop_after_commit(con, **kwargs):
        reindex(con, **kwargs)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(repair, "_reindex", stop_after_commit)
        with pytest.raises(KeyboardInterrupt):
            repair.apply(manifest, "test", target_is_copy=True)
    with closing(sqlite3.connect(db)) as con:
        con.execute("PRAGMA default_cache_size=12345")
    with pytest.raises(SystemExit, match="first seen after a restart"):
        repair.apply(manifest, "retry", target_is_copy=True)
    with pytest.raises(SystemExit, match="writes were made after the repair"):
        repair.rollback(manifest, target_is_copy=True)
    with closing(sqlite3.connect(db)) as con:
        assert con.execute("PRAGMA default_cache_size").fetchone()[0] == 12345     # the later write stands


# ── C8.6-04: an older manifest version is refused, never half-understood ──

def test_a_v2_manifest_is_refused_by_every_command(f1):
    db, manifest = f1
    m = plan_f1(db, manifest)
    manifest.write_text(json.dumps({**m, "version": 2, "state": "applying", "verified_before_commit": True}))
    for call in (lambda: repair.apply(manifest, "test", target_is_copy=True),
                 lambda: repair.rollback(manifest, target_is_copy=True)):
        with pytest.raises(SystemExit, match="re-plan with this version"):
            call()
    assert repair._sha256(db) == m["target_sha256"]


# ── C8.6-05: fork with the keeper registry locked ──

@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork only")
def test_a_child_forked_while_the_registry_is_locked_never_hangs(tmp_path):
    from product import decision_freeze

    decision_freeze.freeze({"decision_id": "dec_fork", "symbol": "S", "as_of": "2026-11-24",
                            "decision": "WATCH", "reason_code": "T"}, path=tmp_path / "f.db")
    try:
        with sqlite_runtime._KEEPERS_LOCK:
            pid = os.fork()
            if pid == 0:
                signal.alarm(5)                          # a hang kills the child: a non-zero status
                sqlite_runtime.close_keepers()
                sqlite_runtime.keep_open(tmp_path / "f.db")
                os._exit(0 if len(sqlite_runtime._INHERITED) == 1 else 3)
        _pid, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        key = (os.getpid(), str((tmp_path / "f.db").resolve()))
        assert sqlite_runtime._KEEPERS[key].execute("SELECT count(*) FROM freezes").fetchone()[0] == 1
    finally:
        sqlite_runtime.close_keepers()


# ── C8.6-06: a hard full-sync error is never downgraded ──

def test_a_full_sync_eio_propagates(tmp_path, monkeypatch):
    import errno
    import fcntl

    def fail_full_sync(*_):
        raise OSError(errno.EIO, "full sync failed")

    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", fail_full_sync)
    monkeypatch.setattr(os, "fsync", lambda _: None)
    with pytest.raises(OSError) as exc:
        repair._fsync_dir(tmp_path)
    assert exc.value.errno == errno.EIO
    fd = os.open(tmp_path / "f", os.O_RDWR | os.O_CREAT)
    try:
        with pytest.raises(OSError) as exc:
            repair._full_fsync(fd)
        assert exc.value.errno == errno.EIO
    finally:
        os.close(fd)


def test_an_unsupported_full_sync_falls_back_to_fsync(tmp_path, monkeypatch):
    import errno
    import fcntl

    synced = []
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda *_: (_ for _ in ()).throw(OSError(errno.ENOTSUP, "no")))
    monkeypatch.setattr(os, "fsync", synced.append)
    fd = os.open(tmp_path / "f", os.O_RDWR | os.O_CREAT)
    try:
        repair._full_fsync(fd)
    finally:
        os.close(fd)
    assert synced == [fd]


# ── C8.6-07: the stage-3 copy is certified only as rollback would accept it ──

@pytest.mark.parametrize("sidecar", ["-journal", "-wal"])
def test_the_stage3_copy_refuses_a_copy_with_a_hot_sidecar(tmp_path, sidecar):
    from tests.test_sqlite_index_repair import _as_written, _block_after, _runbook_text

    root, mdir = tmp_path / "source", tmp_path / "maintenance"
    (root / "product").mkdir(parents=True)
    (mdir / "backup").mkdir(parents=True)
    db = corrupt_index(root / "product" / "counterfactuals.jsonl.freeze.db")
    repair.plan(db, mdir / "stage", mdir / "f1_manifest.json", expect_stale=4)
    dest = mdir / "backup" / "f1_freeze_pre_repair.db"
    dest.write_bytes(db.read_bytes())
    Path(str(dest) + sidecar).write_bytes(b"" if sidecar == "-journal" else b"x" * 32)
    block = _block_after(_runbook_text(), "**F1 only, in the stage 3 SSD tier")
    result = subprocess.run(["bash", "-eu", "-c", _as_written(block, root)],
                            env={**os.environ, "M": str(mdir), "R": str(root)},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "F1_BACKUP_OK" not in result.stdout
    assert dest.read_bytes() == db.read_bytes()


# ── C8.6-08: a failed thread start never suppresses the battery alert ──

def test_the_battery_alert_survives_a_flush_thread_start_failure(monkeypatch):
    from product import host_power_watch as hpw

    sent = []
    watch = hpw.PowerWatch(sent.append, flush=lambda: None)

    def fail_start(_self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(hpw.threading.Thread, "start", fail_start)
    watch.observe({"source": "BATTERY", "percent": 12})
    assert len(sent) == 1 and "BATTERY" in sent[0]
    assert watch.state == "BATTERY" and watch.flush_thread is None


# ── C8.6-09: a second SQLite open of the target inside the window is caught ──

def test_the_audit_catches_reopening_the_exact_target(f1, monkeypatch):
    from tests.test_cycle8_part4 import _locked_opens

    db, manifest = f1
    plan_f1(db, manifest)
    connect = repair._maintenance_connection

    def reopen(path):
        con = connect(path)
        with closing(sqlite3.connect(str(path))):
            pass
        return con

    monkeypatch.setattr(repair, "_maintenance_connection", reopen)
    _events, bad = _locked_opens(db, manifest, monkeypatch, "apply")
    assert [(event, str(args[0])) for event, args in bad] == [("sqlite3.connect", str(db))]
