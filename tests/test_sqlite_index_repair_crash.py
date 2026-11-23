"""F1 repair: HARD process death at every step (os._exit: no finally, no ROLLBACK,
no file close), as a power cut or a killed desk would leave it.

Exceptions raised in-process (tests/test_sqlite_index_repair.py) still run
Python's cleanup; these tests do not. Each crash runs ``apply`` or ``rollback``
in a separate interpreter that dies at one step. Afterwards the ledger must
hold exactly the original or exactly the repaired content (as SQLite itself
recovers it), re-running the same command must recover, and nothing may be
left beside the target but the backup.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from product import sqlite_index_repair as sir
from tests.sqlite_damage import corrupt_index

REPO = Path(__file__).resolve().parents[1]

_RUNNER = r'''
import os, sys
sys.path.insert(0, sys.argv[1])
from product import sqlite_index_repair as sir
point, op, manifest = sys.argv[2], sys.argv[3], sys.argv[4]
die = lambda: os._exit(137)

if point == "before_applying_record":
    real = sir._write_manifest
    sir._write_manifest = lambda p, m: die() if m.get("state") == "applying" else real(p, m)
elif point == "after_applying_record":
    real = sir._reindex
    sir._reindex = lambda con, **kw: die()
elif point == "inside_transaction":
    real_verifier = sir._precommit_verifier
    def verifier(m, **kw):
        check = real_verifier(m, **kw)
        def run(con):
            check(con)
            die()                      # REINDEX done and proven, COMMIT not yet issued
        return run
    sir._precommit_verifier = verifier
elif point == "after_commit":
    real = sir._checkpoint
    calls = {"n": 0}
    def checkpoint(con):
        calls["n"] += 1
        if calls["n"] == 2:            # the checkpoint right after COMMIT
            die()
        return real(con)
    sir._checkpoint = checkpoint
elif point == "before_final_record":
    real = sir._live_content
    calls = {"n": 0}
    def live(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            die()
        return real(*a, **k)
    sir._live_content = live
elif point == "after_rolling_back_record":
    real = sir._write_manifest
    def write(p, m):
        real(p, m)
        if m.get("state") == "rolling_back":
            die()
    sir._write_manifest = write
elif point == "after_restore":
    real = sir._checkpoint
    calls = {"n": 0}
    def checkpoint(con):
        calls["n"] += 1
        if calls["n"] == 2:            # rollback: the checkpoint right after the backup-API restore
            die()
        return real(con)
    sir._checkpoint = checkpoint
elif point == "during_backup_write":
    def half(path, raw):                   # the backup file half written when the power goes
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.write(fd, raw[: len(raw) // 2])
        die()
    sir._write_new_file = half
else:
    raise SystemExit(f"unknown point {point}")

getattr(sir, op)(manifest, "crash test", target_is_copy=True)
print("COMPLETED")
'''


def _crash(op: str, point: str, manifest: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "QT_RUNTIME_ROOT"}
    return subprocess.run([sys.executable, "-c", _RUNNER, str(REPO), point, op, str(manifest)],
                          capture_output=True, text=True, env=env, timeout=120)


def _content_as_sqlite_recovers_it(db: Path, scratch: Path) -> dict:
    """Copy the database WITH its journal files, let SQLite recover the copy, then diagnose it."""
    d = scratch / f"recovered-{os.urandom(4).hex()}"
    d.mkdir()
    copy = d / db.name
    for suffix in ("", "-wal", "-journal"):
        src = Path(str(db) + suffix)
        if src.exists():
            shutil.copy2(src, Path(str(copy) + suffix))
    con = sqlite3.connect(copy)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    return sir._content(sir.diagnose(copy))


@pytest.fixture
def planned(tmp_path):
    (tmp_path / "live").mkdir()
    db = corrupt_index(tmp_path / "live" / "counterfactuals.jsonl.freeze.db")
    m = sir.plan(db, tmp_path / "staging", tmp_path / "manifest.json", expect_stale=4)
    assert m["ready"]
    return db, tmp_path / "manifest.json", m


@pytest.mark.parametrize("point, expect", [
    ("during_backup_write", "original"),         # GPT C8-06: a partial backup, never left untracked
    ("before_applying_record", "original"),
    ("after_applying_record", "original"),
    ("inside_transaction", "original"),          # an uncommitted transaction never survives
    ("after_commit", "repaired"),                # committed frames in the WAL survive
    ("before_final_record", "repaired"),
])
def test_a_hard_crash_during_apply_leaves_original_or_repaired_and_recovers(tmp_path, planned, point, expect):
    db, manifest, m = planned
    r = _crash("apply", point, manifest)
    assert r.returncode == 137 and "COMPLETED" not in r.stdout, r.stderr[-400:]
    crashed = json.loads(manifest.read_text())
    want = m["original_content"] if expect == "original" else m["expected_content"]
    assert sir._same_content(_content_as_sqlite_recovers_it(db, tmp_path), want)

    done = sir.apply(manifest, "recover", target_is_copy=True)    # the same command recovers
    assert done["state"] == "applied"
    assert sir._same_content(_content_as_sqlite_recovers_it(db, tmp_path), m["expected_content"])
    leftovers = sorted(p.name for p in db.parent.iterdir() if p.name != db.name)
    assert all(".pre-reindex-" in name for name in leftovers), leftovers
    assert len(leftovers) <= 1, leftovers
    final = json.loads(manifest.read_text())
    assert Path(final["backup"]).name in leftovers and sir._sha256(Path(final["backup"])) == final["backup_sha256"]
    if point == "during_backup_write":
        partial = crashed["backup_pending"]                      # recorded BEFORE the write began
        assert final["discarded_partial_backups"] == [partial] and not Path(partial).exists()
        assert final["backup"] != partial


@pytest.mark.parametrize("point", ["after_rolling_back_record", "after_restore"])
def test_a_hard_crash_during_rollback_leaves_repaired_or_original_and_recovers(tmp_path, planned, point):
    db, manifest, m = planned
    sir.apply(manifest, "apply", target_is_copy=True)
    r = _crash("rollback", point, manifest)
    assert r.returncode == 137 and "COMPLETED" not in r.stdout, r.stderr[-400:]
    state = _content_as_sqlite_recovers_it(db, tmp_path)
    assert (sir._same_content(state, m["original_content"])
            or sir._same_content(state, m["expected_content"])), "neither original nor repaired"

    done = sir.rollback(manifest, target_is_copy=True)            # the same command recovers
    assert done["state"] == "rolled_back"
    assert sir._same_content(_content_as_sqlite_recovers_it(db, tmp_path), m["original_content"])
    assert json.loads(manifest.read_text())["state"] == "rolled_back"


def test_a_header_mismatch_then_a_crash_after_commit_still_recovers(tmp_path, planned):
    """GPT C8-05: the trial is proven (and that recorded durably) before COMMIT, so after a
    crash past COMMIT both commands tell a proven repair whose header differs from an
    unknown change: apply reaches applied_unverified, rollback restores."""
    db, manifest, m = planned
    m["expected_content"]["header"]["44-91"] = "00" * 48          # the committed header will differ
    manifest.write_text(json.dumps(m))
    r = _crash("apply", "after_commit", manifest)
    assert r.returncode == 137, r.stderr[-400:]
    assert json.loads(manifest.read_text())["verified_before_commit"] is True
    with pytest.raises(SystemExit, match="applied_unverified"):
        sir.apply(manifest, "recover", target_is_copy=True)
    assert json.loads(manifest.read_text())["state"] == "applied_unverified"
    done = sir.rollback(manifest, target_is_copy=True)
    assert done["state"] == "rolled_back" and done["rollback_ok"]
    assert sir._same_content(_content_as_sqlite_recovers_it(db, tmp_path), m["original_content"])


def test_the_lock_dies_with_the_process(tmp_path, planned):
    """After a hard crash, nothing is left holding the database: another
    connection can write immediately, and the next apply does not hang."""
    db, manifest, _m = planned
    assert _crash("apply", "inside_transaction", manifest).returncode == 137
    con = sqlite3.connect(db, timeout=1)
    con.execute("INSERT INTO freezes VALUES ('dec_after_crash', 'fp', 'X', 'd', 'W', 't', '{}')")
    con.commit()
    con.close()
    with pytest.raises(SystemExit, match="changed since the plan"):
        sir.apply(manifest, "recover", target_is_copy=True)
