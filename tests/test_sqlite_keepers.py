"""Cycle 8 benchmark finding: the F1 checkpoint barrier made every per-call decision
freeze pay a full, barriered checkpoint (the last close checkpoints the WAL): p50
280 ms, p99 964 ms, one 110 s stall, against p50 0.95 ms with one idle keeper
connection per process (USB stack, 24 Nov). These tests pin the keeper's behaviour,
never its timing.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from product import decision_freeze, sqlite_runtime


@pytest.fixture(autouse=True)
def _no_keepers_left():
    sqlite_runtime.close_keepers()
    yield
    sqlite_runtime.close_keepers()


def _freeze(db: Path, i: int) -> dict:
    return decision_freeze.freeze({"decision_id": f"dec_keep_{i}", "symbol": "SYM", "as_of": "2026-11-24",
                                   "decision": "WATCH", "reason_code": "T"}, path=db)


def _wal(db: Path) -> Path:
    return Path(str(db) + "-wal")


def test_a_freeze_close_is_not_the_last_close_while_the_keeper_holds_the_store(tmp_path):
    db = tmp_path / "decision_freeze.db"
    for i in range(5):
        _freeze(db, i)
    assert _wal(db).exists() and _wal(db).stat().st_size > 0     # no checkpoint-and-delete per freeze
    key = (os.getpid(), str(db.resolve()))
    assert key in sqlite_runtime._KEEPERS
    sqlite_runtime.close_keepers()                               # the final, barriered checkpoint
    assert not _wal(db).exists() or _wal(db).stat().st_size == 0
    with closing(sqlite3.connect(db)) as con:
        assert con.execute("SELECT count(*) FROM freezes").fetchone()[0] == 5
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.skipif(sys.platform != "darwin", reason="the barrier exists only on macOS")
def test_the_keeper_itself_checkpoints_behind_the_barrier(tmp_path):
    db = tmp_path / "decision_freeze.db"
    _freeze(db, 0)
    keeper = sqlite_runtime._KEEPERS[(os.getpid(), str(db.resolve()))]
    assert keeper.execute("PRAGMA checkpoint_fullfsync").fetchone()[0] == 1


def test_a_forked_child_never_touches_its_parents_keeper(tmp_path):
    db = tmp_path / "decision_freeze.db"
    _freeze(db, 0)
    pid = os.fork()
    if pid == 0:            # the child: the parent's keeper is parked, never used or closed (C8.6-05)
        try:
            sqlite_runtime.close_keepers()
            ok = len(sqlite_runtime._KEEPERS) == 0 and len(sqlite_runtime._INHERITED) == 1
            os._exit(0 if ok else 3)
        finally:
            os._exit(4)
    _pid, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0
    keeper = sqlite_runtime._KEEPERS[(os.getpid(), str(db.resolve()))]
    assert keeper.execute("SELECT count(*) FROM freezes").fetchone()[0] == 1


def test_at_most_max_keepers_are_held(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite_runtime, "MAX_KEEPERS", 3)
    dbs = [tmp_path / f"s{i}.db" for i in range(5)]
    for i, db in enumerate(dbs):
        _freeze(db, i)
    held = [k for k in sqlite_runtime._KEEPERS if k[0] == os.getpid()]
    assert [Path(k[1]).name for k in held] == ["s2.db", "s3.db", "s4.db"]     # the oldest were closed
    assert not _wal(dbs[0]).exists() or _wal(dbs[0]).stat().st_size == 0     # ... with their checkpoint


def test_a_rollback_journal_store_is_never_kept(tmp_path):
    db = tmp_path / "plain.db"
    with closing(sqlite3.connect(db)) as con:
        con.execute("CREATE TABLE t (x)")
    sqlite_runtime.keep_open(db)
    assert (os.getpid(), str(db.resolve())) not in sqlite_runtime._KEEPERS


def test_committed_freezes_survive_a_hard_process_death(tmp_path):
    """No atexit, no keeper close: committed WAL frames are recovered by the next opener."""
    db = tmp_path / "decision_freeze.db"
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "from product import decision_freeze\n"
        "for i in range(20):\n"
        "    decision_freeze.freeze({'decision_id': f'dec_hard_{i}', 'symbol': 'S', 'as_of': '2026-11-24',\n"
        "                            'decision': 'WATCH', 'reason_code': 'T'}, path=sys.argv[1])\n"
        "os._exit(9)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "QT_RUNTIME_ROOT"}
    r = subprocess.run([sys.executable, "-c", code, str(db)], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 9, r.stderr[-500:]
    assert _wal(db).exists()                                     # the keeper never checkpointed it
    with closing(sqlite3.connect(db)) as con:
        assert con.execute("SELECT count(*) FROM freezes").fetchone()[0] == 20
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
