"""Cycle 8, part 4: two independent reviews of 6ebb95bf..62750347 (GPT C8-01..07 and a
Claude subagent), each confirmed finding as a failing test first, then fixed.

| finding | test |
|---|---|
| C8-01 an edited manifest makes rollback install another database | test_rollback_never_installs_other_rows_through_an_edited_manifest |
| C8-02 a verified file re-opened by name (restore source; lock release) | test_rollback_restores_the_bytes_it_proved..., test_nothing_existing_is_opened_by_name... |
| C8-03 a manifest directory or the target swapped after the checks | test_a_manifest_directory_swapped..., test_a_manifest_named..., test_rollback_refuses_a_target_swapped... |
| C8-04 a hard-linked target hides committed WAL frames | test_a_hard_linked_target_is_refused_at_plan_and_at_apply |
| C8-05 header mismatch + crash after COMMIT strands recovery | tests/test_sqlite_index_repair_crash.py |
| C8-06 a crash during the backup write leaves an untracked file | tests/test_sqlite_index_repair_crash.py (during_backup_write) |
| C8-07 a stalled flush blocks the battery warning, polling and stop | test_a_stalled_flush_never_blocks_... |
| S-2 a rollback-journal target cannot recover after a hard crash | test_a_rollback_journal_database_is_refused_at_plan |
| S-4 the stage-3 F1 block deletes the verified SSD copy on a re-run | test_the_stage3_copy_never_deletes_the_verified_copy_on_a_rerun |
| S-5 '..' in an output path is collapsed before symlinks resolve | test_an_output_spelled_with_dotdot_is_refused |
| S-6 /System/Volumes/Data/Volumes/... defeats every /Volumes guard | test_every_live_guard_refuses_every_spelling, test_no_text_only_volumes_guard_is_left |
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest

from product import sqlite_index_repair as repair
from tests.sqlite_damage import corrupt_index, healthy_freeze_db
from tests.test_cycle8_review import f1, plan_f1  # noqa: F401  (the shared fixture)

REPO = Path(__file__).resolve().parents[1]

# Process-wide audit hook (hooks cannot be removed); inert unless a test arms it. Armed, it
# switches ON at the maintenance connection to the target itself (before SQLite takes the lock)
# and OFF when the pin closes, after SQLite released it (GPT C8.5-10).
_OPENED: dict = {"armed": None, "on": False, "events": []}


def _audit(event, args):
    if _OPENED["armed"] and event == "sqlite3.connect" and str(args[0]) == _OPENED["armed"]:
        _OPENED.update(on=True, armed=None)     # consumed once: a SECOND connect is recorded (C8.6-09)
        return
    if _OPENED["on"] and event in ("open", "sqlite3.connect"):
        _OPENED["events"].append((event, args))


sys.addaudithook(_audit)


def freeze_count(db: Path) -> int:
    with closing(sqlite3.connect(db)) as con:
        return con.execute("SELECT count(*) FROM freezes NOT INDEXED").fetchone()[0]


# ── C8-01 / C8-02: rollback restores only proven bytes, and never other rows ──

@pytest.mark.parametrize("edit", ["manifest_only", "manifest_and_backup"])
def test_rollback_never_installs_other_rows_through_an_edited_manifest(f1, edit):
    db, manifest = f1
    plan_f1(db, manifest)
    m = repair.apply(manifest, "test", target_is_copy=True)
    other = healthy_freeze_db(manifest.parent / "one-row.db", n=1)
    m["backup_sha256"] = repair._sha256(other)
    m["original_content"] = repair._content(repair.diagnose(other))     # a consistent edit
    if edit == "manifest_and_backup":
        shutil.copyfile(other, m["backup"])
    manifest.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="rows, schema or settings differ"):
        repair.rollback(manifest, target_is_copy=True, backup_path=other if edit == "manifest_only" else None)
    assert freeze_count(db) == 200 and repair.diagnose(db)["integrity_ok"]


def test_rollback_restores_the_bytes_it_proved_not_what_the_name_holds_later(f1, monkeypatch):
    db, manifest = f1
    plan_f1(db, manifest)
    m = repair.apply(manifest, "test", target_is_copy=True)
    other = healthy_freeze_db(manifest.parent / "one-row.db", n=1)
    real_write = repair._write_manifest

    def write(path, state):
        real_write(path, state)
        if state["state"] == "rolling_back":
            shutil.copyfile(other, m["backup"])     # the backup's NAME now holds another database

    monkeypatch.setattr(repair, "_write_manifest", write)
    r = repair.rollback(manifest, target_is_copy=True)
    assert r["state"] == "rolled_back" and freeze_count(db) == 200
    assert repair._same_content(repair._content(repair.diagnose(db)), m["original_content"])


def _locked_opens(db, manifest, monkeypatch, op):
    """Every open while the lock is held, classified by its FLAGS when it was made."""
    real_close = repair._Pinned.close

    def close(self):
        _OPENED["on"] = False                      # con.close() has already run: the lock is gone
        real_close(self)

    monkeypatch.setattr(repair._Pinned, "close", close)
    _OPENED.update(armed=str(db), on=False, events=[])
    try:
        getattr(repair, op)(manifest, "test", target_is_copy=True)
    finally:
        _OPENED.update(armed=None, on=False)
    bad = []
    for event, args in _OPENED["events"]:
        if event == "sqlite3.connect":
            if args[0] != ":memory:":
                bad.append((event, args))
            continue
        path, _mode, flags = args
        if isinstance(path, int):
            continue                                   # a descriptor this process already holds
        if flags & os.O_CREAT and flags & os.O_EXCL:
            continue                                   # a NEW file: never an existing inode
        if flags & os.O_DIRECTORY and flags & getattr(os, "O_NOFOLLOW", 0):
            continue                                   # can only ever be a directory
        bad.append((event, args))
    return _OPENED["events"], bad


@pytest.mark.parametrize("op", ["apply", "rollback"])
def test_nothing_existing_is_opened_by_name_while_the_lock_is_held(f1, monkeypatch, op):
    """The class behind C8-02: closing ANY descriptor on the live inode releases SQLite's
    POSIX locks, and a name re-opened after a check can be a link to it. While the lock is
    held only NEW files, no-follow directories and in-memory databases are opened."""
    db, manifest = f1
    plan_f1(db, manifest)
    if op == "rollback":
        repair.apply(manifest, "test", target_is_copy=True)
    events, bad = _locked_opens(db, manifest, monkeypatch, op)
    assert events, "the hook saw nothing: the locked window was not observed"
    assert bad == []


def test_the_audit_window_starts_where_the_lock_does(f1, monkeypatch):
    """GPT C8.5-10: an existing-file open right after the lock is taken must be caught."""
    db, manifest = f1
    plan_f1(db, manifest)
    sentinel = manifest.parent / "existing.txt"
    sentinel.write_text("existing")
    real_connect = repair._maintenance_connection

    def connect(path):
        con = real_connect(path)
        with sentinel.open():
            pass
        return con

    monkeypatch.setattr(repair, "_maintenance_connection", connect)
    _events, bad = _locked_opens(db, manifest, monkeypatch, "apply")
    assert [str(a[0]) for _e, a in bad] == [str(sentinel)]


# ── C8-03: swaps after the checks ──

def test_a_manifest_directory_swapped_after_the_checks_cannot_replace_the_database(f1, monkeypatch):
    db, _ = f1
    home = db.parent.parent / "manifests"
    home.mkdir()
    alias = db.parent.parent / "manifest-alias"
    alias.symlink_to(home, target_is_directory=True)
    plan_f1(db, alias / "m.json")
    real_write = repair._write_manifest

    def write(path, state):
        if "backup_pending" in state and alias.resolve() == home:
            alias.unlink()
            alias.symlink_to(db.parent, target_is_directory=True)     # now it names the DB's directory
        real_write(path, state)

    monkeypatch.setattr(repair, "_write_manifest", write)
    repair.apply(alias / "m.json", "test", target_is_copy=True)      # resolved ONCE, at the start
    assert db.read_bytes().startswith(b"SQLite format 3\x00") and repair.diagnose(db)["integrity_ok"]
    assert json.loads((home / "m.json").read_text())["state"] == "applied"
    assert not (db.parent / "m.json").exists()


@pytest.mark.parametrize("name", ["freeze.db", "freeze.db.json", "freeze.db-wal"])
def test_a_manifest_named_like_the_database_is_refused(f1, name):
    db, _ = f1
    with pytest.raises(SystemExit, match=r"\.json file|could replace"):
        plan_f1(db, db.parent.parent / name)
    assert sorted(p.name for p in db.parent.iterdir()) == [db.name]


def test_a_manifest_through_a_symlinked_directory_is_never_written_there(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(SystemExit, match="not its real path"):
        repair._write_manifest(tmp_path / "alias" / "m.json", {"tool": repair.TOOL})
    assert list(real.iterdir()) == []


@pytest.mark.parametrize("swap", ["symlink", "replaced"])
def test_rollback_refuses_a_target_swapped_after_the_guards(f1, monkeypatch, swap):
    db, manifest = f1
    plan_f1(db, manifest)
    repair.apply(manifest, "test", target_is_copy=True)
    other = manifest.parent / "other.db"
    shutil.copyfile(db, other)
    parked = db.with_name("parked.db")
    real_source = repair._verified_rollback_source

    def source(*args):
        res = real_source(*args)
        db.rename(parked)
        if swap == "symlink":
            db.symlink_to(other)
        else:
            shutil.copyfile(other, db)
        return res

    monkeypatch.setattr(repair, "_verified_rollback_source", source)
    with pytest.raises(SystemExit, match="replaced"):
        repair.rollback(manifest, target_is_copy=True)
    assert repair.diagnose(other)["integrity_ok"] and repair.diagnose(parked)["integrity_ok"]
    assert json.loads(manifest.read_text())["state"] == "applied"


# ── C8-04: hard links ──

def test_a_hard_linked_target_is_refused_at_plan_and_at_apply(f1):
    db, manifest = f1
    alias = db.with_name("alias.db")
    os.link(db, alias)
    with pytest.raises(SystemExit, match="2 hard links"):
        plan_f1(alias, manifest)
    alias.unlink()
    m = plan_f1(db, manifest)
    os.link(db, alias)                                # linked after the plan
    with pytest.raises(SystemExit, match="2 hard links"):
        repair.apply(manifest, "test", target_is_copy=True)
    assert repair._sha256(db) == m["target_sha256"]
    assert json.loads(manifest.read_text())["state"] == "planned"


# ── S-2: rollback-journal targets ──

def test_a_rollback_journal_database_is_refused_at_plan(f1):
    db, manifest = f1
    with closing(sqlite3.connect(db)) as con:
        assert con.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    assert db.read_bytes()[18:20] == b"\x01\x01"
    with pytest.raises(SystemExit, match="only WAL-mode"):
        plan_f1(db, manifest)
    assert not manifest.exists()


# ── S-5: '..' ──

def test_an_output_spelled_with_dotdot_is_refused(tmp_path, f1):
    db, manifest = f1
    other = tmp_path / "other"
    other.mkdir()
    (db.parent / "sub").mkdir()
    (other / "link").symlink_to(db.parent / "sub", target_is_directory=True)
    for staging, out in ((tmp_path / "stage", other / "link" / ".." / "f.json"),
                         (tmp_path / "stage", other / "link" / ".." / "m.json"),
                         (other / "link" / ".." / "stage", manifest)):
        with pytest.raises(SystemExit, match=r"'\.\.'"):
            repair.plan(db, staging, out, expect_stale=4)
    with pytest.raises(SystemExit, match=r"'\.\.'"):
        repair.check(db, other / "link" / ".." / "stage")
    assert sorted(p.name for p in db.parent.iterdir()) == [db.name, "sub"]
    assert list((db.parent / "sub").iterdir()) == []


# ── S-4: the stage-3 copy ──

def test_the_stage3_copy_never_deletes_the_verified_copy_on_a_rerun(tmp_path):
    from tests.test_sqlite_index_repair import _as_written, _block_after, _runbook_text

    logs, M = tmp_path / "runtime" / "logs", tmp_path / "M"
    (logs / "product").mkdir(parents=True)
    (M / "backup").mkdir(parents=True)
    target = corrupt_index(logs / "product" / "counterfactuals.jsonl.freeze.db")
    repair.plan(target, M / "f1_staging", M / "f1_manifest.json", expect_stale=4)
    block = _block_after(_runbook_text(), "**F1 only, in the stage 3 SSD tier")

    def run():
        return subprocess.run(["bash", "-eu", "-c", _as_written(block, logs)], capture_output=True, text=True,
                              env={**os.environ, "M": str(M), "R": str(logs)})

    F1B = M / "backup" / "f1_freeze_pre_repair.db"
    r = run()
    assert r.returncode == 0 and "F1_BACKUP_OK" in r.stdout, r.stdout + r.stderr
    first = F1B.read_bytes()
    r = run()                                         # an idempotent re-run before 3b
    assert r.returncode == 0 and "F1_BACKUP_OK" in r.stdout and F1B.read_bytes() == first
    with closing(sqlite3.connect(target)) as con:     # 3b repaired the ledger
        con.execute("REINDEX")
        con.commit()
    r = run()                                         # the block re-run after 3b
    assert r.returncode != 0 and "F1_BACKUP_MISMATCH" in r.stdout
    assert F1B.read_bytes() == first                  # the only off-volume pre-repair copy survives
    assert sorted(p.name for p in (M / "backup").iterdir()) == [F1B.name]      # no temp left


# ── C8-07: the power watch ──

BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t12%; discharging; 0:40 remaining"


def test_a_stalled_flush_never_blocks_the_warning_the_polling_or_the_stop():
    from product import host_power_watch as hpw

    entered, release, stop = threading.Event(), threading.Event(), threading.Event()
    sent, polls = [], []

    def run(_argv):
        polls.append(time.monotonic())
        return BATT

    def flush():
        entered.set()
        release.wait(30)                             # a USB disk that stopped answering

    worker = threading.Thread(target=hpw.run_power_watch, args=(stop,),
                              kwargs={"send": sent.append, "run": run, "poll_s": 0.02, "flush": flush}, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        deadline = time.monotonic() + 2
        while not sent and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(sent) == 1 and "BATTERY" in sent[0]     # warned while the flush is stuck
        seen = len(polls)
        time.sleep(0.2)
        assert len(polls) > seen                            # still polling
        stop.set()
        worker.join(1)
        assert not worker.is_alive()                        # and it stops
    finally:
        stop.set()
        release.set()
        worker.join(2)


def test_only_one_flush_is_ever_outstanding():
    from product import host_power_watch as hpw

    release, calls, clock = threading.Event(), [], [0.0]

    def flush():
        calls.append(1)
        release.wait(30)

    watch = hpw.PowerWatch(lambda _m: None, now=lambda: clock[0], flush=flush, flush_s=30)
    try:
        for step in range(5):
            clock[0] = step * 31.0                          # a flush is due every time
            watch.observe({"source": "BATTERY", "percent": 50})
            time.sleep(0.05)
        assert len(calls) == 1 and watch.flushes_skipped_busy == 4
    finally:
        release.set()
        watch.flush_thread.join(2)
    clock[0] = 1000.0
    watch.observe({"source": "BATTERY", "percent": 50})
    watch.flush_thread.join(2)
    assert len(calls) == 2                                  # free again once the stuck one returned


# ── S-6: one identity-based answer for every "never the live runtime" guard ──

TEXT_SPELLINGS = ["/Volumes", "/Volumes/QuantTermStorage/QuantTerm/runtime", "/volumes/x", "//Volumes/x",
                  "/VOLUMES/QuantTermStorage/x"]


def _guards():
    from product import bhav_store_rebuild as rb
    from product import learning_observation_correction as loc
    from research import counterfactual_reconcile, learning_effectiveness, pit_context_coverage
    from research import store_integrity_audit

    return {
        "bhav_store_rebuild._not_live": lambda p: rb._not_live(Path(p), "x"),
        "learning_observation_correction._not_live": lambda p: loc._not_live(p, "x"),
        "counterfactual_reconcile._refuse_live": lambda p: counterfactual_reconcile._refuse_live(p),
        "pit_context_coverage._refuse_live": pit_context_coverage._refuse_live,
        "store_integrity_audit._refuse_live": store_integrity_audit._refuse_live,
        "learning_effectiveness.build_report": learning_effectiveness.build_report,
    }


@pytest.mark.parametrize("guard", sorted(_guards()))
def test_every_live_guard_refuses_every_spelling(tmp_path, guard):
    fn = _guards()[guard]
    alias = tmp_path / "alias"
    alias.symlink_to("/Volumes/NoSuchVolume", target_is_directory=True)
    spellings = TEXT_SPELLINGS + [str(alias / "runtime")]
    if sys.platform == "darwin":                              # the firmlink spelling exists only there
        spellings.append("/System/Volumes/Data/Volumes/NoSuchVolume/runtime")
    for spelling in spellings:
        with pytest.raises(SystemExit):
            fn(spelling)


def test_identity_catches_a_spelling_realpath_keeps(tmp_path, monkeypatch):
    """Platform-independent: an alias that realpath does NOT normalize (as macOS keeps the
    firmlink spelling) is still caught because it IS the anchor directory."""
    from core import live_volume

    anchor = tmp_path / "anchor"
    (anchor / "vol").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(anchor, target_is_directory=True)
    monkeypatch.setattr(live_volume, "ANCHOR", str(anchor))
    monkeypatch.setattr(live_volume.os.path, "realpath", lambda p: p)
    assert live_volume.on_live_volume(alias / "vol" / "runtime")
    assert not live_volume.on_live_volume(tmp_path / "elsewhere")


@pytest.mark.skipif(sys.platform != "darwin", reason="the firmlink exists only on macOS")
def test_the_firmlink_spelling_is_the_live_volume_on_this_mac():
    from core.live_volume import on_live_volume

    assert on_live_volume("/System/Volumes/Data/Volumes/NoSuchVolume/x")
    assert on_live_volume("/system/volumes/data/volumes/NoSuchVolume/x")
    assert not on_live_volume("/System/Volumes/Data/Users")


def test_no_text_only_volumes_guard_is_left():
    """Every "never the live runtime" guard goes through core.live_volume: a new text-only
    check would bring the firmlink and case spellings back."""
    import re

    pattern = re.compile(r"""startswith\(\s*["'](/Volumes|/volumes)""")
    offenders = []
    for top in ("core", "product", "research", "scan", "data", "execution", "operations", "scripts"):
        for path in (REPO / top).rglob("*.py"):
            if path.name == "live_volume.py":
                continue
            for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(REPO)}:{n}: {line.strip()}")
    assert offenders == []
