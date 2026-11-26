"""Cycle 9, item 1 (GPT review of bf787f41): every writer of an F1 repair manifest is serialized.

``_publish_plan`` checked the manifest, then renamed the new plan over it. Audit §6i
admitted the window: a second command that wrote the manifest between that check and the
rename was overwritten. These tests reproduce it with TWO PROCESSES and deterministic
barriers: the first command is paused at an exact manifest write (after its check, before
its rename), the second runs to completion, then the first is released.

Now plan, apply and rollback each hold the manifest's lock from their first read to their
last write. The lock is a plain file beside the manifest, never replaced, keyed by the
case-folded name in the manifest's physical directory, so every alias of one manifest
meets it. A held lock refuses at once; the kernel releases it when its holder dies.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from product import sqlite_index_repair as repair
from tests.test_cycle8_review import f1, plan_f1  # noqa: F401  (the shared fixture)

REPO = Path(__file__).resolve().parents[1]

_RUNNER = r'''
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
from product import sqlite_index_repair as sir
op, db, manifest, staging, point, gate = sys.argv[2:8]
sir._held_open = lambda _: []
sir._check_desk = lambda *_: None
gate = os.path.abspath(gate)

def barrier(name):
    open(os.path.join(gate, name + ".reached"), "w").close()
    deadline = time.time() + 60
    while not os.path.exists(os.path.join(gate, name + ".go")):
        if time.time() > deadline:
            os._exit(99)                                  # never released: a bounded wait
        time.sleep(0.01)

if point.startswith(("pause-write:", "die-write:")):
    kind, n = point.split(":")
    real, calls = sir._write_manifest, [0]
    def write(path, m):
        calls[0] += 1
        if calls[0] == int(n):
            if kind == "die-write":
                os._exit(137)                             # a hard death holding the lock
            barrier("write")                              # after the command's checks, before its rename
        return real(path, m)
    sir._write_manifest = write
elif point == "die-before-rename":
    real_replace = os.replace
    def replace(src, dst):
        if str(dst).endswith(".json"):
            os._exit(137)                                 # the temp file is written and synced; no rename
        return real_replace(src, dst)
    os.replace = replace

try:
    if op == "plan":
        out = sir.plan(db, staging, manifest, expect_stale=4)
    elif op == "apply":
        out = sir.apply(manifest, "test", target_is_copy=True)
    else:
        out = sir.rollback(manifest, target_is_copy=True)
except SystemExit as exc:
    print(json.dumps({"refused": str(exc)}))
    sys.exit(2)
print(json.dumps({"state": out.get("state"), "plan_id": out.get("plan_id"), "backup": out.get("backup")}))
'''


def _env(tmp_path: Path) -> dict:
    (tmp_path / "child-runtime").mkdir(exist_ok=True)
    return {**os.environ, "QT_RUNTIME_ROOT": str(tmp_path / "child-runtime"),
            "KITE_ACCESS_TOKEN": "", "TELEGRAM_BOT_TOKEN": "", "DEEPSEEK_API_KEY": ""}


class _Cmd:
    def __init__(self, tmp_path: Path, op: str, db: Path, manifest: Path, point: str = "none",
                 staging: str = "stage-x"):
        self.gate = tmp_path / f"gate-{op}-{time.monotonic_ns()}"
        self.gate.mkdir()
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _RUNNER, str(REPO), op, str(db), str(manifest),
             str(tmp_path / f"{staging}-{time.monotonic_ns()}"), point, str(self.gate)],
            env=_env(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_reached(self, name: str = "write") -> None:
        deadline = time.time() + 60
        while not (self.gate / f"{name}.reached").exists():
            assert self.proc.poll() is None, self.proc.communicate()
            assert time.time() < deadline, "the paused command never reached its barrier"
            time.sleep(0.01)

    def release(self, name: str = "write") -> None:
        (self.gate / f"{name}.go").touch()

    def result(self) -> tuple[int, dict]:
        out, err = self.proc.communicate(timeout=120)
        lines = [line for line in out.splitlines() if line.startswith("{")]
        return self.proc.returncode, (json.loads(lines[-1]) if lines else {"stderr": err[-800:]})


def _run(tmp_path, op, db, manifest, point="none") -> tuple[int, dict]:
    return _Cmd(tmp_path, op, db, manifest, point).result()


def _stale(db: Path, tmp_path: Path) -> int:
    return repair.check(db, tmp_path / f"chk-{time.monotonic_ns()}")["stale_total"]


# ── the reproduced race: a check, a concurrent writer, then the rename ──

def test_a_plan_paused_after_its_check_refuses_a_concurrent_apply(f1, tmp_path):
    db, manifest = f1
    plan_f1(db, manifest)
    paused = _Cmd(tmp_path, "plan", db, manifest, "pause-write:1")
    paused.wait_reached()                         # the plan's re-check passed; its rename is next
    rc, out = _run(tmp_path, "apply", db, manifest)
    paused.release()
    prc, planned = paused.result()
    m = json.loads(manifest.read_text())
    # Before the lock: the apply repaired the target and recorded its backup, then the plan's
    # rename erased that record (state planned, no backup) beside a repaired database.
    assert rc == 2 and "in use by plan" in out["refused"], out
    assert prc == 0 and m["plan_id"] == planned["plan_id"] and "backup" not in m
    assert _stale(db, tmp_path) == 4                                       # never repaired
    assert repair.apply(manifest, "test", target_is_copy=True)["state"] == "applied"   # the retry


def test_two_concurrent_plans_publish_exactly_one(f1, tmp_path):
    db, manifest = f1
    plan_f1(db, manifest)
    paused = _Cmd(tmp_path, "plan", db, manifest, "pause-write:1")
    paused.wait_reached()
    rc, out = _run(tmp_path, "plan", db, manifest)
    paused.release()
    prc, planned = paused.result()
    # Before the lock: both reported success; the second's plan was silently replaced.
    assert rc == 2 and "in use by plan" in out["refused"], out
    assert prc == 0 and json.loads(manifest.read_text())["plan_id"] == planned["plan_id"]


@pytest.mark.parametrize("second", ["rollback", "plan", "apply"])
def test_an_apply_paused_before_its_final_record_refuses_every_other_command(f1, tmp_path, second):
    db, manifest = f1
    plan_f1(db, manifest)
    paused = _Cmd(tmp_path, "apply", db, manifest, "pause-write:4")   # repaired and proven; record next
    paused.wait_reached()
    on_disk = manifest.read_text()
    rc, out = _run(tmp_path, second, db, manifest)
    assert manifest.read_text() == on_disk                            # nothing written beside it
    paused.release()
    prc, applied = paused.result()
    assert rc == 2 and "in use by apply" in out["refused"], out
    assert prc == 0 and applied["state"] == "applied"
    m = json.loads(manifest.read_text())
    assert m["state"] == "applied" and m["backup"] == applied["backup"] and Path(m["backup"]).is_file()
    assert repair.rollback(manifest, target_is_copy=True)["state"] == "rolled_back"


def test_a_rollback_paused_mid_record_refuses_a_re_plan(f1, tmp_path):
    db, manifest = f1
    plan_f1(db, manifest)
    applied = repair.apply(manifest, "test", target_is_copy=True)
    paused = _Cmd(tmp_path, "rollback", db, manifest, "pause-write:1")   # before 'rolling_back'
    paused.wait_reached()
    rc, out = _run(tmp_path, "plan", db, manifest)
    paused.release()
    prc, rolled = paused.result()
    assert rc == 2 and "in use by rollback" in out["refused"], out
    assert prc == 0 and rolled["state"] == "rolled_back"
    m = plan_f1(db, manifest)                          # afterwards a re-plan keeps the history
    assert applied["backup"] in m["earlier_backups"] and Path(applied["backup"]).is_file()


# ── aliases: every spelling of one manifest meets the same lock ──

def _paused_plan(tmp_path, db, manifest) -> _Cmd:
    plan_f1(db, manifest)
    paused = _Cmd(tmp_path, "plan", db, manifest, "pause-write:1")
    paused.wait_reached()
    return paused


def test_a_symlinked_directory_alias_meets_the_same_lock(f1, tmp_path):
    db, manifest = f1
    (tmp_path / "alias").symlink_to(manifest.parent, target_is_directory=True)
    paused = _paused_plan(tmp_path, db, manifest)
    try:
        rc, out = _run(tmp_path, "apply", db, tmp_path / "alias" / manifest.name)
    finally:
        paused.release()
    assert paused.result()[0] == 0
    assert rc == 2 and "in use by plan" in out["refused"], out


def test_a_differently_cased_name_meets_the_same_lock(f1, tmp_path):
    db, manifest = f1
    paused = _paused_plan(tmp_path, db, manifest)
    try:
        rc, out = _run(tmp_path, "plan", db, manifest.with_name(manifest.name.upper()))
    finally:
        paused.release()
    assert paused.result()[0] == 0
    assert rc == 2 and "in use by plan" in out["refused"], out


def test_a_hard_linked_manifest_is_refused_by_every_command(f1, tmp_path):
    db, manifest = f1
    plan_f1(db, manifest)
    before = manifest.read_text()
    os.link(manifest, tmp_path / "second-name.json")
    for call in (lambda: repair.apply(manifest, "test", target_is_copy=True),
                 lambda: repair.rollback(manifest, target_is_copy=True),
                 lambda: plan_f1(db, manifest)):
        with pytest.raises(SystemExit, match="hard links"):
            call()
    assert manifest.read_text() == before and _stale(db, tmp_path) == 4


# ── release, interrupted publication, lock identity ──

def test_a_command_that_dies_holding_the_lock_leaves_a_retry_free(f1, tmp_path):
    db, manifest = f1
    plan_f1(db, manifest)
    rc, _ = _run(tmp_path, "apply", db, manifest, "die-write:2")      # backup recorded, then killed
    assert rc == 137
    dead = json.loads(manifest.read_text())["backup_pending"]
    m = repair.apply(manifest, "test", target_is_copy=True)         # the kernel released the lock
    assert m["state"] == "applied" and m["earlier_backups"] == [dead]   # the dead attempt's backup recorded
    assert Path(dead).is_file() and Path(m["backup"]).is_file() and m["backup"] != dead


def test_an_interrupted_publication_keeps_the_old_record_whole(f1, tmp_path):
    db, manifest = f1
    plan_f1(db, manifest)
    applied = repair.apply(manifest, "test", target_is_copy=True)
    assert repair.rollback(manifest, target_is_copy=True)["state"] == "rolled_back"
    before = manifest.read_bytes()
    rc, _ = _run(tmp_path, "plan", db, manifest, "die-before-rename")
    assert rc == 137 and manifest.read_bytes() == before             # the old record, byte for byte
    m = plan_f1(db, manifest)                                         # the retry proceeds
    assert m["state"] == "planned" and applied["backup"] in m["earlier_backups"]


def test_a_lock_removed_while_held_refuses_the_write(f1, monkeypatch):
    db, manifest = f1
    plan_f1(db, manifest)
    before = manifest.read_text()
    quiet_copy = repair._quiet_copy

    def copy_then_remove_lock(src, dst):
        copied = quiet_copy(src, dst)
        repair._lock_path(manifest.resolve()).unlink()             # a second holder could now lock a new file
        return copied

    monkeypatch.setattr(repair, "_quiet_copy", copy_then_remove_lock)
    with pytest.raises(SystemExit, match="removed or replaced while it was held"):
        repair.plan(db, manifest.parent / "stage2", manifest)
    assert manifest.read_text() == before


def test_a_manifest_is_never_written_without_its_lock(tmp_path):
    real = tmp_path.resolve() / "m.json"
    with pytest.raises(SystemExit, match="without its lock"):
        repair._write_manifest(real, {"tool": repair.TOOL})
    assert not real.exists()


def test_the_lock_is_released_after_a_refusal_and_never_created_beside_the_database(f1, tmp_path):
    db, manifest = f1
    plan_f1(db, manifest)
    with pytest.raises(SystemExit, match="owner's approval"):
        repair.apply(manifest, "", target_is_copy=True)
    assert repair.apply(manifest, "test", target_is_copy=True)["state"] == "applied"
    with pytest.raises(SystemExit, match="database's directory"):
        repair.plan(db, tmp_path / "stage9", db.parent / "beside.json")
    assert sorted(p.name for p in db.parent.iterdir() if not p.name.startswith(db.name)) == []


@pytest.mark.parametrize("command", ["apply", "rollback"])
def test_a_manifest_moved_beside_the_database_is_refused_before_any_lock_file(f1, command):
    """apply / rollback check the manifest's place against the target it names BEFORE they
    create its lock, so nothing is ever created in the live database's directory."""
    db, manifest = f1
    plan_f1(db, manifest)
    beside = db.parent / "moved.json"
    beside.write_text(manifest.read_text())
    call = (lambda: repair.apply(beside, "test", target_is_copy=True)) if command == "apply" else (
        lambda: repair.rollback(beside, target_is_copy=True))
    with pytest.raises(SystemExit, match="database's directory"):
        call()
    assert sorted(p.name for p in db.parent.iterdir() if not p.name.startswith(db.name)) == ["moved.json"]
