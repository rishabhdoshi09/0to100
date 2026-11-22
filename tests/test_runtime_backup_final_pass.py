"""The runbook's final backup pass refreshes and verifies every file written after the bulk pass began (GPT QT-07, cycle 6)."""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runtime_backup_final_pass.sh"
pytestmark = pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")


def _run(src, dst, stamp):
    return subprocess.run(["bash", str(SCRIPT), str(src), str(dst), str(stamp)],
                          capture_output=True, text=True)


def test_final_backup_refreshes_same_size_db(tmp_path):
    """GPT's QT-07 sketch, adapted: it parsed a single rsync line from the runbook;
    the final pass is now this script, and the realistic race is a same-size
    rewrite whose mtime falls in the SAME second the backup copy recorded."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    stamp = tmp_path / "bulk_started"
    stamp.write_text("")
    os.utime(stamp, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    db = src / "paper.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE state(value TEXT)")
        con.execute("INSERT INTO state VALUES ('before')")
    os.utime(db, ns=(1_700_000_001_100_000_000, 1_700_000_001_100_000_000))
    shutil.copy2(db, dst / db.name)                     # the bulk pass copy
    with sqlite3.connect(db) as con:
        con.execute("UPDATE state SET value='after!'")
    # same size, mtime in the same whole second: the quick check cannot see it
    os.utime(db, ns=(1_700_000_001_900_000_000, 1_700_000_001_900_000_000))
    assert db.stat().st_size == (dst / db.name).stat().st_size

    out = _run(src, dst, stamp)
    assert out.returncode == 0, out.stderr
    assert "FINAL_PASS_OK rechecked=1" in out.stdout
    with sqlite3.connect(dst / db.name) as con:
        assert con.execute("SELECT value FROM state").fetchone() == ("after!",)


def test_final_pass_mirrors_deletions_and_new_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "logs").mkdir(parents=True)
    dst.mkdir()
    stamp = tmp_path / "bulk_started"
    stamp.write_text("")
    os.utime(stamp, (1_700_000_000, 1_700_000_000))
    (dst / "stale.json").write_text("{}")
    (src / "logs" / "new.json").write_text('{"a": 1}')
    out = _run(src, dst, stamp)
    assert out.returncode == 0, out.stderr
    assert not (dst / "stale.json").exists()
    assert (dst / "logs" / "new.json").read_text() == '{"a": 1}'


def test_final_pass_refuses_without_the_bulk_stamp(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "d").mkdir()
    out = _run(tmp_path / "s", tmp_path / "d", tmp_path / "missing")
    assert out.returncode == 2 and "no bulk-pass stamp" in out.stderr


# ── GPT cycle-6 verification pass (R8, R10) ──


# R8
def test_final_pass_verifies_same_size_copy_with_preserved_mtime(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    stamp = tmp_path / "stamp"
    original, update = src / "autopilot.json", tmp_path / "update.json"
    stamp.touch()
    original.write_text('{"qty":1}')
    update.write_text('{"qty":2}')
    for p in (stamp, original, update):
        os.utime(p, (1_700_000_000, 1_700_000_000))
    shutil.copy2(original, dst / original.name)
    shutil.copy2(update, original)  # Published after bulk, with an older mtime.
    out = _run(src, dst, stamp)
    assert out.returncode == 0, out.stderr
    assert (dst / original.name).read_bytes() == original.read_bytes()

# R10
def test_runbook_stops_before_start_on_a_paper_hash_failure(tmp_path):
    runbook = SCRIPT.parents[1] / "docs/runbooks/2026-11-22-paper-upgrade-and-store-repair.md"
    stage = runbook.read_text().split("## 6. Deploy", 1)[1]
    block = stage.split("```bash\n", 1)[1].split("```", 1)[0]
    start = block.index("venv/bin/python -m product.host_install install")
    end = block.index("venv/bin/python -m product.host_install status")
    commands = block[start:end].replace("venv/bin/python", "desk")
    stubs = """
shasum() { return 1; }
desk() {
  case "$*" in *--no-start*) return 0;; esac
  touch "$M/started"
}
"""
    subprocess.run(["bash", "-c", stubs + commands],
                   env={**os.environ, "M": str(tmp_path), "R": str(tmp_path)})
    assert not (tmp_path / "started").exists()
