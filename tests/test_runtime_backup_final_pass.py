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
    # tmp_path source and backup share one filesystem; the script refuses that
    # outside tests (test_the_backup_must_be_on_another_filesystem)
    return subprocess.run(["bash", str(SCRIPT), str(src), str(dst), str(stamp)],
                          capture_output=True, text=True, env={**os.environ, "QT_FINAL_PASS_ALLOW_SAME_FS": "1"})


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


# ── Cycle 7 (relay review of 011dffb0): unchanged-file corruption, aliasing ──

def _env(**extra):
    return {**os.environ, "QT_FINAL_PASS_ALLOW_SAME_FS": "1", **extra}


def _run2(*argv):
    return subprocess.run(["bash", str(SCRIPT), *map(str, argv)], capture_output=True, text=True, env=_env())


def _baseline(dst, out):
    """A stage-1 style manifest of the backup: 'sha256  relpath' lines."""
    lines = []
    for p in sorted(Path(dst).rglob("*")):
        if p.is_file() and not p.is_symlink():
            import hashlib
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dst)}")
    Path(out).write_text("\n".join(lines) + "\n")
    return out


def test_corruption_of_an_unchanged_backup_file_fails_with_a_verified_baseline(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "ledger.db").write_bytes(b"A" * 4096)
    shutil.copy2(src / "ledger.db", dst / "ledger.db")
    base = _baseline(dst, tmp_path / "baseline.sha256")       # verified at stage 1
    stamp = tmp_path / "stamp"
    stamp.write_text("")
    os.utime(stamp, (2_000_000_000, 2_000_000_000))             # nothing written since
    st = (dst / "ledger.db").stat()
    (dst / "ledger.db").write_bytes(b"B" * 4096)                # silent corruption of the BACKUP
    os.utime(dst / "ledger.db", ns=(st.st_atime_ns, st.st_mtime_ns))
    out = _run2(src, dst, stamp, "--baseline", base)
    assert out.returncode != 0 and "FINAL_PASS_OK" not in out.stdout


def test_without_a_baseline_unchanged_file_corruption_is_not_detectable(tmp_path):
    """Documents the limit: rsync's quick check and the post-stamp delta never
    read an unchanged file, which is why the runbook passes --baseline."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "ledger.db").write_bytes(b"A" * 4096)
    shutil.copy2(src / "ledger.db", dst / "ledger.db")
    stamp = tmp_path / "stamp"
    stamp.write_text("")
    os.utime(stamp, (2_000_000_000, 2_000_000_000))
    st = (dst / "ledger.db").stat()
    (dst / "ledger.db").write_bytes(b"B" * 4096)
    os.utime(dst / "ledger.db", ns=(st.st_atime_ns, st.st_mtime_ns))
    out = _run2(src, dst, stamp)
    assert out.returncode == 0 and "NO_BASELINE" in out.stdout


@pytest.mark.parametrize("case", ["same", "dst_inside_src", "src_inside_dst"])
def test_aliased_or_nested_paths_are_refused(tmp_path, case):
    stamp = tmp_path / "stamp"
    stamp.write_text("")
    if case == "src_inside_dst":
        dst = tmp_path / "outer"
        src = dst / "inner"
        src.mkdir(parents=True)
    else:
        src = tmp_path / "src"
        src.mkdir()
        dst = src if case == "same" else src / "backup"
        dst.mkdir(exist_ok=True)
    (src / "a").write_text("x")
    out = _run2(src, dst, stamp)
    assert out.returncode == 2 and "refusing" in out.stderr
    assert (src / "a").exists()                                   # nothing was deleted


def test_the_backup_must_be_on_another_filesystem(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    stamp = tmp_path / "stamp"
    stamp.write_text("")
    out = subprocess.run(["bash", str(SCRIPT), str(src), str(dst), str(stamp)], capture_output=True, text=True,
                         env={k: v for k, v in os.environ.items() if k != "QT_FINAL_PASS_ALLOW_SAME_FS"})
    assert out.returncode == 2 and "same filesystem" in out.stderr


def test_verify_only_checks_the_baseline_without_copying(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a.json").write_text('{"a": 1}')
    shutil.copy2(src / "a.json", dst / "a.json")
    base = _baseline(dst, tmp_path / "baseline.sha256")
    (src / "new.json").write_text("{}")                          # must NOT be copied in verify-only
    stamp = tmp_path / "stamp"
    stamp.write_text("")
    out = _run2(src, dst, stamp, "--baseline", base, "--verify-only")
    assert out.returncode == 0 and "BASELINE_OK" in out.stdout and not (dst / "new.json").exists()
    (dst / "a.json").write_text('{"a": 2}')
    out = _run2(src, dst, stamp, "--baseline", base, "--verify-only")
    assert out.returncode == 1 and "BASELINE_MISMATCH" in out.stderr


def test_a_baseline_with_no_usable_entries_is_refused(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a").write_text("x")
    shutil.copy2(src / "a", dst / "a")
    bad = tmp_path / "bad.sha256"
    bad.write_text("not-a-hash  a\n")
    stamp = tmp_path / "stamp"
    stamp.write_text("")
    os.utime(stamp, (2_000_000_000, 2_000_000_000))
    out = _run2(src, dst, stamp, "--baseline", bad, "--verify-only")
    assert out.returncode == 2 and "malformed" in out.stderr


def test_relative_stamp_and_baseline_paths_work(tmp_path):
    """Found by the cycle-7 benchmark: the delta search runs inside the source, so a
    relative stamp path must be made absolute first."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "a").write_text("x")
    shutil.copy2(src / "a", dst / "a")
    _baseline(dst, tmp_path / "baseline.sha256")
    (tmp_path / "stamp").write_text("")
    out = subprocess.run(["bash", str(SCRIPT), "src", "dst", "stamp", "--baseline", "baseline.sha256"],
                         capture_output=True, text=True, cwd=tmp_path, env=_env())
    assert out.returncode == 0, out.stderr
    assert "baseline_verified=1" in out.stdout


def test_a_file_deleted_since_the_bulk_pass_is_mirrored_not_a_baseline_failure(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    for n in ("keep", "gone"):
        (src / n).write_text(n)
        shutil.copy2(src / n, dst / n)
    base = _baseline(dst, tmp_path / "baseline.sha256")
    stamp = tmp_path / "stamp"
    stamp.write_text("")
    (src / "gone").unlink()                                      # the desk deleted it during the bulk pass
    out = _run2(src, dst, stamp, "--baseline", base)
    assert out.returncode == 0, out.stderr
    assert "baseline_verified=1" in out.stdout and not (dst / "gone").exists()


# ── GPT cycle-7 review (C7-01..C7-03): sketches as written ──

# GPT C7-01 (cycle-7 review sketch)
@pytest.mark.parametrize("mode", ["newline_delta", "empty", "malformed"])
def test_cycle7_review_final_checks_unchanged_backup(tmp_path, mode):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    ledger = src / "ledger.db"
    ledger.write_bytes(b"good")
    (src / "healthy").write_bytes(b"healthy")
    shutil.copy2(ledger, dst / ledger.name)
    shutil.copy2(src / "healthy", dst / "healthy")
    base = _baseline(dst, tmp_path / "baseline")
    stamp = tmp_path / "stamp"
    stamp.touch()
    os.utime(stamp, (2_000_000_000, 2_000_000_000))

    if mode == "newline_delta":
        added = src / "new\nledger.db"
        added.write_bytes(b"fresh")
        os.utime(added, (2_000_000_002, 2_000_000_002))
    elif mode == "empty":
        base.write_text("")
    else:
        base.write_text("bad-hash  ledger.db\n" + next(
            line for line in base.read_text().splitlines(keepends=True)
            if line.endswith("  healthy\n")
        ))

    (dst / ledger.name).write_bytes(b"bad!")
    shutil.copystat(ledger, dst / ledger.name)
    out = _run2(src, dst, stamp, "--baseline", base)
    assert out.returncode != 0 and "FINAL_PASS_OK" not in out.stdout

# GPT C7-02 (cycle-7 review sketch)
def test_cycle7_review_verify_only_checks_a_changed_baseline_entry(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    ledger = src / "ledger.db"
    ledger.write_bytes(b"bulk")
    shutil.copy2(ledger, dst / ledger.name)
    base = _baseline(dst, tmp_path / "baseline")
    stamp = tmp_path / "stamp"
    stamp.touch()
    os.utime(stamp, (2_000_000_000, 2_000_000_000))
    ledger.write_bytes(b"live")
    os.utime(ledger, (2_000_000_002, 2_000_000_002))
    (dst / ledger.name).write_bytes(b"bad!")

    out = _run2(src, dst, stamp, "--baseline", base, "--verify-only")
    assert (dst / ledger.name).read_bytes() == b"bad!"
    assert out.returncode != 0 and "BASELINE_OK" not in out.stdout

# GPT C7-03 (cycle-7 review sketch)
def test_cycle7_review_final_pass_preserves_directory_name_bytes(tmp_path):
    src, sibling, dst = [tmp_path / n for n in ("source\n", "source", "backup")]
    for path in (src, sibling, dst):
        path.mkdir()
    (src / "ledger.db").write_bytes(b"correct")
    shutil.copy2(src / "ledger.db", dst / "ledger.db")
    (sibling / "other.db").write_bytes(b"wrong tree")
    base = _baseline(dst, tmp_path / "baseline")
    stamp = tmp_path / "stamp"
    stamp.touch()
    os.utime(stamp, (2_000_000_000, 2_000_000_000))

    out = _run2(src, dst, stamp, "--baseline", base)
    assert (dst / "ledger.db").exists(), out.stdout
    assert (dst / "ledger.db").read_bytes() == b"correct"
    assert not (dst / "other.db").exists()


# Each guard on its own (the layers above would also fail these runs, later):

def _corrupt_setup(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "ledger.db").write_bytes(b"good")
    (src / "healthy").write_bytes(b"healthy")
    for n in ("ledger.db", "healthy"):
        shutil.copy2(src / n, dst / n)
    base = _baseline(dst, tmp_path / "baseline")
    stamp = tmp_path / "stamp"
    stamp.touch()
    os.utime(stamp, (2_000_000_000, 2_000_000_000))
    return src, dst, base, stamp


def test_a_newline_in_a_changed_file_name_is_refused_outright(tmp_path):
    src, dst, base, stamp = _corrupt_setup(tmp_path)
    added = src / "new\nledger.db"
    added.write_bytes(b"fresh")
    os.utime(added, (2_000_000_002, 2_000_000_002))
    out = _run2(src, dst, stamp, "--baseline", base)
    assert out.returncode == 2 and "newline" in out.stderr


def test_a_malformed_baseline_line_is_refused_even_beside_good_ones(tmp_path):
    src, dst, base, stamp = _corrupt_setup(tmp_path)
    base.write_text("bad-hash  ledger.db\n" + base.read_text())
    out = _run2(src, dst, stamp, "--baseline", base)
    assert out.returncode == 2 and "malformed" in out.stderr


def test_verify_only_reports_a_changed_entry_as_a_baseline_mismatch(tmp_path):
    src, dst, base, stamp = _corrupt_setup(tmp_path)
    (src / "ledger.db").write_bytes(b"live")
    os.utime(src / "ledger.db", (2_000_000_002, 2_000_000_002))
    (dst / "ledger.db").write_bytes(b"bad!")
    out = _run2(src, dst, stamp, "--baseline", base, "--verify-only")
    assert out.returncode == 1 and "BASELINE_MISMATCH ledger.db" in out.stderr


# ── GPT cycle-7 verification pass (C7V): sketches ──

# GPT C7V-03 (cycle-7 verification sketch)
def test_c7v_verify_only_requires_a_regular_backup_file(tmp_path):
    src, dst, base, stamp = _corrupt_setup(tmp_path)
    ledger = dst / "ledger.db"
    ledger.unlink()
    ledger.symlink_to(src / ledger.name)

    result = _run2(src, dst, stamp, "--baseline", base, "--verify-only")
    assert result.returncode != 0
    assert "BASELINE_OK" not in result.stdout

# GPT C7V-04 (cycle-7 verification sketch)
def test_c7v_refuses_newline_in_resolved_source(tmp_path):
    src, sibling, dst = [tmp_path / name for name in ("source\n", "source", "backup")]
    for directory in (src, sibling, dst):
        directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(src, target_is_directory=True)
    (src / "ledger.db").write_bytes(b"correct")
    shutil.copy2(src / "ledger.db", dst / "ledger.db")
    base = _baseline(dst, tmp_path / "baseline")
    stamp = tmp_path / "stamp"
    stamp.touch()
    os.utime(stamp, (2_000_000_000, 2_000_000_000))
    wrong = sibling / "other.db"
    wrong.write_bytes(b"wrong tree")
    os.utime(wrong, (2_000_000_002, 2_000_000_002))

    result = _run2(alias, dst, stamp, "--baseline", base)
    assert result.returncode == 2 and "newline" in result.stderr
    assert (dst / "ledger.db").read_bytes() == b"correct"
    assert not (dst / "other.db").exists()


# ── Stage 1 (22 Nov 2026): paths the stage-1 backup deliberately excluded ──


def _stage1_tree(tmp_path):
    """A backup that holds everything except one large static log, which
    stage 1 excluded; the baseline is that backup's manifest."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "logs").mkdir(parents=True)
    dst.mkdir()
    (src / "ledger.db").write_bytes(b"L" * 2048)
    (src / "logs" / "cron.log").write_bytes(b"static September log")
    (dst / "logs").mkdir()
    shutil.copy2(src / "ledger.db", dst / "ledger.db")
    base = _baseline(dst, tmp_path / "baseline.sha256")
    stamp = tmp_path / "stamp"
    stamp.write_text("")
    os.utime(stamp, (2_000_000_000, 2_000_000_000))              # nothing written since
    return src, dst, base, stamp


def test_without_exclude_a_stage1_excluded_log_is_copied_in_the_window_and_fails_coverage(tmp_path):
    """The gap: rsync --delete copies the excluded 10 GB log inside the
    downtime, and the copy is then neither in the delta nor in the baseline."""
    src, dst, base, stamp = _stage1_tree(tmp_path)
    out = _run2(src, dst, stamp, "--baseline", base)
    assert out.returncode == 1 and "UNCOVERED logs/cron.log" in out.stderr
    assert (dst / "logs" / "cron.log").exists()


def test_an_excluded_path_is_neither_copied_nor_required_by_the_baseline(tmp_path):
    src, dst, base, stamp = _stage1_tree(tmp_path)
    out = _run2(src, dst, stamp, "--baseline", base, "--exclude", "logs/cron.log")
    assert out.returncode == 0, out.stderr
    assert "FINAL_PASS_OK" in out.stdout and "excluded=1" in out.stdout
    assert not (dst / "logs" / "cron.log").exists()
    out = _run2(src, dst, stamp, "--baseline", base, "--verify-only", "--exclude", "logs/cron.log")
    assert out.returncode == 0 and "BASELINE_OK" in out.stdout


def test_exclude_does_not_hide_other_files_or_corruption(tmp_path):
    src, dst, base, stamp = _stage1_tree(tmp_path)
    st = (dst / "ledger.db").stat()
    (dst / "ledger.db").write_bytes(b"X" * 2048)                 # corrupt a NON-excluded backup file
    os.utime(dst / "ledger.db", ns=(st.st_atime_ns, st.st_mtime_ns))
    out = _run2(src, dst, stamp, "--baseline", base, "--exclude", "logs/cron.log")
    assert out.returncode == 1 and "BASELINE_MISMATCH ledger.db" in out.stderr
    # an exclude names one literal path: a sibling with a shared prefix is still covered
    (dst / "ledger.db").write_bytes(b"L" * 2048)
    (src / "logs" / "cron.log.2").write_bytes(b"x")
    out = _run2(src, dst, stamp, "--baseline", base, "--exclude", "logs/cron.log")
    assert out.returncode == 1 and "UNCOVERED logs/cron.log.2" in out.stderr


def test_an_excluded_file_changed_after_the_stamp_is_not_recopied(tmp_path):
    src, dst, base, stamp = _stage1_tree(tmp_path)
    os.utime(src / "logs" / "cron.log", (2_000_000_005, 2_000_000_005))
    out = _run2(src, dst, stamp, "--baseline", base, "--exclude", "logs/cron.log")
    assert out.returncode == 0 and "rechecked=0" in out.stdout
    assert not (dst / "logs" / "cron.log").exists()


def test_an_excluded_directory_covers_everything_under_it(tmp_path):
    src, dst, base, stamp = _stage1_tree(tmp_path)
    (src / "logs" / "big").mkdir()
    (src / "logs" / "big" / "a.log").write_bytes(b"a")
    out = _run2(src, dst, stamp, "--baseline", base, "--exclude", "logs/cron.log", "--exclude", "logs/big/")
    assert out.returncode == 0 and "excluded=2" in out.stdout
    assert not (dst / "logs" / "big").exists()


@pytest.mark.parametrize("bad", ["/logs/cron.log", "logs/*.log", "logs/cron.lo?", "logs/[c]ron.log",
                                 "logs/../ledger.db", "./logs", "logs//cron.log", "logs\\cron.log",
                                 "logs/cron.log\n", ""])
def test_exclude_accepts_only_a_literal_relative_path(tmp_path, bad):
    src, dst, base, stamp = _stage1_tree(tmp_path)
    out = _run2(src, dst, stamp, "--baseline", base, "--exclude", bad)
    assert out.returncode == 2 and "refusing" in out.stderr
    assert not (dst / "logs" / "cron.log").exists()


def test_an_excluded_file_already_in_the_backup_is_kept_and_not_required(tmp_path):
    """An earlier partial pass left the excluded log in the backup, outside the
    baseline: verify-only must not call it UNCOVERED, and --delete keeps it."""
    src, dst, base, stamp = _stage1_tree(tmp_path)
    shutil.copy2(src / "logs" / "cron.log", dst / "logs" / "cron.log")
    out = _run2(src, dst, stamp, "--baseline", base, "--verify-only")
    assert out.returncode == 1 and "UNCOVERED logs/cron.log" in out.stderr
    out = _run2(src, dst, stamp, "--baseline", base, "--verify-only", "--exclude", "logs/cron.log")
    assert out.returncode == 0 and "BASELINE_OK" in out.stdout
    (src / "logs" / "cron.log").unlink()
    out = _run2(src, dst, stamp, "--baseline", base, "--exclude", "logs/cron.log")
    assert out.returncode == 0 and (dst / "logs" / "cron.log").exists()


def test_an_excluded_path_listed_in_the_baseline_is_outside_the_claim(tmp_path):
    """An exclusion takes the path out of the backup's claim entirely: a
    baseline entry for it is not demanded from the backup."""
    src, dst, base, stamp = _stage1_tree(tmp_path)
    shutil.copy2(src / "logs" / "cron.log", dst / "logs" / "cron.log")
    base = _baseline(dst, tmp_path / "baseline2.sha256")
    (dst / "logs" / "cron.log").unlink()                         # freed from the backup later
    out = _run2(src, dst, stamp, "--baseline", base, "--verify-only")
    assert out.returncode == 1 and "BASELINE_MISMATCH logs/cron.log" in out.stderr
    out = _run2(src, dst, stamp, "--baseline", base, "--verify-only", "--exclude", "logs/cron.log")
    assert out.returncode == 0 and "BASELINE_OK" in out.stdout
