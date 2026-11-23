"""Stage 1 finding F1: a decision-freeze ledger whose primary-key index holds
stale entries (index pages persisted, table pages lost at a power cut).

The fixture builds that exact damage deterministically: the real table holds
the current rows, and the PRIMARY KEY index is given the b-tree of a WITHOUT
ROWID table with the same (key, rowid) records plus stale keys that point at
rowids the table re-used. integrity_check then reports exactly what production
reported: ``wrong # of entries in index sqlite_autoindex_freezes_1``.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import types
from pathlib import Path

import pytest

from product import sqlite_index_repair as sir

from tests.sqlite_damage import STALE_KEYS as STALE, corrupt_index as _corrupt, healthy_freeze_db as _healthy


@pytest.fixture
def damaged(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    return _corrupt(live / "counterfactuals.jsonl.freeze.db")


def _integrity(path: Path) -> list[str]:
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        return [r[0] for r in con.execute("PRAGMA integrity_check")]
    finally:
        con.close()


def _plan(tmp_path, target, **kw):
    return sir.plan(target, tmp_path / "staging", tmp_path / "manifest.json", **kw)


# ── the finding, reproduced ──

def test_the_fixture_reproduces_the_production_signature(damaged):
    assert _integrity(damaged) == ["wrong # of entries in index sqlite_autoindex_freezes_1"]
    con = sqlite3.connect(damaged)
    assert con.execute("SELECT count(*) FROM freezes NOT INDEXED").fetchone()[0] == 200
    assert con.execute("SELECT count(*) FROM freezes INDEXED BY sqlite_autoindex_freezes_1").fetchone()[0] == 204
    # a stale key resolves through the index to ANOTHER decision's row
    row = con.execute("SELECT rowid, symbol FROM freezes WHERE freeze_id=?", (STALE[0],)).fetchone()
    con.close()
    assert row == (150, "SYM150")


def test_check_diagnoses_a_copy_and_never_touches_the_target(tmp_path, damaged):
    before = (damaged.stat().st_mtime_ns, sir._sha256(damaged))
    d = sir.check(damaged, tmp_path / "staging")
    assert d["stale_total"] == 4 and d["missing_total"] == 0 and d["lookup_mismatch_total"] == 0
    assert d["index_damage_only"] and not d["integrity_ok"]
    assert sorted(e["key"][0] for e in d["indexes"]["sqlite_autoindex_freezes_1"]["stale"]) == sorted(STALE)
    assert (damaged.stat().st_mtime_ns, sir._sha256(damaged)) == before
    assert not any(Path(str(damaged) + s).exists() for s in ("-wal", "-shm", "-journal"))


# ── plan ──

def test_plan_proves_the_repair_before_anything_is_applied(tmp_path, damaged):
    m = _plan(tmp_path, damaged, expect_stale=4)
    assert m["ready"] and m["state"] == "planned", m["gates"]
    assert all(m["gates"].values()) and set(m["gates"]) >= {
        "index_damage_only", "repaired_integrity_ok", "indexes_consistent_after", "rows_identical",
        "schema_identical", "foreign_keys_no_worse", "only_index_entries_removed", "expected_stale"}
    assert m["before"]["tables"]["freezes"]["rows"] == 200 == m["after"]["tables"]["freezes"]["rows"]
    assert m["before"]["tables"] == m["after"]["tables"]
    assert sorted(e["key"][0] for e in m["stale_entries"]["sqlite_autoindex_freezes_1"]) == sorted(STALE)
    assert sorted(e["rowid"] for e in m["stale_entries"]["sqlite_autoindex_freezes_1"]) == [150, 151, 152, 153]
    assert _integrity(Path(m["repaired"])) == ["ok"]
    assert m["after"]["header"]["journal_header"] == "wal"           # the journal mode survives
    assert json.loads((tmp_path / "manifest.json").read_text())["plan_id"] == m["plan_id"]


def test_plan_never_writes_the_target(tmp_path, damaged):
    st, sha = damaged.stat(), sir._sha256(damaged)
    _plan(tmp_path, damaged)
    assert (damaged.stat().st_mtime_ns, damaged.stat().st_size, sir._sha256(damaged)) == (st.st_mtime_ns, st.st_size, sha)
    assert sorted(p.name for p in damaged.parent.iterdir()) == [damaged.name]


def test_an_unexpected_stale_count_blocks_the_plan(tmp_path, damaged):
    m = _plan(tmp_path, damaged, expect_stale=3)
    assert not m["ready"] and m["gates"]["expected_stale"] is False


def test_a_healthy_database_is_nothing_to_repair(tmp_path):
    db = _healthy(tmp_path / "ok.db")
    m = _plan(tmp_path, db)
    assert m["state"] == "nothing_to_repair" and not m["ready"] and m["nothing_to_repair"]
    with pytest.raises(SystemExit, match="not 'planned'"):
        sir.apply(tmp_path / "manifest.json", "owner words", target_is_copy=True)


def test_table_damage_is_refused_not_reindexed(tmp_path):
    db = _healthy(tmp_path / "t.db", n=400)
    con = sqlite3.connect(db)
    page = con.execute("PRAGMA page_size").fetchone()[0]
    root = con.execute("SELECT rootpage FROM sqlite_master WHERE name='freezes'").fetchone()[0]
    con.close()
    with open(db, "r+b") as fh:                         # an invalid b-tree page type on the TABLE root
        fh.seek((root - 1) * page)
        fh.write(b"\xff")
    # the only safe outcomes: refused outright, or planned as not ready (damage not index-only)
    try:
        m = _plan(tmp_path, db)
    except SystemExit as exc:
        assert "refusing" in str(exc)
    else:
        assert not m["ready"] and not m["gates"]["index_damage_only"], m["gates"]
        with pytest.raises(SystemExit, match="gates did not all pass"):
            sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)


def test_a_hot_wal_or_journal_is_refused_at_plan(tmp_path, damaged):
    Path(str(damaged) + "-wal").write_bytes(b"x" * 32)
    with pytest.raises(SystemExit, match="-wal"):
        _plan(tmp_path, damaged)
    Path(str(damaged) + "-wal").unlink()
    Path(str(damaged) + "-journal").write_bytes(b"")
    with pytest.raises(SystemExit, match="-journal"):
        _plan(tmp_path, damaged)


def test_a_source_that_moves_during_the_copy_is_refused(tmp_path, damaged, monkeypatch):
    real = os.stat
    calls = {"n": 0}

    def moving(path, *a, **k):
        st = real(path, *a, **k)
        if Path(path) == damaged:
            calls["n"] += 1
            if calls["n"] == 2:                          # the 'after' stat of the copy
                return types.SimpleNamespace(st_size=st.st_size, st_mtime_ns=st.st_mtime_ns + 1,
                                             st_ctime_ns=st.st_ctime_ns)
        return st

    monkeypatch.setattr(sir.os, "stat", moving)
    with pytest.raises(SystemExit, match="changed while it was copied"):
        sir.check(damaged, tmp_path / "staging")


@pytest.mark.parametrize("where", ["staging", "out"])
def test_plan_output_is_never_written_under_volumes(tmp_path, damaged, where):
    staging, out = tmp_path / "staging", tmp_path / "m.json"
    if where == "staging":
        staging = Path("/Volumes/QuantTermStorage/QuantTerm/staging")
    else:
        out = Path("/volumes/QuantTermStorage/m.json")             # case-insensitive (cycle 7)
    with pytest.raises(SystemExit, match="under /Volumes"):
        sir.plan(damaged, staging, out)


# ── apply ──

def test_apply_needs_words(tmp_path, damaged):
    _plan(tmp_path, damaged)
    for words in ("", "   "):
        with pytest.raises(SystemExit, match="owner's approval"):
            sir.apply(tmp_path / "manifest.json", words, target_is_copy=True)
    assert _integrity(damaged) != ["ok"]


def test_apply_repairs_backs_up_and_is_idempotent(tmp_path, damaged):
    original_sha = sir._sha256(damaged)
    original_rows = sir.diagnose(sir.check(damaged, tmp_path / "s0")["copy"])["tables"]
    _plan(tmp_path, damaged, expect_stale=4)
    m = sir.apply(tmp_path / "manifest.json", "repair the freeze index", target_is_copy=True)
    assert m["state"] == "applied" and m["applied_ok"]
    assert _integrity(damaged) == ["ok"] and sir._sha256(damaged) == m["repaired_sha256"]
    assert sir._sha256(Path(m["backup"])) == original_sha == m["target_sha256"]
    assert sir.diagnose(sir.check(damaged, tmp_path / "s1")["copy"])["tables"] == original_rows
    con = sqlite3.connect(damaged)
    assert con.execute("SELECT count(*) FROM freezes WHERE freeze_id IN (?,?,?,?)", STALE).fetchone()[0] == 0
    for fid, sym in con.execute("SELECT freeze_id, symbol FROM freezes NOT INDEXED").fetchall()[:50]:
        assert con.execute("SELECT symbol FROM freezes WHERE freeze_id=?", (fid,)).fetchone()[0] == sym
    con.close()
    for suffix in ("-wal", "-shm"):
        Path(str(damaged) + suffix).unlink(missing_ok=True)
    again = sir.apply(tmp_path / "manifest.json", "repair the freeze index", target_is_copy=True)
    assert again.get("noop") is True
    assert sir.plan(damaged, tmp_path / "staging2", tmp_path / "m2.json")["state"] == "nothing_to_repair"


def test_apply_refuses_a_target_changed_since_the_plan(tmp_path, damaged):
    _plan(tmp_path, damaged)
    con = sqlite3.connect(damaged)
    con.execute("INSERT INTO freezes VALUES ('dec_new', 'fp', 'NEW', '2026-11-23', 'WATCH', 't', '{}')")
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    for suffix in ("-wal", "-shm"):
        Path(str(damaged) + suffix).unlink(missing_ok=True)
    with pytest.raises(SystemExit, match="changed since the plan"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)


def test_apply_refuses_a_changed_repaired_copy(tmp_path, damaged):
    m = _plan(tmp_path, damaged)
    with open(m["repaired"], "r+b") as fh:
        fh.seek(5000)
        fh.write(b"\x00\x01")
    with pytest.raises(SystemExit, match="repaired copy changed"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    assert sir._sha256(damaged) == m["target_sha256"]


def test_apply_refuses_a_hot_wal_that_appeared_after_the_plan(tmp_path, damaged):
    _plan(tmp_path, damaged)
    Path(str(damaged) + "-wal").write_bytes(b"x" * 64)
    with pytest.raises(SystemExit, match="-wal"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    assert sorted(p.name for p in damaged.parent.iterdir()) == [damaged.name, damaged.name + "-wal"]


def test_copy_mode_is_never_honoured_under_volumes(tmp_path, damaged):
    m = _plan(tmp_path, damaged)
    for fake in ("/Volumes/QuantTermStorage/QuantTerm/runtime/logs/product/x.db",
                 "/volumes/QuantTermStorage/QuantTerm/runtime/logs/product/x.db"):
        m2 = {**m, "target": fake}
        (tmp_path / "fake.json").write_text(json.dumps(m2))
        with pytest.raises(SystemExit, match="never honoured under /Volumes|canonical"):
            sir.apply(tmp_path / "fake.json", "words", target_is_copy=True)


def test_without_copy_mode_the_desk_check_runs(tmp_path, damaged, monkeypatch):
    _plan(tmp_path, damaged)
    import product.bhav_store_rebuild as rb
    monkeypatch.setattr(rb, "_desk_running", lambda: ["launchd:com.quantterm.desk:123"])
    with pytest.raises(SystemExit, match="desk is running"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=False)
    assert _integrity(damaged) != ["ok"]
    # refused BEFORE anything was written beside the live target
    assert sorted(p.name for p in damaged.parent.iterdir()) == [damaged.name]


# ── interruption, recovery, rollback ──

def test_an_apply_interrupted_before_the_swap_rolls_back_cleanly(tmp_path, damaged, monkeypatch):
    m = _plan(tmp_path, damaged)
    real = os.replace

    def crash_on_swap(src, dst):
        if Path(dst) == damaged:                          # the swap itself
            raise KeyboardInterrupt("power cut")
        return real(src, dst)

    monkeypatch.setattr(sir.os, "replace", crash_on_swap)
    with pytest.raises(KeyboardInterrupt):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    monkeypatch.setattr(sir.os, "replace", real)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["state"] == "swapping" and sir._sha256(damaged) == m["target_sha256"]
    r = sir.rollback(tmp_path / "manifest.json", target_is_copy=True)
    assert r["state"] == "rolled_back" and sir._sha256(damaged) == m["target_sha256"]
    assert not list(damaged.parent.glob(".reindex-*"))      # the verified copy was not left behind


def test_a_crash_while_recording_swapping_leaves_nothing_behind(tmp_path, damaged, monkeypatch):
    m = _plan(tmp_path, damaged)
    real = sir._write_manifest

    def crash(path, data):
        if data.get("state") == "swapping":
            raise KeyboardInterrupt("power cut")
        return real(path, data)

    monkeypatch.setattr(sir, "_write_manifest", crash)
    with pytest.raises(KeyboardInterrupt):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    monkeypatch.undo()
    assert json.loads((tmp_path / "manifest.json").read_text())["state"] == "planned"
    assert sir._sha256(damaged) == m["target_sha256"]
    assert not list(damaged.parent.glob(".reindex-*")) and not list(damaged.parent.glob(".backup-*"))
    assert sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)["applied_ok"]


def test_an_apply_interrupted_after_the_swap_is_recovered_then_reapplied(tmp_path, damaged, monkeypatch):
    m = _plan(tmp_path, damaged)
    monkeypatch.setattr(sir, "_verify_live", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt("crash")))
    with pytest.raises(KeyboardInterrupt):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    monkeypatch.undo()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["state"] == "swapping" and sir._sha256(damaged) == m["repaired_sha256"]
    with pytest.raises(SystemExit, match="not 'planned'"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    r = sir.rollback(tmp_path / "manifest.json", target_is_copy=True)
    assert r["state"] == "rolled_back" and r["rolled_back_from"] == "interrupted apply"
    assert sir._sha256(damaged) == m["target_sha256"]
    m2 = sir.plan(damaged, tmp_path / "staging2", tmp_path / "m2.json")
    assert m2["ready"]
    assert sir.apply(tmp_path / "m2.json", "words", target_is_copy=True)["applied_ok"]


def test_rollback_of_a_verified_apply_and_its_refusals(tmp_path, damaged):
    m = _plan(tmp_path, damaged)
    sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    r = sir.rollback(tmp_path / "manifest.json", target_is_copy=True)
    assert r["state"] == "rolled_back" and sir._sha256(damaged) == m["target_sha256"]
    again = sir.rollback(tmp_path / "manifest.json", target_is_copy=True)    # repeat: no-op
    assert again["state"] == "rolled_back"


def test_rollback_never_discards_writes_made_after_a_verified_apply(tmp_path, damaged):
    _plan(tmp_path, damaged)
    sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    con = sqlite3.connect(damaged)
    con.execute("INSERT INTO freezes VALUES ('dec_after', 'fp', 'AFTER', '2026-11-24', 'WATCH', 't', '{}')")
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    for suffix in ("-wal", "-shm"):
        Path(str(damaged) + suffix).unlink(missing_ok=True)
    with pytest.raises(SystemExit, match="would discard"):
        sir.rollback(tmp_path / "manifest.json", target_is_copy=True)


def test_a_live_rollback_needs_words(tmp_path, damaged):
    _plan(tmp_path, damaged)
    sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    with pytest.raises(SystemExit, match="live rollback needs the owner's approval"):
        sir.rollback(tmp_path / "manifest.json", target_is_copy=False)


def test_rollback_uses_a_verified_external_backup_when_the_recorded_one_is_gone(tmp_path, damaged):
    m = _plan(tmp_path, damaged)
    ssd = tmp_path / "ssd_copy.db"
    shutil.copy2(damaged, ssd)
    applied = sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    Path(applied["backup"]).unlink()
    with pytest.raises(SystemExit, match="nothing to roll back"):
        sir.rollback(tmp_path / "manifest.json", target_is_copy=True)
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not it")
    with pytest.raises(SystemExit, match="does not match the pre-repair"):
        sir.rollback(tmp_path / "manifest.json", target_is_copy=True, backup_path=bad)
    r = sir.rollback(tmp_path / "manifest.json", target_is_copy=True, backup_path=ssd)
    assert r["rollback_ok"] and sir._sha256(damaged) == m["target_sha256"]


def test_the_cli_exit_codes(tmp_path, damaged):
    assert sir.main(["check", "--target", str(damaged), "--staging", str(tmp_path / "s")]) == 1
    assert sir.main(["plan", "--target", str(damaged), "--staging", str(tmp_path / "s"),
                     "--out", str(tmp_path / "m.json"), "--expect-stale", "4"]) == 0
    assert sir.main(["apply", "--manifest", str(tmp_path / "m.json"), "--owner-approval", "words",
                     "--target-is-copy"]) == 0
    assert sir.main(["check", "--target", str(damaged), "--staging", str(tmp_path / "s")]) == 0


def _bad_freelist(db: Path) -> None:
    """Readable, NON-index damage: the header's freelist count is wrong."""
    import struct
    con = sqlite3.connect(db)
    con.execute("DELETE FROM freezes WHERE rowid > 120")
    con.commit()
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    free = con.execute("PRAGMA freelist_count").fetchone()[0]
    con.close()
    for suffix in ("-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    assert free > 0
    with open(db, "r+b") as fh:
        fh.seek(36)
        fh.write(struct.pack(">I", free + 3))


@pytest.mark.parametrize("with_index_damage", [False, True])
def test_damage_beyond_the_indexes_is_never_ready(tmp_path, with_index_damage):
    db = (_corrupt if with_index_damage else _healthy)(tmp_path / "f.db")
    _bad_freelist(db)
    assert any("Freelist" in m for m in _integrity(db))
    m = _plan(tmp_path, db)
    assert not m["ready"] and m["gates"]["index_damage_only"] is False, m["gates"]
    with pytest.raises(SystemExit, match="gates did not all pass"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)


def test_a_repair_that_changed_any_row_is_never_ready(tmp_path, damaged, monkeypatch):
    real = sir.diagnose
    calls = {"n": 0}

    def tampered(path):
        d = real(path)
        calls["n"] += 1
        if calls["n"] == 2:                              # the REPAIRED copy
            d["tables"]["freezes"] = {**d["tables"]["freezes"], "digest": "0" * 64}
        return d

    monkeypatch.setattr(sir, "diagnose", tampered)
    m = _plan(tmp_path, damaged)
    assert not m["ready"] and m["gates"]["rows_identical"] is False


# ── the runbook's own F1 commands (stage 2 plan, stage 3b apply/check, rollback), run as written ──

REPO = Path(__file__).resolve().parents[1]


def _runbook_f1_blocks():
    text = (REPO / "docs/runbooks/2026-11-22-paper-upgrade-and-store-repair.md").read_text()
    plan = text.split("**F1 (read-only, optional)", 1)[1].split("```bash\n", 1)[1].split("```", 1)[0]
    stage = text.split("## 3b. Repair", 1)[1].split("\n## 4.", 1)[0]
    apply = stage.split("```bash\n", 1)[1].split("```", 1)[0]
    rollback = stage.split("**Rollback**", 1)[1].split("```bash\n", 1)[1].split("```", 1)[0]
    return plan, apply, rollback


def _as_written(block: str, logs: Path) -> str:
    import sys as _sys
    return (block.replace("\\\n", " ")
            .replace("/Users/apple/0to100/venv/bin/python", _sys.executable)
            .replace("/Volumes/QuantTermStorage/QuantTerm/runtime/logs", str(logs))
            .replace("cd $M/code", f"cd {REPO}")
            .replace('"<the owner\'s words>"', '"repair freeze index (test)"')
            .replace(" [--backup $M/backup/f1_freeze_pre_repair.db]", ""))


def test_the_runbook_f1_commands_run_as_written(tmp_path):
    import subprocess
    logs, M, stubs = tmp_path / "runtime" / "logs", tmp_path / "M", tmp_path / "stubs"
    (logs / "product").mkdir(parents=True)
    M.mkdir()
    stubs.mkdir()
    for tool, rc in (("pgrep", 1), ("launchctl", 0)):              # the desk is stopped
        (stubs / tool).write_text(f"#!/bin/sh\nexit {rc}\n")
        (stubs / tool).chmod(0o755)
    target = _corrupt(logs / "product" / "counterfactuals.jsonl.freeze.db")
    original = sir._sha256(target)
    env = {**os.environ, "M": str(M), "R": str(logs), "PATH": f"{stubs}:{os.environ['PATH']}"}
    env.pop("QT_RUNTIME_ROOT", None)
    plan, apply, rollback = _runbook_f1_blocks()

    def run(block):
        return subprocess.run(["bash", "-eu", "-c", _as_written(block, logs)], capture_output=True, text=True, env=env)

    r = run(plan)
    assert r.returncode == 0, r.stderr
    m = json.loads((M / "f1_manifest.json").read_text())
    assert m["ready"] and m["before"]["stale_total"] == 4 and m["before"]["missing_total"] == 0
    r = run(apply)
    assert r.returncode == 0, r.stderr                   # apply, then check exits 0
    assert json.loads((M / "f1_manifest.json").read_text())["state"] == "applied"
    assert _integrity(target) == ["ok"]
    r = run(rollback)
    assert r.returncode == 0, r.stderr
    assert sir._sha256(target) == original
