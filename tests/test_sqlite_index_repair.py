"""Stage 1 finding F1: a decision-freeze ledger whose primary-key index holds
stale entries (index pages persisted, table pages lost at a power cut).

tests/sqlite_damage.py builds that exact damage deterministically. The repair
(v2, after the cycle-8 GPT review) runs REINDEX IN PLACE under SQLite's
exclusive lock; these tests pin the plan's proofs, every refusal, every crash
point, rollback, the production guards and the runbook's own commands.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

from product import sqlite_index_repair as sir
from tests.sqlite_damage import STALE_KEYS as STALE, corrupt_index as _corrupt, healthy_freeze_db as _healthy

REPO = Path(__file__).resolve().parents[1]


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


def _content_of(path: Path, tmp_path: Path) -> dict:
    copy = tmp_path / f"content-{os.urandom(4).hex()}.db"
    shutil.copy2(path, copy)
    try:
        return sir._content(sir.diagnose(copy))
    finally:
        copy.unlink()


def _plan(tmp_path, target, **kw):
    return sir.plan(target, tmp_path / "staging", tmp_path / "manifest.json", **kw)


def _db_in_subdir(tmp_path, maker, **kw):
    d = tmp_path / "db"
    d.mkdir(exist_ok=True)
    return maker(d / "f.db", **kw)


def _leave_hot_wal(db: Path) -> None:
    """A committed write still only in the WAL, with no connection open (a crashed writer)."""
    con = sqlite3.connect(db)
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("INSERT INTO freezes VALUES ('dec_in_wal', 'fp', 'WAL', '2026-11-23', 'WATCH', 't', '{}')")
    con.commit()
    kept = Path(str(db) + "-wal").read_bytes()
    con.close()                                           # a clean close would checkpoint and delete it
    Path(str(db) + "-wal").write_bytes(kept)


_WRITER = """
import sqlite3, sys
con = sqlite3.connect(sys.argv[1], timeout=0.2)
try:
    con.execute("INSERT INTO freezes VALUES ('dec_' || hex(randomblob(8)), 'fp', 'X', 'd', 'W', 't', '{}')")
    con.commit()
    print("WROTE")
except sqlite3.OperationalError as exc:
    print("BLOCKED", exc)
"""


def _other_process_write(db: Path) -> str:
    return subprocess.run([sys.executable, "-c", _WRITER, str(db)], capture_output=True, text=True).stdout.strip()


# ── the finding, reproduced ──

def test_the_fixture_reproduces_the_production_signature(damaged):
    assert _integrity(damaged) == ["wrong # of entries in index sqlite_autoindex_freezes_1"]
    con = sqlite3.connect(damaged)
    assert con.execute("SELECT count(*) FROM freezes NOT INDEXED").fetchone()[0] == 200
    assert con.execute("SELECT count(*) FROM freezes INDEXED BY sqlite_autoindex_freezes_1").fetchone()[0] == 204
    row = con.execute("SELECT rowid, symbol FROM freezes WHERE freeze_id=?", (STALE[0],)).fetchone()
    con.close()
    assert row == (150, "SYM150")              # a stale key resolves to ANOTHER decision's row


def test_check_diagnoses_a_copy_and_never_touches_the_target(tmp_path, damaged):
    before = (damaged.stat().st_mtime_ns, sir._sha256(damaged))
    d = sir.check(damaged, tmp_path / "staging")
    assert d["stale_total"] == 4 and d["missing_total"] == 0 and d["lookup_mismatch_total"] == 0
    assert d["index_damage_only"] and not d["integrity_ok"] and not d["inconclusive"]
    assert sorted(e["key"][0] for e in d["indexes"]["sqlite_autoindex_freezes_1"]["stale"]) == sorted(STALE)
    assert (damaged.stat().st_mtime_ns, sir._sha256(damaged)) == before
    assert sorted(p.name for p in damaged.parent.iterdir()) == [damaged.name]


def test_check_is_inconclusive_with_a_hot_wal(tmp_path):
    db = _db_in_subdir(tmp_path, _healthy)
    _leave_hot_wal(db)
    d = sir.check(db, tmp_path / "staging")
    assert d["inconclusive"] and d["hot_journal"]
    assert sir.main(["check", "--target", str(db), "--staging", str(tmp_path / "s")]) == sir.EXIT_INCONCLUSIVE


# ── plan ──

def test_plan_proves_the_repair_before_anything_is_applied(tmp_path, damaged):
    m = _plan(tmp_path, damaged, expect_stale=4)
    assert m["ready"] and m["state"] == "planned", m["gates"]
    assert all(m["gates"].values()) and set(m["gates"]) >= set(sir.REQUIRED_GATES) | {"expected_stale"}
    assert m["before"]["tables"]["freezes"]["rows"] == 200 == m["after"]["tables"]["freezes"]["rows"]
    assert m["before"]["tables"] == m["after"]["tables"]
    assert sorted(e["key"][0] for e in m["stale_entries"]["sqlite_autoindex_freezes_1"]) == sorted(STALE)
    assert sorted(e["rowid"] for e in m["stale_entries"]["sqlite_autoindex_freezes_1"]) == [150, 151, 152, 153]
    assert _integrity(Path(m["repaired"])) == ["ok"]
    assert m["after"]["header"]["journal_header"] == "wal"
    assert m["expected_content"]["tables"] == m["original_content"]["tables"]
    assert m["expected_content"]["index_digests"] != m["original_content"]["index_digests"]


def test_plan_never_writes_the_target(tmp_path, damaged):
    st, sha = damaged.stat(), sir._sha256(damaged)
    _plan(tmp_path, damaged)
    assert (damaged.stat().st_mtime_ns, damaged.stat().st_size, sir._sha256(damaged)) == (st.st_mtime_ns, st.st_size, sha)
    assert sorted(p.name for p in damaged.parent.iterdir()) == [damaged.name]


def test_an_unexpected_stale_count_blocks_the_plan(tmp_path, damaged):
    m = _plan(tmp_path, damaged, expect_stale=3)
    assert not m["ready"] and m["gates"]["expected_stale"] is False


def test_a_healthy_database_is_nothing_to_repair(tmp_path):
    db = _db_in_subdir(tmp_path, _healthy)
    m = _plan(tmp_path, db)
    assert m["state"] == "nothing_to_repair" and not m["ready"] and m["nothing_to_repair"]
    with pytest.raises(SystemExit, match="not 'planned'"):
        sir.apply(tmp_path / "manifest.json", "owner words", target_is_copy=True)


def test_table_damage_is_refused_not_reindexed(tmp_path):
    db = _db_in_subdir(tmp_path, _healthy, n=400)
    con = sqlite3.connect(db)
    page = con.execute("PRAGMA page_size").fetchone()[0]
    root = con.execute("SELECT rootpage FROM sqlite_master WHERE name='freezes'").fetchone()[0]
    con.close()
    with open(db, "r+b") as fh:                         # an invalid b-tree page type on the TABLE root
        fh.seek((root - 1) * page)
        fh.write(b"\xff")
    try:
        m = _plan(tmp_path, db)
    except SystemExit as exc:
        assert "refusing" in str(exc)
    else:
        assert not m["ready"] and not m["gates"]["index_damage_only"], m["gates"]


def _bad_freelist(db: Path) -> None:
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
    db = _db_in_subdir(tmp_path, _corrupt if with_index_damage else _healthy)
    _bad_freelist(db)
    assert any("Freelist" in m for m in _integrity(db))
    m = _plan(tmp_path, db)
    assert not m["ready"] and m["gates"]["index_damage_only"] is False, m["gates"]
    with pytest.raises(SystemExit, match="gates did not all pass"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)


@pytest.mark.parametrize("field", ["tables", "header"])
def test_a_repair_that_changed_rows_or_header_is_never_ready(tmp_path, damaged, monkeypatch, field):
    real = sir.diagnose
    calls = {"n": 0}

    def tampered(path):
        d = real(path)
        calls["n"] += 1
        if calls["n"] == 2:                              # the REPAIRED copy
            if field == "tables":
                d["tables"]["freezes"] = {**d["tables"]["freezes"], "digest": "0" * 64}
            else:
                d["header"] = {**d["header"], "44-91": "00" * 48}        # a field REINDEX must not touch
        return d

    monkeypatch.setattr(sir, "diagnose", tampered)
    m = _plan(tmp_path, damaged)
    assert not m["ready"]
    assert m["gates"]["rows_identical" if field == "tables" else "schema_identical"] is False


def test_a_hot_wal_or_journal_is_refused_at_plan(tmp_path):
    db = _db_in_subdir(tmp_path, _corrupt)
    _leave_hot_wal(db)
    with pytest.raises(SystemExit, match="-wal"):
        _plan(tmp_path, db)
    Path(str(db) + "-wal").unlink()
    Path(str(db) + "-journal").write_bytes(b"")
    with pytest.raises(SystemExit, match="-journal"):
        _plan(tmp_path, db)


def test_a_source_that_moves_during_the_copy_is_refused(tmp_path, damaged, monkeypatch):
    real = os.stat
    calls = {"n": 0}

    def moving(path, *a, **k):
        st = real(path, *a, **k)
        if Path(path) == damaged:
            calls["n"] += 1
            if calls["n"] == 2:                          # the 'after' stat of the copy
                return types.SimpleNamespace(st_dev=st.st_dev, st_ino=st.st_ino, st_size=st.st_size,
                                             st_mtime_ns=st.st_mtime_ns + 1, st_ctime_ns=st.st_ctime_ns)
        return st

    monkeypatch.setattr(sir.os, "stat", moving)
    with pytest.raises(SystemExit, match="changed"):
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


@pytest.mark.parametrize("out_kind", ["target", "wal", "beside", "staging_beside", "unrelated_file"])
def test_plan_outputs_never_touch_the_target_or_its_directory(tmp_path, damaged, out_kind):
    original = damaged.read_bytes()
    staging, out = tmp_path / "staging", tmp_path / "m.json"
    if out_kind == "target":
        out = damaged
    elif out_kind == "wal":
        out = Path(str(damaged) + "-wal")
    elif out_kind == "beside":
        out = damaged.with_name("manifest.json")
    elif out_kind == "staging_beside":
        staging = damaged.parent / "staging"
    else:
        out.write_text("precious notes")
    expect = {"target": "would overwrite the database", "wal": "would overwrite the database",
              "beside": "must not be in the database's directory",
              "staging_beside": "must not be in the database's directory",
              "unrelated_file": "is not a sqlite_index_repair manifest"}[out_kind]
    with pytest.raises(SystemExit, match=expect):
        sir.plan(damaged, staging, out)
    assert damaged.read_bytes() == original
    if out_kind == "unrelated_file":
        assert out.read_text() == "precious notes"


def test_a_symlinked_target_is_refused(tmp_path, damaged):
    alias = damaged.with_name("alias.db")
    alias.symlink_to(damaged)
    with pytest.raises(SystemExit, match="is a symlink; name the database itself"):
        _plan(tmp_path, alias)


def test_tables_come_from_sqlite_metadata_not_a_name_pattern(tmp_path, damaged):
    con = sqlite3.connect(damaged)
    con.execute("CREATE TABLE sqliteX(k TEXT PRIMARY KEY) WITHOUT ROWID")
    con.execute("CREATE TABLE sqlite1(k TEXT)")                  # a rowid table the old LIKE pattern skipped
    con.execute("INSERT INTO sqlite1 VALUES ('evidence')")
    con.commit()
    con.close()
    for suffix in ("-wal", "-shm"):
        Path(str(damaged) + suffix).unlink(missing_ok=True)
    m = _plan(tmp_path, damaged)
    assert any("sqliteX" in u for u in m["before"]["unsupported"]) and not m["ready"]
    assert "sqlite1" in m["before"]["tables"]


def test_a_shadowed_rowid_uses_an_unshadowed_alias_or_is_refused():
    con = sqlite3.connect(":memory:")
    con.executescript("CREATE TABLE t(rowid INTEGER, evidence BLOB);"
                      "INSERT INTO t(_rowid_, rowid, evidence) VALUES (1, 7, X'AB');")
    alias = sir._rowid_alias(con, "t")
    assert alias == "_rowid_"
    before = sir._row_digest(con, "t", alias)
    con.execute("UPDATE t SET _rowid_=2")
    assert sir._row_digest(con, "t", alias) != before
    con.execute("CREATE TABLE u(rowid INTEGER, _rowid_ INTEGER, oid INTEGER)")
    assert sir._rowid_alias(con, "u") is None


def test_the_unique_probe_uses_the_index_collation(tmp_path, damaged):
    con = sqlite3.connect(damaged)
    con.executescript("CREATE TABLE labels(k TEXT COLLATE NOCASE);"
                      "CREATE UNIQUE INDEX labels_binary ON labels(k COLLATE BINARY);"
                      "INSERT INTO labels VALUES ('a'), ('A');")
    con.close()
    for suffix in ("-wal", "-shm"):
        Path(str(damaged) + suffix).unlink(missing_ok=True)
    m = _plan(tmp_path, damaged)
    assert m["after"]["indexes"]["labels_binary"]["lookup_mismatch_count"] == 0 and m["ready"], m["gates"]


# ── apply ──

def test_apply_needs_words(tmp_path, damaged):
    _plan(tmp_path, damaged)
    for words in ("", "   "):
        with pytest.raises(SystemExit, match="owner's approval"):
            sir.apply(tmp_path / "manifest.json", words, target_is_copy=True)
    assert _integrity(damaged) != ["ok"]


def test_apply_repairs_in_place_backs_up_and_is_idempotent(tmp_path, damaged):
    original_sha = sir._sha256(damaged)
    inode = damaged.stat().st_ino
    m = _plan(tmp_path, damaged, expect_stale=4)
    a = sir.apply(tmp_path / "manifest.json", "repair the freeze index", target_is_copy=True)
    assert a["state"] == "applied" and a["applied_ok"]
    assert damaged.stat().st_ino == inode                       # in place: no rename, no new file
    assert _integrity(damaged) == ["ok"]
    assert _content_of(damaged, tmp_path) == m["expected_content"]
    assert sir._sha256(Path(a["backup"])) == original_sha == m["target_sha256"]
    con = sqlite3.connect(damaged)
    assert con.execute("SELECT count(*) FROM freezes WHERE freeze_id IN (?,?,?,?)", STALE).fetchone()[0] == 0
    for fid, sym in con.execute("SELECT freeze_id, symbol FROM freezes NOT INDEXED").fetchall()[:50]:
        assert con.execute("SELECT symbol FROM freezes WHERE freeze_id=?", (fid,)).fetchone()[0] == sym
    con.close()
    again = sir.apply(tmp_path / "manifest.json", "repair the freeze index", target_is_copy=True)
    assert again.get("noop") is True
    assert sir.plan(damaged, tmp_path / "staging2", tmp_path / "m2.json")["state"] == "nothing_to_repair"


def test_another_process_cannot_write_during_the_apply(tmp_path, damaged, monkeypatch):
    """The exclusive lock holds through the checks, the backup copy (read through
    the pinned descriptor: closing any other descriptor would drop it), the
    REINDEX and the verification."""
    _plan(tmp_path, damaged)
    seen = []
    real_backup, real_reindex = sir._durable_backup, sir._reindex

    def backup_then_probe(*a, **k):
        real_backup(*a, **k)
        seen.append(("after backup", _other_process_write(damaged)))

    def reindex_then_probe(con, **kw):
        real_reindex(con, **kw)
        seen.append(("after reindex", _other_process_write(damaged)))

    monkeypatch.setattr(sir, "_durable_backup", backup_then_probe)
    monkeypatch.setattr(sir, "_reindex", reindex_then_probe)
    sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    assert [s[0] for s in seen] == ["after backup", "after reindex"]
    assert all(s[1].startswith("BLOCKED") for s in seen), seen
    assert _other_process_write(damaged) == "WROTE"             # released when apply returns


def test_a_writer_that_committed_since_the_plan_is_never_overwritten(tmp_path, damaged):
    _plan(tmp_path, damaged)
    assert _other_process_write(damaged) == "WROTE"
    with pytest.raises(SystemExit, match="changed since the plan"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    con = sqlite3.connect(damaged)
    assert con.execute("SELECT count(*) FROM freezes NOT INDEXED").fetchone()[0] == 201
    con.close()


def test_a_committed_write_left_in_the_wal_is_seen_not_lost(tmp_path):
    db = _db_in_subdir(tmp_path, _corrupt)
    _plan(tmp_path, db)
    _leave_hot_wal(db)
    with pytest.raises(SystemExit, match="changed since the plan"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    con = sqlite3.connect(db)
    assert con.execute("SELECT count(*) FROM freezes NOT INDEXED WHERE freeze_id='dec_in_wal'").fetchone()[0] == 1
    con.close()


def test_an_edited_manifest_cannot_install_another_database(tmp_path, damaged):
    m = _plan(tmp_path, damaged)
    other = _healthy(tmp_path / "replacement.db", n=1)
    m.update(repaired=str(other), repaired_sha256=sir._sha256(other))
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    con = sqlite3.connect(damaged)
    assert con.execute("SELECT count(*) FROM freezes NOT INDEXED").fetchone()[0] == 200   # not the 1-row file
    con.close()
    assert _integrity(damaged) == ["ok"]


@pytest.mark.parametrize("gate", ["rows_identical", "index_damage_only", "indexes_consistent_after"])
def test_apply_rechecks_every_required_gate_not_just_ready(tmp_path, damaged, gate):
    m = _plan(tmp_path, damaged)
    m["gates"][gate] = False                              # an edited manifest that still says ready
    (tmp_path / "manifest.json").write_text(json.dumps({**m, "ready": True}))
    with pytest.raises(SystemExit, match="gates did not all pass"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    assert _integrity(damaged) != ["ok"]


def test_copy_mode_is_never_honoured_under_volumes(tmp_path, damaged):
    m = _plan(tmp_path, damaged)
    for fake in ("/Volumes/QuantTermStorage/QuantTerm/runtime/logs/product/x.db",
                 "/volumes/QuantTermStorage/QuantTerm/runtime/logs/product/x.db"):
        (tmp_path / "fake.json").write_text(json.dumps({**m, "target": fake}))
        with pytest.raises(SystemExit, match="never honoured under /Volumes"):
            sir.apply(tmp_path / "fake.json", "words", target_is_copy=True)


def test_without_copy_mode_the_desk_check_runs_before_anything_is_written(tmp_path, damaged, monkeypatch):
    _plan(tmp_path, damaged)
    import product.bhav_store_rebuild as rb
    monkeypatch.setattr(rb, "_desk_running", lambda: ["launchd:com.quantterm.desk:123"])
    with pytest.raises(SystemExit, match="desk is running"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=False)
    assert _integrity(damaged) != ["ok"]
    assert sorted(p.name for p in damaged.parent.iterdir()) == [damaged.name]


def test_another_open_connection_refuses_the_apply(tmp_path, damaged):
    _plan(tmp_path, damaged)
    holder = sqlite3.connect(damaged)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(SystemExit, match="another connection holds"):
            sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    finally:
        holder.rollback()
        holder.close()
    assert _integrity(damaged) != ["ok"]


# ── interruption, recovery, rollback ──

def test_a_crash_before_the_reindex_leaves_the_original_and_reapplies(tmp_path, damaged, monkeypatch):
    m = _plan(tmp_path, damaged)
    monkeypatch.setattr(sir, "_reindex", lambda con, **kw: (_ for _ in ()).throw(KeyboardInterrupt("power cut")))
    with pytest.raises(KeyboardInterrupt):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    monkeypatch.undo()
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["state"] == "applying" and _content_of(damaged, tmp_path) == m["original_content"]
    backup = saved["backup"]
    a = sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)    # finishes the repair
    # every attempt writes and syncs its OWN backup; the earlier one is kept and recorded,
    # never reused (its durability is unknown) and never deleted (GPT C8.5-01, C8.5-06)
    assert a["state"] == "applied" and a["backup"] != backup and backup in a["earlier_backups"]
    assert sorted(p.name for p in damaged.parent.glob("*.pre-reindex-*")) == sorted(
        Path(n).name for n in (backup, a["backup"]))
    assert sir._sha256(Path(backup)) == sir._sha256(Path(a["backup"])) == a["backup_sha256"]


def test_a_crash_inside_the_reindex_transaction_rolls_back(tmp_path, damaged, monkeypatch):
    m = _plan(tmp_path, damaged)

    def crash_mid_transaction(con, **kw):
        con.execute("BEGIN EXCLUSIVE")
        con.execute("REINDEX")
        raise KeyboardInterrupt("power cut before COMMIT")

    monkeypatch.setattr(sir, "_reindex", crash_mid_transaction)
    with pytest.raises(KeyboardInterrupt):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    monkeypatch.undo()
    assert _content_of(damaged, tmp_path) == m["original_content"]       # the transaction never committed
    assert sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)["applied_ok"]


def test_a_crash_after_the_commit_is_finished_not_redone(tmp_path, damaged, monkeypatch):
    m = _plan(tmp_path, damaged)
    real = sir._live_content
    calls = {"n": 0}

    def crash_on_verify(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("power cut after COMMIT")
        return real(*a, **k)

    monkeypatch.setattr(sir, "_live_content", crash_on_verify)
    with pytest.raises(KeyboardInterrupt):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    monkeypatch.undo()
    assert json.loads((tmp_path / "manifest.json").read_text())["state"] == "applying"
    assert _content_of(damaged, tmp_path) == m["expected_content"]
    again = sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    assert again.get("noop") is True and again["state"] == "applied"


def test_rollback_restores_the_original_content_and_is_idempotent(tmp_path, damaged):
    m = _plan(tmp_path, damaged)
    sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    r = sir.rollback(tmp_path / "manifest.json", target_is_copy=True)
    assert r["state"] == "rolled_back" and sir._same_content(_content_of(damaged, tmp_path), m["original_content"])
    assert _integrity(damaged) == ["wrong # of entries in index sqlite_autoindex_freezes_1"]
    assert sir.rollback(tmp_path / "manifest.json", target_is_copy=True)["state"] == "rolled_back"


@pytest.mark.parametrize("state", ["applied", "applying"])
def test_rollback_never_discards_writes_made_after_the_repair(tmp_path, damaged, state):
    _plan(tmp_path, damaged)
    sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    if state == "applying":                     # as if the final manifest write was lost (GPT F1-02)
        saved = json.loads((tmp_path / "manifest.json").read_text())
        (tmp_path / "manifest.json").write_text(json.dumps({**saved, "state": "applying"}))
    assert _other_process_write(damaged) == "WROTE"
    newer = _content_of(damaged, tmp_path)
    with pytest.raises(SystemExit, match="would discard"):
        sir.rollback(tmp_path / "manifest.json", target_is_copy=True)
    assert _content_of(damaged, tmp_path) == newer


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
    assert r["rollback_ok"] and sir._same_content(_content_of(damaged, tmp_path), m["original_content"])


def test_the_cli_exit_codes(tmp_path, damaged):
    assert sir.main(["check", "--target", str(damaged), "--staging", str(tmp_path / "s")]) == 1
    assert sir.main(["plan", "--target", str(damaged), "--staging", str(tmp_path / "s"),
                     "--out", str(tmp_path / "m.json"), "--expect-stale", "4"]) == 0
    assert sir.main(["apply", "--manifest", str(tmp_path / "m.json"), "--owner-approval", "words",
                     "--target-is-copy"]) == 0
    assert sir.main(["check", "--target", str(damaged), "--staging", str(tmp_path / "s")]) == 0


# ── the runbook's own F1 commands, run as written ──

def _runbook_text() -> str:
    return (REPO / "docs/runbooks/2026-11-22-paper-upgrade-and-store-repair.md").read_text()


def _block_after(text: str, marker: str) -> str:
    return text.split(marker, 1)[1].split("```bash\n", 1)[1].split("```", 1)[0]


def _as_written(block: str, logs: Path) -> str:
    return (block.replace("\\\n", " ")
            .replace("/Users/apple/0to100/venv/bin/python", sys.executable)
            .replace("/Volumes/QuantTermStorage/QuantTerm/runtime/logs", str(logs))
            .replace("cd $M/code", f"cd {REPO}")
            .replace('"<the owner\'s words>"', '"repair freeze index (test)"'))


def test_the_runbook_f1_commands_run_as_written(tmp_path):
    logs, M, stubs = tmp_path / "runtime" / "logs", tmp_path / "M", tmp_path / "stubs"
    (logs / "product").mkdir(parents=True)
    (M / "backup").mkdir(parents=True)
    stubs.mkdir()
    for tool, rc in (("pgrep", 1), ("launchctl", 0)):              # the desk is stopped
        (stubs / tool).write_text(f"#!/bin/sh\nexit {rc}\n")
        (stubs / tool).chmod(0o755)
    target = _corrupt(logs / "product" / "counterfactuals.jsonl.freeze.db")
    env = {**os.environ, "M": str(M), "R": str(logs), "PATH": f"{stubs}:{os.environ['PATH']}"}
    env.pop("QT_RUNTIME_ROOT", None)
    text = _runbook_text()
    stage3b = text.split("## 3b. Repair", 1)[1].split("\n## 4.", 1)[0]
    blocks = {
        "plan": _block_after(text, "**F1 (read-only, optional)"),
        "stage3_copy": _block_after(text, "**F1 only, in the stage 3 SSD tier"),
        "apply": _block_after(stage3b, "**Apply**"),
        "rollback": _block_after(stage3b, "**Rollback from the recorded backup**"),
        "rollback_ssd": _block_after(stage3b, "**Rollback from the stage 3 SSD copy**"),
    }

    def run(name):
        return subprocess.run(["bash", "-eu", "-c", _as_written(blocks[name], logs)],
                              capture_output=True, text=True, env=env)

    r = run("plan")
    assert r.returncode == 0, r.stderr
    m = json.loads((M / "f1_manifest.json").read_text())
    assert m["ready"] and m["before"]["stale_total"] == 4 and m["before"]["missing_total"] == 0
    assert run("stage3_copy").returncode == 0                      # the SSD copy matches the plan
    r = run("apply")
    assert r.returncode == 0, r.stderr
    assert json.loads((M / "f1_manifest.json").read_text())["state"] == "applied"
    assert _integrity(target) == ["ok"]
    r = run("rollback")
    assert r.returncode == 0, r.stderr
    assert sir._same_content(_content_of(target, tmp_path), m["original_content"])
    sir.apply(M / "f1_manifest.json", "again", target_is_copy=True)    # in-process: the stubs cover subprocesses only
    Path(json.loads((M / "f1_manifest.json").read_text())["backup"]).unlink()
    r = run("rollback_ssd")
    assert r.returncode == 0, r.stderr
    assert sir._same_content(_content_of(target, tmp_path), m["original_content"])


def test_the_runbook_stage3_copy_fails_on_a_mismatch(tmp_path):
    logs, M = tmp_path / "runtime" / "logs", tmp_path / "M"
    (logs / "product").mkdir(parents=True)
    (M / "backup").mkdir(parents=True)
    target = _corrupt(logs / "product" / "counterfactuals.jsonl.freeze.db")
    sir.plan(target, M / "f1_staging", M / "f1_manifest.json", expect_stale=4)
    _healthy(target.with_name("other.db"), n=1).replace(target)      # the source changed after the plan
    block = _block_after(_runbook_text(), "**F1 only, in the stage 3 SSD tier")
    r = subprocess.run(["bash", "-eu", "-c", _as_written(block, logs)], capture_output=True, text=True,
                       env={**os.environ, "M": str(M), "R": str(logs)})
    assert r.returncode != 0 and "F1_BACKUP_MISMATCH" in (r.stdout + r.stderr)


def test_already_applied_is_never_claimed_over_new_damage(tmp_path, damaged):
    """GPT V2-02: the state proof carries the integrity result. A repaired ledger that
    later shows damage only integrity_check sees (a wrong freelist count: rows, index
    entries and lookups are unchanged, and the header bytes compared exclude it) must
    not be reported as 'already applied'."""
    import struct
    _plan(tmp_path, damaged)
    sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    con = sqlite3.connect(damaged)
    free = con.execute("PRAGMA freelist_count").fetchone()[0]
    con.close()
    with open(damaged, "r+b") as fh:
        fh.seek(36)
        fh.write(struct.pack(">I", free + 3))
    assert any("Freelist" in m for m in _integrity(damaged))
    with pytest.raises(SystemExit, match="neither the planned original nor the repaired"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)


def test_check_never_stages_in_the_database_directory(tmp_path, damaged):
    """check, like plan, keeps its copies away from the database's own directory."""
    with pytest.raises(SystemExit, match="must not be in the database's directory"):
        sir.check(damaged, damaged.parent / "staging")
    assert sorted(p.name for p in damaged.parent.iterdir()) == [damaged.name]


def test_an_interrupted_rollback_from_applied_unverified_still_recovers(tmp_path, damaged, monkeypatch):
    """GPT (cycle 8, part 3): rollback accepts a header-only mismatch for
    applied_unverified. A crash after it records rolling_back must not lose that
    acceptance on the retry."""
    m = _plan(tmp_path, damaged)
    m["expected_content"]["header"] = {**m["expected_content"]["header"], "44-91": "00" * 48}
    (tmp_path / "manifest.json").write_text(json.dumps(m))       # the committed header will differ
    with pytest.raises(SystemExit, match="applied_unverified"):
        sir.apply(tmp_path / "manifest.json", "words", target_is_copy=True)
    assert json.loads((tmp_path / "manifest.json").read_text())["state"] == "applied_unverified"
    real = sir._write_manifest

    def crash_after_rolling_back(path, data):
        real(path, data)
        if data.get("state") == "rolling_back":
            raise KeyboardInterrupt("power cut")

    monkeypatch.setattr(sir, "_write_manifest", crash_after_rolling_back)
    with pytest.raises(KeyboardInterrupt):
        sir.rollback(tmp_path / "manifest.json", target_is_copy=True)
    monkeypatch.undo()
    r = sir.rollback(tmp_path / "manifest.json", target_is_copy=True)
    assert r["state"] == "rolled_back"
    assert sir._same_content(_content_of(damaged, tmp_path), m["original_content"])
