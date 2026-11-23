"""Repair a SQLite database whose INDEXES disagree with their tables (Stage 1 finding F1).

    plan     --target <db> --staging <dir you own> --out <manifest.json> [--expect-stale N]
    apply    --manifest <manifest.json> --owner-approval "<words>" [--target-is-copy]
    rollback --manifest <manifest.json> --owner-approval "<words>" [--target-is-copy] [--backup <copy>]
    check    --target <db> --staging <dir you own>        (read-only diagnosis of a byte copy)

``plan`` and ``check`` never open the target with SQLite. They take a stable byte
copy: one read through one descriptor that also hashes it, with the descriptor's
and the path's (device, inode, size, mtime, ctime) identical before and after,
and no hot ``-wal``/``-journal`` before or after. ``plan`` then runs the SAME
repair routine ``apply`` runs, on a second copy, and proves:

* the damage is index-only (every ``integrity_check`` message is one of the index
  forms); table damage is refused, because REINDEX cannot repair it;
* after the repair ``integrity_check`` is ``ok``; every index holds exactly its
  table's (key, rowid) entries, read through a scan SQLite reports as COVERING;
  every unique key resolves through its index, with the index's own collation,
  to its own row;
* every row is unchanged: the same rowids (through an unshadowed rowid alias)
  and the same content digest over every column, for every rowid table SQLite's
  ``table_list`` reports (``sqlite_sequence`` and ``sqlite_stat*`` included);
* the schema (every ``sqlite_master`` entry except root pages), the header
  bytes REINDEX must not touch (0-23 and 40-91), and the full foreign-key
  violation list are unchanged.

``apply`` repairs the LIVE database IN PLACE, inside SQLite's own locking: no
file is renamed or replaced. It opens the target with ``locking_mode=EXCLUSIVE``
(no other connection can read or write until it closes), full-sync barriers and
``synchronous=FULL``; checkpoints; verifies the target's sha256 is the planned
one; writes and syncs a byte backup; records ``applying`` durably; runs REINDEX
in one transaction; checkpoints; and proves the result on a byte copy against
the plan's digests, all before releasing the lock. A crash rolls the
transaction back. Re-running ``apply`` after a crash finishes or redoes the
repair, after success it is a no-op; a healthy database plans as
``nothing_to_repair``.

``rollback`` restores the backup through SQLite's backup API under the same
exclusive lock, only while the target holds exactly the repaired content: in
every state an unknown content digest is refused, so writes made after the
repair are never discarded. Both need the owner's words for a live target and a
stopped desk; copy mode is never honoured under /Volumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat as S
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from product.bhav_store_rebuild import (
    _check_desk,
    _fsync_dir,
    _full_fsync,
    _held_open,
    _not_live,
    _on_live_volume,
    _sha256,
    _write_manifest,
)

TOOL = "sqlite_index_repair"
VERSION = 2
INDEX_DAMAGE = (
    re.compile(r"^wrong # of entries in index \S+$"),
    re.compile(r"^row \d+ missing from index \S+$"),
    re.compile(r"^non-unique entry in index \S+$"),
)
# Header bytes REINDEX may change (measured, cycle 8): change counter 24-27, page
# count 28-31, freelist 32-39, version-valid-for 92-95, SQLite version 96-99.
# The schema cookie (40-43) is kept apart: REINDEX leaves it, but a restore
# through the backup API bumps it (it is a change counter, not content).
HEADER_FIXED = ((0, 24), (40, 44), (44, 92))
SCHEMA_COOKIE = "40-43"
HEADER_PRAGMAS = ("page_size", "user_version", "application_id", "encoding", "auto_vacuum")
_LIST_LIMIT = 200
EXIT_INCONCLUSIVE = 3


# ── identity, copies, sidecars ───────────────────────────────────────────────

def _ident(st: os.stat_result) -> tuple[int, int, int, int, int]:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _sidecars(db: Path) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for suffix in ("-wal", "-shm", "-journal"):
        p = Path(str(db) + suffix)
        out[suffix] = p.stat().st_size if p.exists() else None
    return out


def _hot(side: dict[str, int | None]) -> list[str]:
    """Sidecars that mean the main file alone is not the database."""
    hot = []
    if side.get("-wal"):
        hot.append(f"-wal ({side['-wal']} bytes)")
    if side.get("-journal") is not None:
        hot.append("-journal")
    return hot


def _stable_copy(src: Path, dst: Path) -> dict[str, Any]:
    """One read of ``src`` through one descriptor that hashes and writes ``dst``.

    Refused when the path's and the descriptor's (device, inode, size, mtime,
    ctime) differ, or move, between before and after (GPT F1-05, cycle 8).
    """
    path_before = os.stat(src)
    h = hashlib.sha256()
    with open(src, "rb") as inp, open(dst, "xb") as out:
        fd_before = os.fstat(inp.fileno())
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
            out.write(chunk)
        out.flush()
        _full_fsync(out.fileno())
        fd_after = os.fstat(inp.fileno())
    path_after = os.stat(src)
    if not (_ident(path_before) == _ident(fd_before) == _ident(fd_after) == _ident(path_after)):
        dst.unlink(missing_ok=True)
        raise SystemExit(f"refusing: {src} changed (or was replaced) while it was copied; a writer is active")
    sha = h.hexdigest()
    if _sha256(dst) != sha:
        dst.unlink(missing_ok=True)
        raise SystemExit(f"refusing: the copy of {src.name} does not hash to the bytes read")
    return {"sha256": sha, "size": path_before.st_size, "mtime_ns": path_before.st_mtime_ns}


def _quiet_copy(src: Path, dst: Path) -> dict[str, Any]:
    """A stable copy taken with no hot sidecar before or after it."""
    before = _sidecars(src)
    if _hot(before):
        raise SystemExit(f"refusing: {src.name} has {_hot(before)}: the main file alone is not the database "
                         "(stop its writers so SQLite checkpoints, then retry)")
    copy = _stable_copy(src, dst)
    after = _sidecars(src)
    if _hot(after) or after.get("-wal") != before.get("-wal") or after.get("-journal") != before.get("-journal"):
        dst.unlink(missing_ok=True)
        raise SystemExit(f"refusing: {src.name}'s journal changed while it was copied; a writer is active")
    copy["sidecars"] = before
    return copy


# ── diagnosis (on copies only) ───────────────────────────────────────────────

def _open_ro(path: Path) -> sqlite3.Connection:
    # immutable: no locks, no -wal/-shm created; exact because the copy has no WAL
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _val(v: Any) -> tuple[str, Any]:
    return (type(v).__name__, v.hex() if isinstance(v, (bytes, bytearray, memoryview)) else v)


def _unval(v: tuple[str, Any]) -> Any:
    return bytes.fromhex(v[1]) if v[0] in ("bytes", "bytearray", "memoryview") else v[1]


def _tables(con: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """Every rowid table in ``main`` (internal sqlite_* included), and unsupported objects.

    SQLite's own ``table_list`` metadata, not a name pattern: ``LIKE 'sqlite_%'``
    treats ``_`` as a wildcard and skipped a user table named ``sqliteX`` (GPT F1-06).
    """
    tables, unsupported = [], []
    for row in con.execute("PRAGMA main.table_list"):
        _schema, name, kind, _ncol, wr, _strict = row
        if name in ("sqlite_schema", "sqlite_master", "sqlite_temp_schema", "sqlite_temp_master"):
            continue
        if kind == "view":
            continue
        if kind != "table":
            unsupported.append(f"{kind} {name}")
        elif wr:
            unsupported.append(f"table {name} (WITHOUT ROWID)")
        else:
            tables.append(name)
    return sorted(tables), unsupported


def _rowid_alias(con: sqlite3.Connection, table: str) -> str | None:
    """A rowid spelling no user column shadows (GPT F1-06), or None."""
    cols = {str(r[1]).lower() for r in con.execute(f"PRAGMA table_xinfo({_q(table)})")}
    for alias in ("_rowid_", "rowid", "oid"):
        if alias not in cols:
            return alias
    return None


def _index_columns(con: sqlite3.Connection, index: str) -> list[tuple[str, str]] | None:
    """(column, collation) of the plain key columns of ``index``; None for an expression index."""
    cols = []
    for row in con.execute(f"PRAGMA index_xinfo({_q(index)})"):
        _seq, cid, name, _desc, coll, key = row
        if not key:
            continue
        if cid == -2 or name is None:
            return None
        cols.append((name, coll or "BINARY"))
    return cols


def _row_digest(con: sqlite3.Connection, table: str, alias: str) -> dict[str, Any]:
    h = hashlib.sha256()
    n = 0
    for row in con.execute(f"SELECT {alias}, * FROM {_q(table)} NOT INDEXED ORDER BY {alias}"):
        h.update(json.dumps([_val(v) for v in row], default=str, separators=(",", ":")).encode())
        h.update(b"\n")
        n += 1
    return {"rows": n, "digest": h.hexdigest()}


def _covering(con: sqlite3.Connection, sql: str) -> bool:
    plan = " ".join(str(r[-1]) for r in con.execute("EXPLAIN QUERY PLAN " + sql))
    return "COVERING INDEX" in plan.upper()


def _index_check(con: sqlite3.Connection, table: str, alias: str, index: str,
                 cols: list[tuple[str, str]], unique: bool) -> dict[str, Any] | None:
    sel = ", ".join(_q(c) for c, _ in cols)
    via_index = f"SELECT {sel}, {alias} FROM {_q(table)} INDEXED BY {_q(index)}"
    if not _covering(con, via_index):
        return None                         # the values would not come from the index b-tree
    idx = Counter(tuple(_val(v) for v in r) for r in con.execute(via_index))
    tab = Counter(tuple(_val(v) for v in r) for r in con.execute(
        f"SELECT {sel}, {alias} FROM {_q(table)} NOT INDEXED"))
    stale = list((idx - tab).elements())
    missing = list((tab - idx).elements())
    digest = hashlib.sha256(json.dumps(sorted(idx.elements(), key=repr), default=str).encode()).hexdigest()
    lookup_bad = []
    if unique:
        where = " AND ".join(f"{_q(c)} = ? COLLATE {coll}" for c, coll in cols)   # GPT F1-09
        for key in tab:
            values = [_unval(v) for v in key[:-1]]
            if any(v is None for v in values):
                continue                    # NULLs are not unique keys
            got = [r[0] for r in con.execute(
                f"SELECT {alias} FROM {_q(table)} INDEXED BY {_q(index)} WHERE {where}", values)]
            if got != [key[-1][1]]:
                lookup_bad.append({"key": [v[1] for v in key[:-1]], "rowid": key[-1][1], "resolved_to": got})
    return {"table": table, "columns": [c for c, _ in cols], "collations": [k for _, k in cols],
            "unique": unique, "index_entries": sum(idx.values()), "table_rows": sum(tab.values()),
            "entries_digest": digest,
            "stale": [{"key": [v[1] for v in e[:-1]], "rowid": e[-1][1]} for e in stale[:_LIST_LIMIT]],
            "stale_count": len(stale),
            "missing": [{"key": [v[1] for v in e[:-1]], "rowid": e[-1][1]} for e in missing[:_LIST_LIMIT]],
            "missing_count": len(missing),
            "lookup_mismatches": lookup_bad[:_LIST_LIMIT], "lookup_mismatch_count": len(lookup_bad)}


def _header(path: Path) -> dict[str, str]:
    with open(path, "rb") as fh:
        raw = fh.read(100)
    return {f"{a}-{b - 1}": raw[a:b].hex() for a, b in HEADER_FIXED} | {
        "journal_header": {1: "rollback", 2: "wal"}.get(raw[18], str(raw[18]))}


def diagnose(path: str | Path) -> dict[str, Any]:
    """Read-only diagnosis of a database COPY (never call it on a live file)."""
    path = Path(path)
    con = _open_ro(path)
    try:
        integrity = [r[0] for r in con.execute("PRAGMA integrity_check(1000)")]
        schema = [list(r) for r in con.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name")]
        settings = {p: con.execute(f"PRAGMA {p}").fetchone()[0] for p in HEADER_PRAGMAS}
        fk = sorted([list(r) for r in con.execute("PRAGMA foreign_key_check")], key=repr)
        tables, unsupported = _tables(con)
        digests: dict[str, Any] = {}
        indexes: dict[str, Any] = {}
        for t in tables:
            alias = _rowid_alias(con, t)
            if alias is None:
                unsupported.append(f"table {t} (every rowid spelling is a column)")
                continue
            digests[t] = _row_digest(con, t, alias)
            for _seq, name, unique, _origin, partial in con.execute(f"PRAGMA index_list({_q(t)})").fetchall():
                cols = _index_columns(con, name)
                if cols is None or partial:
                    unsupported.append(f"index {name} ({'expression' if cols is None else 'partial'})")
                    continue
                checked = _index_check(con, t, alias, name, cols, bool(unique))
                if checked is None:
                    unsupported.append(f"index {name} (no covering scan)")
                    continue
                indexes[name] = checked
    finally:
        con.close()
    index_only = all(any(p.match(m) for p in INDEX_DAMAGE) for m in integrity) if integrity != ["ok"] else True
    return {
        "integrity": integrity,
        "integrity_ok": integrity == ["ok"],
        "index_damage_only": index_only,
        "schema": schema,
        "schema_sha256": hashlib.sha256(json.dumps(schema).encode()).hexdigest(),
        "settings": settings,
        "header": _header(path),
        "foreign_key_violations": fk[:_LIST_LIMIT],
        "foreign_key_violation_count": len(fk),
        "foreign_key_sha256": hashlib.sha256(json.dumps(fk, default=str).encode()).hexdigest(),
        "tables": digests,
        "indexes": indexes,
        "unsupported": unsupported,
        "stale_total": sum(i["stale_count"] for i in indexes.values()),
        "missing_total": sum(i["missing_count"] for i in indexes.values()),
        "lookup_mismatch_total": sum(i["lookup_mismatch_count"] for i in indexes.values()),
    }


def _diagnose_or_refuse(path: Path) -> dict[str, Any]:
    try:
        return diagnose(path)
    except sqlite3.DatabaseError as exc:
        raise SystemExit(f"refusing: the copy cannot be read as a database ({exc}); "
                         "this is not index-only damage and REINDEX cannot repair it") from None


def _consistent(d: dict[str, Any]) -> bool:
    return d["integrity_ok"] and not (d["stale_total"] or d["missing_total"] or d["lookup_mismatch_total"])


def _same_content(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Equal content, the schema cookie aside."""
    strip = lambda c: {**c, "header": {k: v for k, v in c["header"].items() if k != SCHEMA_COOKIE}}
    return strip(a) == strip(b)


def _content(d: dict[str, Any]) -> dict[str, Any]:
    """What a repair must preserve (tables) and what it produces (index entries)."""
    return {"tables": d["tables"], "schema_sha256": d["schema_sha256"], "header": d["header"],
            "settings": d["settings"], "foreign_key_sha256": d["foreign_key_sha256"],
            "index_digests": {n: i["entries_digest"] for n, i in d["indexes"].items()}}


# ── paths ────────────────────────────────────────────────────────────────────

def _entry(target: Path) -> Path:
    """The directory entry: parent resolved, final name kept (GPT C7-04)."""
    return Path(target).parent.resolve() / Path(target).name


def _check_target(target: Path) -> Path:
    entry = _entry(target)
    try:
        st = os.lstat(entry)
    except FileNotFoundError:
        raise SystemExit(f"refusing: no database at {target}") from None
    if S.S_ISLNK(st.st_mode):
        raise SystemExit(f"refusing: {target} is a symlink; name the database itself (GPT F1-04)")
    if not S.S_ISREG(st.st_mode):
        raise SystemExit(f"refusing: {target} is not a regular file")
    return entry


def _check_outputs(entry: Path, staging: Path, out: Path) -> None:
    """Outputs live on a disk the operator owns, away from the target's directory (GPT F1-04)."""
    _not_live(staging, "staging")
    _not_live(out.parent, "the manifest")
    target_dir = entry.parent
    protected = {entry, *(Path(str(entry) + s) for s in ("-wal", "-shm", "-journal"))}
    for what, p in (("the manifest", out), ("staging", staging)):
        resolved = Path(p).parent.resolve() / Path(p).name
        # one rule: nothing in the database's directory (which also holds its sidecars)
        if resolved == target_dir or target_dir in resolved.parents or resolved.parent == target_dir:
            if resolved in protected or resolved.resolve() in protected:
                raise SystemExit(f"refusing: {what} would overwrite the database or its journal")
            raise SystemExit(f"refusing: {what} must not be in the database's directory ({target_dir})")
    if out.exists():
        try:
            prior = json.loads(out.read_text())
        except Exception:
            prior = None
        if not (isinstance(prior, dict) and prior.get("tool") == TOOL):
            raise SystemExit(f"refusing: {out} exists and is not a {TOOL} manifest")


# ── the repair routine (one code path for plan's copy and apply's live target) ──

def _maintenance_connection(path: Path) -> sqlite3.Connection:
    """Exclusive, full-sync connection: no other connection can read or write until it closes."""
    con = sqlite3.connect(str(path), timeout=2.0, isolation_level=None)
    try:
        con.execute("PRAGMA busy_timeout=2000")
        con.execute("PRAGMA locking_mode=EXCLUSIVE")
        con.execute("PRAGMA synchronous=FULL")
        con.execute("PRAGMA fullfsync=ON")
        con.execute("PRAGMA checkpoint_fullfsync=ON")
        con.execute("BEGIN EXCLUSIVE")          # takes the lock; kept after COMMIT in EXCLUSIVE mode
        con.execute("COMMIT")
    except sqlite3.OperationalError as exc:
        con.close()
        raise SystemExit(f"refusing: another connection holds {path.name} ({exc}); stop its writers first") from None
    return con


def _checkpoint(con: sqlite3.Connection) -> None:
    busy, _log, _done = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    if busy:
        raise SystemExit("refusing: the WAL could not be checkpointed (busy)")


def _reindex(con: sqlite3.Connection) -> None:
    con.execute("BEGIN EXCLUSIVE")
    try:
        con.execute("REINDEX")
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    _checkpoint(con)


# ── plan ─────────────────────────────────────────────────────────────────────

def check(target: str | Path, staging: str | Path) -> dict[str, Any]:
    entry = _check_target(Path(target))
    staging = _not_live(Path(staging), "staging")
    work = staging / f"check-{time.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True)
    side = _sidecars(entry)
    if _hot(side):
        # the main file alone is not the database: a verdict on it would be wrong (GPT F1-05)
        return {"target": str(entry), "sidecars": side, "hot_journal": _hot(side), "inconclusive": True,
                "integrity": [], "stale_total": None, "missing_total": None, "lookup_mismatch_total": None,
                "unsupported": [], "target_sha256": None}
    copy = _quiet_copy(entry, work / "copy.db")
    out = _diagnose_or_refuse(work / "copy.db")
    out.update({"target": str(entry), "target_sha256": copy["sha256"], "sidecars": side,
                "hot_journal": [], "inconclusive": False, "copy": str(work / "copy.db")})
    return out


def plan(target: str | Path, staging: str | Path, out: str | Path, *, expect_stale: int | None = None) -> dict[str, Any]:
    entry = _check_target(Path(target))
    staging, out = Path(staging), Path(out)
    _check_outputs(entry, staging, out)
    staging = staging.resolve()
    plan_id = uuid.uuid4().hex
    work = staging / f"plan-{time.strftime('%Y%m%d_%H%M%S')}-{plan_id[:8]}"
    work.mkdir(parents=True)
    original, repaired = work / "original.db", work / "repaired.db"
    copy = _quiet_copy(entry, original)
    before = _diagnose_or_refuse(original)
    m: dict[str, Any] = {
        "tool": TOOL, "version": VERSION, "plan_id": plan_id,
        "planned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target": str(entry), "target_sha256": copy["sha256"], "target_size": copy["size"],
        "target_mtime_ns": copy["mtime_ns"], "sidecars_at_plan": copy["sidecars"],
        "work": str(work), "original_copy": str(original), "before": before,
    }
    if _consistent(before):
        m.update({"state": "nothing_to_repair", "ready": False, "nothing_to_repair": True,
                  "gates": {"damage_found": False}})
        _write_manifest(out, m)
        return m

    shutil.copy2(original, repaired)
    con = _maintenance_connection(repaired)
    try:
        _reindex(con)
    finally:
        con.close()
    repaired_sha = _sha256(repaired)              # digest first, then the proof of those bytes (GPT F1-03)
    after = _diagnose_or_refuse(repaired)
    if _sha256(repaired) != repaired_sha:
        raise SystemExit("refusing: the repaired copy changed while it was diagnosed")
    gates = {
        "damage_found": True,
        "index_damage_only": bool(before["index_damage_only"]),
        "supported_schema": not before["unsupported"] and not after["unsupported"],
        "repaired_integrity_ok": bool(after["integrity_ok"]),
        "indexes_consistent_after": _consistent(after),
        "rows_identical": before["tables"] == after["tables"],
        "schema_identical": (before["schema_sha256"] == after["schema_sha256"]
                             and before["settings"] == after["settings"] and before["header"] == after["header"]),
        "foreign_keys_unchanged": before["foreign_key_sha256"] == after["foreign_key_sha256"],
        "no_leftover_sidecars": not any(Path(str(repaired) + s).exists() for s in ("-wal", "-shm", "-journal")),
        "only_index_entries_removed": after["stale_total"] == 0 and after["missing_total"] == 0
                                       and all(i["index_entries"] == i["table_rows"] for i in after["indexes"].values()),
    }
    if expect_stale is not None:
        gates["expected_stale"] = before["stale_total"] == int(expect_stale) and before["missing_total"] == 0
    m.update({
        "after": after,
        "expected_content": _content(after),
        "original_content": _content(before),
        "repaired": str(repaired), "repaired_sha256": repaired_sha,
        "stale_entries": {name: i["stale"] for name, i in before["indexes"].items() if i["stale_count"]},
        "missing_entries": {name: i["missing"] for name, i in before["indexes"].items() if i["missing_count"]},
        "expect_stale": expect_stale,
        "gates": gates, "ready": all(gates.values()),
        "state": "planned",
    })
    _write_manifest(out, m)
    return m


# ── apply / rollback (in place, under SQLite's exclusive lock) ───────────────

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


REQUIRED_GATES = ("damage_found", "index_damage_only", "supported_schema", "repaired_integrity_ok",
                  "indexes_consistent_after", "rows_identical", "schema_identical", "foreign_keys_unchanged",
                  "only_index_entries_removed")


def _load(manifest_path: Path) -> dict[str, Any]:
    m = json.loads(Path(manifest_path).read_text())
    if m.get("tool") != TOOL or m.get("version") != VERSION:
        raise SystemExit(f"refusing: not a {TOOL} v{VERSION} manifest")
    return m


def _guards(m: dict[str, Any], target_is_copy: bool) -> Path:
    named = Path(m["target"])
    if target_is_copy and (_on_live_volume(named) or _on_live_volume(_entry(named))
                           or _on_live_volume(named.resolve())):
        raise SystemExit("refusing: --target-is-copy is never honoured under /Volumes")
    target = _check_target(named)
    if target != named:
        raise SystemExit("refusing: the manifest's target is not a canonical directory entry")
    _check_desk(target, target_is_copy)
    holders = [p for p in _held_open(target) if p != str(os.getpid())]
    if holders:
        raise SystemExit(f"refusing: {target.name} is held open by {holders} (stop the desk first)")
    return target


class _Pinned:
    """One read-only descriptor on the live target, opened BEFORE the SQLite
    connection and closed AFTER it.

    POSIX drops every fcntl lock a process holds on a file when the process
    closes ANY descriptor on it. Opening and closing the target with ordinary
    file I/O while SQLite holds its exclusive lock would silently release that
    lock to other processes (the classic "separate close()" corruption). Every
    byte read of the live target goes through this one descriptor instead.
    """

    def __init__(self, target: Path):
        self.target = target
        self.fd = os.open(str(target), os.O_RDONLY)
        self.identity = _ident(os.fstat(self.fd))[:2]

    def check_same_file(self) -> None:
        if _ident(os.stat(self.target))[:2] != self.identity:
            raise SystemExit("refusing: the target was replaced while it was locked")

    def copy_to(self, dst: Path) -> str:
        h = hashlib.sha256()
        with open(dst, "xb") as out:
            offset = 0
            while True:
                chunk = os.pread(self.fd, 1 << 20, offset)
                if not chunk:
                    break
                h.update(chunk)
                out.write(chunk)
                offset += len(chunk)
            out.flush()
            _full_fsync(out.fileno())
        if _sha256(dst) != h.hexdigest():
            dst.unlink(missing_ok=True)
            raise SystemExit("refusing: a copy of the target does not hash to the bytes read")
        return h.hexdigest()

    def close(self) -> None:
        os.close(self.fd)


def _live_content(pin: _Pinned, work: Path) -> tuple[str, dict[str, Any]]:
    """sha256 and content digests of the (locked, checkpointed) live target, via a byte copy."""
    pin.check_same_file()
    copy = work / f"live-{uuid.uuid4().hex[:8]}.db"
    sha = pin.copy_to(copy)
    try:
        return sha, _content(_diagnose_or_refuse(copy))
    finally:
        copy.unlink(missing_ok=True)


def _durable_backup(pin: _Pinned, backup: Path, expect_sha: str) -> None:
    tmp = backup.with_name(f".backup-{uuid.uuid4().hex[:8]}")
    try:
        if pin.copy_to(tmp) != expect_sha:
            raise SystemExit("refusing: the backup does not match the target's planned sha256")
        os.replace(tmp, backup)
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(backup.parent)                              # the backup is durable BEFORE the repair


def apply(manifest_path: str | Path, owner_approval: str, *, target_is_copy: bool = False) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    m = _load(manifest_path)
    if not str(owner_approval or "").strip():
        raise SystemExit("refusing: apply needs the owner's approval in words (--owner-approval)")
    if m.get("state") not in {"planned", "applying", "applied", "rolled_back"}:
        # rolled_back: the plan's proof holds again once the original content is back
        raise SystemExit(f"refusing: the manifest is in state {m.get('state')!r}, not 'planned'")
    gates = m.get("gates") or {}
    if not (m.get("ready") and all(gates.get(g) is True for g in REQUIRED_GATES)):
        raise SystemExit(f"refusing: the plan's gates did not all pass: {gates}")
    target = _guards(m, target_is_copy)
    work = Path(m["work"])
    pin = _Pinned(target)
    try:
        con = _maintenance_connection(target)
    except BaseException:
        pin.close()
        raise
    try:
        _checkpoint(con)
        if _hot(_sidecars(target)):
            raise SystemExit("refusing: the target still has a hot journal after the checkpoint")
        sha, content = _live_content(pin, work)
        if _same_content(content, m["expected_content"]):        # already repaired (or a crash after COMMIT)
            m.update({"state": "applied", "applied_ok": True, "applied_at": m.get("applied_at") or _now()})
            _write_manifest(manifest_path, m)
            return {**m, "noop": True}
        if not _same_content(content, m["original_content"]):
            raise SystemExit("refusing: the target is neither the planned original nor the repaired database "
                             "(it changed since the plan; re-plan)")
        backup = Path(m["backup"]) if m.get("backup") else target.with_name(
            f"{target.name}.pre-reindex-{time.strftime('%Y%m%d_%H%M%S')}")
        if not (backup.is_file() and _sha256(backup) == sha):
            if backup.exists():
                backup = target.with_name(f"{target.name}.pre-reindex-{time.strftime('%Y%m%d_%H%M%S')}-{sha[:8]}")
            _durable_backup(pin, backup, sha)                 # the exact bytes about to be repaired
        m.update({"state": "applying", "backup": str(backup), "backup_sha256": sha,
                  "owner_approval": owner_approval.strip(), "apply_started_at": _now()})
        _write_manifest(manifest_path, m)
        _reindex(con)
        sha_after, content_after = _live_content(pin, work)
        ok = _same_content(content_after, m["expected_content"])
        m.update({"state": "applied" if ok else "applied_unverified", "applied_ok": ok,
                  "applied_sha256": sha_after, "bytes_match_plan": sha_after == m["repaired_sha256"],
                  "applied_at": _now()})
        _write_manifest(manifest_path, m)
    finally:
        con.close()
        pin.close()                                            # only after SQLite released its locks
    if not ok:
        raise SystemExit("the repaired target does not match the plan's proof: run rollback")
    return m


def rollback(manifest_path: str | Path, owner_approval: str = "", *, target_is_copy: bool = False,
             backup_path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    m = _load(manifest_path)
    target = _entry(Path(m["target"]))
    live = not target_is_copy or _on_live_volume(target) or _on_live_volume(target.resolve())
    if live and not str(owner_approval or "").strip():
        raise SystemExit("refusing: a live rollback needs the owner's approval in words (--owner-approval)")
    if m.get("state") not in {"applying", "applied", "applied_unverified", "rolling_back", "rolled_back"}:
        raise SystemExit(f"nothing to roll back: the manifest is in state {m.get('state')!r}")
    recorded = str(backup_path or m.get("backup") or "").strip()
    backup = Path(recorded) if recorded else None
    if backup is None or not backup.is_file():
        raise SystemExit("nothing to roll back: no backup recorded or found (pass --backup <verified copy>)")
    if _sha256(backup) not in {m["target_sha256"], m.get("backup_sha256")}:
        raise SystemExit("refusing: the backup does not match the pre-repair sha256")
    target = _guards(m, target_is_copy)
    work = Path(m["work"])
    pin = _Pinned(target)
    try:
        con = _maintenance_connection(target)
    except BaseException:
        pin.close()
        raise
    try:
        _checkpoint(con)
        _sha, content = _live_content(pin, work)
        if _same_content(content, m["original_content"]):
            m.update({"state": "rolled_back", "rollback_ok": True, "rolled_back_at": _now()})
            _write_manifest(manifest_path, m)
            return m                                           # already the pre-repair database
        if not _same_content(content, m["expected_content"]):
            # in EVERY state: anything else holds writes made after the repair (GPT F1-02)
            raise SystemExit("refusing: the target holds neither the repaired nor the original content "
                             "(writes were made after the repair); rolling back would discard them")
        m.update({"state": "rolling_back", "rollback_started_at": _now(),
                  "rollback_approval": str(owner_approval or "").strip()})
        _write_manifest(manifest_path, m)
        src = _open_ro(backup)
        try:
            src.backup(con)                                    # under the same exclusive lock
        finally:
            src.close()
        _checkpoint(con)
        _sha, restored = _live_content(pin, work)
        ok = _same_content(restored, m["original_content"])
        m.update({"state": "rolled_back" if ok else "rollback_unverified", "rollback_ok": ok,
                  "rolled_back_at": _now()})
        _write_manifest(manifest_path, m)
    finally:
        con.close()
        pin.close()
    if not ok:
        raise SystemExit("rollback did not restore the pre-repair content")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--target", required=True)
    p.add_argument("--staging", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expect-stale", type=int, default=None)
    c = sub.add_parser("check")
    c.add_argument("--target", required=True)
    c.add_argument("--staging", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--manifest", required=True)
    a.add_argument("--owner-approval", required=True)
    a.add_argument("--target-is-copy", action="store_true", help="an isolated copy (never honoured under /Volumes)")
    r = sub.add_parser("rollback")
    r.add_argument("--manifest", required=True)
    r.add_argument("--owner-approval", default="")
    r.add_argument("--target-is-copy", action="store_true")
    r.add_argument("--backup", default=None, help="a verified copy to restore when the recorded backup is gone")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        m = plan(args.target, args.staging, args.out, expect_stale=args.expect_stale)
        print(json.dumps({k: m.get(k) for k in ("state", "ready", "gates", "target_sha256", "repaired_sha256")}
                         | {"stale_total": m["before"]["stale_total"], "missing_total": m["before"]["missing_total"],
                            "integrity_before": m["before"]["integrity"][:5]}, indent=1, default=str))
        if m.get("nothing_to_repair"):
            return 0
        return 0 if m.get("ready") else 1
    if args.cmd == "check":
        d = check(args.target, args.staging)
        print(json.dumps({k: d.get(k) for k in ("integrity", "stale_total", "missing_total", "lookup_mismatch_total",
                                                "unsupported", "hot_journal", "inconclusive", "target_sha256")},
                         indent=1, default=str))
        if d.get("inconclusive"):
            return EXIT_INCONCLUSIVE
        return 0 if _consistent(d) else 1
    if args.cmd == "apply":
        m = apply(args.manifest, args.owner_approval, target_is_copy=args.target_is_copy)
        print(json.dumps({k: m.get(k) for k in ("state", "applied_ok", "noop", "backup", "bytes_match_plan")}, indent=1))
        return 0
    m = rollback(args.manifest, args.owner_approval, target_is_copy=args.target_is_copy, backup_path=args.backup)
    print(json.dumps({k: m.get(k) for k in ("state", "rollback_ok", "target_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
