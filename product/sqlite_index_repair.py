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

Only WAL-mode databases are planned (the freeze ledger is one): under the
exclusive lock a rollback-journal database keeps its ``-journal`` file, so a
crashed apply could never be re-run. A target with more than one hard link is
refused: SQLite names ``-wal`` after the path it opened, so through a second
name committed frames are invisible.

``apply`` repairs the LIVE database IN PLACE, inside SQLite's own locking: no
file is renamed or replaced. It opens the target with ``locking_mode=EXCLUSIVE``
(no other connection can read or write until it closes), full-sync barriers and
``synchronous=FULL``; checkpoints; proves the live content is the planned
original; writes and syncs a FRESH byte backup at a name recorded BEFORE it is
written (an earlier attempt's backup is recorded, never reused or deleted);
records ``applying`` durably; runs REINDEX in one transaction, proves
the trial against the plan and records that proof durably, all BEFORE COMMIT;
checkpoints; and proves the committed result, all before releasing the lock. A
crash rolls the transaction back. Re-running ``apply`` after a crash finishes or
redoes the repair, after success it is a no-op; a healthy database plans as
``nothing_to_repair``.

While the lock is held nothing is opened BY NAME except new files created with
``O_EXCL|O_NOFOLLOW`` (the backup, manifest temp files). POSIX releases every
lock a process holds on a file when it closes ANY descriptor on it, and a name
re-opened after a check can have become a link to the live file. So every proof
of the live file is made on its bytes, read once through one pinned descriptor
and diagnosed as an in-memory database, and every new file is verified through
the descriptor that created it (cycle-8 review, part 4).

``rollback`` proves the backup (sha256 AND content) in memory before the lock,
then, under the same exclusive lock and only while the target holds exactly the
repaired content, restores THAT in-memory database through SQLite's backup API.
A repair changes index entries only, so a rollback must too: the backup's rows,
schema, settings and foreign keys must equal the live database's, compared
between the two databases themselves, never through the manifest (an edited
manifest cannot make rollback install another database), and both must be
fully covered by the row proof (no WITHOUT ROWID or other unsupported object).
A header that differs from the plan's is accepted only when it is exactly the
header apply observed after its own proven commit (``applied_unverified``). In every state an
unknown content digest is refused, so writes made after the repair are never
discarded. Both need the owner's words for a live target and a stopped desk;
copy mode is never honoured under /Volumes (any spelling: core.live_volume).
"""
from __future__ import annotations

import argparse
import contextlib
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
    _full_fsync,
    _held_open,
    _not_live,
    _on_live_volume,
    _sha256,
)

TOOL = "sqlite_index_repair"
VERSION = 3      # v3 (cycle 8, part 6): attempt-bound trial proofs, applied_header, no backup reuse;
                 # a v2 manifest carries other recovery semantics and is refused: re-plan
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
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CHUNK = 1 << 20
#: What a repair (and so its rollback) must never change, compared database to database.
PRESERVED = ("tables", "schema_sha256", "settings", "foreign_key_sha256")
_UNVERIFIED = ("the content was proven before COMMIT, but the committed file's header differs from the plan's "
               "(state applied_unverified): inspect with check; rollback accepts this state")
_UNVERIFIED_RESTART = ("the repaired content was proven before COMMIT, but the header differs from the plan's and was "
                       "first seen after a restart, so it may be a later write (state applied_unverified): the "
                       "database is repaired; an automatic rollback will refuse it; restore the stage-3 SSD copy by "
                       "hand only after inspecting with check")


# ── identity, copies, sidecars ───────────────────────────────────────────────

def _fsync_dir(directory: Path) -> None:
    """Make an entry change in ``directory`` durable, through a descriptor that can only
    be a DIRECTORY: ``O_DIRECTORY|O_NOFOLLOW`` fails on a symlink or a file, so a name
    swapped after the checks can never yield (and then close) a descriptor on the live
    database while SQLite holds its lock (GPT C8.5-03). EINVAL/ENOTSUP from a filesystem
    that cannot sync a directory is tolerated; any other error is raised."""
    import errno

    fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)
    try:
        try:
            _full_fsync(fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)):
                raise
    finally:
        os.close(fd)


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


def _header_bytes(raw: bytes) -> dict[str, str]:
    return {f"{a}-{b - 1}": raw[a:b].hex() for a, b in HEADER_FIXED} | {
        "journal_header": {1: "rollback", 2: "wal"}.get(raw[18], str(raw[18]))}


def _header(path: Path) -> dict[str, str]:
    with open(path, "rb") as fh:
        return _header_bytes(fh.read(100))


def _memory_db(raw: bytes) -> sqlite3.Connection:
    """The exact bytes as an in-memory database: no file is opened, by name or otherwise.

    ``sqlite3_deserialize`` cannot use a WAL-mode image, so the write/read version
    bytes (18, 19) are set to rollback mode IN THE COPY; header proofs are always
    taken from the original bytes. A restore from it into a WAL database keeps
    the destination in WAL mode (SQLite's backup sets the version back).
    """
    image = bytearray(raw)
    if len(image) >= 100 and image[18] == 2 and image[19] == 2:
        image[18] = image[19] = 1
    con = sqlite3.connect(":memory:")
    try:
        con.deserialize(bytes(image))
    except BaseException:
        con.close()
        raise
    return con


def _diagnose_bytes(raw: bytes) -> dict[str, Any]:
    """``diagnose`` of a database image held in memory (identical result: tested)."""
    try:
        con = _memory_db(raw)
        try:
            out = _diagnose_connection(con)
        finally:
            con.close()
        out["header"] = _header_bytes(raw)
    except (sqlite3.DatabaseError, IndexError) as exc:
        raise SystemExit(f"refusing: the bytes cannot be read as a database ({exc}); "
                         "this is not index-only damage and REINDEX cannot repair it") from None
    return out


def diagnose(path: str | Path) -> dict[str, Any]:
    """Read-only diagnosis of a database COPY (never call it on a live file)."""
    path = Path(path)
    con = _open_ro(path)
    try:
        out = _diagnose_connection(con)
    finally:
        con.close()
    out["header"] = _header(path)
    return out


def _diagnose_connection(con: sqlite3.Connection) -> dict[str, Any]:
    """Everything but the file header, through ``con`` (also inside a transaction)."""
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
    index_only = all(any(p.match(m) for p in INDEX_DAMAGE) for m in integrity) if integrity != ["ok"] else True
    return {
        "integrity": integrity,
        "integrity_ok": integrity == ["ok"],
        "index_damage_only": index_only,
        "schema": schema,
        "schema_sha256": hashlib.sha256(json.dumps(schema).encode()).hexdigest(),
        "settings": settings,
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


def _same_content(a: dict[str, Any], b: dict[str, Any], *, ignore_header: bool = False) -> bool:
    """Equal content, the schema cookie aside (the whole header too with ``ignore_header``)."""
    def strip(c: dict[str, Any]) -> dict[str, Any]:
        header = {} if ignore_header else {k: v for k, v in (c.get("header") or {}).items() if k != SCHEMA_COOKIE}
        return {**c, "header": header}
    return strip(a) == strip(b)


def _rows_preserved(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same rows, schema, settings and foreign keys: what a repair, and so its rollback,
    may never change. Compared between two databases' OWN proofs, never through the
    manifest, so an edited manifest cannot make rollback install another database
    (GPT C8-01, cycle-8 review part 4)."""
    return all(a.get(k) == b.get(k) for k in PRESERVED)


def _content(d: dict[str, Any]) -> dict[str, Any]:
    """The state proof: what a repair must preserve (tables, schema, settings, FKs) and
    what it produces (index entries AND integrity: GPT V2-02, a sorted entry digest
    alone cannot see an index whose entries are right but out of order)."""
    return {"tables": d["tables"], "schema_sha256": d["schema_sha256"], "header": d.get("header") or {},
            "settings": d["settings"], "foreign_key_sha256": d["foreign_key_sha256"],
            "index_digests": {n: i["entries_digest"] for n, i in d["indexes"].items()},
            "integrity": d["integrity"], "unsupported": sorted(d["unsupported"]),
            "stale_total": d["stale_total"], "missing_total": d["missing_total"],
            "lookup_mismatch_total": d["lookup_mismatch_total"]}


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
    if st.st_nlink != 1:
        # SQLite names -wal/-shm after the PATH it opened: through a second name, frames
        # committed through the other name's WAL are invisible, and a repair through it would
        # back up and rewrite the shared inode without them (GPT C8-04).
        raise SystemExit(f"refusing: {target} has {st.st_nlink} hard links; name the database's only entry")
    return entry


def _identity(p: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(p)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _no_dotdot(p: Path, what: str) -> None:
    """'..' is resolved AFTER symlinks by the kernel but textually by abspath, so a
    checked spelling could land elsewhere (cycle-8 review part 4): never accept one."""
    if ".." in Path(p).parts:
        raise SystemExit(f"refusing: {what} must not contain '..' ({p}); name the directory directly")


def _existing_chain(p: Path) -> list[Path]:
    """``p`` resolved (symlinks followed), then each existing ancestor up to the root."""
    p = Path(os.path.abspath(p))
    while not p.exists() and p != p.parent:
        p = p.parent
    p = p.resolve()
    return [p, *p.parents]


def _check_staging(entry: Path, staging: Path) -> None:
    """``check``'s copies follow the same rule as ``plan``'s outputs."""
    _no_dotdot(staging, "staging")
    _not_live(staging, "staging")
    target_dir = _identity(entry.parent)
    if any(_identity(c) == target_dir for c in _existing_chain(Path(staging))):
        raise SystemExit(f"refusing: staging must not be in the database's directory ({entry.parent})")


def _check_outputs(entry: Path, staging: Path, out: Path) -> None:
    """Outputs live on a disk the operator owns, never in the database's directory.

    Compared by FILESYSTEM IDENTITY up the resolved path, so a symlinked staging
    directory or a differently cased spelling cannot reach it (GPT F1-04, V2-04).
    Placement only: the manifest already at ``out`` is read under its lock
    (_replaceable_prior), so no file is created before these refusals ran.
    """
    _no_dotdot(staging, "staging")
    _no_dotdot(out, "the manifest")
    _not_live(staging, "staging")
    _not_live(out.parent, "the manifest")
    target_dir = _identity(entry.parent)
    target_id = _identity(entry)
    for what, p in (("the manifest", out), ("staging", staging)):
        if target_id is not None and _identity(Path(p)) == target_id:
            raise SystemExit(f"refusing: {what} would overwrite the database or its journal")
        if Path(str(entry) + "-wal") == Path(p) or str(p).startswith(str(entry) + "-"):
            raise SystemExit(f"refusing: {what} would overwrite the database or its journal")
        chain = _existing_chain(Path(p) if what == "staging" else Path(p).parent)
        if any(_identity(c) == target_dir for c in chain):
            raise SystemExit(f"refusing: {what} must not be in the database's directory ({entry.parent})")


def _replaceable_prior(out: Path) -> dict[str, Any] | None:
    """The manifest at ``out`` if a new plan may replace it (None when there is none). Read
    under the manifest's lock, so no apply or rollback can write it between this read and the
    plan's publication (GPT cycle-9 item 1)."""
    prior = _prior_manifest(out)
    if prior is not None:
        state = prior.get("state")
        in_flight = [k for k in ("backup", "backup_pending") if prior.get(k)]
        # Replaceable: an untouched plan, or a ROLLED-BACK attempt (its backup history is carried
        # into the new plan). Anything else is what rollback needs (cycle-8 review, parts 7-8).
        if not (state == "rolled_back" or (state in ("planned", "nothing_to_repair") and not in_flight)):
            raise SystemExit(f"refusing: {out} is the record of an apply (state {state!r}, "
                             f"{', '.join(in_flight) or 'no backup yet'}); rollback needs it: plan to a new --out")
    return prior


def _prior_manifest(out: Path) -> dict[str, Any] | None:
    """The manifest already at ``out`` (None when there is none); anything else there is refused."""
    if not _manifest_exists(out):
        return None
    try:
        prior = json.loads(_read_manifest_text(out))
    except (OSError, ValueError):
        prior = None
    if not (isinstance(prior, dict) and prior.get("tool") == TOOL):
        raise SystemExit(f"refusing: {out} exists and is not a {TOOL} manifest")
    return prior


def _publish_plan(out: Path, m: dict[str, Any], prior: dict[str, Any] | None) -> None:
    """Write the plan only if nothing wrote ``out`` while it was made: an apply that published
    its record in the meantime must never be overwritten (GPT C8.7-04). Every command of this
    tool now holds the manifest's lock from its first read to its last write, which closes the
    check-to-rename window between two commands (cycle-9 item 1); this re-check still catches
    a writer outside the tool (an edit, a restore, an older version)."""
    if _prior_manifest(out) != prior:
        raise SystemExit(f"refusing: {out} changed while the plan was made (another command wrote it); "
                         "inspect it, then plan to a new --out")
    if prior is not None:
        history = set(prior.get("earlier_backups") or [])
        history.update(n for n in (prior.get("backup"), prior.get("backup_pending")) if n)
        if history:
            m["earlier_backups"] = sorted(history)            # an earlier attempt's backups stay recorded
            m["replanned_over"] = {"state": prior.get("state"), "plan_id": prior.get("plan_id")}
    _write_manifest(out, m)


def _manifest_path(path: str | Path) -> Path:
    """The manifest's REAL location, resolved ONCE (symlinks before '..', as the kernel
    does) and never again: every later read and write goes to this path, so a name swapped
    to a symlink afterwards is replaced, never followed (GPT C8-03, C8.5-07)."""
    _no_dotdot(Path(path), "the manifest")
    real = Path(os.path.realpath(os.fspath(path)))
    if real.suffix.lower() != ".json":
        raise SystemExit(f"refusing: the manifest must be a .json file, not {real.name}")
    return real


def _check_manifest_place(real: Path, target: Path) -> None:
    """Never a name that could replace the database, its journals or its backups were a
    directory swapped later: not beginning with the database's name, not in its directory.
    ``real`` is already resolved and is NOT resolved again."""
    if real.name.lower().startswith(target.name.lower()):
        raise SystemExit(f"refusing: the manifest name {real.name} could replace a file beside the database")
    if _identity(real.parent) == _identity(target.parent):
        raise SystemExit(f"refusing: the manifest must not be in the database's directory ({target.parent})")


#: Manifest directories this process holds locked: real manifest path -> (directory fd, its identity).
_HELD: dict[str, tuple[int, tuple[int, int]]] = {}


def _check_single_link(real: Path, dir_fd: int | None = None) -> None:
    """A hard-linked manifest has a second name the lock may not cover: refused. (A name
    swapped to a SYMLINK is not followed: the write replaces the entry, GPT C8-03.)"""
    try:
        st = (os.stat(real.name, dir_fd=dir_fd, follow_symlinks=False) if dir_fd is not None
              else os.lstat(real))
    except FileNotFoundError:
        return
    if S.S_ISREG(st.st_mode) and st.st_nlink != 1:
        raise SystemExit(f"refusing: the manifest {real} has {st.st_nlink} hard links; an alias could be "
                         "written without its lock")


@contextlib.contextmanager
def _manifest_lock(real: Path, command: str, target_dir: Path):
    """Exclusive, non-blocking lock on the manifest's DIRECTORY for a whole plan / apply / rollback.

    Serializes every writer of the manifest (cycle-9 item 1): the plan's check-to-rename
    window, two applies, an apply beside a rollback. The lock is the directory itself, opened
    once (``O_DIRECTORY|O_NOFOLLOW``) and pinned: no lock file exists that a cleanup could
    delete or replace (GPT C9B-01), nothing is ever created to take it (C9B-07), and every
    spelling of the manifest (a symlinked or firmlinked directory, another case) meets the
    same lock. All reads and writes of the manifest under it go through that descriptor, so
    a directory renamed or swapped afterwards can never redirect them. Fails closed: a held
    lock refuses at once, never waits; the kernel releases it when its holder exits or dies.
    Commands on manifests in one directory are serialized. Lock order is always this lock
    first, then SQLite's exclusive lock on the target.
    """
    import fcntl

    real = Path(real)
    try:
        dfd = os.open(str(real.parent), os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)
    except OSError as exc:
        raise SystemExit(f"refusing: the manifest's directory {real.parent} cannot be opened as a directory "
                         f"itself ({exc}); is it a symlink?") from None
    try:
        st = os.fstat(dfd)
        identity = (st.st_dev, st.st_ino)
        if identity == _identity(Path(target_dir)):
            raise SystemExit(f"refusing: the manifest must not be in the database's directory ({target_dir})")
        try:
            fcntl.flock(dfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"refusing: another plan/apply/rollback holds the manifest directory {real.parent}; "
                             "wait for it to finish, then re-run") from None
        except OSError as exc:
            raise SystemExit(f"refusing: the manifest directory {real.parent} cannot be locked ({exc})") from None
        if _identity(real.parent) != identity:
            raise SystemExit(f"refusing: the manifest's directory {real.parent} was moved while it was locked")
        _check_single_link(real, dfd)
        _HELD[str(real)] = (dfd, identity)
        try:
            yield
        finally:
            _HELD.pop(str(real), None)
    finally:
        os.close(dfd)                                  # releases the lock


@contextlib.contextmanager
def _locked_record(real: Path, command: str):
    """apply / rollback: the manifest's place is checked against the target it names, the
    lock is taken (and its directory compared with the target's), then the caller re-reads
    the manifest under it."""
    peek = _load(real)
    target = _entry(Path(peek["target"]))
    _check_manifest_place(real, target)
    with _manifest_lock(real, command, target.parent):
        yield


def _require_lock(path: Path) -> int:
    """The held directory descriptor of ``path``'s manifest; refused when not held, or when the
    path no longer names the locked directory."""
    held = _HELD.get(str(path))
    if held is None:
        raise SystemExit(f"refusing: {path.name} would be written without its lock")
    dfd, identity = held
    if _identity(Path(path).parent) != identity:
        raise SystemExit(f"refusing: the manifest's directory {Path(path).parent} was moved or replaced while locked")
    return dfd


def _read_manifest_text(path: Path) -> str:
    """The manifest's text: through the locked directory when held (never redirected)."""
    held = _HELD.get(str(path))
    if held is None:
        return Path(path).read_text()
    fd = os.open(Path(path).name, os.O_RDONLY | _NOFOLLOW, dir_fd=held[0])
    with os.fdopen(fd, "r") as fh:
        return fh.read()


def _manifest_exists(path: Path) -> bool:
    held = _HELD.get(str(path))
    if held is None:
        return Path(path).exists()
    try:
        os.stat(Path(path).name, dir_fd=held[0], follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _write_manifest(path: Path, m: dict[str, Any]) -> None:
    """Durable manifest write through an EXCLUSIVELY created, no-follow temp file.

    The shared writer opened a predictable ``<name>.tmp-<pid>`` with ``"w"``: a
    symlink placed there truncated whatever it pointed at, the database included
    (GPT V2-04). ``os.replace`` then swaps the directory entry, never its target.
    Only under the manifest's lock, and only through its locked directory descriptor
    (cycle-9 item 1, GPT C9B-01/07).
    """
    import errno

    path = Path(path)
    if os.path.realpath(path.parent) != str(path.parent):
        raise SystemExit(f"refusing: the manifest's directory {path.parent} is not its real path (a symlink?)")
    dfd = _require_lock(path)
    tmp = f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o644, dir_fd=dfd)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(m, indent=2, default=str))
            fh.flush()
            _full_fsync(fh.fileno())
        os.replace(tmp, path.name, src_dir_fd=dfd, dst_dir_fd=dfd)
    except BaseException:
        try:
            os.unlink(tmp, dir_fd=dfd)
        except FileNotFoundError:
            pass
        raise
    try:
        _full_fsync(dfd)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)):
            raise


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


def _reindex(con: sqlite3.Connection, verify: Any = None) -> None:
    """REINDEX in one transaction; ``verify(con)`` runs BEFORE COMMIT and aborts it by raising."""
    con.execute("BEGIN EXCLUSIVE")
    try:
        con.execute("REINDEX")
        if verify is not None:
            verify(con)
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    _checkpoint(con)


# ── plan ─────────────────────────────────────────────────────────────────────

def check(target: str | Path, staging: str | Path) -> dict[str, Any]:
    entry = _check_target(Path(target))
    _check_staging(entry, Path(staging))
    staging = Path(staging).resolve()
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
    out = _manifest_path(out)
    _check_manifest_place(out, entry)
    with _manifest_lock(out, "plan", entry.parent):
        return _plan_locked(entry, staging, out, expect_stale)


def _plan_locked(entry: Path, staging: Path, out: Path, expect_stale: int | None) -> dict[str, Any]:
    prior = _replaceable_prior(out)
    staging = staging.resolve()
    plan_id = uuid.uuid4().hex
    work = staging / f"plan-{time.strftime('%Y%m%d_%H%M%S')}-{plan_id[:8]}"
    work.mkdir(parents=True)
    original, repaired = work / "original.db", work / "repaired.db"
    copy = _quiet_copy(entry, original)
    before = _diagnose_or_refuse(original)
    if before["header"].get("journal_header") != "wal":
        raise SystemExit("refusing: only WAL-mode databases are supported. Under the exclusive lock a "
                         "rollback-journal database keeps its -journal file, so a crashed apply could not be "
                         "re-run (cycle-8 review, part 4); the freeze ledger is WAL")
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
        _publish_plan(out, m, prior)
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
    _publish_plan(out, m, prior)
    return m


# ── apply / rollback (in place, under SQLite's exclusive lock) ───────────────

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


REQUIRED_GATES = ("damage_found", "index_damage_only", "supported_schema", "repaired_integrity_ok",
                  "indexes_consistent_after", "rows_identical", "schema_identical", "foreign_keys_unchanged",
                  "only_index_entries_removed")


def _load(manifest_path: Path) -> dict[str, Any]:
    m = json.loads(_read_manifest_text(Path(manifest_path)))
    if m.get("tool") != TOOL or m.get("version") != VERSION:
        raise SystemExit(f"refusing: not a {TOOL} v{VERSION} manifest (an older version's manifest carries other "
                         "recovery rules: re-plan with this version)")
    return m


def _guards(m: dict[str, Any], target_is_copy: bool) -> tuple[Path, tuple[int, int]]:
    """The checked target and its (device, inode): every later open must be that inode."""
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
    st = os.lstat(target)
    return target, (st.st_dev, st.st_ino)


class _Pinned:
    """One read-only, no-follow descriptor on the live target, opened BEFORE the SQLite
    connection and closed AFTER it, on the exact inode the guards checked.

    POSIX drops every fcntl lock a process holds on a file when the process
    closes ANY descriptor on it. Opening and closing the target with ordinary
    file I/O while SQLite holds its exclusive lock would silently release that
    lock to other processes (the classic "separate close()" corruption). Every
    byte read of the live target goes through this one descriptor instead.
    """

    def __init__(self, target: Path, identity: tuple[int, int]):
        self.target = target
        try:
            self.fd = os.open(str(target), os.O_RDONLY | _NOFOLLOW)
        except OSError as exc:
            raise SystemExit(f"refusing: {target.name} could not be opened as itself ({exc}); "
                             "it was replaced after the checks") from None
        st = os.fstat(self.fd)
        if not S.S_ISREG(st.st_mode) or st.st_nlink != 1 or (st.st_dev, st.st_ino) != tuple(identity):
            os.close(self.fd)                       # no SQLite connection yet: no lock to lose
            raise SystemExit(f"refusing: {target.name} was replaced (or linked) after the checks")
        self.identity = (st.st_dev, st.st_ino)

    def check_same_file(self) -> None:
        """The NAME still is the pinned inode, the only link to it, and not a symlink."""
        try:
            st = os.lstat(self.target)
        except OSError:
            raise SystemExit("refusing: the target was moved while it was locked") from None
        if (S.S_ISLNK(st.st_mode) or (st.st_dev, st.st_ino) != self.identity
                or os.fstat(self.fd).st_nlink != 1):
            raise SystemExit("refusing: the target was replaced (or linked) while it was locked")

    def require_wal(self) -> None:
        """Checked through the pin BEFORE SQLite opens the file: opening a rollback-journal
        database under the exclusive lock would leave its -journal behind (GPT C8.5-05)."""
        if os.pread(self.fd, 100, 0)[18:20] != b"\x02\x02":
            raise SystemExit(f"refusing: {self.target.name} is not in WAL mode (it changed since the plan)")

    def read_all(self) -> bytes:
        chunks, offset = [], 0
        while chunk := os.pread(self.fd, _CHUNK, offset):
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        os.close(self.fd)


def _live_content(pin: _Pinned) -> tuple[str, dict[str, Any], bytes]:
    """sha256, content proof and bytes of the (locked, checkpointed) live target, from ONE
    read through the pinned descriptor, diagnosed in memory. No copy is written and re-opened
    by name: that name could have become a link to the live file (GPT C8-02)."""
    pin.check_same_file()
    raw = pin.read_all()
    return hashlib.sha256(raw).hexdigest(), _content(_diagnose_bytes(raw)), raw


def _write_new_file(path: Path, raw: bytes) -> None:
    """``raw`` as a NEW file (``O_EXCL|O_NOFOLLOW``), F_FULLFSYNC'd, verified by reading it back
    through the descriptor that created it (never re-opened by name), its directory synced."""
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o644)
    try:
        view, offset = memoryview(raw), 0
        while offset < len(raw):
            offset += os.write(fd, view[offset:offset + _CHUNK])
        _full_fsync(fd)
        h, offset = hashlib.sha256(), 0
        while chunk := os.pread(fd, _CHUNK, offset):
            h.update(chunk)
            offset += len(chunk)
    finally:
        os.close(fd)
    if h.hexdigest() != hashlib.sha256(raw).hexdigest():
        # left in place (its name is recorded as backup_pending): nothing is ever deleted
        raise SystemExit(f"refusing: {path.name} does not read back as the bytes written")
    _fsync_dir(path.parent)


def _durable_backup(backup: Path, raw: bytes, expect_sha: str) -> None:
    """The exact bytes about to be repaired, at the name recorded BEFORE this write: a crash
    leaves at most that one file, which the next run records among earlier_backups (GPT C8-06)."""
    if hashlib.sha256(raw).hexdigest() != expect_sha:
        raise SystemExit("refusing: the backup does not match the target's planned sha256")
    _write_new_file(backup, raw)                           # durable BEFORE the repair


def _all_gates_pass(m: dict[str, Any]) -> bool:
    """Every required gate AND every conditional one the plan recorded (expected_stale: GPT V2-06)."""
    gates = m.get("gates") or {}
    return (bool(m.get("ready")) and all(gates.get(g) is True for g in REQUIRED_GATES)
            and all(v is True for v in gates.values()))


def _precommit_verifier(m: dict[str, Any], on_verified: Any = None):
    """Proves the trial repair against the plan INSIDE the transaction, before COMMIT (GPT V2-06):
    a mismatch rolls the transaction back, so nothing unproven is ever committed. ``on_verified``
    records the proof durably, still before COMMIT, so a crash after COMMIT can tell a proven
    repair from an unknown change (GPT C8-05)."""
    expected = m["expected_content"]

    def verify(con: sqlite3.Connection) -> None:
        got = _content(_diagnose_connection(con))
        if not _same_content(got, expected, ignore_header=True):
            raise SystemExit("refusing: the trial repair does not match the plan's proof; "
                             "the transaction was rolled back and nothing was committed")
        if on_verified is not None:
            on_verified()
    return verify


def _fresh_backup_path(target: Path, sha: str) -> Path:
    """A name no earlier attempt used (a random part every time), so a crashed attempt's
    removed partial is never confused with the next attempt's backup (cycle-8 review part 4)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    while True:
        path = target.with_name(f"{target.name}.pre-reindex-{stamp}-{sha[:8]}-{uuid.uuid4().hex[:6]}")
        if not (path.exists() or path.is_symlink()):
            return path


def _record_earlier_backups(m: dict[str, Any]) -> None:
    """Backups (or partial backups) of earlier attempts are RECORDED, never reused and never
    deleted: a name in an editable manifest proves no ownership, a matching hash proves no
    durability, and the bytes about to be repaired are already in memory, so every attempt
    writes its own fresh, synced backup (GPT C8.5-01, C8.5-06)."""
    names = set(m.get("earlier_backups") or [])
    names.update(n for n in (m.get("backup"), m.get("backup_pending")) if n)
    m["earlier_backups"] = sorted(names)


def apply(manifest_path: str | Path, owner_approval: str, *, target_is_copy: bool = False) -> dict[str, Any]:
    manifest_path = _manifest_path(manifest_path)                  # resolved once, never again
    with _locked_record(manifest_path, "apply"):
        return _apply_locked(manifest_path, owner_approval, target_is_copy)


def _apply_locked(manifest_path: Path, owner_approval: str, target_is_copy: bool) -> dict[str, Any]:
    m = _load(manifest_path)                                       # re-read under the lock
    if not str(owner_approval or "").strip():
        raise SystemExit("refusing: apply needs the owner's approval in words (--owner-approval)")
    if m.get("state") not in {"planned", "applying", "applied", "rolled_back"}:
        # rolled_back: the plan's proof holds again once the original content is back
        raise SystemExit(f"refusing: the manifest is in state {m.get('state')!r}, not 'planned'")
    if not _all_gates_pass(m):
        raise SystemExit(f"refusing: the plan's gates did not all pass: {m.get('gates')}")
    target, identity = _guards(m, target_is_copy)
    _check_manifest_place(manifest_path, target)
    pin = _Pinned(target, identity)
    try:
        pin.require_wal()
        con = _maintenance_connection(target)
    except BaseException:
        pin.close()
        raise
    ok = False
    try:
        pin.check_same_file()                  # SQLite opened the name the pin holds
        _checkpoint(con)
        if _hot(_sidecars(target)):
            raise SystemExit("refusing: the target still has a hot journal after the checkpoint")
        sha, content, raw = _live_content(pin)
        if _same_content(content, m["expected_content"]):        # already repaired (or a crash after COMMIT)
            m.update({"state": "applied", "applied_ok": True, "applied_at": m.get("applied_at") or _now()})
            _write_manifest(manifest_path, m)
            return {**m, "noop": True}
        if (m.get("state") == "applying" and m.get("attempt_id")
                and m.get("verified_attempt") == m.get("attempt_id")
                and _same_content(content, m["expected_content"], ignore_header=True)):
            # THIS attempt's trial was proven before COMMIT and the final record was cut off; only
            # the header differs from the plan's. The repaired content stands (applied_unverified),
            # but a header FIRST SEEN after a restart could be a later write, so it is not recorded
            # as one rollback may accept: an automatic rollback refuses (GPT C8-05, C8.5-05, C8.6-03)
            m.update({"state": "applied_unverified", "applied_ok": False, "header_first_seen_on_restart": True,
                      "applied_at": m.get("applied_at") or _now()})
            _write_manifest(manifest_path, m)
            raise SystemExit(_UNVERIFIED_RESTART)
        if not _same_content(content, m["original_content"]):
            raise SystemExit("refusing: the target is neither the planned original nor the repaired database "
                             "(it changed since the plan; re-plan)")
        _record_earlier_backups(m)
        backup = _fresh_backup_path(target, sha)
        # Recorded BEFORE it is written: a crash in between leaves a name the next run
        # records among earlier_backups, never an untracked file (GPT C8-06).
        m.update({"backup_pending": str(backup)})
        _write_manifest(manifest_path, m)
        _durable_backup(backup, raw, sha)                     # the exact bytes about to be repaired
        attempt = uuid.uuid4().hex
        m.update({"state": "applying", "backup": str(backup), "backup_sha256": sha, "attempt_id": attempt,
                  "owner_approval": owner_approval.strip(), "apply_started_at": _now()})
        m.pop("applied_header", None)
        _write_manifest(manifest_path, m)

        def record_trial() -> None:
            m.update({"verified_before_commit": True, "verified_attempt": attempt, "trial_verified_at": _now()})
            _write_manifest(manifest_path, m)

        _reindex(con, verify=_precommit_verifier(m, on_verified=record_trial))
        sha_after, content_after, _raw = _live_content(pin)
        ok = _same_content(content_after, m["expected_content"])
        header_only = not ok and _same_content(content_after, m["expected_content"], ignore_header=True)
        m.update({"state": "applied" if ok else "applied_unverified" if header_only else "apply_mismatch",
                  "applied_ok": ok, "applied_sha256": sha_after,
                  "bytes_match_plan": sha_after == m["repaired_sha256"], "applied_at": _now()})
        if header_only:
            m["applied_header"] = content_after["header"]        # the one header rollback may accept
        _write_manifest(manifest_path, m)
        if not ok and not header_only:
            raise SystemExit("the trial was proven before COMMIT, but the file read back through the pinned "
                             "descriptor is not the planned result (state apply_mismatch): stop, inspect with "
                             "check, do not roll back blindly")
    finally:
        con.close()
        pin.close()                                            # only after SQLite released its locks
    if not ok:
        raise SystemExit(_UNVERIFIED)
    return m


def _read_whole(path: Path) -> bytes:
    """A regular file's bytes through ONE no-follow descriptor, unchanged from open to close."""
    try:
        fd = os.open(str(path), os.O_RDONLY | _NOFOLLOW)
    except OSError as exc:
        raise SystemExit(f"refusing: {path} cannot be opened as a regular file ({exc})") from None
    try:
        before = os.fstat(fd)
        if not S.S_ISREG(before.st_mode):
            raise SystemExit(f"refusing: {path} is not a regular file")
        chunks, offset = [], 0
        while chunk := os.pread(fd, _CHUNK, offset):
            chunks.append(chunk)
            offset += len(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        named = os.lstat(path)
    except OSError:
        named = None
    if named is None or not (_ident(before) == _ident(after) == _ident(named)):
        raise SystemExit(f"refusing: {path} changed (or was replaced) while it was read")
    return b"".join(chunks)


def _verified_rollback_source(backup: Path, target: Path, m: dict[str, Any]) -> tuple[sqlite3.Connection, dict]:
    """The backup's bytes, read ONCE before the lock and proven by sha256 AND content to be the
    pre-repair ledger, held as an in-memory database with its proof. The restore reads only that
    connection: no name is re-opened after the proof (GPT V2-01, C8-02)."""
    if _identity(backup) == _identity(target):
        raise SystemExit("refusing: the backup is the live database itself")
    side = _sidecars(backup)
    if _hot(side):
        raise SystemExit(f"refusing: the backup has {_hot(side)}: the file alone is not the database")
    raw = _read_whole(backup)
    if hashlib.sha256(raw).hexdigest() not in {m["target_sha256"], m.get("backup_sha256")}:
        raise SystemExit("refusing: the backup does not match the pre-repair sha256")
    content = _content(_diagnose_bytes(raw))
    if not _same_content(content, m["original_content"]):
        raise SystemExit("refusing: the backup's content is not the pre-repair content")
    return _memory_db(raw), content


def rollback(manifest_path: str | Path, owner_approval: str = "", *, target_is_copy: bool = False,
             backup_path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = _manifest_path(manifest_path)
    with _locked_record(manifest_path, "rollback"):
        return _rollback_locked(manifest_path, owner_approval, target_is_copy, backup_path)


def _rollback_locked(manifest_path: Path, owner_approval: str, target_is_copy: bool,
                     backup_path: str | Path | None) -> dict[str, Any]:
    m = _load(manifest_path)                                       # re-read under the lock
    target = _entry(Path(m["target"]))
    live = not target_is_copy or _on_live_volume(target) or _on_live_volume(target.resolve())
    if live and not str(owner_approval or "").strip():
        raise SystemExit("refusing: a live rollback needs the owner's approval in words (--owner-approval)")
    if m.get("state") not in {"applying", "applied", "applied_unverified", "apply_mismatch",
                              "rolling_back", "rolled_back"}:
        raise SystemExit(f"nothing to roll back: the manifest is in state {m.get('state')!r}")
    recorded = str(backup_path or m.get("backup") or "").strip()
    backup = Path(recorded) if recorded else None
    if backup is None or not backup.is_file():
        raise SystemExit("nothing to roll back: no backup recorded or found (pass --backup <verified copy>)")
    target, identity = _guards(m, target_is_copy)
    _check_manifest_place(manifest_path, target)
    src, src_content = _verified_rollback_source(backup, target, m)
    try:
        pin = _Pinned(target, identity)
        try:
            pin.require_wal()
            con = _maintenance_connection(target)
        except BaseException:
            pin.close()
            raise
        try:
            pin.check_same_file()
            _checkpoint(con)
            _sha, content, _raw = _live_content(pin)
            if _same_content(content, m["original_content"]):
                m.update({"state": "rolled_back", "rollback_ok": True, "rolled_back_at": _now()})
                _write_manifest(manifest_path, m)
                return m                                       # already the pre-repair database
            # Exactly the repaired content, or (applied_unverified) the repaired content with exactly
            # the header apply OBSERVED after its proven commit. Anything else, a later header write
            # included, holds writes made after the repair (GPT F1-02, C8.5-05).
            observed = m.get("applied_header")
            repaired = _same_content(content, m["expected_content"]) or (
                bool(observed) and _same_content(content, m["expected_content"], ignore_header=True)
                and _same_content({"header": content["header"]}, {"header": observed}))
            if not repaired:
                raise SystemExit("refusing: the target holds neither the repaired nor the original content "
                                 "(writes were made after the repair); rolling back would discard them")
            if src_content.get("unsupported") or content.get("unsupported"):
                raise SystemExit("refusing: a database object is outside the row proof (e.g. a WITHOUT ROWID "
                                 f"table): {sorted(set(src_content.get('unsupported') or []) | set(content.get('unsupported') or []))}")
            if content["header"].get("journal_header") != "wal" or src_content["header"].get("journal_header") != "wal":
                raise SystemExit("refusing: the restore needs a WAL-mode source and destination")
            if not _rows_preserved(src_content, content):
                raise SystemExit("refusing: the backup's rows, schema or settings differ from the live database's; "
                                 "a repair changes index entries only, so its rollback must too")
            m.update({"state": "rolling_back", "rollback_started_at": _now(),
                      "rollback_approval": str(owner_approval or "").strip()})
            _write_manifest(manifest_path, m)
            src.backup(con)                                    # under the same exclusive lock
            _checkpoint(con)
            _sha, restored, _raw = _live_content(pin)
            ok = _same_content(restored, m["original_content"])
            m.update({"state": "rolled_back" if ok else "rollback_unverified", "rollback_ok": ok,
                      "rolled_back_at": _now()})
            _write_manifest(manifest_path, m)
        finally:
            con.close()
            pin.close()
    finally:
        src.close()
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
