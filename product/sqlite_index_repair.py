"""Repair a SQLite database whose INDEXES disagree with their tables (Stage 1 finding F1).

    plan     --target <db> --staging <dir you own> --out <manifest.json> [--expect-stale N]
    apply    --manifest <manifest.json> --owner-approval "<words>" [--target-is-copy]
    rollback --manifest <manifest.json> --owner-approval "<words>" [--target-is-copy] [--backup <copy>]
    check    --target <db> --staging <dir you own>        (read-only diagnosis of a byte copy)

The production database is never opened by ``plan`` or ``check``: they take a
stable byte copy (the source's size, mtime and ctime must not move during the
single read that also hashes it) and work on that.

``plan`` runs ``REINDEX`` on a second copy and proves the result before anything
can be applied:

* the damage is index-only: every ``integrity_check`` message is one of the
  index forms (``wrong # of entries in index``, ``row N missing from index``,
  ``non-unique entry in index``); table damage is not repairable by REINDEX and
  is refused;
* after the repair ``integrity_check`` is ``ok``, every index holds exactly its
  table's (key, rowid) entries, and every unique key resolves through its index
  to its own row;
* every table row is unchanged: the same rowids, and the same content digest
  over every column of every row, table by table;
* the schema (every ``sqlite_master`` entry except root pages) and the header
  settings are unchanged, and foreign-key checks are no worse.

``apply`` needs the owner's words and a stopped desk (copy mode only away from
/Volumes). It re-verifies the target's and the repaired copy's sha256, refuses a
hot journal or open handles, writes a verified backup next to the target and
syncs it, records ``swapping`` durably, then swaps the verified copy in with one
atomic rename and re-checks the result. ``rollback`` restores the backup from a
verified apply (only while the target still holds exactly the repaired bytes) or
from any interrupted state. Re-running ``apply`` after it succeeded is a no-op;
a healthy database plans as ``nothing_to_repair``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from product.bhav_store_rebuild import (
    _check_desk,
    _durable_copy,
    _fsync_dir,
    _full_fsync,
    _held_open,
    _not_live,
    _on_live_volume,
    _sha256,
    _write_manifest,
)

TOOL = "sqlite_index_repair"
VERSION = 1
INDEX_DAMAGE = (
    re.compile(r"^wrong # of entries in index \S+$"),
    re.compile(r"^row \d+ missing from index \S+$"),
    re.compile(r"^non-unique entry in index \S+$"),
)
HEADER_PRAGMAS = ("page_size", "user_version", "application_id", "encoding", "auto_vacuum")
_LIST_LIMIT = 200


# ── copies ───────────────────────────────────────────────────────────────────

def _ident(st: os.stat_result) -> tuple[int, int, int]:
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _stable_copy(src: Path, dst: Path) -> dict[str, Any]:
    """One read of ``src`` that hashes and writes ``dst``; refuses if ``src`` moved meanwhile."""
    before = os.stat(src)
    h = hashlib.sha256()
    with open(src, "rb") as inp, open(dst, "xb") as out:
        for chunk in iter(lambda: inp.read(1 << 20), b""):
            h.update(chunk)
            out.write(chunk)
        out.flush()
        _full_fsync(out.fileno())
    after = os.stat(src)
    if _ident(before) != _ident(after):
        dst.unlink(missing_ok=True)
        raise SystemExit(f"refusing: {src} changed while it was copied (a writer is active; stop it first)")
    sha = h.hexdigest()
    if _sha256(dst) != sha:
        dst.unlink(missing_ok=True)
        raise SystemExit(f"refusing: the copy of {src.name} does not hash to the bytes read")
    return {"sha256": sha, "size": before.st_size, "mtime_ns": before.st_mtime_ns}


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


# ── diagnosis (on copies only) ───────────────────────────────────────────────

def _open_ro(path: Path) -> sqlite3.Connection:
    # immutable: no locks, no -wal/-shm created; exact because the copy has no WAL
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _val(v: Any) -> tuple[str, Any]:
    return (type(v).__name__, v.hex() if isinstance(v, (bytes, bytearray, memoryview)) else v)


def _tables(con: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """User rowid tables, and unsupported objects (WITHOUT ROWID or virtual tables)."""
    tables, unsupported = [], []
    for name, sql in con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
        text = (sql or "").upper()
        if text.startswith("CREATE VIRTUAL TABLE") or re.search(r"WITHOUT\s+ROWID", text):
            unsupported.append(f"table {name}")
        else:
            tables.append(name)
    return tables, unsupported


def _index_columns(con: sqlite3.Connection, index: str) -> list[str] | None:
    """Plain key columns of ``index``; None for an expression index."""
    cols = []
    for row in con.execute(f"PRAGMA index_xinfo({_q(index)})"):
        _seq, cid, name, _desc, _coll, key = row
        if not key:
            continue
        if cid == -2 or name is None:
            return None
        cols.append(name)
    return cols


def _row_digest(con: sqlite3.Connection, table: str) -> dict[str, Any]:
    h = hashlib.sha256()
    n = 0
    for row in con.execute(f"SELECT rowid, * FROM {_q(table)} NOT INDEXED ORDER BY rowid"):
        h.update(json.dumps([_val(v) for v in row], default=str, separators=(",", ":")).encode())
        h.update(b"\n")
        n += 1
    return {"rows": n, "digest": h.hexdigest()}


def _index_check(con: sqlite3.Connection, table: str, index: str, cols: list[str], unique: bool) -> dict[str, Any]:
    sel = ", ".join(_q(c) for c in cols)
    # covering query: the values come from the index b-tree itself
    idx = Counter(tuple(_val(v) for v in r) for r in con.execute(
        f"SELECT {sel}, rowid FROM {_q(table)} INDEXED BY {_q(index)}"))
    tab = Counter(tuple(_val(v) for v in r) for r in con.execute(
        f"SELECT {sel}, rowid FROM {_q(table)} NOT INDEXED"))
    stale = list((idx - tab).elements())
    missing = list((tab - idx).elements())
    lookup_bad = []
    if unique:
        where = " AND ".join(f"{_q(c)} = ?" for c in cols)
        for key in tab:
            values = [v[1] if v[0] not in ("bytes", "bytearray", "memoryview") else bytes.fromhex(v[1])
                      for v in key[:-1]]
            if any(v is None for v in values):
                continue                    # NULLs are not unique keys
            got = [r[0] for r in con.execute(
                f"SELECT rowid FROM {_q(table)} INDEXED BY {_q(index)} WHERE {where}", values)]
            if got != [key[-1][1]]:
                lookup_bad.append({"key": [v[1] for v in key[:-1]], "rowid": key[-1][1], "resolved_to": got})
    return {"table": table, "columns": cols, "unique": unique,
            "index_entries": sum(idx.values()), "table_rows": sum(tab.values()),
            "stale": [{"key": [v[1] for v in e[:-1]], "rowid": e[-1][1]} for e in stale[:_LIST_LIMIT]],
            "stale_count": len(stale),
            "missing": [{"key": [v[1] for v in e[:-1]], "rowid": e[-1][1]} for e in missing[:_LIST_LIMIT]],
            "missing_count": len(missing),
            "lookup_mismatches": lookup_bad[:_LIST_LIMIT], "lookup_mismatch_count": len(lookup_bad)}


def diagnose(path: str | Path) -> dict[str, Any]:
    """Read-only diagnosis of a database COPY (never call it on a live file)."""
    con = _open_ro(Path(path))
    try:
        integrity = [r[0] for r in con.execute("PRAGMA integrity_check(1000)")]
        schema = [list(r) for r in con.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name")]
        header = {p: con.execute(f"PRAGMA {p}").fetchone()[0] for p in HEADER_PRAGMAS}
        with open(path, "rb") as fh:
            raw = fh.read(100)
        header["journal_header"] = {1: "rollback", 2: "wal"}.get(raw[18], raw[18])
        fk = [list(r) for r in con.execute("PRAGMA foreign_key_check")][:_LIST_LIMIT]
        tables, unsupported = _tables(con)
        digests = {t: _row_digest(con, t) for t in tables}
        if con.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_sequence'").fetchone():
            digests["sqlite_sequence"] = _row_digest(con, "sqlite_sequence")
        indexes: dict[str, Any] = {}
        for t in tables:
            for _seq, name, unique, _origin, partial in con.execute(f"PRAGMA index_list({_q(t)})").fetchall():
                cols = _index_columns(con, name)
                if cols is None or partial:
                    unsupported.append(f"index {name} ({'expression' if cols is None else 'partial'})")
                    continue
                indexes[name] = _index_check(con, t, name, cols, bool(unique))
    finally:
        con.close()
    index_only = all(any(p.match(m) for p in INDEX_DAMAGE) for m in integrity) if integrity != ["ok"] else True
    return {
        "integrity": integrity,
        "integrity_ok": integrity == ["ok"],
        "index_damage_only": index_only,
        "schema": schema,
        "schema_sha256": hashlib.sha256(json.dumps(schema).encode()).hexdigest(),
        "header": header,
        "foreign_key_violations": fk,
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


# ── plan ─────────────────────────────────────────────────────────────────────

def _entry(target: Path) -> Path:
    """The directory entry os.replace writes: parent resolved, final name kept (GPT C7-04)."""
    return Path(target).parent.resolve() / Path(target).name


def check(target: str | Path, staging: str | Path) -> dict[str, Any]:
    target = Path(target)
    staging = _not_live(Path(staging), "staging")
    work = staging / f"check-{time.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True)
    side = _sidecars(target)
    copy = _stable_copy(target, work / "copy.db")
    out = _diagnose_or_refuse(work / "copy.db")
    out.update({"target": str(_entry(target)), "target_sha256": copy["sha256"], "sidecars": side,
                "hot_journal": _hot(side), "copy": str(work / "copy.db")})
    return out


def plan(target: str | Path, staging: str | Path, out: str | Path, *, expect_stale: int | None = None) -> dict[str, Any]:
    target = Path(target)
    entry = _entry(target)
    staging = _not_live(Path(staging), "staging")
    out = Path(out)
    _not_live(out.parent, "the manifest")
    if not target.is_file():
        raise SystemExit(f"refusing: no database at {target}")
    side = _sidecars(target)
    hot = _hot(side)
    if hot:
        raise SystemExit(f"refusing: {target.name} has {hot}: the main file alone is not the database "
                         "(stop its writers so SQLite checkpoints, then plan)")
    plan_id = uuid.uuid4().hex
    work = staging / f"plan-{time.strftime('%Y%m%d_%H%M%S')}-{plan_id[:8]}"
    work.mkdir(parents=True)
    original, repaired = work / "original.db", work / "repaired.db"
    copy = _stable_copy(target, original)
    before = _diagnose_or_refuse(original)
    m: dict[str, Any] = {
        "tool": TOOL, "version": VERSION, "plan_id": plan_id,
        "planned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target": str(entry), "target_resolved": str(target.resolve()),
        "target_sha256": copy["sha256"], "target_size": copy["size"], "target_mtime_ns": copy["mtime_ns"],
        "sidecars_at_plan": side, "work": str(work), "original_copy": str(original),
        "before": before,
    }
    if _consistent(before):
        m.update({"state": "nothing_to_repair", "ready": False, "nothing_to_repair": True,
                  "gates": {"damage_found": False}})
        _write_manifest(out, m)
        return m

    shutil.copy2(original, repaired)
    con = sqlite3.connect(str(repaired))
    try:
        con.execute("REINDEX")
        con.commit()
        if str(before["header"].get("journal_header")) == "wal":
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()
    for suffix in ("-wal", "-shm"):
        leftover = Path(str(repaired) + suffix)
        if leftover.exists() and leftover.stat().st_size == 0:
            leftover.unlink()
    after = _diagnose_or_refuse(repaired)
    tables_equal = before["tables"] == after["tables"]
    gates = {
        "damage_found": True,
        "index_damage_only": bool(before["index_damage_only"]),
        "supported_schema": not before["unsupported"],
        "repaired_integrity_ok": bool(after["integrity_ok"]),
        "indexes_consistent_after": _consistent(after),
        "rows_identical": tables_equal,
        "schema_identical": before["schema_sha256"] == after["schema_sha256"] and before["header"] == after["header"],
        "foreign_keys_no_worse": len(after["foreign_key_violations"]) <= len(before["foreign_key_violations"]),
        "no_leftover_sidecars": not any(Path(str(repaired) + s).exists() for s in ("-wal", "-shm", "-journal")),
        "only_index_entries_removed": after["stale_total"] == 0 and after["missing_total"] == 0
                                       and all(i["index_entries"] == i["table_rows"] for i in after["indexes"].values()),
    }
    if expect_stale is not None:
        gates["expected_stale"] = before["stale_total"] == int(expect_stale) and before["missing_total"] == 0
    m.update({
        "after": after,
        "repaired": str(repaired), "repaired_sha256": _sha256(repaired),
        "stale_entries": {name: i["stale"] for name, i in before["indexes"].items() if i["stale_count"]},
        "missing_entries": {name: i["missing"] for name, i in before["indexes"].items() if i["missing_count"]},
        "expect_stale": expect_stale,
        "gates": gates, "ready": all(gates.values()),
        "state": "planned",
    })
    _write_manifest(out, m)
    return m


# ── apply / rollback ─────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _target_of(m: dict[str, Any]) -> Path:
    target = Path(m["target"])
    if _entry(target) != target:
        raise SystemExit("refusing: the manifest's target is not a canonical directory entry")
    return target


def _require_quiet(target: Path, target_is_copy: bool) -> None:
    _check_desk(target, target_is_copy)
    hot = _hot(_sidecars(target))
    if hot:
        raise SystemExit(f"refusing: {target.name} has {hot} (a writer ran since the plan; stop it and re-plan)")
    holders = _held_open(target) if target.exists() else []
    if holders:
        raise SystemExit(f"refusing: {target.name} is held open by {holders} (stop the desk first)")


def _verify_live(target: Path, expect_sha: str) -> bool:
    if not target.is_file() or _sha256(target) != expect_sha:
        return False
    con = _open_ro(target)        # immutable read: exact, since there is no WAL (checked above)
    try:
        return [r[0] for r in con.execute("PRAGMA integrity_check")] == ["ok"]
    finally:
        con.close()


def apply(manifest_path: str | Path, owner_approval: str, *, target_is_copy: bool = False) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    if m.get("tool") != TOOL:
        raise SystemExit("refusing: not a sqlite_index_repair manifest")
    if not str(owner_approval or "").strip():
        raise SystemExit("refusing: apply needs the owner's approval in words (--owner-approval)")
    target = _target_of(m)
    if m.get("state") == "applied":
        if target.is_file() and _sha256(target) == m["repaired_sha256"]:
            return {**m, "noop": True}                        # idempotent re-run
        raise SystemExit("refusing: the manifest says applied but the target no longer holds the repaired bytes")
    if m.get("state") != "planned":
        raise SystemExit(f"refusing: the manifest is in state {m.get('state')!r}, not 'planned' "
                         "(an interrupted apply is recovered with rollback, then re-planned)")
    if not m.get("ready"):
        raise SystemExit(f"refusing: the plan's gates did not all pass: {m.get('gates')}")
    if target_is_copy and (_on_live_volume(target) or _on_live_volume(target.resolve())):
        raise SystemExit("refusing: --target-is-copy is never honoured under /Volumes")
    repaired = Path(m["repaired"])
    _require_quiet(target, target_is_copy)
    if _sha256(target) != m["target_sha256"]:
        raise SystemExit("refusing: the target changed since the plan (re-plan)")
    if _sha256(repaired) != m["repaired_sha256"]:
        raise SystemExit("refusing: the repaired copy changed since the plan (re-plan)")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(f"{target.name}.pre-reindex-{stamp}")
    btmp = _durable_copy(target, target.parent, ".backup-", m["target_sha256"])
    os.replace(btmp, backup)
    _fsync_dir(target.parent)                     # the backup is durable BEFORE anything is replaced
    candidate = _durable_copy(repaired, target.parent, ".reindex-", m["repaired_sha256"])
    swapped = False
    try:
        m.update({"state": "swapping", "backup": str(backup), "owner_approval": owner_approval.strip(),
                  "swap_started_at": _now()})
        _write_manifest(manifest_path, m)
        _require_quiet(target, target_is_copy)    # nothing started while the copies were written
        if _sha256(target) != m["target_sha256"]:
            m.update({"state": "planned", "swap_aborted_at": _now()})
            _write_manifest(manifest_path, m)
            raise SystemExit("refusing: the target changed during the apply (re-plan)")
        os.replace(candidate, target)
        swapped = True
    finally:
        if not swapped:
            Path(candidate).unlink(missing_ok=True)   # never leave the verified copy beside the target
    _fsync_dir(target.parent)
    ok = _verify_live(target, m["repaired_sha256"])
    m.update({"state": "applied" if ok else "swap_unverified", "applied_ok": ok, "applied_at": _now()})
    _write_manifest(manifest_path, m)
    if not ok:
        raise SystemExit("the target does not verify after the swap: run rollback")
    return m


def rollback(manifest_path: str | Path, owner_approval: str = "", *, target_is_copy: bool = False,
             backup_path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    if m.get("tool") != TOOL:
        raise SystemExit("refusing: not a sqlite_index_repair manifest")
    target = _target_of(m)
    live = not target_is_copy or _on_live_volume(target) or _on_live_volume(target.resolve())
    if live and not str(owner_approval or "").strip():
        raise SystemExit("refusing: a live rollback needs the owner's approval in words (--owner-approval)")
    interrupted = m.get("state") in {"swapping", "swap_unverified", "rolling_back", "rollback_unverified"}
    if not interrupted and m.get("state") not in {"applied", "rolled_back"}:
        raise SystemExit(f"nothing to roll back: the manifest is in state {m.get('state')!r}")
    recorded = str(backup_path or m.get("backup") or "").strip()
    backup = Path(recorded) if recorded else None
    if backup is None or not backup.is_file():
        raise SystemExit("nothing to roll back: no backup recorded or found (pass --backup <verified copy>)")
    if _sha256(backup) != m["target_sha256"]:
        raise SystemExit("refusing: the backup does not match the pre-repair sha256")
    _require_quiet(target, target_is_copy)
    current = _sha256(target) if target.is_file() else None
    if current == m["target_sha256"]:
        m.update({"state": "rolled_back", "rollback_ok": True, "rolled_back_at": _now()})
        _write_manifest(manifest_path, m)
        return m                                               # already the pre-repair bytes
    if current != m["repaired_sha256"] and not interrupted:
        raise SystemExit("refusing: the target changed after the apply (a writer saved newer data); "
                         "rolling back would discard it")
    candidate = _durable_copy(backup, target.parent, ".rollback-", m["target_sha256"])
    m.update({"state": "rolling_back", "rollback_started_at": _now(),
              "rollback_approval": str(owner_approval or "").strip()})
    _write_manifest(manifest_path, m)
    os.replace(candidate, target)
    _fsync_dir(target.parent)
    ok = target.is_file() and _sha256(target) == m["target_sha256"]
    m.update({"state": "rolled_back" if ok else "rollback_unverified", "rollback_ok": ok,
              "rolled_back_from": "interrupted apply" if interrupted else "verified apply",
              "rolled_back_at": _now()})
    _write_manifest(manifest_path, m)
    if not ok:
        raise SystemExit("rollback did not restore the pre-repair bytes")
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
        return 0 if (m.get("ready") or m.get("nothing_to_repair")) else 1
    if args.cmd == "check":
        d = check(args.target, args.staging)
        print(json.dumps({k: d[k] for k in ("integrity", "stale_total", "missing_total", "lookup_mismatch_total",
                                            "unsupported", "hot_journal", "target_sha256")}, indent=1, default=str))
        return 0 if _consistent(d) else 1
    if args.cmd == "apply":
        m = apply(args.manifest, args.owner_approval, target_is_copy=args.target_is_copy)
        print(json.dumps({k: m.get(k) for k in ("state", "applied_ok", "noop", "backup", "repaired_sha256")}, indent=1))
        return 0
    m = rollback(args.manifest, args.owner_approval, target_is_copy=args.target_is_copy, backup_path=args.backup)
    print(json.dumps({k: m.get(k) for k in ("state", "rollback_ok", "target_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
