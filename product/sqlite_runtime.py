"""Shared SQLite connection policy for QuantTerm product stores.

WAL + busy_timeout are the default. Callers still own schema CREATE.
"""
from __future__ import annotations

import atexit
import os
import sqlite3
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from core.runtime_paths import logs_dir, logs_path

#: Idle keeper connections, at most this many per process (launchd's default soft
#: limit is 256 descriptors; each keeper holds the database, its -wal and -shm).
MAX_KEEPERS = 8
_KEEPERS: "OrderedDict[tuple[int, str], sqlite3.Connection]" = OrderedDict()
_KEEPERS_LOCK = threading.Lock()
_ATEXIT = {"registered_pid": None}
_INHERITED: list[sqlite3.Connection] = []

try:                                            # bound at import: the fork hook imports nothing
    import ctypes as _ctypes

    _PIN = _ctypes.pythonapi.Py_IncRef
    _PY_OBJECT = _ctypes.py_object
except (ImportError, AttributeError):           # no way to pin in a forked child: no keepers at all
    _PIN = _PY_OBJECT = None


def _forget_inherited_keepers() -> None:
    """In a forked child: a fresh lock (the parent's may have been held at fork) and an empty
    registry. The parent's connections are parked, never used, evicted or closed here:
    SQLite forbids using a connection across fork (GPT C8.6-05). QuantTerm does not fork
    Python workers (macOS multiprocessing spawns; subprocesses exec); this makes it safe if
    something ever does."""
    global _KEEPERS_LOCK
    _KEEPERS_LOCK = threading.Lock()
    for con in _KEEPERS.values():
        # An extra reference the child never releases: the connection is never deallocated,
        # so it is never closed, not even when the child's interpreter shuts down. Closing it
        # here would let SQLite checkpoint from the parent's stale view and delete a WAL that
        # holds other processes' commits (cycle-8 review, part 7: 15 committed rows became 5).
        _PIN(_PY_OBJECT(con))
        _INHERITED.append(con)
    _KEEPERS.clear()
    _ATEXIT["registered_pid"] = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_inherited_keepers)


def keep_open(path: str | Path) -> None:
    """Hold one idle connection to a WAL store for the life of this process.

    A store that opens a connection per write (decision freezes: up to ~3,000 a
    day) makes every close the LAST close, and the last close checkpoints the WAL.
    With the F1 checkpoint barrier (``F_FULLFSYNC``) that costs, measured on the
    owner's USB stack on 24 Nov: p50 280 ms, p99 964 ms per write, and a 110 s
    stall in a 60 s sustained run. With one idle keeper the per-write close is not
    the last: p50 0.95 ms, p99 5.1 ms. Checkpoints then happen through SQLite's
    autocheckpoint (every ~1,000 WAL pages) and at the keeper's close at exit,
    each still behind the barrier, so the F1 reordering stays impossible.

    The keeper reads once so it holds the WAL's shared lock (that is what makes
    another connection's close "not the last"), never starts a transaction, and is
    closed (with its final, barriered checkpoint) at interpreter exit. Keyed by
    pid: a forked child never touches its parent's keeper. Best effort: a store
    that cannot be kept open keeps working exactly as before.
    """
    if _PIN is None:
        return                                  # a child could not pin an inherited keeper (GPT C8.7-05)
    target = str(Path(path).resolve())
    key = (os.getpid(), target)
    evicted: list[sqlite3.Connection] = []
    with _KEEPERS_LOCK:
        if key in _KEEPERS:
            _KEEPERS.move_to_end(key)
            return
        try:
            con = sqlite3.connect(target, timeout=30.0, check_same_thread=False)
        except sqlite3.Error:
            return
        try:
            con.execute("PRAGMA busy_timeout=30000")
            # this read opens the WAL and takes its shared lock, held until close: that is what
            # makes another connection's close "not the last" (tests/test_sqlite_keepers.py)
            mode = str(con.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            if mode != "wal":
                con.close()
                return
            con.execute("PRAGMA synchronous=NORMAL")
            from core.atomic_io import make_sqlite_checkpoint_safe

            make_sqlite_checkpoint_safe(con)                  # its exit checkpoint is barriered too
            con.execute("SELECT count(*) FROM sqlite_master").fetchone()   # belt and braces: a read
        except sqlite3.Error:
            con.close()
            return
        _KEEPERS[key] = con
        while len(_KEEPERS) > MAX_KEEPERS:
            _old_key, old = _KEEPERS.popitem(last=False)
            evicted.append(old)
        if _ATEXIT["registered_pid"] != key[0]:
            atexit.register(close_keepers)
            _ATEXIT["registered_pid"] = key[0]
    for old in evicted:
        _close_quietly(old)


def _close_quietly(con: sqlite3.Connection) -> None:
    try:
        con.close()                                           # the final checkpoint, barriered
    except sqlite3.Error as exc:
        print(f"[sqlite_runtime] keeper close failed: {exc}", file=sys.stderr, flush=True)


def close_keepers() -> None:
    """Close this process's keepers (registered at exit; also for tests and maintenance)."""
    pid = os.getpid()
    with _KEEPERS_LOCK:
        mine = [(k, c) for k, c in _KEEPERS.items() if k[0] == pid]
        for k, _c in mine:
            del _KEEPERS[k]
    for _k, con in mine:
        _close_quietly(con)


def connect_readonly(path: str | Path, *, timeout: float = 5.0) -> sqlite3.Connection | None:
    """Open another component's store for READING without side effects.

    A plain sqlite3.connect() creates a missing database file. A reader that
    does that leaves an empty, schema-less store behind for every later reader
    to trip over. Returns None when the store does not exist yet; mode=ro still
    sees committed WAL rows.
    """
    target = Path(path)
    if not target.exists():
        return None
    con = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout)
    con.row_factory = sqlite3.Row
    return con


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def connect(path: str | Path, *, timeout: float = 30.0, wal: bool = True) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(target), timeout=timeout)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    if wal:
        con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA foreign_keys=ON")
    # The journal mode lives in the FILE: a wal=False connection to a store that is
    # already WAL still checkpoints it, so the barrier follows the file (GPT F1-08).
    if wal or str(con.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal":
        from core.atomic_io import make_sqlite_checkpoint_safe

        make_sqlite_checkpoint_safe(con)        # real checkpoint barriers on macOS (finding F1)
    return con


def ensure_columns(con: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Add missing columns on an existing table. Safe after a killed CREATE.

    ``table`` and column names must be literals from our schema code.
    """
    if not table.isidentifier() or not table.replace("_", "").isalnum():
        raise ValueError("invalid table")
    existing = {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if not existing:
        return
    for name, decl in columns.items():
        if not name.isidentifier():
            raise ValueError("invalid column")
        if name in existing:
            continue
        con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def integrity_ok(path: str | Path) -> dict[str, object]:
    con = connect(path)
    try:
        row = con.execute("PRAGMA integrity_check").fetchone()
        wal = con.execute("PRAGMA journal_mode").fetchone()
        result = str(row[0] if row else "")
        return {
            "path": str(path),
            "ok": result.lower() == "ok",
            "integrity": result,
            "journal_mode": str(wal[0] if wal else ""),
        }
    finally:
        con.close()


def bootstrap_product_stores() -> dict[str, object]:
    """Create/upgrade required product schemas. Safe on existing DBs."""
    opened: list[str] = []
    errors: list[str] = []
    loaders = (
        ("pit_warehouse", "product.pit_warehouse", "_connect"),
        ("decision_journal", "product.decision_journal", "_connect"),
        ("candidate_lifecycle", "product.candidate_lifecycle", "_connect"),
        ("opportunity_memory", "product.opportunity_memory", "_connect"),
        ("learning_ledger", "product.learning_ledger", "_connect"),
        ("decision_freeze", "product.decision_freeze", "_connect"),
        ("pit_debt", "product.pit_debt", "_connect"),
        ("job_store", "research.autonomy.job_store", "JobStore"),
    )
    for name, module, attr in loaders:
        try:
            mod = __import__(module, fromlist=[attr])
            obj = getattr(mod, attr)
            if attr == "JobStore":
                root = Path(__file__).resolve().parents[1]
                obj(logs_dir() / "autonomy" / "jobs.db")
            else:
                con = obj()
                con.close()
            opened.append(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}"[:200])
    return {"ok": not errors, "opened": opened, "errors": errors}
