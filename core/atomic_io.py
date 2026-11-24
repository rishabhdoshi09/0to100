"""Crash- and concurrency-safe replacement of durable files.

Every write goes to its own temporary file in the target's directory (same
filesystem, so ``os.replace`` is atomic), is flushed and fsynced, then replaces
the target in one step. Readers see either the old file or the new one, never a
half-written one, and concurrent writers never share a temporary inode (the
fixed ``<name>.tmp`` idiom let one writer publish or delete another's
half-written file). On any failure the previous file is left untouched and the
temporary file is removed.

Last writer wins. Callers that must not lose a concurrent update still need
their own lock or generation check around read-modify-write.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, IO

_DEFAULT_MODE = 0o644


def durable_sync(fd: int) -> None:
    """Force ``fd``'s data to stable storage, not just to the drive's cache.

    On macOS ``fsync`` only hands data to the drive; ``F_FULLFSYNC`` makes the
    drive flush its write cache too (Apple's guidance for durability). On the
    owner's USB-backed runtime that costs ~0.4 s per call, so only state whose
    loss matters (paper book, owner pause/halt) asks for it. Elsewhere, or if
    the filesystem refuses the request, plain ``fsync`` is used.
    """
    if sys.platform == "darwin":
        try:
            import fcntl

            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            return
        except (OSError, AttributeError, ValueError):
            pass
    os.fsync(fd)


def _sync_directory(directory: Path) -> None:
    """Make the rename itself durable (best effort; not every FS supports it)."""
    try:
        dfd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        durable_sync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _replace_with(target: Path, fill: Callable[[IO[Any]], None], *, mode: str, encoding: str | None,
                  newline: str | None, fsync: bool, durable: bool = False) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        perms = target.stat().st_mode & 0o777
    except FileNotFoundError:
        perms = _DEFAULT_MODE
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        os.fchmod(fd, perms)
        with os.fdopen(fd, mode, encoding=encoding, newline=newline) as handle:
            fill(handle)
            handle.flush()
            if durable:
                durable_sync(handle.fileno())
            elif fsync:
                os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        if durable:
            _sync_directory(target.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8", fsync: bool = True,
                      durable: bool = False) -> Path:
    """``durable=True``: full flush to stable storage (file and directory); see ``durable_sync``."""
    return _replace_with(Path(path), lambda fh: fh.write(text), mode="w", encoding=encoding,
                         newline=None, fsync=fsync, durable=durable)


def atomic_write_bytes(path: str | Path, data: bytes, *, fsync: bool = True) -> Path:
    return _replace_with(Path(path), lambda fh: fh.write(data), mode="wb", encoding=None,
                         newline=None, fsync=fsync)


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    default: Callable[[Any], Any] | None = str,
    sort_keys: bool = False,
    fsync: bool = True,
) -> Path:
    # Serialize first: an unserializable payload must not touch the file at all.
    text = json.dumps(payload, indent=indent, default=default, sort_keys=sort_keys)
    return atomic_write_text(path, text, fsync=fsync)


def atomic_write_with(
    path: str | Path,
    fill: Callable[[IO[str]], None],
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
    fsync: bool = True,
) -> Path:
    """Stream into the temporary file (CSV writers etc.), then replace atomically."""
    return _replace_with(Path(path), fill, mode="w", encoding=encoding, newline=newline, fsync=fsync)


def quarantine_torn_jsonl_tail(path: str | Path) -> Path | None:
    """Recover a JSONL ledger whose LAST line was cut by a crash; call it under the writer lock.

    An append-only ledger written with plain appends can be left with an unterminated,
    unparseable final line. Every reader that validates lines then refuses the whole file,
    and the pipeline behind it stops for good (lifecycle audit F3/F5, 24 Nov). This moves
    exactly those trailing bytes to a durable ``<name>.torn-<stamp>`` sidecar (kept for
    forensics, never deleted) and truncates the ledger to its last complete line, syncing
    each step. A complete last line, even one missing its newline, is left alone, and so is
    any invalid line BEFORE the last: that is corruption, not an interrupted append, and the
    reader must keep refusing it. Returns the sidecar path, or None when nothing was cut.
    """
    import json
    import time
    import uuid

    target = Path(path)
    try:
        raw = target.read_bytes()
    except FileNotFoundError:
        return None
    if not raw or raw.endswith(b"\n"):
        return None
    cut = raw.rfind(b"\n") + 1
    tail = raw[cut:]
    try:
        json.loads(tail.decode("utf-8"))
        return None                                # a complete row: not torn
    except (UnicodeDecodeError, ValueError):
        pass
    side = target.with_name(f"{target.name}.torn-{time.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}")
    fd = os.open(str(side), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, tail)
        durable_sync(fd)
    finally:
        os.close(fd)
    _sync_directory(side.parent)
    with target.open("r+b") as fh:
        fh.truncate(cut)
        fh.flush()
        durable_sync(fh.fileno())
    print(f"[LEDGER] quarantined a torn last line of {target.name} ({len(tail)} bytes) to {side.name}",
          file=sys.stderr, flush=True)
    return side


def make_sqlite_durable(connection: Any) -> None:
    """Ask SQLite for real stable-storage syncs on macOS (``F_FULLFSYNC``).

    For stores whose loss costs money or truth (paper F&O book, OMS orders,
    protection plans, decision freezes). With ``synchronous=FULL`` each commit
    then waits for one full flush: measured on the owner's USB stack (24 Nov),
    p50 58-85 ms, p99 ~0.6-0.8 s. A store that also closes its last connection
    per write pays a barriered checkpoint on top (p50 280 ms): keep it open
    (``product.sqlite_runtime.keep_open``). Every WAL store needs at least
    :func:`make_sqlite_checkpoint_safe`.
    """
    if sys.platform != "darwin":
        return
    import sqlite3

    try:
        connection.execute("PRAGMA fullfsync=ON")
        connection.execute("PRAGMA checkpoint_fullfsync=ON")
    except sqlite3.Error as exc:      # e.g. a closed connection: reported, the store still works
        print(f"[atomic_io] durable syncs not set: {exc}", file=sys.stderr, flush=True)


def make_sqlite_checkpoint_safe(connection: Any) -> None:
    """Make a WAL store's checkpoints real write barriers on macOS (Stage 1 finding F1).

    On macOS ``fsync()`` does not flush the drive's cache; only ``F_FULLFSYNC``
    does. On the owner's storage stack (APFS in a sparsebundle on an exFAT USB
    disk) both F1 losses were writes from the last seconds before a clamshell
    sleep dropped the disk, lost out of order. A WAL checkpoint
    copies pages into the database and then lets the WAL be overwritten: without
    a real barrier between the two, a power cut keeps some new pages and loses
    others. On 19 Nov 2026 that left a decision-freeze ledger with index pages
    from one transaction and table pages from an older one.

    ``checkpoint_fullfsync`` makes the checkpoint syncs ``F_FULLFSYNC``: one full
    flush per checkpoint, at any ``synchronous`` level. A persistent connection
    checkpoints rarely (every ~1000 WAL pages), so its cost is negligible. A store
    that opens a connection per write checkpoints on every close (the last
    connection to a WAL database checkpoints it and deletes the WAL), so each write
    pays one flush: that is exactly the step F1's writer lacked a barrier for.
    With ``synchronous`` at NORMAL or below, commits never sync, so ``fullfsync``
    is also set: it only adds the WAL-header sync. With FULL it would add a full
    flush to every commit, which is :func:`make_sqlite_durable`'s deliberate
    choice. Call it after the store's ``journal_mode`` and ``synchronous`` pragmas.
    """
    if sys.platform != "darwin":
        return
    import sqlite3

    try:
        connection.execute("PRAGMA checkpoint_fullfsync=ON")
        level = connection.execute("PRAGMA synchronous").fetchone()[0]
        if int(level) <= 1:
            connection.execute("PRAGMA fullfsync=ON")
    except sqlite3.Error as exc:      # e.g. a closed or read-only connection: the store still works
        print(f"[atomic_io] checkpoint barrier not set: {exc}", file=sys.stderr, flush=True)
