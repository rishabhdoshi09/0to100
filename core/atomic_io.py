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


def make_sqlite_durable(connection: Any) -> None:
    """Ask SQLite for real stable-storage syncs on macOS (``F_FULLFSYNC``).

    For stores whose loss costs money or truth and that commit rarely (paper
    F&O book, OMS orders, protection plans). Busy stores (job ledgers, caches)
    must not use it: each commit then waits ~0.4 s on the owner's USB disk.
    """
    if sys.platform != "darwin":
        return
    try:
        connection.execute("PRAGMA fullfsync=ON")
        connection.execute("PRAGMA checkpoint_fullfsync=ON")
    except Exception:
        pass
