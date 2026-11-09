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
import tempfile
from pathlib import Path
from typing import Any, Callable, IO

_DEFAULT_MODE = 0o644


def _replace_with(target: Path, fill: Callable[[IO[Any]], None], *, mode: str, encoding: str | None,
                  newline: str | None, fsync: bool) -> Path:
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
            if fsync:
                os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8", fsync: bool = True) -> Path:
    return _replace_with(Path(path), lambda fh: fh.write(text), mode="w", encoding=encoding,
                         newline=None, fsync=fsync)


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
