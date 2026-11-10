"""Parse a large durable JSON artifact once per version, for READ-ONLY callers.

One Home request parsed the ~1 MB recommendation file and the scan file a dozen
times through independent readers. Durable writers replace files atomically
(core.atomic_io: a new inode per write), so ``(inode, mtime_ns, size)`` changes
on every published version and identifies it exactly. A version that changes
while it is being parsed is returned but not cached.

The returned object is SHARED between callers and threads: callers must treat
it as read-only (copy before mutating). Use this only where that holds.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[tuple[int, int, int], Any]] = {}
_MAX_ENTRIES = 16


def file_identity(path: str | Path) -> tuple[int, int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _default_loader(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_shared(path: str | Path, loader: Callable[[Path], Any] | None = None) -> Any:
    """Return the parsed artifact, reusing the parse while the file is unchanged.

    ``loader`` (default: json) may raise; errors propagate and nothing is cached.
    A missing file returns ``None`` without calling the loader.
    """
    target = Path(path)
    key = str(target)
    ident = file_identity(target)
    if ident is None:
        with _LOCK:
            _CACHE.pop(key, None)
        return None
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None and hit[0] == ident:
            return hit[1]
    value = (loader or _default_loader)(target)
    if file_identity(target) == ident:
        with _LOCK:
            if len(_CACHE) >= _MAX_ENTRIES and key not in _CACHE:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = (ident, value)
    return value


def clear() -> None:
    with _LOCK:
        _CACHE.clear()
