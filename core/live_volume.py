"""Is a path on the live runtime's volumes? One answer for every "never the live runtime" guard.

The production runtime lives under ``/Volumes`` (the APFS sparsebundle on the
USB disk). Maintenance and research tools must refuse it, and they used to
compare path TEXT: ``str(path).startswith("/Volumes/")``. Text is not identity
on macOS:

* the volumes are case-insensitive, so ``/volumes/...`` reaches the same files;
* ``/Volumes`` is a firmlink of the data volume, so
  ``/System/Volumes/Data/Volumes/...`` reaches them too, and ``realpath`` keeps
  that spelling (cycle-8 review, part 4).

``on_live_volume`` answers by identity as well: the path, symlinks resolved, is
live when any of its ancestors IS ``/Volumes`` (same device and inode). The
walk goes from the root down and stops at the first match, so it never stats
anything inside a mounted volume (a dead USB mount can block a stat).
"""
from __future__ import annotations

import os
from pathlib import Path

ANCHOR = "/Volumes"


def _text_live(text: str) -> bool:
    low = text.lower().rstrip("/")
    return low == ANCHOR.lower() or low.startswith(ANCHOR.lower() + "/")


def _anchor_identity() -> tuple[int, int] | None:
    try:
        st = os.stat(ANCHOR)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _identity_live(resolved: str, anchor: tuple[int, int]) -> bool:
    parts = Path(resolved).parts
    for depth in range(1, len(parts) + 1):
        try:
            st = os.stat(Path(*parts[:depth]))
        except OSError:
            return False                      # below here nothing exists to be the anchor
        if (st.st_dev, st.st_ino) == anchor:
            return True
    return False


def on_live_volume(path: str | os.PathLike[str]) -> bool:
    """True when ``path`` (as spelled, or with symlinks resolved) is ``/Volumes`` or below it."""
    text = os.path.abspath(os.fspath(path))
    if _text_live(text):
        return True                           # decided before anything inside the mount is touched
    resolved = os.path.realpath(text)
    if _text_live(resolved):
        return True
    anchor = _anchor_identity()
    return anchor is not None and _identity_live(resolved, anchor)


def refuse_live(path: str | os.PathLike[str], message: str) -> Path:
    """``path`` resolved, or SystemExit(message) when it is on the live volumes."""
    if on_live_volume(path):
        raise SystemExit(message)
    return Path(path).resolve()
