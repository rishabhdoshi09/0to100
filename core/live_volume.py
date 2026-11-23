"""Is a path on the live runtime's volumes? One answer for every "never the live runtime" guard.

The production runtime lives under ``/Volumes`` (the APFS sparsebundle on the
USB disk). Maintenance and research tools must refuse it, and they used to
compare path TEXT: ``str(path).startswith("/Volumes/")``. Text is not identity
on macOS:

* the volumes are case-insensitive, so ``/volumes/...`` reaches the same files;
* ``/Volumes`` is a firmlink of the data volume, so
  ``/System/Volumes/Data/Volumes/...`` reaches them too, and ``realpath`` keeps
  that spelling (cycle-8 review, part 4).

``on_live_volume`` answers by identity as well: the path is live when any
component met while resolving it, one component at a time and through every
symlink as the kernel would, IS ``/Volumes`` (same device and inode). So
``link/..`` means what the kernel makes of it (GPT C8.5-04), an anchor behind
an alias of an alias is found (GPT C8.6-02), and the walk stops at the anchor,
never stat-ing anything inside a mounted volume (a dead USB mount can block a
stat).

A spelling that only READS as /Volumes (``/Volumes/link/../x`` with the link
pointing elsewhere) is refused too: deliberately conservative.
"""
from __future__ import annotations

import os
from pathlib import Path
from stat import S_ISLNK

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


MAX_LINKS = 40                                  # as the kernel's MAXSYMLINKS


def _identity_live(raw: str, anchor: tuple[int, int]) -> bool:
    """Resolve ``raw`` one component at a time, as the kernel does, and report whether ANY
    component on the way, including those reached through symlinks, IS the anchor.

    Symlinks are expanded in place (an absolute target restarts at '/'); '..' applies to
    the resolved, symlink-free path, so a textual parent is the real one. Each component
    is lstat-ed and compared BEFORE anything below it is touched, so an anchor hidden
    behind an alias is found (GPT C8.6-02) and nothing inside a mount below the anchor is
    ever stat-ed (a dead USB mount can block a stat). More than MAX_LINKS symlinks
    (a loop) counts as live: refusing is the safe answer.
    """
    absolute = raw if raw.startswith("/") else os.getcwd().rstrip("/") + "/" + raw
    pending = [part for part in absolute.split("/") if part]
    resolved, links = "/", 0
    while pending:
        part = pending.pop(0)
        if part == ".":
            continue
        if part == "..":
            resolved = os.path.dirname(resolved.rstrip("/")) or "/"
            try:
                st = os.lstat(resolved)
            except OSError:
                return False
            if (st.st_dev, st.st_ino) == anchor:
                return True
            continue
        candidate = resolved.rstrip("/") + "/" + part
        try:
            st = os.lstat(candidate)
        except OSError:
            return False                       # nothing below a missing component exists
        if S_ISLNK(st.st_mode):
            links += 1
            if links > MAX_LINKS:
                return True
            target = os.readlink(candidate)
            if target.startswith("/"):
                resolved = "/"
            pending = [p for p in target.split("/") if p] + pending
            continue
        if (st.st_dev, st.st_ino) == anchor:
            return True
        resolved = candidate
    return False


def on_live_volume(path: str | os.PathLike[str]) -> bool:
    """True when ``path`` is ``/Volumes`` or below it: by text, by identity, or resolved."""
    raw = os.fspath(path)
    if _text_live(os.path.abspath(raw)):
        return True                            # decided before anything inside the mount is touched
    anchor = _anchor_identity()
    if anchor is not None and _identity_live(raw, anchor):
        return True
    return _text_live(os.path.realpath(raw))   # no anchor on this host (Linux): text only


def refuse_live(path: str | os.PathLike[str], message: str) -> Path:
    """``path`` resolved, or SystemExit(message) when it is on the live volumes (checked as
    spelled BEFORE resolving, so a dead mount is never traversed to answer)."""
    if on_live_volume(path):
        raise SystemExit(message)
    return Path(path).resolve()
