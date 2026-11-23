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
prefix of it, AS SPELLED, IS ``/Volumes`` (same device and inode). Each prefix
is resolved by the kernel itself, so ``link/..`` means what the kernel makes of
it (symlink first, then '..'), not what textual normalization would (GPT
C8.5-04). The walk goes from the root down and stops at the first match, so a
spelling that reaches the mounts through ``/Volumes`` never stats anything
inside a mounted volume (a dead USB mount can block a stat); a symlink
component is text-checked from its link before it is followed.

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


def _prefixes(raw: str) -> list[str]:
    """``/a``, ``/a/b``, ... of ``raw`` made absolute, WITHOUT normalizing '.' or '..'."""
    absolute = raw if raw.startswith("/") else os.getcwd().rstrip("/") + "/" + raw
    out, current = [], ""
    for part in absolute.split("/"):
        if part:
            current += "/" + part
            out.append(current)
    return out


def _identity_live(raw: str, anchor: tuple[int, int]) -> bool:
    for prefix in _prefixes(raw):
        try:
            link = os.lstat(prefix)
        except OSError:
            return False                       # below here nothing exists to be the anchor
        if S_ISLNK(link.st_mode):
            target = os.readlink(prefix)
            joined = target if target.startswith("/") else os.path.dirname(prefix) + "/" + target
            if _text_live(os.path.abspath(joined)):
                return True                    # decided before following it into a mount
        try:
            st = os.stat(prefix)
        except OSError:
            return False
        if (st.st_dev, st.st_ino) == anchor:
            return True
    return False


def on_live_volume(path: str | os.PathLike[str]) -> bool:
    """True when ``path`` is ``/Volumes`` or below it: by text, by identity, or resolved."""
    raw = os.fspath(path)
    if _text_live(os.path.abspath(raw)):
        return True                            # decided before anything inside the mount is touched
    anchor = _anchor_identity()
    if anchor is not None and _identity_live(raw, anchor):
        return True
    return _text_live(os.path.realpath(raw))   # e.g. a dangling link into an absent volume


def refuse_live(path: str | os.PathLike[str], message: str) -> Path:
    """``path`` resolved, or SystemExit(message) when it is on the live volumes (checked as
    spelled BEFORE resolving, so a dead mount is never traversed to answer)."""
    if on_live_volume(path):
        raise SystemExit(message)
    return Path(path).resolve()
