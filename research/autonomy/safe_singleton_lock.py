"""Stable-inode single-instance lock for the autonomy supervisor.

A flock file must not be unlinked while contenders may be waiting on its inode.  Keeping the path
stable means every process contends on the same kernel lock.  The PID is written only after the
exclusive lock is held, so a losing contender cannot truncate the visible owner identity.
"""
from __future__ import annotations

import os
from pathlib import Path

_INSTALLED = False


class StableSingleInstanceLock:
    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def acquire(self) -> bool:
        try:
            import fcntl

            handle = self.path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                return False
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            self._fh = handle
            return True
        except Exception:
            # QuantTerm's production launcher is macOS/Linux and therefore has flock.
            # Do not fall back to an unlink/O_EXCL scheme with different ownership semantics.
            try:
                if self._fh is not None:
                    self._fh.close()
            except Exception:
                pass
            self._fh = None
            return False

    def release(self) -> None:
        handle = self._fh
        self._fh = None
        if handle is None:
            return
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass
        # Never unlink. Future contenders must open the same inode.


def install_supervisor_singleton_lock() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from research.autonomy import supervisor as SUP

    SUP.SingleInstanceLock = StableSingleInstanceLock
    _INSTALLED = True
