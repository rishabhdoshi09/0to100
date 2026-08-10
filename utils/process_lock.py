"""Cross-platform process file locks (Windows + POSIX).

Uses fcntl.flock on Unix and msvcrt byte-range locks on Windows.
Never invents ownership — acquire fails closed when the lock is held.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import IO, Optional


class ProcessFileLock:
    """Exclusive non-blocking file lock that dies with the process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: Optional[IO[str]] = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        if self._handle is not None:
            return True
        try:
            handle = self.path.open("a+", encoding="utf-8")
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                # Ensure at least one byte exists for msvcrt.locking.
                if handle.read(1) == "":
                    handle.write("0")
                    handle.flush()
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    handle.close()
                    return False
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    handle.close()
                    return False
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            self._handle = handle
            return True
        except Exception:
            try:
                if self._handle is not None:
                    self._handle.close()
            except Exception:
                pass
            self._handle = None
            # Last-resort exclusive create (weaker: stale file if crash before release)
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self._handle = self.path.open("a+", encoding="utf-8")
                return True
            except FileExistsError:
                return False
            except Exception:
                return False

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            try:
                self.path.unlink(missing_ok=True)
            except Exception:
                pass
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()
        except Exception:
            pass
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass

    def __enter__(self) -> "ProcessFileLock":
        if not self.acquire():
            raise RuntimeError(f"Could not acquire lock: {self.path}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
