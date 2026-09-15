"""Fail-closed guard for the installed QuantTerm durable runtime.

The installed host must never keep trading/research children alive after its
runtime disappears, and it must never create a replacement directory on the
system disk. The guard therefore performs only existence/identity/write-through
checks against the already-adopted runtime root. It creates no parent paths.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.runtime_paths import runtime_root

MANIFEST_REL = Path("state") / "host_runtime_manifest.json"
DEFAULT_INTERVAL_S = 15.0


class RuntimeStorageUnavailable(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_through_probe(root: Path) -> None:
    """Prove the existing directory is writable without ever creating parents."""
    fd = -1
    probe = ""
    try:
        fd, probe = tempfile.mkstemp(prefix=".quantterm-runtime-probe.", dir=str(root))
        os.write(fd, b"quantterm-runtime-ok\n")
        os.fsync(fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if probe:
            try:
                os.unlink(probe)
            except OSError:
                pass


@dataclass(frozen=True)
class RuntimeStorageIdentity:
    configured_root: str
    resolved_root: str
    manifest_sha256: str
    st_dev: int
    st_ino: int


def establish_runtime_storage_identity(root: Path | None = None) -> RuntimeStorageIdentity:
    configured = Path(root or runtime_root()).expanduser()
    if not configured.exists() or not configured.is_dir():
        raise RuntimeStorageUnavailable(f"runtime root is missing: {configured}")
    try:
        resolved = configured.resolve(strict=True)
        stat = resolved.stat()
        _write_through_probe(resolved)
    except Exception as exc:
        raise RuntimeStorageUnavailable(
            f"runtime root is not safely readable/writable: {configured}: {type(exc).__name__}"
        ) from exc
    manifest = resolved / MANIFEST_REL
    manifest_hash = ""
    if manifest.exists():
        try:
            manifest_hash = _sha256(manifest)
        except Exception as exc:
            raise RuntimeStorageUnavailable(
                f"runtime identity manifest is unreadable: {manifest}: {type(exc).__name__}"
            ) from exc
    return RuntimeStorageIdentity(
        configured_root=str(configured),
        resolved_root=str(resolved),
        manifest_sha256=manifest_hash,
        st_dev=int(stat.st_dev),
        st_ino=int(stat.st_ino),
    )


def verify_runtime_storage(identity: RuntimeStorageIdentity) -> tuple[bool, str]:
    configured = Path(identity.configured_root)
    try:
        if not configured.exists() or not configured.is_dir():
            return False, f"runtime root disappeared: {configured}"
        resolved = configured.resolve(strict=True)
        if str(resolved) != identity.resolved_root:
            return False, f"runtime root changed identity: {resolved} != {identity.resolved_root}"
        stat = resolved.stat()
        manifest = resolved / MANIFEST_REL
        if identity.manifest_sha256:
            if not manifest.is_file():
                return False, f"runtime identity manifest disappeared: {manifest}"
            if _sha256(manifest) != identity.manifest_sha256:
                return False, "runtime identity manifest changed unexpectedly"
        elif int(stat.st_dev) != identity.st_dev or int(stat.st_ino) != identity.st_ino:
            return False, "runtime directory device/inode changed unexpectedly"
        _write_through_probe(resolved)
        return True, ""
    except Exception as exc:
        return False, f"runtime storage verification failed: {type(exc).__name__}: {exc}"


class RuntimeStorageGuard:
    """Low-frequency monitor used by the installed host entrypoint."""

    def __init__(self, *, root: Path | None = None, interval_s: float = DEFAULT_INTERVAL_S):
        self.identity = establish_runtime_storage_identity(root)
        self.interval_s = max(0.2, float(interval_s))
        # Deliberately latched until wait_until_recovered() acknowledges the
        # recovery. A very fast remount must not erase the loss event before
        # the supervisor has stopped every child.
        self.lost = threading.Event()
        self.shutdown = threading.Event()
        self.last_error = ""
        self._thread: threading.Thread | None = None
        self._on_loss: Callable[[str], None] | None = None

    def check(self) -> bool:
        ok, reason = verify_runtime_storage(self.identity)
        self.last_error = "" if ok else reason
        return ok

    def start(self, on_loss: Callable[[str], None]) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._on_loss = on_loss
        self._thread = threading.Thread(
            target=self._loop,
            name="quantterm-runtime-storage-guard",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        notified = False
        while not self.shutdown.wait(self.interval_s):
            ok = self.check()
            if ok:
                if not self.lost.is_set():
                    notified = False
                continue
            self.lost.set()
            if not notified and self._on_loss is not None:
                notified = True
                self._on_loss(self.last_error)

    def wait_until_recovered(
        self,
        *,
        should_stop: Callable[[], bool] | None = None,
        prepare: Callable[[], None] | None = None,
    ) -> bool:
        """Wait for the exact adopted runtime, optionally preparing its mount.

        ``prepare`` may attach an already-configured external volume/sparsebundle
        but must never create a replacement runtime. Identity verification still
        happens afterwards and remains the authority for clearing the latched
        loss event.
        """
        while not self.shutdown.is_set() and not (should_stop and should_stop()):
            if prepare is not None:
                try:
                    prepare()
                except Exception as exc:
                    self.last_error = (
                        f"runtime recovery preparation failed: {type(exc).__name__}: {exc}"
                    )[:500]
                    time.sleep(min(5.0, self.interval_s))
                    continue
            if self.check():
                self.lost.clear()
                return True
            time.sleep(min(5.0, self.interval_s))
        return False

    def close(self) -> None:
        self.shutdown.set()
