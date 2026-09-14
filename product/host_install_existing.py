"""Strict installed-host adapter for an already-adopted durable runtime.

The generic host installer is allowed to initialise a new persistent runtime on a
first install.  That behaviour is inappropriate for the macOS external-APFS
path: if the removable volume disappears during an update, creating the same
/Volumes/... path on the internal disk would split durable state.

This adapter replaces only the runtime-root adoption primitive for the duration
of one install.  The root must already exist, resolve strictly, remain outside
the checkout, and pass a write-through probe.  It never creates the root or any
parent directory.  The canonical host_install workflow still owns migration,
preflight, service installation, rollback, exact-SHA checks and startup proof.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

from product import host_install as HI


def ensure_existing_persistent_runtime_root(
    path: Path, *, repo_root: Path = HI.REPO_ROOT,
) -> Path:
    configured = Path(path).expanduser()
    if not configured.exists() or not configured.is_dir():
        raise HI.HostInstallError(
            f"QT_RUNTIME_ROOT must already exist for strict host installation: {configured}"
        )
    try:
        root = configured.resolve(strict=True)
        repo = Path(repo_root).expanduser().resolve(strict=True)
    except Exception as exc:
        raise HI.HostInstallError(
            f"cannot resolve strict existing runtime root: {type(exc).__name__}: {exc}"
        ) from exc

    if root == repo:
        raise HI.HostInstallError("QT_RUNTIME_ROOT must be outside the source checkout")
    try:
        root.relative_to(repo)
    except ValueError:
        pass
    else:
        raise HI.HostInstallError("QT_RUNTIME_ROOT must be outside the source checkout")

    if (str(root) + "/").startswith(HI.EPHEMERAL_PREFIXES):
        raise HI.HostInstallError(f"QT_RUNTIME_ROOT is on an ephemeral filesystem: {root}")

    fd = -1
    probe = ""
    try:
        fd, probe = tempfile.mkstemp(prefix=".quantterm-install-probe.", dir=str(root))
        os.write(fd, b"quantterm-existing-runtime-ok\n")
        os.fsync(fd)
    except Exception as exc:
        raise HI.HostInstallError(
            f"QT_RUNTIME_ROOT is not safely writable: {root}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if probe:
            try:
                os.unlink(probe)
            except OSError:
                pass
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    original = HI.ensure_persistent_runtime_root
    HI.ensure_persistent_runtime_root = ensure_existing_persistent_runtime_root
    try:
        return int(HI.main(["install", *args]))
    finally:
        HI.ensure_persistent_runtime_root = original


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
