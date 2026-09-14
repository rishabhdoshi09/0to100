"""Fail-closed installer adapter for an already-adopted durable runtime.

The generic host installer is intentionally able to initialise a new persistent
runtime on a first install. That behaviour must never be reachable from the
macOS external-APFS update path: if removable storage disappears during an
update, creating or migrating into the same /Volumes/... pathname on the system
disk would split durable state.

This adapter therefore makes two guarantees for one canonical install:
1. the configured runtime root must already exist, resolve strictly and pass a
   write-through probe without creating any parent; and
2. runtime migration becomes adoption-only: the existing host runtime manifest
   must be readable and valid. No source inventory copy or mkdir path is allowed.

The canonical product.host_install workflow still owns service definition,
preflight, rollback, exact-SHA checks and startup proof.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

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


def adopt_existing_runtime(
    target_root: Path, *, repo_root: Path = HI.REPO_ROOT, build_sha: str = "",
) -> dict[str, Any]:
    """Adopt only an already-initialised host runtime; never migrate/copy data."""
    root = ensure_existing_persistent_runtime_root(target_root, repo_root=repo_root)
    marker = root / HI.MANIFEST_REL
    try:
        if not marker.is_file():
            raise HI.HostInstallError(
                f"strict host installation requires existing runtime manifest: {marker}"
            )
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except HI.HostInstallError:
        raise
    except Exception as exc:
        raise HI.HostInstallError(
            f"runtime manifest is unreadable during strict adoption: {marker}: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(payload, dict) or not payload.get("initialized_at"):
        raise HI.HostInstallError(f"runtime manifest is invalid: {marker}")

    # Re-verify after reading the manifest. If storage vanished or changed while
    # adoption was in flight, fail before persisting a pointer or service state.
    verified = ensure_existing_persistent_runtime_root(root, repo_root=repo_root)
    if verified != root:
        raise HI.HostInstallError("runtime root changed identity during strict adoption")

    try:
        HI.write_runtime_pointer(root, repo_root=repo_root)
    except Exception as exc:
        raise HI.HostInstallError(f"cannot persist runtime-root pointer: {exc}") from exc
    return {"state": "ADOPTED", "runtime_root": str(root), "manifest": payload}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    original_root = HI.ensure_persistent_runtime_root
    original_migrate = HI.migrate_repo_runtime
    HI.ensure_persistent_runtime_root = ensure_existing_persistent_runtime_root
    HI.migrate_repo_runtime = adopt_existing_runtime
    try:
        return int(HI.main(["install", *args]))
    finally:
        HI.ensure_persistent_runtime_root = original_root
        HI.migrate_repo_runtime = original_migrate


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
