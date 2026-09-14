"""Fail-closed installer adapter for an already-adopted durable runtime.

The generic host installer may initialise a new persistent runtime. The macOS
external-APFS update path must never do that: if removable storage disappears
mid-install, no later mkdir/write may be redirected to an internal /Volumes/...
pathname.

Strict mode therefore pins an open directory descriptor to the verified runtime
for the entire install. Runtime-owned probes, directory creation and manifest
writes are performed relative to that descriptor. Normal detach is held busy by
the descriptor; abrupt device loss makes descriptor operations fail rather than
retargeting an absolute pathname.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Mapping, Sequence

from product import host_install as HI

_PINNED_ROOT: Path | None = None
_PINNED_ROOT_FD: int | None = None


def _dir_flags() -> int:
    return os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))


def _close_pinned_root() -> None:
    global _PINNED_ROOT, _PINNED_ROOT_FD
    if _PINNED_ROOT_FD is not None:
        try:
            os.close(_PINNED_ROOT_FD)
        except OSError:
            pass
    _PINNED_ROOT = None
    _PINNED_ROOT_FD = None


def _pin_root(root: Path) -> None:
    global _PINNED_ROOT, _PINNED_ROOT_FD
    if _PINNED_ROOT_FD is not None:
        if _PINNED_ROOT != root:
            raise HI.HostInstallError(
                f"strict installer runtime changed during install: {_PINNED_ROOT} -> {root}"
            )
        return
    try:
        fd = os.open(str(root), _dir_flags())
        path_stat = root.stat()
        fd_stat = os.fstat(fd)
        if (int(path_stat.st_dev), int(path_stat.st_ino)) != (
            int(fd_stat.st_dev), int(fd_stat.st_ino)
        ):
            os.close(fd)
            raise HI.HostInstallError("runtime identity changed while pinning strict install root")
    except HI.HostInstallError:
        raise
    except Exception as exc:
        raise HI.HostInstallError(
            f"cannot pin strict runtime root: {root}: {type(exc).__name__}: {exc}"
        ) from exc
    _PINNED_ROOT = root
    _PINNED_ROOT_FD = fd


def _require_pin() -> tuple[Path, int]:
    if _PINNED_ROOT is None or _PINNED_ROOT_FD is None:
        raise HI.HostInstallError("strict runtime is not pinned")
    return _PINNED_ROOT, _PINNED_ROOT_FD


def _open_relative_dir(parts: Sequence[str], *, create: bool = False) -> int:
    _root, root_fd = _require_pin()
    current = os.dup(root_fd)
    try:
        for part in parts:
            if not part or part in {".", ".."} or "/" in part:
                raise HI.HostInstallError(f"unsafe strict runtime path component: {part!r}")
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=current)
                except FileExistsError:
                    pass
            nxt = os.open(part, _dir_flags(), dir_fd=current)
            os.close(current)
            current = nxt
        return current
    except Exception:
        try:
            os.close(current)
        except OSError:
            pass
        raise


def _relative_runtime_path(path: Path) -> tuple[str, ...]:
    root, _fd = _require_pin()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    try:
        rel = target.relative_to(root)
    except ValueError as exc:
        raise HI.HostInstallError(f"strict runtime write escaped pinned root: {target}") from exc
    if not rel.parts:
        raise HI.HostInstallError("strict runtime write cannot target the root directory")
    return rel.parts


def _anchored_probe(prefix: str = ".quantterm-install-probe") -> None:
    _root, root_fd = _require_pin()
    name = f"{prefix}.{os.getpid()}.{time.time_ns()}"
    fd = -1
    try:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=root_fd,
        )
        os.write(fd, b"quantterm-existing-runtime-ok\n")
        os.fsync(fd)
    except Exception as exc:
        raise HI.HostInstallError(
            f"pinned QT_RUNTIME_ROOT is not safely writable: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(name, dir_fd=root_fd)
        except OSError:
            pass


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

    _pin_root(root)
    _anchored_probe()
    return root


def _read_json_relative(parts: Sequence[str]) -> dict[str, Any]:
    if not parts:
        raise HI.HostInstallError("strict JSON read requires a relative file path")
    parent_fd = -1
    fd = -1
    try:
        parent_fd = _open_relative_dir(parts[:-1])
        fd = os.open(parts[-1], os.O_RDONLY, dir_fd=parent_fd)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        raise HI.HostInstallError(
            f"strict runtime file is unreadable: {'/'.join(parts)}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def strict_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    parts = _relative_runtime_path(path)
    parent_fd = _open_relative_dir(parts[:-1])
    name = parts[-1]
    tmp = f".{name}.tmp.{os.getpid()}.{time.time_ns()}"
    fd = -1
    try:
        data = json.dumps(dict(payload), indent=2, default=str).encode("utf-8")
        fd = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    except Exception as exc:
        raise HI.HostInstallError(
            f"strict runtime JSON write failed for {'/'.join(parts)}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp, dir_fd=parent_fd)
        except OSError:
            pass
        os.close(parent_fd)


def adopt_existing_runtime(
    target_root: Path, *, repo_root: Path = HI.REPO_ROOT, build_sha: str = "",
) -> dict[str, Any]:
    """Adopt only an already-initialised host runtime; never migrate/copy data."""
    root = ensure_existing_persistent_runtime_root(target_root, repo_root=repo_root)
    parts = HI.MANIFEST_REL.parts
    payload = _read_json_relative(parts)
    if not payload.get("initialized_at"):
        raise HI.HostInstallError(
            f"strict host installation requires valid existing runtime manifest: {root / HI.MANIFEST_REL}"
        )

    # Probe through the pinned descriptor again before changing local service
    # metadata. A lost device fails here without creating any pathname.
    _anchored_probe(prefix=".quantterm-adopt-probe")
    try:
        HI.write_runtime_pointer(root, repo_root=repo_root)
    except Exception as exc:
        raise HI.HostInstallError(f"cannot persist runtime-root pointer: {exc}") from exc
    return {"state": "ADOPTED", "runtime_root": str(root), "manifest": payload}


def _ensure_service_log_dir() -> None:
    # Creating subdirectories is safe here because it is relative to the pinned
    # external-runtime descriptor, never an absolute /Volumes pathname.
    logs_fd = _open_relative_dir(("logs",), create=True)
    try:
        try:
            os.mkdir("service", 0o700, dir_fd=logs_fd)
        except FileExistsError:
            pass
        service_fd = os.open("service", _dir_flags(), dir_fd=logs_fd)
        os.close(service_fd)
    except Exception as exc:
        raise HI.HostInstallError(
            f"cannot prepare pinned runtime service-log directory: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        os.close(logs_fd)


def strict_install_service_definition(
    *, runtime_root: Path, build_sha: str, env_file: str = "", repo_root: Path = HI.REPO_ROOT,
    python: str | None = None, manager: str | None = None, home: Path | None = None,
) -> dict[str, Any]:
    selected = manager or HI.service_manager()
    root = ensure_existing_persistent_runtime_root(runtime_root, repo_root=repo_root)
    if selected != "launchd":
        # Strict-existing mode is currently used only by the macOS APFS path.
        # Refuse silent expansion to another service manager until its complete
        # runtime-write surface receives the same anchored treatment.
        raise HI.HostInstallError(
            f"strict existing-runtime installer only supports launchd, got {selected}"
        )

    _ensure_service_log_dir()
    python = python or sys.executable
    path, label = HI._service_paths(selected, home=home)
    path.parent.mkdir(parents=True, exist_ok=True)  # local LaunchAgents path, not runtime state
    content = HI.render_launchd_plist(
        repo_root=repo_root,
        runtime_root=root,
        python=python,
        build_sha=build_sha,
        env_file=env_file,
    )
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    backup = path.with_suffix(path.suffix + ".previous")
    if previous is not None:
        backup.write_text(previous, encoding="utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return {
        "manager": selected,
        "path": str(path),
        "label": label,
        "backup": str(backup) if previous is not None else "",
    }


def strict_run_required_preflight(
    runtime_root: Path, build_sha: str, env_file: str = "",
) -> dict[str, Any]:
    root = ensure_existing_persistent_runtime_root(runtime_root)
    HI._prepare_process_environment(root, build_sha, env_file)
    from product import host_preflight as HP

    original_runtime_check = HP.check_runtime_root
    original_db_check = HP.check_database_writable

    def check_runtime_root_strict():
        try:
            ensure_existing_persistent_runtime_root(root)
        except Exception as exc:
            return HP._bad(
                "runtime_root",
                f"strict runtime root unavailable: {type(exc).__name__}: {exc}"[:180],
                path=str(root),
            )
        return HP._ok(
            "runtime_root",
            f"{root} (persistent, pinned, writable)",
            path=str(root),
            persistent=True,
            pinned=True,
        )

    def check_database_writable_strict():
        db_fd = -1
        file_fd = -1
        name = f".preflight-sqlite-probe.{os.getpid()}.{time.time_ns()}"
        try:
            # Prove the sqlite runtime itself works without needing an absolute
            # filesystem pathname, then prove the pinned database directory is
            # writable with an anchored durable probe.
            with sqlite3.connect(":memory:") as conn:
                conn.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY)")
                conn.execute("INSERT INTO probe DEFAULT VALUES")
                conn.commit()
            db_fd = _open_relative_dir(("db",), create=True)
            file_fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=db_fd,
            )
            os.write(file_fd, b"sqlite-runtime-and-pinned-db-write-ok\n")
            os.fsync(file_fd)
            return HP._ok("database_writable", f"sqlite available; pinned db writable under {root / 'db'}")
        except Exception as exc:
            return HP._bad(
                "database_writable",
                f"strict pinned db probe failed: {type(exc).__name__}: {exc}"[:160],
                path=str(root / "db"),
            )
        finally:
            if file_fd >= 0:
                os.close(file_fd)
            if db_fd >= 0:
                try:
                    os.unlink(name, dir_fd=db_fd)
                except OSError:
                    pass
                os.close(db_fd)

    HP.check_runtime_root = check_runtime_root_strict
    HP.check_database_writable = check_database_writable_strict
    try:
        report = HP.run_host_preflight(skip_network=False)
    finally:
        HP.check_runtime_root = original_runtime_check
        HP.check_database_writable = original_db_check

    if report.get("verdict") != HP.READY:
        blockers = "; ".join(
            f"{b.get('check')}: {b.get('detail')}" for b in report.get("blockers") or []
        )
        raise HI.HostInstallError(
            f"host preflight blocked installation: {blockers or 'unknown blocker'}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    originals = {
        "root": HI.ensure_persistent_runtime_root,
        "migrate": HI.migrate_repo_runtime,
        "atomic": HI._atomic_json,
        "service": HI.install_service_definition,
        "preflight": HI.run_required_preflight,
    }
    HI.ensure_persistent_runtime_root = ensure_existing_persistent_runtime_root
    HI.migrate_repo_runtime = adopt_existing_runtime
    HI._atomic_json = strict_atomic_json
    HI.install_service_definition = strict_install_service_definition
    HI.run_required_preflight = strict_run_required_preflight
    try:
        return int(HI.main(["install", *args]))
    finally:
        HI.ensure_persistent_runtime_root = originals["root"]
        HI.migrate_repo_runtime = originals["migrate"]
        HI._atomic_json = originals["atomic"]
        HI.install_service_definition = originals["service"]
        HI.run_required_preflight = originals["preflight"]
        _close_pinned_root()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
