"""Single source of truth for where QuantTerm's mutable runtime state lives.

Every durable artifact the product writes — logs, SQLite stores, scans,
recommendations, evidence — belongs under one root.  ``QT_RUNTIME_ROOT`` wins.
After a host migration, an ignored checkout-local pointer records the adopted
persistent root so CLI tools started outside the service manager do not fall
back to stale repo-local state.

Installed-host processes additionally set ``QT_RUNTIME_ROOT_REQUIRE_EXISTING``.
In that mode a vanished configured runtime is an error at path resolution time,
not a directory downstream helpers are allowed to recreate on another disk.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_VAR = "QT_RUNTIME_ROOT"
REQUIRE_EXISTING_ENV = "QT_RUNTIME_ROOT_REQUIRE_EXISTING"
POINTER_NAME = ".quantterm_runtime_root"
_TRUTHY = {"1", "true", "yes", "on"}


def runtime_pointer_path(repo_root: Path | None = None) -> Path:
    return Path(repo_root or REPO_ROOT) / POINTER_NAME


def _validated_pointer_target(raw: str, *, repo_root: Path) -> Path:
    candidate = Path(raw.strip()).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError(f"{POINTER_NAME} must contain an absolute path")
    resolved = candidate.resolve()
    repo = Path(repo_root).resolve()
    if resolved == repo or repo in resolved.parents:
        raise RuntimeError(f"{POINTER_NAME} must point outside the source checkout")
    return resolved


def read_runtime_pointer(repo_root: Path | None = None) -> Path | None:
    repo = Path(repo_root or REPO_ROOT)
    pointer = runtime_pointer_path(repo)
    if not pointer.exists():
        return None
    try:
        raw = pointer.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read {pointer}: {exc}") from exc
    if not raw.strip():
        raise RuntimeError(f"{pointer} is empty")
    return _validated_pointer_target(raw, repo_root=repo)


def write_runtime_pointer(target: Path, *, repo_root: Path | None = None) -> Path:
    """Atomically point checkout-local CLI tools at the adopted persistent root."""
    repo = Path(repo_root or REPO_ROOT)
    resolved = _validated_pointer_target(str(target), repo_root=repo)
    pointer = runtime_pointer_path(repo)
    tmp = pointer.with_suffix(pointer.suffix + ".tmp")
    tmp.write_text(str(resolved) + "\n", encoding="utf-8")
    os.replace(tmp, pointer)
    return pointer


def _require_existing_runtime() -> bool:
    return os.environ.get(REQUIRE_EXISTING_ENV, "").strip().lower() in _TRUTHY


def _existing_runtime(path: Path) -> Path:
    """Resolve an already-existing runtime or fail without creating anything."""
    candidate = Path(path).expanduser()
    try:
        if not candidate.is_dir():
            raise RuntimeError(f"configured QuantTerm runtime root is missing: {candidate}")
        return candidate.resolve(strict=True)
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(
            f"configured QuantTerm runtime root is unavailable: {candidate}: {type(exc).__name__}"
        ) from exc


def runtime_root() -> Path:
    """Base directory for all mutable runtime state.

    Environment wins because tests and service managers intentionally redirect
    the process.  The durable pointer is the second choice; only a checkout
    that has never been migrated falls back to the repository itself.

    In installed-host strict mode there is deliberately no fallback: the chosen
    persistent root must already exist. This prevents a child from recreating a
    vanished external-runtime path during the storage watchdog detection window.
    """
    strict = _require_existing_runtime()
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        root = Path(override).expanduser()
        return _existing_runtime(root) if strict else root
    pointer = read_runtime_pointer()
    if pointer is not None:
        return _existing_runtime(pointer) if strict else pointer
    if strict:
        raise RuntimeError(
            "installed QuantTerm requires an existing configured persistent runtime root"
        )
    return REPO_ROOT


def runtime_path(*parts: str | os.PathLike[str]) -> Path:
    return runtime_root().joinpath(*[str(part) for part in parts])


def logs_dir() -> Path:
    return runtime_path("logs")


def logs_path(*parts: str | os.PathLike[str]) -> Path:
    return logs_dir().joinpath(*[str(part) for part in parts])


def ensure_logs_path(*parts: str | os.PathLike[str]) -> Path:
    target = logs_path(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def is_redirected() -> bool:
    return runtime_root().resolve() != REPO_ROOT.resolve()
