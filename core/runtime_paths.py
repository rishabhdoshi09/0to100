"""Single source of truth for where QuantTerm's mutable runtime state lives.

Every durable artifact the product writes — logs, SQLite stores, scans,
recommendations, evidence — belongs under one root.  ``QT_RUNTIME_ROOT`` wins.
After a host migration, an ignored checkout-local pointer records the adopted
persistent root so CLI tools started outside the service manager do not fall
back to stale repo-local state.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_VAR = "QT_RUNTIME_ROOT"
POINTER_NAME = ".quantterm_runtime_root"


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


def runtime_root() -> Path:
    """Base directory for all mutable runtime state.

    Environment wins because tests and service managers intentionally redirect
    the process.  The durable pointer is the second choice; only a checkout
    that has never been migrated falls back to the repository itself.
    """
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    pointer = read_runtime_pointer()
    if pointer is not None:
        return pointer
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
