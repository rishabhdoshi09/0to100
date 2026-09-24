"""Immutable storage for canonical signal-registry definitions.

The mutable eligibility snapshot in ``signal_registry.json`` describes current
evidence counts.  This store separately preserves the exact signal definitions
for every registry version referenced by replay/calibration provenance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_path
from scan import signal_registry as SR

SCHEMA_VERSION = 1
DEFAULT_DIR = logs_path("scan", "signal_registry_versions")


def canonical_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_version": SR.registry_version(),
        "definitions": {key: dict(value) for key, value in SR.SIGNAL_DEFINITIONS.items()},
        "signal_ids": list(SR.signal_ids()),
    }


def _atomic_create(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(dict(payload), sort_keys=True, indent=2), encoding="utf-8")
    try:
        # Never overwrite an immutable version.  ``xb`` is the authority check;
        # the temp file only keeps serialization away from the final pathname.
        with open(path, "xb") as out:
            out.write(tmp.read_bytes())
            out.flush()
            os.fsync(out.fileno())
    finally:
        tmp.unlink(missing_ok=True)


def persist_canonical_version(*, directory: str | Path | None = None) -> dict[str, Any]:
    root = Path(directory) if directory is not None else DEFAULT_DIR
    manifest = canonical_manifest()
    target = root / f"{manifest['registry_version']}.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("immutable signal-registry version collision")
        return {**existing, "cache_hit": True}
    _atomic_create(target, manifest)
    return {**manifest, "cache_hit": False}


def load_version(version: str, *, directory: str | Path | None = None) -> dict[str, Any]:
    value = str(version or "").strip()
    if not value or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise ValueError("invalid signal registry version")
    root = Path(directory) if directory is not None else DEFAULT_DIR
    target = root / f"{value}.json"
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if str(payload.get("registry_version") or "") != value:
        raise ValueError("signal registry provenance mismatch")
    return dict(payload)
