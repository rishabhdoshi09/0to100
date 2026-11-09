"""Persisted Recommendations desk from the last whole-market scan.

GET reads this file. Scan Now writes it. Page-open does not rescan.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping
from core.runtime_paths import logs_path

DEFAULT_RECO_PATH = logs_path("product/latest_recommendations.json")


def _publication_lock_path() -> Path:
    override = str(os.environ.get("QT_RECO_PUBLISH_LOCK") or "").strip()
    return Path(override) if override else logs_path("product", ".recommendations_publish.lock")


@contextmanager
def publication_lock(timeout_s: float = 600.0) -> Iterator[None]:
    """One cross-process critical section for publishing a scan generation.

    The scan pipeline holds it from saving a generation's recommendations to
    saving that scan, so the pair commits together. The API's background
    rebuild holds it for its generation check + save, so a rebuild for an older
    generation can never overwrite a newer publication. The OS releases the
    lock if the holder dies. Raises TimeoutError instead of waiting forever.
    """
    import fcntl

    path = _publication_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"recommendation publication lock busy for {timeout_s:.0f}s")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def save_recommendations(payload: Mapping[str, Any], path: str | Path = DEFAULT_RECO_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    from core.atomic_io import atomic_write_text
    atomic_write_text(target, json.dumps(dict(payload), indent=2, default=str))
    return target


def load_recommendations(path: str | Path = DEFAULT_RECO_PATH) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0) or 0) < 4:
            return None
        if not isinstance(payload.get("categories"), list):
            return None
        return payload
    except Exception:
        return None


def reco_matches_scan(
    saved: Mapping[str, Any] | None,
    *,
    scan_scanned_at: str,
    long_term_scanned_at: str,
) -> bool:
    if not saved:
        return False
    return (
        str(saved.get("scan_scanned_at") or "") == str(scan_scanned_at or "")
        and str(saved.get("long_term_scanned_at") or "") == str(long_term_scanned_at or "")
    )
