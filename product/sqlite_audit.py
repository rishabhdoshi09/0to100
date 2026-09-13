"""Read-only SQLite runtime diagnostics.

This module is deliberately non-repairing.  QuantTerm uses WAL for many product
stores, and a plain ``sqlite3 -readonly`` probe can fail with SQLITE_CANTOPEN
when SQLite cannot create/open the WAL shared-memory sidecar even though the
base database is readable.  The audit therefore records both an authoritative
read-only open attempt and a diagnostic immutable base-file attempt without
changing journal mode or writing recovery state.

An immutable success is *not* proof that the live database is current when a
WAL exists: immutable mode intentionally ignores uncheckpointed WAL content.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3
from typing import Iterable
from urllib.parse import quote

from core.runtime_paths import logs_dir


SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


@dataclass(frozen=True)
class Probe:
    ok: bool
    result: str
    error: str = ""


@dataclass(frozen=True)
class DatabaseAudit:
    path: str
    size_bytes: int
    wal_present: bool
    shm_present: bool
    journal_present: bool
    readonly: Probe
    immutable_base: Probe
    classification: str
    note: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return payload


def _uri(path: Path, *, immutable: bool = False) -> str:
    # quote() keeps slash separators while escaping spaces and other URI chars.
    encoded = quote(str(path.resolve()), safe="/")
    suffix = "&immutable=1" if immutable else ""
    return f"file:{encoded}?mode=ro{suffix}"


def _quick_check(path: Path, *, immutable: bool = False) -> Probe:
    try:
        con = sqlite3.connect(_uri(path, immutable=immutable), uri=True, timeout=2.0)
    except Exception as exc:
        return Probe(False, "", f"{type(exc).__name__}: {exc}"[:300])
    try:
        con.execute("PRAGMA query_only=ON")
        row = con.execute("PRAGMA quick_check").fetchone()
        result = str(row[0] if row else "")
        return Probe(result.lower() == "ok", result, "")
    except Exception as exc:
        return Probe(False, "", f"{type(exc).__name__}: {exc}"[:300])
    finally:
        con.close()


def classify(*, readonly_ok: bool, immutable_ok: bool, wal_present: bool) -> tuple[str, str]:
    """Classify probe results without overstating what immutable mode proves."""
    if readonly_ok:
        return "READONLY_OK", "Read-only quick_check succeeded against SQLite's normal live view."
    if immutable_ok and wal_present:
        return (
            "WAL_READONLY_SIDECAR_CONSTRAINT",
            "Normal read-only open failed but the immutable base file is readable. "
            "Because a WAL exists, immutable mode ignores uncheckpointed WAL content; "
            "this points to a read-only/WAL sidecar constraint, not proven corruption.",
        )
    if immutable_ok:
        return (
            "READONLY_OPEN_CONSTRAINT",
            "Normal read-only open failed but the immutable base file passed quick_check. "
            "The base file is readable; inspect permissions/path/locking before any repair.",
        )
    return (
        "BASE_FILE_UNREADABLE_OR_INTEGRITY_FAILURE",
        "Neither normal read-only nor immutable base-file quick_check succeeded. "
        "This requires targeted investigation before the store is trusted or repaired.",
    )


def audit_database(path: str | Path) -> DatabaseAudit:
    target = Path(path)
    readonly = _quick_check(target, immutable=False)
    immutable = _quick_check(target, immutable=True)
    wal = Path(str(target) + "-wal").exists()
    shm = Path(str(target) + "-shm").exists()
    journal = Path(str(target) + "-journal").exists()
    classification, note = classify(
        readonly_ok=readonly.ok,
        immutable_ok=immutable.ok,
        wal_present=wal,
    )
    try:
        size = int(target.stat().st_size)
    except OSError:
        size = -1
    return DatabaseAudit(
        path=str(target),
        size_bytes=size,
        wal_present=wal,
        shm_present=shm,
        journal_present=journal,
        readonly=readonly,
        immutable_base=immutable,
        classification=classification,
        note=note,
    )


def discover_databases(root: str | Path | None = None) -> list[Path]:
    base = Path(root) if root is not None else logs_dir()
    if not base.exists():
        return []
    return sorted(
        path for path in base.rglob("*")
        if path.is_file() and path.suffix.lower() in SQLITE_SUFFIXES
    )


def audit_runtime(root: str | Path | None = None) -> list[DatabaseAudit]:
    return [audit_database(path) for path in discover_databases(root)]


def summary(rows: Iterable[DatabaseAudit]) -> dict[str, object]:
    items = list(rows)
    counts: dict[str, int] = {}
    for row in items:
        counts[row.classification] = counts.get(row.classification, 0) + 1
    return {
        "databases": len(items),
        "classifications": counts,
        "needs_investigation": sum(
            1 for row in items
            if row.classification == "BASE_FILE_UNREADABLE_OR_INTEGRITY_FAILURE"
        ),
        "readonly_failures": sum(1 for row in items if not row.readonly.ok),
        "immutable_base_failures": sum(1 for row in items if not row.immutable_base.ok),
    }
