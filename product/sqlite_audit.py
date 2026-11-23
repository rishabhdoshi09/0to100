"""Read-only SQLite runtime diagnostics.

This module is deliberately non-repairing.  QuantTerm uses WAL for many product
stores, and a plain ``sqlite3 -readonly`` probe can fail with SQLITE_CANTOPEN
when SQLite cannot create/open the WAL shared-memory sidecar even though the
base database is readable.  The audit therefore records both an authoritative
read-only open attempt and a diagnostic immutable base-file attempt without
changing journal mode or writing recovery state.

An immutable success is *not* proof that the live database is current when a
WAL exists: immutable mode intentionally ignores uncheckpointed WAL content.

The probe is the FULL ``PRAGMA integrity_check`` by default. ``quick_check``
skips the comparison of index entries with table rows by design. On the
production Mac's SQLite 3.45.1 it reported ``ok`` for a store whose index holds
stale entries, which is exactly Stage 1 finding F1 (22-23 Nov 2026: the
decision-freeze ledger passed quick_check and was classified READONLY_OK). Other
builds, such as CI's, flag the entry count; ``integrity_check`` does on all. ``quick=True`` remains available for a fast pass
and says what it cannot see.

A live view that opens and reports a diagnostic is an integrity failure, never a
sidecar constraint, even when the immutable base passes: the damage may sit in
the WAL (GPT F1-07, cycle 8). Measured limit (SQLite 3.45): ``integrity_check``
on a read-only connection does not verify CHECK constraints; a read-write
connection does. The audit stays read-only, so CHECK violations are out of its
reach.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sqlite3
from typing import Iterable
from urllib.parse import quote

from core.runtime_paths import logs_dir


SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


QUICK_CHECK_LIMIT = "quick_check cannot see index entries that disagree with their tables (finding F1)"
# SQLite result codes that mean the DATA is damaged, whatever operation raised them
# (GPT V2-05): a malformed schema committed only to the WAL raises while the check
# reads the schema, after a successful open.
_CORRUPTION_CODES = {11, 26}             # SQLITE_CORRUPT, SQLITE_NOTADB (primary codes)


@dataclass(frozen=True)
class Probe:
    ok: bool
    result: str
    error: str = ""
    check: str = "integrity_check"
    corruption: bool = False              # an error whose SQLite code means damaged data


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


def _is_corruption(exc: BaseException) -> bool:
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None and (int(code) & 0xFF) in _CORRUPTION_CODES:
        return True
    text = str(exc).lower()
    return "malformed" in text or "not a database" in text


def _check(path: Path, *, immutable: bool = False, quick: bool = False) -> Probe:
    pragma = "quick_check" if quick else "integrity_check"
    try:
        con = sqlite3.connect(_uri(path, immutable=immutable), uri=True, timeout=2.0)
    except Exception as exc:
        return Probe(False, "", f"{type(exc).__name__}: {exc}"[:300], pragma, _is_corruption(exc))
    try:
        con.execute("PRAGMA query_only=ON")
        rows = [str(r[0]) for r in con.execute(f"PRAGMA {pragma}(20)").fetchall()]
        result = "; ".join(rows)[:1000]
        return Probe(rows == ["ok"], result, "", pragma)
    except Exception as exc:
        return Probe(False, "", f"{type(exc).__name__}: {exc}"[:300], pragma, _is_corruption(exc))
    finally:
        con.close()


def _quick_check(path: Path, *, immutable: bool = False) -> Probe:
    """Kept for callers of the old name; it is the quick (index-blind) probe."""
    return _check(path, immutable=immutable, quick=True)


NEEDS_INVESTIGATION = ("BASE_FILE_UNREADABLE_OR_INTEGRITY_FAILURE", "LIVE_VIEW_INTEGRITY_FAILURE")


def classify(*, readonly_ok: bool, immutable_ok: bool, wal_present: bool,
             check: str = "integrity_check", readonly_diagnostic: bool = False) -> tuple[str, str]:
    """Classify probe results without overstating what immutable mode proves.

    ``readonly_diagnostic``: the normal read-only open SUCCEEDED and the check
    returned diagnostics (as opposed to the open or the check raising).
    """
    if readonly_ok:
        note = f"Read-only {check} succeeded against SQLite's normal live view."
        if check == "quick_check":
            note += " " + QUICK_CHECK_LIMIT + "."
        return "READONLY_OK", note
    if readonly_diagnostic:
        return (
            "LIVE_VIEW_INTEGRITY_FAILURE",
            f"The normal read-only view (including any WAL) opened and failed {check}. "
            "This is an integrity failure, not a sidecar constraint, even if the base file passes.",
        )
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
            f"Normal read-only open failed but the immutable base file passed {check}. "
            "The base file is readable; inspect permissions/path/locking before any repair.",
        )
    return (
        "BASE_FILE_UNREADABLE_OR_INTEGRITY_FAILURE",
        f"Neither normal read-only nor immutable base-file {check} succeeded. "
        "This requires targeted investigation before the store is trusted or repaired.",
    )


def audit_database(path: str | Path, *, quick: bool = False) -> DatabaseAudit:
    target = Path(path)
    readonly = _check(target, immutable=False, quick=quick)
    immutable = _check(target, immutable=True, quick=quick)
    wal = Path(str(target) + "-wal").exists()
    shm = Path(str(target) + "-shm").exists()
    journal = Path(str(target) + "-journal").exists()
    classification, note = classify(
        readonly_ok=readonly.ok,
        immutable_ok=immutable.ok,
        wal_present=wal,
        check=readonly.check,
        readonly_diagnostic=(not readonly.ok and (not readonly.error or readonly.corruption)),
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


def audit_runtime(root: str | Path | None = None, *, quick: bool = False) -> list[DatabaseAudit]:
    return [audit_database(path, quick=quick) for path in discover_databases(root)]


def summary(rows: Iterable[DatabaseAudit]) -> dict[str, object]:
    items = list(rows)
    counts: dict[str, int] = {}
    for row in items:
        counts[row.classification] = counts.get(row.classification, 0) + 1
    checks = sorted({row.readonly.check for row in items})
    return {
        "databases": len(items),
        "check": checks[0] if len(checks) == 1 else checks,
        "limit": QUICK_CHECK_LIMIT if "quick_check" in checks else "",
        "classifications": counts,
        "needs_investigation": sum(1 for row in items if row.classification in NEEDS_INVESTIGATION),
        "readonly_failures": sum(1 for row in items if not row.readonly.ok),
        "immutable_base_failures": sum(1 for row in items if not row.immutable_base.ok),
    }
