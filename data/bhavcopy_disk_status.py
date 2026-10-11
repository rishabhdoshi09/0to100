"""Read-only NSE bhavcopy CSV directory metadata; no pandas or pickle import.

A dashboard needs the newest *on-disk session filename*, NOT thousands of
in-memory OHLCV DataFrames. Treat this as file inventory only: loaded=False
and no claim that a session's CSV content has been validated for trading.
The authoritative scan process still uses data.bhavcopy_store.
"""
from __future__ import annotations

from datetime import date
import os
from pathlib import Path
import threading
from typing import Any

from core.runtime_paths import logs_path

_LOCK = threading.Lock()
_LAST_SIGNATURE: tuple[Any, ...] | None = None
_LAST_DATES: tuple[date, ...] = ()
_MIN_DAYS = 60


def _signature(root: Path) -> tuple[Any, ...] | None:
    try:
        stat = root.stat()
        if not root.is_dir():
            return None
        return (str(root), stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return None


def disk_dates(root: Path | None = None) -> tuple[date, ...]:
    """Invalidate the metadata-only cache when directory identity changes."""
    global _LAST_SIGNATURE, _LAST_DATES
    path = Path(root) if root is not None else logs_path("bhav")
    sig = _signature(path)
    if sig is None:
        return ()
    with _LOCK:
        if sig == _LAST_SIGNATURE:
            return _LAST_DATES
        rows: list[date] = []
        try:
            with os.scandir(path) as items:
                for entry in items:
                    name = entry.name
                    if (len(name) != 12 or not name.endswith(".csv")
                            or not name[:8].isascii() or not name[:8].isdecimal()):
                        continue
                    try:
                        rows.append(date(int(name[4:8]), int(name[2:4]), int(name[:2])))
                    except ValueError:
                        continue
        except OSError:
            return ()
        # Publishing a new official session while we enumerate must not
        # cache a torn set as the complete file inventory.
        if _signature(path) != sig:
            return ()
        _LAST_SIGNATURE = sig
        _LAST_DATES = tuple(sorted(rows))
        return _LAST_DATES


def _latest_csv_has_basic_integrity(path: Path) -> bool:
    """Cheap superficial validity check, NOT full OHLCV or source certification.

    A file named for today's market session is not proof that its download
    finished. Reject empty files, directories, symlinks and a non-CSV header.
    Actual row/content verification stays in the authoritative scanner.
    """
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size < 1000:
            return False
        with path.open("rb") as handle:
            header = handle.readline(2048).decode("latin-1").upper().lstrip("\ufeff")
        return "," in header and "SYMBOL" in header and "CLOSE" in header
    except (OSError, UnicodeError):
        return False


def unloaded_status(root: Path | None = None) -> dict[str, Any]:
    """File-inventory truth only; zero loaded symbols, no session approval."""
    path = Path(root) if root is not None else logs_path("bhav")
    dates = disk_dates(path)
    latest = dates[-1] if dates else None
    validated = bool(
        latest is not None
        and _latest_csv_has_basic_integrity(
            path / f"{latest.strftime('%d%m%Y')}.csv"
        )
    )
    cache = path / "store_cache.pkl"
    return {
        "ready": False,
        "symbols": 0,
        "sessions": 0,
        "latest_date": "",
        "csv_files": len(dates),
        # Do not promote a filename into official session freshness if its
        # content is clearly corrupt or incomplete.
        "csv_latest_date": latest.isoformat() if validated and latest else "",
        "csv_latest_named_date": latest.isoformat() if latest else "",
        "csv_latest_validated": validated,
        "csv_inventory_error": "LATEST_CSV_INVALID" if latest and not validated else "",
        "cache_exists": cache.is_file(),
        "cache_path": str(cache),
        "bhavcopy_dir": str(path),
        "minimum_sessions": _MIN_DAYS,
        "source": "official_nse_bhavcopy",
        "metadata_only": True,
    }
