"""Read-only NSE bhavcopy CSV directory metadata; no pandas or pickle import.

A dashboard needs the newest *on-disk session filename*, NOT thousands of
in-memory OHLCV DataFrames. Treat this as file inventory only: loaded=False
and no claim that a session's CSV content has been validated for trading.
The authoritative scan process still uses data.bhavcopy_store.
"""
from __future__ import annotations

from datetime import date, datetime
import csv
import math
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


def _latest_csv_has_basic_integrity(path: Path, *, expected_date: date | None = None) -> bool:
    """Bounded structural check, never full OHLCV/source certification.

    NSE sec_bhavdata_full must have the fields the canonical pandas parser
    actually consumes. Validate a few realistic EQ bars and, when DATE1 is
    present, make sure a renamed prior-session file cannot masquerade as
    the latest archive. Scan execution still requires the full loaded store.
    """
    required = (
        "SYMBOL", "SERIES", "OPEN_PRICE", "HIGH_PRICE",
        "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY",
    )
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size < 1000:
            return False
        if expected_date is None:
            expected_date = date(
                int(path.stem[4:8]), int(path.stem[2:4]), int(path.stem[:2])
            )
        with path.open("r", encoding="latin-1", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = {
                str(name).strip().lstrip("\ufeff").upper(): name
                for name in (reader.fieldnames or []) if name
            }
            if not all(field in columns for field in required):
                return False
            valid_eq = 0
            for index, row in enumerate(reader):
                if index >= 256:
                    break
                if str(row.get(columns["SERIES"]) or "").strip().upper() != "EQ":
                    continue
                if not str(row.get(columns["SYMBOL"]) or "").strip():
                    continue
                if "DATE1" in columns:
                    raw_day = str(row.get(columns["DATE1"]) or "").strip()
                    try:
                        data_day = datetime.strptime(raw_day, "%d-%b-%Y").date()
                    except ValueError:
                        try:
                            data_day = date.fromisoformat(raw_day)
                        except ValueError:
                            return False
                    if data_day != expected_date:
                        return False
                try:
                    prices = [
                        float(str(row.get(columns[key]) or "").strip())
                        for key in ("OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE")
                    ]
                    volume = float(str(row.get(columns["TTL_TRD_QNTY"]) or "").strip())
                except (ValueError, TypeError):
                    continue
                op, hi, lo, cl = prices
                if (all(math.isfinite(v) for v in (*prices, volume))
                        and lo > 0 and lo <= min(op, cl) <= max(op, cl) <= hi
                        and volume >= 0):
                    valid_eq += 1
                    if valid_eq >= 3:
                        return True
        return False
    except (OSError, UnicodeError, ValueError, csv.Error):
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
