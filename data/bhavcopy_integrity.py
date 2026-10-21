"""Validate embedded NSE archive trading dates before trusting filenames.

sec_bhavdata_full_DDMMYYYY.csv has a DATE1 column.  An exchange holiday can
return the previous session's archive with HTTP 200, so neither response status
nor filename is sufficient evidence of a trading session.
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime
from pathlib import Path
from typing import TextIO


_DATE_FORMATS = ("%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d")


def parse_archive_date(value: str) -> date | None:
    value = str(value or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def _validate_csv(handle: TextIO, expected: date, *, full: bool) -> tuple[bool, str]:
    try:
        reader = csv.reader(handle)
        header = [str(col).strip().upper() for col in next(reader, [])]
        required = {"DATE1", "SYMBOL", "SERIES", "CLOSE_PRICE", "TTL_TRD_QNTY"}
        if not required.issubset(header):
            return False, "MISSING_REQUIRED_COLUMNS"
        date_idx = header.index("DATE1")
        checked = 0
        for row in reader:
            if not row or not any(str(cell).strip() for cell in row):
                continue
            if len(row) <= date_idx:
                return False, "DATE1_MISSING_ROW"
            embedded = parse_archive_date(row[date_idx])
            if embedded is None:
                return False, "DATE1_INVALID"
            if embedded != expected:
                return False, f"DATE1_MISMATCH: expected={expected.isoformat()} actual={embedded.isoformat()}"
            checked += 1
            if not full:
                break
        if not checked:
            return False, "ARCHIVE_EMPTY"
        return True, "OK"
    except (csv.Error, UnicodeError, OSError, ValueError) as exc:
        return False, f"ARCHIVE_PARSE_ERROR:{type(exc).__name__}"


def validate_archive_bytes(payload: bytes, expected: date) -> tuple[bool, str]:
    """Validate *every* DATE1 before persisting a newly downloaded archive."""
    with io.StringIO(payload.decode("latin-1")) as stream:
        return _validate_csv(stream, expected, full=True)


def validate_archive_path(path: str | Path, expected: date, *, full: bool = False) -> tuple[bool, str]:
    """Check an existing file (fast header/first row for cataloging).

    The canonical _read_day parser independently checks every DATE1 cell before
    producing OHLCV, so catalog speed never becomes publication authority.
    """
    try:
        with Path(path).open("r", encoding="latin-1", newline="") as stream:
            return _validate_csv(stream, expected, full=full)
    except OSError as exc:
        return False, f"ARCHIVE_READ_ERROR:{type(exc).__name__}"
