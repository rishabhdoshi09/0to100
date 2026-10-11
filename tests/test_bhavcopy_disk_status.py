"""Cheap read-only NSE file inventory: never imports price-engine pandas."""
from __future__ import annotations

from datetime import date
import os
import sys

from data import bhavcopy_disk_status as disk
from data import bhavcopy_runtime as runtime


def test_disk_inventory_reports_date_without_claiming_ohlcv_ready(tmp_path):
    path = tmp_path / "bhav"
    path.mkdir()
    (path / "09102026.csv").write_text(
        "SYMBOL,SERIES,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE\n"
        + ("TCS,EQ,100,101,99,100\n" * 100),
        encoding="utf-8",
    )
    (path / "11132026.csv").write_text("bad", encoding="utf-8")
    (path / "invalid.csv").write_text("bad", encoding="utf-8")
    result = disk.unloaded_status(path)
    assert result["metadata_only"] is True
    assert result["csv_latest_date"] == "2026-10-09"
    assert result["csv_files"] == 1
    assert result["csv_latest_validated"] is True
    assert result["csv_inventory_error"] == ""
    assert result["ready"] is False
    assert result["symbols"] == 0
    assert result["sessions"] == 0
    assert result["latest_date"] == ""


def test_changed_or_missing_disk_inventory_is_invalidated(tmp_path):
    path = tmp_path / "bhav"
    path.mkdir()
    first = path / "08102026.csv"
    first.write_text("TEST", encoding="utf-8")
    assert disk.disk_dates(path) == (date(2026, 10, 8),)

    second = path / "09102026.csv"
    second.write_text("TEST", encoding="utf-8")
    old = path.stat().st_mtime_ns
    os.utime(path, ns=(old + 2_000_000_000, old + 2_000_000_000))
    assert disk.disk_dates(path) == (date(2026, 10, 8), date(2026, 10, 9))

    second.unlink()
    newer = path.stat().st_mtime_ns
    os.utime(path, ns=(newer + 2_000_000_000, newer + 2_000_000_000))
    assert disk.disk_dates(path) == (date(2026, 10, 8),)

    missing = path / "unmounted"
    assert disk.unloaded_status(missing)["ready"] is False
    assert disk.disk_dates(missing) == ()
    assert not missing.exists()


def test_bhavcopy_status_metadata_only_never_imports_heavy_store(monkeypatch):
    import data.bhavcopy_disk_status as disk_module

    calls = []
    marker = {
        "ready": False, "sessions": 0, "latest_date": "",
        "csv_latest_date": "2026-10-09", "csv_files": 1800,
        "metadata_only": True,
    }
    monkeypatch.delitem(sys.modules, "data.bhavcopy_store", raising=False)
    monkeypatch.setattr(disk_module, "unloaded_status", lambda: calls.append("disk") or marker)
    monkeypatch.setattr(
        runtime, "_store_module",
        lambda: (_ for _ in ()).throw(
            AssertionError("metadata-only API imported the full pandas price engine")
        ),
    )
    assert runtime.status(load_cache=False) is marker
    assert calls == ["disk"]


def test_official_date_current_does_not_promote_unloaded_store_to_scan_ready():
    from datetime import datetime

    data = {
        "ready": False, "sessions": 0, "latest_date": "",
        "csv_latest_date": "2026-10-09", "csv_files": 1826,
        "metadata_only": True,
    }
    freshness = runtime.official_history_freshness(
        data, now=datetime(2026, 10, 11, 16, 45), holidays=set(),
        load_cache=False, require_store=False,
    )
    assert freshness["current"] is True
    assert freshness["store_loaded"] is False
    strict = runtime.official_history_freshness(
        data, now=datetime(2026, 10, 11, 16, 45), holidays=set(),
        load_cache=False, require_store=True,
    )
    assert strict["current"] is False
    assert strict["usable_for_scan"] is False
    assert strict["reason_code"] == "HISTORY_NOT_READY"


def test_bogus_newer_csv_filename_never_makes_history_current(tmp_path):
    from datetime import datetime
    path = tmp_path / "bhav"
    path.mkdir()
    (path / "09102026.csv").write_text(
        "SYMBOL,SERIES,OPEN_PRICE,CLOSE_PRICE\n"
        + ("TCS,EQ,100,101\n" * 100),
        encoding="utf-8",
    )
    assert disk.unloaded_status(path)["csv_latest_date"] == "2026-10-09"
    # An unfinished file can appear while the other process is publishing
    # an archive. Read-only status must not call it current or make new dirs.
    (path / "12102026.csv").write_text("INCOMPLETE", encoding="utf-8")
    result = disk.unloaded_status(path)
    assert result["csv_latest_named_date"] == "2026-10-12"
    assert result["csv_latest_date"] == ""
    assert result["csv_latest_validated"] is False
    assert result["csv_inventory_error"] == "LATEST_CSV_INVALID"
    freshness = runtime.official_history_freshness(
        result, now=datetime(2026, 10, 12, 19, 0),
        holidays=set(), load_cache=False, require_store=False,
    )
    assert freshness["current"] is False
    assert freshness["usable_for_scan"] is False


def test_zero_byte_or_non_csv_latest_file_is_not_verified(tmp_path):
    path = tmp_path / "bhav"
    path.mkdir()
    filename = path / "09102026.csv"
    filename.write_bytes(b"")
    assert disk.unloaded_status(path)["csv_latest_date"] == ""
    filename.write_text("HTTP 200 OK " * 150, encoding="utf-8")
    assert disk.unloaded_status(path)["csv_latest_date"] == ""
