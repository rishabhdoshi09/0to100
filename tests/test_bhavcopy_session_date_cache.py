"""Read-only session-date metadata must be cheap, current, and fail closed."""
from __future__ import annotations

from datetime import date
import os

from data import bhavcopy_store as store


def test_bhavcopy_dates_index_reuses_unmodified_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_BHAV_DIR", tmp_path)
    (tmp_path / "09102026.csv").write_text("dummy", encoding="utf-8")
    got = store._dates_on_disk()
    assert got == [date(2026, 10, 9)]

    # The cached date index should be returned by copy, never mutable global state.
    got.clear()
    assert store._dates_on_disk() == [date(2026, 10, 9)]

    # Writer publishes another date: directory metadata identity invalidates it.
    (tmp_path / "08102026.csv").write_text("dummy", encoding="utf-8")
    before_ns = tmp_path.stat().st_mtime_ns
    os.utime(tmp_path, ns=(before_ns + 2_000_000_000, before_ns + 2_000_000_000))
    assert store._dates_on_disk() == [date(2026, 10, 8), date(2026, 10, 9)]


def test_bhavcopy_dates_index_ignores_invalid_names_and_does_not_use_cache_after_unmount(tmp_path, monkeypatch):
    root = tmp_path / "volume"
    root.mkdir()
    monkeypatch.setattr(store, "_BHAV_DIR", root)
    (root / "32132026.csv").write_text("invalid", encoding="utf-8")
    (root / "non_date.csv").write_text("invalid", encoding="utf-8")
    (root / "09102026.csv").write_text("valid", encoding="utf-8")
    assert store._dates_on_disk() == [date(2026, 10, 9)]

    # If external volume disappears the verified dates cannot remain current.
    monkeypatch.setattr(store, "_BHAV_DIR", root / "unmounted")
    assert store._dates_on_disk() == []


def test_bhavcopy_dates_index_detects_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_BHAV_DIR", tmp_path)
    filename = tmp_path / "09102026.csv"
    filename.write_text("history", encoding="utf-8")
    assert store._dates_on_disk() == [date(2026, 10, 9)]
    filename.unlink()
    mtime = tmp_path.stat().st_mtime_ns
    os.utime(tmp_path, ns=(mtime + 2_000_000_000, mtime + 2_000_000_000))
    assert store._dates_on_disk() == []
