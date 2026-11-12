"""_dates_on_disk() stat()ed ~1,755 archives per call; /api/health, Home and the
dashboard call it on every poll and starved the API on the owner's USB HDD."""
from __future__ import annotations

import os
from datetime import date

import data.bhavcopy_store as B


def test_validated_dates_are_cached_per_directory_identity(monkeypatch, tmp_path):
    for name in ("03112026.csv", "04112026.csv"):
        (tmp_path / name).write_text("x")
    old = 1_000_000_000
    for p in tmp_path.iterdir():
        os.utime(p, (old, old))
    os.utime(tmp_path, (old, old))
    monkeypatch.setattr(B, "_BHAV_DIR", tmp_path)
    monkeypatch.setattr(B, "_DATES_CACHE", None)
    monkeypatch.setattr(B, "_SESSION_LISTING", None)
    monkeypatch.setattr(B, "_holidays", lambda: set())
    stats = []
    monkeypatch.setattr(B, "_is_valid_archive", lambda path, day: stats.append(path.name) or True)

    assert B._dates_on_disk() == [date(2026, 11, 3), date(2026, 11, 4)]
    assert B._dates_on_disk() == [date(2026, 11, 3), date(2026, 11, 4)]
    assert len(stats) == 2                      # second call served from cache

    (tmp_path / "05112026.csv").write_text("x")  # a download renames into the dir
    os.utime(tmp_path / "05112026.csv", (old, old))
    os.utime(tmp_path, (old + 10, old + 10))
    assert B._dates_on_disk()[-1] == date(2026, 11, 5)

    monkeypatch.setattr(B.time, "monotonic", lambda: 10_000_000.0)  # TTL expired
    before = len(stats)
    B._dates_on_disk()
    assert len(stats) == before + 3
