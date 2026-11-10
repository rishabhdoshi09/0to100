"""QT-PRA-074: the official-history listing must not re-list 1,754 archives per call.

On the owner's Mac Home called ``_dates_on_disk`` 5 times per request, each a
glob + strptime + stat over every archive on the external sparsebundle
(~0.3 s per call). The listing is now cached by directory identity, while each
file is still validated by its own stat identity, so additions, removals,
renames and in-place overwrites are all reflected on the next call.
"""
from __future__ import annotations

import os
from datetime import date

import pytest

from data import bhavcopy_store as BS


def _csv(day: str) -> bytes:
    lines = [
        "SYMBOL, SERIES, DATE1, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, TTL_TRD_QNTY",
        f"INFY, EQ, {day}, 100, 102, 99, 101, 150000",
        f"TCS, EQ, {day}, 200, 202, 199, 201, 100000",
    ]
    return ("\n".join(lines) + "\n").encode("latin-1")


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "_BHAV_DIR", tmp_path)
    monkeypatch.setattr(BS, "_holidays", lambda: {"2026-10-20"})
    monkeypatch.setattr(BS, "_SESSION_LISTING", None)
    return tmp_path


def _age_dir(path, seconds_ago=60):
    import time

    ns = int((time.time() - seconds_ago) * 1e9)
    os.utime(path, ns=(ns, ns))


def test_listing_tracks_add_remove_rename_and_in_place_overwrite(store_dir):
    (store_dir / "19102026.csv").write_bytes(_csv("19-Oct-2026"))
    (store_dir / "20102026.csv").write_bytes(_csv("20-Oct-2026"))   # holiday: excluded
    (store_dir / "notes.txt").write_text("x")
    (store_dir / "store_cache.pkl").write_bytes(b"x")
    assert BS._dates_on_disk() == [date(2026, 10, 19)]

    (store_dir / "21102026.csv").write_bytes(_csv("21-Oct-2026"))
    assert BS._dates_on_disk() == [date(2026, 10, 19), date(2026, 10, 21)]

    os.rename(store_dir / "21102026.csv", store_dir / "22102026.csv")    # mislabelled now
    assert BS._dates_on_disk() == [date(2026, 10, 19)]

    (store_dir / "22102026.csv").unlink()
    (store_dir / "22102026.csv").write_bytes(_csv("22-Oct-2026"))
    assert BS._dates_on_disk() == [date(2026, 10, 19), date(2026, 10, 22)]

    # In-place overwrite (no directory change): per-file validation still applies.
    with open(store_dir / "19102026.csv", "wb") as fh:
        fh.write(_csv("18-Oct-2026") + b"padding-to-change-the-size\n")
    assert BS._dates_on_disk() == [date(2026, 10, 22)]


def test_unchanged_directory_is_listed_once(store_dir, monkeypatch):
    for day in ("19102026", "21102026", "22102026"):
        stamp = f"{day[:2]}-Oct-{day[4:]}"
        (store_dir / f"{day}.csv").write_bytes(_csv(stamp))
    scans = {"n": 0}
    real = BS.os.scandir

    def counting(path):
        scans["n"] += 1
        return real(path)

    monkeypatch.setattr(BS.os, "scandir", counting)
    _age_dir(store_dir, 60)
    first = BS._dates_on_disk()
    for _ in range(5):
        assert BS._dates_on_disk() == first
    assert scans["n"] == 1

    _age_dir(store_dir, 30)          # directory changed (still quiet)
    assert BS._dates_on_disk() == first
    assert scans["n"] == 2


def test_a_just_modified_directory_is_never_cached(store_dir, monkeypatch):
    """Coarse-mtime filesystems: a listing taken in the same tick as a write
    must not be trusted, or an archive added in that tick would stay invisible."""
    (store_dir / "19102026.csv").write_bytes(_csv("19-Oct-2026"))
    _age_dir(store_dir, 0)
    assert BS._dates_on_disk() == [date(2026, 10, 19)]
    # Same-tick addition that does not move the (coarse) directory mtime.
    st = os.stat(store_dir)
    (store_dir / "21102026.csv").write_bytes(_csv("21-Oct-2026"))
    os.utime(store_dir, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert BS._dates_on_disk() == [date(2026, 10, 19), date(2026, 10, 21)]


def test_holiday_calendar_change_invalidates_listing(store_dir, monkeypatch):
    (store_dir / "19102026.csv").write_bytes(_csv("19-Oct-2026"))
    assert BS._dates_on_disk() == [date(2026, 10, 19)]
    monkeypatch.setattr(BS, "_holidays", lambda: {"2026-10-19"})
    assert BS._dates_on_disk() == []


def test_missing_store_dir_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "_BHAV_DIR", tmp_path / "absent")
    assert BS._dates_on_disk() == []
