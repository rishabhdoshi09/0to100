"""Regression: an HTTP 200 holiday archive must not fabricate a candle."""
from __future__ import annotations

import pickle
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from data import bhavcopy_store as BS
from data.bhavcopy_integrity import validate_archive_bytes, validate_archive_path
from data.bhavcopy_runtime import official_history_freshness


def _csv(day: str, *, second_day: str | None = None) -> bytes:
    lines = [
        "SYMBOL, SERIES, DATE1, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, TTL_TRD_QNTY",
        f"INFY, EQ, {day}, 100, 102, 99, 101, 150000",
        f"TCS, EQ, {second_day or day}, 200, 202, 199, 201, 100000",
    ]
    return ("\n".join(lines) + "\n").encode("latin-1")


def test_valid_archive_reports_matching_embedded_session(tmp_path):
    expected = date(2026, 10, 19)
    payload = _csv("19-Oct-2026")
    assert validate_archive_bytes(payload, expected) == (True, "OK")
    path = tmp_path / "19102026.csv"
    path.write_bytes(payload)
    assert validate_archive_path(path, expected) == (True, "OK")


def test_fake_holiday_archive_rejected_even_on_http_200():
    valid, reason = validate_archive_bytes(_csv("19-Oct-2026"), date(2026, 10, 20))
    assert not valid
    assert "DATE1_MISMATCH" in reason
    assert "2026-10-19" in reason


def test_mixed_date_archive_rejected_not_just_first_row():
    valid, reason = validate_archive_bytes(
        _csv("19-Oct-2026", second_day="20-Oct-2026"), date(2026, 10, 19)
    )
    assert not valid
    assert "DATE1_MISMATCH" in reason


@pytest.mark.parametrize("payload", [b"no,archive\nhello,world\n", b""])
def test_non_archive_response_rejected(payload):
    assert not validate_archive_bytes(payload, date(2026, 10, 19))[0]


def test_reader_and_catalog_exclude_mislabeled_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "_BHAV_DIR", tmp_path)
    (tmp_path / "19102026.csv").write_bytes(_csv("19-Oct-2026"))
    (tmp_path / "20102026.csv").write_bytes(_csv("19-Oct-2026"))
    assert BS._read_day(date(2026, 10, 19)) is not None
    assert BS._read_day(date(2026, 10, 20)) is None
    assert BS._dates_on_disk() == [date(2026, 10, 19)]


def test_existing_bad_archive_never_counts_as_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "_BHAV_DIR", tmp_path)
    (tmp_path / "21102026.csv").write_bytes(_csv("19-Oct-2026"))
    assert BS._download_day(date(2026, 10, 21)) is False
    assert (tmp_path / "21102026.csv").exists()  # preserve evidence, no overwrite


def test_http_200_mismatched_date_is_not_written(tmp_path, monkeypatch):
    import requests

    class Response:
        status_code = 200
        content = _csv("19-Oct-2026") * 20  # comfortably above minimum size

    monkeypatch.setattr(requests, "get", lambda *_a, **_k: Response())
    monkeypatch.setattr(BS, "_BHAV_DIR", tmp_path)
    assert BS._download_day(date(2026, 10, 21)) is False
    assert not (tmp_path / "21102026.csv").exists()


def test_holiday_is_never_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "_BHAV_DIR", tmp_path)
    def never(*_a, **_k):
        raise AssertionError("No HTTP request on NSE holiday")
    import requests
    monkeypatch.setattr(requests, "get", never)
    assert BS._download_day(date(2026, 10, 20)) is False


def test_old_pickle_with_fabricated_holiday_last_day_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "_BHAV_DIR", tmp_path)
    monkeypatch.setattr(BS, "_PKL", tmp_path / "store_cache.pkl")
    (tmp_path / "19102026.csv").write_bytes(_csv("19-Oct-2026"))
    with BS._PKL.open("wb") as fh:
        pickle.dump({"store": {"INFY": pd.DataFrame({"close": [101]})},
                     "last_day": date(2026, 10, 20), "sessions": 100}, fh)
    assert BS._load_pkl() is False
    assert BS._PKL.exists()


def test_holiday_dated_history_cannot_report_ready():
    history = {"ready": True, "sessions": 500, "latest_date": "2026-10-20"}
    f = official_history_freshness(
        history, now=datetime(2026, 10, 21, 8, 30),
        holidays={"2026-10-20"}, load_cache=False, require_store=False,
    )
    assert f["usable_for_scan"] is False
    assert f["current"] is False
    assert f["reason_code"] == "HISTORY_NON_SESSION_DATE"


def test_candidate_days_exclude_bundled_holiday(monkeypatch):
    class ClockDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 21)
    monkeypatch.setattr(BS, "date", ClockDate)
    days = BS._trading_days_back(10)
    assert date(2026, 10, 20) not in days
    assert date(2026, 10, 19) in days
