"""Canonical pandas OHLCV parser must not advance market session on bad data."""
from __future__ import annotations

from datetime import date

from data import bhavcopy_store as store


DAY = date(2026, 10, 9)
HEADER = "SYMBOL,SERIES,DATE1,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY\n"


def _save(monkeypatch, tmp_path, rows, header=HEADER):
    monkeypatch.setattr(store, "_BHAV_DIR", tmp_path)
    store._day_path(DAY).write_text(header + rows, encoding="utf-8")


def test_canonical_equity_day_with_matching_embedded_date_parses(monkeypatch, tmp_path):
    _save(monkeypatch, tmp_path, "TCS,EQ,09-Oct-2026,100,101,99,100,500\n")
    frame = store._read_day(DAY)
    assert frame is not None
    assert frame["symbol"].tolist() == ["TCS"]
    assert frame["close"].tolist() == [100.0]
    assert str(frame["date"].iloc[0].date()) == "2026-10-09"


def test_bhavcopy_with_no_equity_rows_is_not_a_completed_day(monkeypatch, tmp_path):
    _save(monkeypatch, tmp_path, "TCS,BE,09-Oct-2026,100,101,99,100,500\n")
    assert store._read_day(DAY) is None


def test_renamed_prior_day_is_not_valid_session_history(monkeypatch, tmp_path):
    _save(monkeypatch, tmp_path, "TCS,EQ,08-Oct-2026,100,101,99,100,500\n")
    assert store._read_day(DAY) is None


def test_equity_archive_without_a_valid_close_is_not_a_completed_day(monkeypatch, tmp_path):
    _save(monkeypatch, tmp_path, "TCS,EQ,09-Oct-2026,100,101,99,NOT_A_PRICE,500\n")
    assert store._read_day(DAY) is None


def test_legacy_equity_file_without_date1_keeps_previous_parser_compatibility(monkeypatch, tmp_path):
    header = "SYMBOL,SERIES,OPEN_PRICE,HIGH_PRICE,LOW_PRICE,CLOSE_PRICE,TTL_TRD_QNTY\n"
    _save(monkeypatch, tmp_path, "TCS,EQ,100,101,99,100,500\n", header=header)
    frame = store._read_day(DAY)
    assert frame is not None
    assert frame["symbol"].tolist() == ["TCS"]
