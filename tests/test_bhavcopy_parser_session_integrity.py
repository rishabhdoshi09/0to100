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


def test_full_rebuild_never_claims_a_newest_file_that_failed_to_parse(monkeypatch):
    import pandas as pd
    older = date(2026, 10, 8)
    latest_bad = DAY
    bars = pd.DataFrame({
        "symbol": ["TCS"] * 30,
        "date": [pd.Timestamp(older)] * 30,
        "open": [100.0] * 30, "high": [101.0] * 30,
        "low": [99.0] * 30, "close": [100.0] * 30,
        "volume": [500.0] * 30,
    })
    monkeypatch.setattr(store, "_read_day",
                        lambda d: bars.copy() if d == older else None)
    monkeypatch.setattr(store, "_save_pkl", lambda: None)
    assert store._full_build([older, latest_bad]) == 1
    assert store._store_last_day == older
    assert store._store_sessions == 1


def test_incremental_append_never_claims_newer_corrupt_archive(monkeypatch):
    import pandas as pd
    baseline = date(2026, 10, 7)
    valid = date(2026, 10, 8)
    latest_bad = DAY
    preexisting = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [100.0], "volume": [500.0],
    }, index=[pd.Timestamp(baseline)])
    next_bar = pd.DataFrame({
        "symbol": ["TCS"], "date": [pd.Timestamp(valid)],
        "open": [101.0], "high": [102.0], "low": [100.0],
        "close": [101.0], "volume": [600.0],
    })
    monkeypatch.setattr(store, "_store", {"TCS": preexisting})
    monkeypatch.setattr(store, "_store_last_day", baseline)
    monkeypatch.setattr(store, "_store_sessions", 1)
    monkeypatch.setattr(store, "_read_day",
                        lambda d: next_bar.copy() if d == valid else None)
    monkeypatch.setattr(store, "_save_pkl", lambda: None)
    store._incremental_append([valid, latest_bad])
    assert store._store_last_day == valid
    assert store._store_sessions == 2
    assert len(store._store["TCS"]) == 2
