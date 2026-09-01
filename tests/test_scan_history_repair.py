from __future__ import annotations

from datetime import date, timedelta

import pytest

import scan.bulk_fetcher as bulk


@pytest.fixture(autouse=True)
def _clear_repair_caches():
    with bulk._lock:
        bulk._kite_cache.clear()
        bulk._yf_cache.clear()
    yield
    with bulk._lock:
        bulk._kite_cache.clear()
        bulk._yf_cache.clear()


def _candles(days: int = 90):
    start = date(2026, 4, 1)
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        price = 100 + i * 0.1
        rows.append({
            "date": d.isoformat(),
            "open": price,
            "high": price + 2,
            "low": price - 2,
            "close": price + 1,
            "volume": 500_000,
        })
    return rows


class _FakeDataClient:
    def profile(self):
        return {"user_id": "paper-test"}

    def instruments(self, exchange="NSE"):
        assert exchange == "NSE"
        return [
            {"exchange": "NSE", "instrument_type": "EQ", "tradingsymbol": "NEWSTOCK", "instrument_token": 101},
            {"exchange": "NFO", "instrument_type": "FUT", "tradingsymbol": "NEWSTOCK26SEP", "instrument_token": 202},
        ]

    def historical(self, token, frm, to, interval="day"):
        assert token == 101
        assert interval == "day"
        assert frm < to
        return _candles()


def test_missing_current_nse_equity_is_repaired_via_data_only_kite(monkeypatch):
    monkeypatch.setattr(bulk, "_bhav_symbols", lambda: {"EXISTING"})

    result = bulk.backfill_missing(
        ["EXISTING", "NEWSTOCK"],
        client=_FakeDataClient(),
    )

    assert result["missing"] == 1
    assert result["attempted"] == 1
    assert result["loaded"] == 1
    assert result["failed"] == 0
    assert result["source"] == "zerodha_kite_data_only"
    assert "NEWSTOCK" in bulk.cached_symbols()
    frame = bulk.get_cached("NEWSTOCK")
    assert frame is not None
    assert len(frame) == 90
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_history_repair_leaves_unresolvable_symbol_missing(monkeypatch):
    monkeypatch.setattr(bulk, "_bhav_symbols", lambda: set())

    result = bulk.backfill_missing(["NOTINMASTER"], client=_FakeDataClient())

    assert result["missing"] == 1
    assert result["attempted"] == 0
    assert result["loaded"] == 0
    assert result["unresolved"] == 1
    assert "NOTINMASTER" not in bulk.cached_symbols()
