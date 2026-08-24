"""Canonical outcome resolver: trading sessions on official bars, never quotes."""
from __future__ import annotations

import pandas as pd


def _bars(closes, start="2026-08-03"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 2 for c in closes],
            "low": [c - 2 for c in closes],
            "close": closes,
            "volume": 1,
        },
        index=idx,
    )


def test_short_history_stays_pending(monkeypatch):
    from core.outcome_resolver import session_close_return

    monkeypatch.setattr(
        "data.bhavcopy_store.get_ohlcv",
        lambda s: _bars([100, 101, 102]),
    )
    assert session_close_return("HAL", "2026-08-03", horizon=5) is None


def test_session_close_uses_horizon_th_close_not_last(monkeypatch):
    from core.outcome_resolver import session_close_return

    # iloc[5] = 110; last bar = 200 would contaminate a delayed resolver
    closes = [100, 101, 102, 103, 104, 110, 150, 200]
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda s: _bars(closes))
    got = session_close_return("HAL", "2026-08-03", horizon=5)
    assert got is not None
    exit_px, pct = got
    assert exit_px == 110
    assert pct == 10.0


def test_first_touch_stop_before_target(monkeypatch):
    from core.outcome_resolver import first_touch_path

    idx = pd.bdate_range("2026-08-03", periods=4)
    df = pd.DataFrame(
        {
            "high": [101, 107, 108, 109],
            "low": [99, 95, 94, 93],
            "close": [100, 96, 95, 94],
        },
        index=idx,
    )
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda s: df)
    got = first_touch_path("HAL", "2026-08-03", 100.0, 96.0, 106.0, horizon=15)
    assert got is not None
    price, pct, worked = got
    assert worked == 0
    assert price == 96.0
    assert abs(pct - (-4.0)) < 1e-6


def test_first_touch_target_when_stop_not_hit(monkeypatch):
    from core.outcome_resolver import first_touch_path

    idx = pd.bdate_range("2026-08-03", periods=3)
    df = pd.DataFrame(
        {
            "high": [101, 107, 108],
            "low": [99, 100, 101],
            "close": [100, 106, 107],
        },
        index=idx,
    )
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda s: df)
    got = first_touch_path("HAL", "2026-08-03", 100.0, 96.0, 106.0)
    assert got is not None
    price, pct, worked = got
    assert worked == 1
    assert price == 106.0
    assert abs(pct - 6.0) < 1e-6


def test_first_touch_stays_open_inside_horizon(monkeypatch):
    from core.outcome_resolver import first_touch_path

    idx = pd.bdate_range("2026-08-03", periods=3)
    df = pd.DataFrame(
        {
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100, 101, 102],
        },
        index=idx,
    )
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda s: df)
    assert first_touch_path("HAL", "2026-08-03", 100.0, 96.0, 106.0) is None


def _wait_frame(highs, lows, closes, start="2026-08-03"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=idx)


def test_wait_patience_offered_later_entry(monkeypatch):
    from core.outcome_resolver import WAIT_OFFERED, wait_patience

    # decision day 100, next session pulls back to 96 (< 97), then drifts
    highs = [102, 101, 101, 101, 101, 101]
    lows = [99, 96, 98, 98, 98, 98]
    closes = [100, 97, 98, 98, 98, 99]
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv",
                        lambda s: _wait_frame(highs, lows, closes))
    got = wait_patience("HAL", "2026-08-03", 100.0, horizon=5)
    assert got is not None
    _px, _pct, result = got
    assert result == WAIT_OFFERED


def test_wait_patience_ran_away_without_entry(monkeypatch):
    from core.outcome_resolver import WAIT_RAN_AWAY, wait_patience

    highs = [101, 103, 106, 108, 110, 112]
    lows = [99, 100, 101, 102, 103, 104]
    closes = [100, 102, 105, 107, 109, 111]
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv",
                        lambda s: _wait_frame(highs, lows, closes))
    got = wait_patience("HAL", "2026-08-03", 100.0, horizon=5)
    assert got is not None
    assert got[2] == WAIT_RAN_AWAY


def test_wait_patience_failed_setup(monkeypatch):
    from core.outcome_resolver import WAIT_FAILED, wait_patience

    highs = [101, 101, 101, 101, 101, 101]
    lows = [99, 99, 98.5, 99, 99, 99]
    closes = [100, 100, 99.5, 100, 100, 99.8]
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv",
                        lambda s: _wait_frame(highs, lows, closes))
    got = wait_patience("HAL", "2026-08-03", 100.0, horizon=5)
    assert got is not None
    assert got[2] == WAIT_FAILED
