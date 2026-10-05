from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from product.fo_live_history import current_frame, prepare_history
from product.fo_runtime import _daily_prefilter


DAY = date(2026, 10, 5)
NOW = datetime(2026, 10, 5, 10, 0, 0)


def history():
    return pd.DataFrame({"open": 100., "high": 101., "low": 99.,
                         "close": 100., "volume": 1_000_000.},
                        index=pd.date_range(end="2026-10-02", periods=80))


def quote():
    return {"ohlc": {"open": 100., "high": 106., "low": 99., "close": 100.},
            "last_price": 105., "volume": 2_000_000., "timestamp": NOW}


def test_current_kite_bar_reaches_prefilter_without_mutating_cached_history():
    cached = history()
    original = cached.copy(deep=True)
    assert not _daily_prefilter(cached)["eligible"]
    frames, state = prepare_history(["AAA"], history_getter=lambda s: cached,
        client=SimpleNamespace(quote=lambda keys: {"NSE:AAA": quote()}), as_of=DAY, now=NOW)
    assert state["ready"] and state["symbols"] == 1
    assert state["history_source"] == "kite_snapshot"
    assert frames["AAA"].index[-1].date() == DAY
    assert _daily_prefilter(frames["AAA"])["eligible"]
    pd.testing.assert_frame_equal(cached, original)


@pytest.mark.parametrize("change,reason", [
    ({"timestamp": NOW - timedelta(days=3)}, "QUOTE_SESSION_MISMATCH"),
    ({"timestamp": NOW - timedelta(minutes=4)}, "QUOTE_STALE"),
    ({"timestamp": None}, "QUOTE_TIMESTAMP_UNAVAILABLE"),
    ({"volume": None}, "CURRENT_OHLCV_UNAVAILABLE"),
    ({"last_price": float("nan")}, "CURRENT_OHLCV_INVALID"),
    ({"ohlc": {"open": 100., "high": 90., "low": 99.}}, "CURRENT_OHLCV_INVALID"),
])
def test_current_session_requires_trusted_complete_quotes(change, reason):
    q = quote()
    q.update(change)
    frame, actual = current_frame(history(), q, as_of=DAY, now=NOW)
    assert frame is None and actual == reason


def test_replaces_same_day_bar_and_preserves_prior_baseline():
    cached = history()
    cached.loc[pd.Timestamp(DAY)] = [100., 999., 99., 999., 50_000_000.]
    frame, _ = current_frame(cached, quote(), as_of=DAY, now=NOW)
    assert len(frame) == len(cached)
    assert frame.iloc[-1]["high"] == 106.
    assert frame.iloc[-1]["close"] == 105.
    assert _daily_prefilter(frame)["eligible"]
    assert cached.iloc[-1]["high"] == 999.


def test_partial_quote_coverage_is_explicit_and_never_uses_old_bar():
    frames, state = prepare_history(["AAA", "BBB"], history_getter=lambda s: history(),
        client=SimpleNamespace(quote=lambda keys: {"NSE:AAA": quote()}), as_of=DAY, now=NOW)
    assert set(frames) == {"AAA"}
    assert state["requested_symbols"] == 2
    assert state["rejected"] == {"BBB": "QUOTE_TIMESTAMP_UNAVAILABLE"}


def test_read_failure_returns_blocked_details_without_reading_history():
    def fail(keys):
        raise ConnectionError("unavailable")
    frames, state = prepare_history(["AAA"],
        history_getter=lambda s: pytest.fail("do not read unquotable history"),
        client=SimpleNamespace(quote=fail), as_of=DAY, now=NOW)
    assert not frames and not state["ready"]
    assert state["rejected"]["AAA"] == "CURRENT_QUOTE_READ_FAILED:ConnectionError"
