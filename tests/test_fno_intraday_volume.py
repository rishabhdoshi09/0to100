"""Clock-matched NSE volume tests: no future bars, incomplete sessions or daily look-ahead."""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from product.fno_intraday_volume import same_clock_rvol
from product.fo_runtime import _daily_prefilter


AS_OF = date(2026, 10, 14)
IST = ZoneInfo("Asia/Kolkata")


class Client:
    def __init__(self, *, missing=False, days=7):
        self.calls = []
        self.bars = []
        for delta in range(1, 25):
            day = AS_OF - timedelta(days=delta)
            if day.weekday() >= 5:
                continue
            for minute in range(9 * 60 + 15, 15 * 60 + 30):
                if missing and minute % 3:
                    continue
                self.bars.append({
                    "date": datetime.combine(day, datetime.min.time(), IST).replace(
                        hour=minute // 60, minute=minute % 60),
                    "volume": 1000 if minute <= 10 * 60 + 30 else 1_000_000,
                })
            days -= 1
            if days <= 0:
                break

    def historical(self, instrument_token, start, end, interval="minute"):
        self.calls.append((instrument_token, start, end, interval))
        return self.bars


def test_same_clock_current_partial_volume_ignores_future_minute_bars():
    client = Client()
    result = same_clock_rvol(
        client=client, instrument_token=1234, as_of=AS_OF,
        observed_at=datetime(2026, 10, 14, 10, 30, 10, tzinfo=IST),
        cumulative_volume=152_000,
    )
    assert result["available"] is True
    assert result["prior_sessions"] >= 5
    assert result["rvol"] == 2.0
    assert client.calls[0][-1] == "minute"


def test_invalid_or_incomplete_prior_minute_evidence_fails_closed():
    client = Client(missing=True)
    result = same_clock_rvol(
        client=client, instrument_token=1234, as_of=AS_OF,
        observed_at=datetime(2026, 10, 14, 10, 30, tzinfo=IST),
        cumulative_volume=250_000,
    )
    assert result["available"] is False
    assert result["reason"] == "SAME_CLOCK_HISTORY_INSUFFICIENT"


def test_stale_or_missing_instrument_cannot_grant_rvol():
    client = Client()
    res = same_clock_rvol(
        client=client, instrument_token=0, as_of=AS_OF,
        observed_at=datetime(2026, 10, 14, 10, 30, tzinfo=IST),
        cumulative_volume=250_000,
    )
    assert not res["available"]
    assert not client.calls
    res = same_clock_rvol(
        client=client, instrument_token=123, as_of=AS_OF,
        observed_at=datetime(2026, 10, 13, 10, 30, tzinfo=IST),
        cumulative_volume=250_000,
    )
    assert not res["available"]


def test_time_volume_changes_prefilter_decision_not_just_display():
    frame = pd.DataFrame({
        "close": [100.0] * 59 + [110.0],
        "high": [102.0] * 59 + [111.0],
        "low": [98.0] * 59 + [109.0],
        "volume": [1_000_000] * 59 + [40_000],
    })
    assert not _daily_prefilter(frame)["eligible"]
    assert _daily_prefilter(frame, defer_volume=True)["eligible"]
    blocked = _daily_prefilter(frame, intraday_rvol=1.1)
    assert not blocked["eligible"] and blocked["reason"] == "SAME_CLOCK_VOLUME_BELOW_THRESHOLD"
    accepted = _daily_prefilter(frame, intraday_rvol=1.8)
    assert accepted["eligible"]
    assert accepted["daily_rvol"] < 1
    assert accepted["volume_basis"] == "KITE_PRIOR_MINUTE_CUMULATIVE"
