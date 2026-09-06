from __future__ import annotations

from datetime import date, datetime

from research.autonomy import schedules as SCH


def test_persisted_iso_holiday_string_is_not_a_session():
    holiday = datetime(2026, 10, 2, 10, 0)
    assert SCH._is_session_day(holiday, {"2026-10-02"}) is False
    assert SCH.market_is_open(datetime(2026, 10, 2, 10, 30), {"2026-10-02"}) is False


def test_date_object_holiday_is_also_supported():
    holiday = datetime(2026, 10, 2, 10, 0)
    assert SCH._is_session_day(holiday, {date(2026, 10, 2)}) is False


def test_last_completed_session_skips_persisted_friday_holiday():
    # Monday morning before the cash close: Friday was a persisted holiday, so
    # Thursday is the most recent completed NSE session.
    monday = datetime(2026, 10, 5, 10, 0)
    assert SCH.last_completed_session_date(monday, {"2026-10-02"}) == "2026-10-01"
