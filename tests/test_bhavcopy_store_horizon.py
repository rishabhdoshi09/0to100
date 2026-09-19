from __future__ import annotations

from datetime import date

from data.bhavcopy_store import _trading_days_back


def test_bhavcopy_candidate_horizon_uses_calendar_allowance_not_weekday_multiplier():
    days = _trading_days_back(500)

    assert days
    assert 0 <= (date.today() - days[0]).days <= 2
    assert all(day.weekday() < 5 for day in days)
    # 500 sessions need roughly 775 calendar days, which contains about 553
    # weekdays. The old bug produced 775 weekdays (~3 years) and hundreds of
    # unnecessary network requests on a cold start.
    assert 500 <= len(days) <= 560
    assert (days[0] - days[-1]).days < 775
