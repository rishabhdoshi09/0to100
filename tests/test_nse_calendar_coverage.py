"""Session counting for evidence fails closed when the holiday calendar cannot vouch for a year."""

from __future__ import annotations

from datetime import date

import research.intelligence.data.nse_calendar as CAL
import scan.signal_backtest as SB

FULL_2026 = CAL.load_holidays()


def test_only_years_with_a_published_calendar_are_covered():
    assert 2026 in CAL.covered_years(FULL_2026)
    assert CAL.covered_years({"2027-01-26"}) == set()                    # one amendment is not a calendar
    assert CAL.covered_years({"garbage", "2026-02-30"}) == set()
    assert CAL.calendar_gap(date(2026, 3, 1), date(2026, 4, 1), FULL_2026) == ""
    assert CAL.calendar_gap(date(2026, 12, 20), date(2027, 1, 5), FULL_2026) == "EXCHANGE_CALENDAR_NOT_COVERED:2027"
    assert CAL.calendar_gap(date(2026, 3, 1), date(2026, 4, 1), set()) == "EXCHANGE_CALENDAR_UNAVAILABLE"


def test_forward_window_skips_holidays_and_refuses_uncovered_years():
    window = CAL.forward_sessions(date(2026, 11, 18), 10, FULL_2026)
    assert window[0] == date(2026, 11, 19) and window[-1] == date(2026, 12, 3)
    assert date(2026, 11, 24) not in window                               # Guru Nanak Jayanti
    assert CAL.forward_sessions(date(2026, 12, 22), 10, FULL_2026) is None  # reaches into 2027
    assert CAL.forward_sessions(date(2026, 11, 18), 10, set()) is None


def test_strict_ordinals_never_count_an_uncovered_year_as_weekdays():
    assert CAL.strict_session_ordinals(["2026-11-06", "2026-11-09", "2026-11-11"], FULL_2026) == {
        "2026-11-06": 1, "2026-11-09": 2, "2026-11-11": 3}
    assert CAL.strict_session_ordinals(["2025-06-02", "2025-06-03"], FULL_2026) is None
    assert SB._session_ordinals(["2025-06-02", "2026-06-03"]) is None     # the walk-forward fallback too
