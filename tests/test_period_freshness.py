"""Filings-period honesty — frozen Screener tables must not look current."""
from __future__ import annotations

from datetime import date

from fundamentals.period_freshness import (
    expected_latest_quarter,
    normalize_period_points,
    pack_filings_stale,
    pack_latest_period,
    pack_needs_filings_retry,
    prefer_fresher_pack,
    quarters_behind,
)


AS_OF = date(2026, 8, 17)


def _eimco_consolidated():
    return {
        "quarterly_results": [
            {"": "Sales+", "Dec 2023": 48, "Mar 2024": 84},
        ],
        "profit_loss": [
            {"": "Sales+", "Mar 2019": 185, "Mar 2020": 108, "Mar 2021": 126,
             "Mar 2022": 84, "Mar 2023": 173, "Mar 2024": 228},
        ],
    }


def _eimco_standalone():
    return {
        "quarterly_results": [
            {"": "Sales+", "Mar 2026": 70, "Jun 2026": 78},
        ],
        "profit_loss": [
            {"": "Sales+", "Mar 2025": 300, "Mar 2026": 350},
        ],
    }


def test_expected_quarter_on_17_aug_is_june():
    assert expected_latest_quarter(AS_OF) == date(2026, 6, 1)


def test_mar_2024_is_stale_in_aug_2026():
    pack = _eimco_consolidated()
    latest, label = pack_latest_period(pack)
    assert label == "Mar 2024"
    assert quarters_behind(latest, AS_OF) >= 2
    assert pack_filings_stale(pack, as_of=AS_OF)
    assert pack_needs_filings_retry(pack, as_of=AS_OF)


def test_jun_2026_is_current_on_17_aug():
    pack = _eimco_standalone()
    latest, label = pack_latest_period(pack)
    assert label == "Jun 2026"
    assert quarters_behind(latest, AS_OF) == 0
    assert not pack_filings_stale(pack, as_of=AS_OF)
    assert not pack_needs_filings_retry(pack, as_of=AS_OF)


def test_prefer_fresher_keeps_standalone_jun_2026():
    chosen = prefer_fresher_pack(_eimco_consolidated(), _eimco_standalone())
    _, label = pack_latest_period(chosen)
    assert label == "Jun 2026"


def test_attempted_flag_stops_retry_loop():
    pack = {**_eimco_consolidated(), "_filings_refresh_attempted": True}
    assert pack_filings_stale(pack, as_of=AS_OF)
    assert not pack_needs_filings_retry(pack, as_of=AS_OF)


def test_empty_tables_are_not_treated_as_frozen_filings():
    pack = {"about": "test co", "quarterly_results": []}
    assert pack_latest_period(pack) == (None, "")
    assert not pack_filings_stale(pack, as_of=AS_OF)
    assert not pack_needs_filings_retry(pack, as_of=AS_OF)


def test_normalize_drops_ttm_and_sorts():
    points = [
        {"period": "Jun 2026", "value": 78},
        {"period": "Mar 2024", "value": 84},
        {"period": "TTM", "value": 999},
        {"period": "Mar 2026", "value": 70},
    ]
    out = normalize_period_points(points)
    assert [row["period"] for row in out] == ["Mar 2024", "Mar 2026", "Jun 2026"]
    assert out[-1]["value"] == 78
