"""Phase II: period alignment, freshness, quality gates, FEATURE-002 ops states."""
from __future__ import annotations

from data.period_alignment import ANNUAL, NINE_MONTH, QUARTER, YTD, classify_period, consol_label
from data.universe_freshness import REQUIRE_BAR_ON_SESSION, investability
from research.data_foundation.quality import (
    EVENT_DATE_ONLY,
    EVENT_TIMESTAMP_STRONG,
    FUNDAMENTAL_MISSING,
    FUNDAMENTAL_PIT_STRONG,
    event_quality,
    fundamental_quality,
)
from research.feature002.acceptance import (
    HEALTHY_COLLECTING,
    NO_POST_ACTIVATION_SCAN,
    evaluate_first_real_scan,
    operational_state,
)
from research.feature002.constants import UNTIL_MATURE


def test_nine_month_is_not_a_quarter():
    q = classify_period(
        period="Quarterly", period_start="2024-10-01", period_end="2024-12-31",
        cumulative="Non-cumulative", relating_to="Third Quarter",
    )
    assert q["period_kind"] == QUARTER
    assert q["quarterly_usable"] is True
    nine = classify_period(
        period="Quarterly", period_start="2024-04-01", period_end="2024-12-31",
        cumulative="Cumulative", relating_to="Third Quarter",
    )
    assert nine["period_kind"] == NINE_MONTH
    assert nine["quarterly_usable"] is False
    ytd = classify_period(
        period="Quarterly", period_start="2024-04-01", period_end="2024-09-30",
        cumulative="Cumulative",
    )
    assert ytd["period_kind"] == YTD
    assert ytd["quarterly_usable"] is False
    ann = classify_period(period="Annual", period_start="2024-04-01", period_end="2025-03-31")
    assert ann["period_kind"] == ANNUAL
    assert ann["quarterly_usable"] is False


def test_consol_not_mixed_silently():
    assert consol_label("Consolidated") == "CONSOLIDATED"
    assert consol_label("Non-Consolidated") == "STANDALONE"


def test_stale_and_delist_and_holiday_rules():
    cal = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    dead = investability(
        symbol="DEAD", as_of="2024-01-05", listed=True, delisted=True,
        last_bar="2023-06-01", calendar=cal,
    )
    assert dead["tradable"] is False
    assert dead["reason"] == "delisted"
    susp = investability(
        symbol="HALT", as_of="2024-01-05", listed=True, delisted=False,
        last_bar="2024-01-05", calendar=cal, suspended=True,
    )
    assert susp["tradable"] is False
    same = investability(
        symbol="AAA", as_of="2024-01-05", listed=True, delisted=False,
        last_bar="2024-01-05", calendar=cal, max_stale_sessions=REQUIRE_BAR_ON_SESSION,
    )
    assert same["tradable"] is True
    miss = investability(
        symbol="AAA", as_of="2024-01-05", listed=True, delisted=False,
        last_bar="2024-01-04", calendar=cal, max_stale_sessions=REQUIRE_BAR_ON_SESSION,
    )
    assert miss["tradable"] is False
    assert miss["reason"] == "missing_session_bar"
    # one official session gap allowed when max_stale=1 (halt / holiday handling)
    halt = investability(
        symbol="AAA", as_of="2024-01-05", listed=True, delisted=False,
        last_bar="2024-01-04", calendar=cal, max_stale_sessions=1,
    )
    assert halt["tradable"] is True
    hard = investability(
        symbol="AAA", as_of="2024-01-05", listed=True, delisted=False,
        last_bar="2024-01-01", calendar=cal, max_stale_sessions=1,
    )
    assert hard["tradable"] is False
    relist = investability(
        symbol="NEW", as_of="2024-01-05", listed=True, delisted=False,
        last_bar="2024-01-05", calendar=cal,
    )
    assert relist["tradable"] is True


def test_quality_tokens():
    assert fundamental_quality(None) == FUNDAMENTAL_MISSING
    assert fundamental_quality({
        "available_at": "2024-11-03", "source": "nse_xbrl", "raw_hash": "abc",
        "quarterly_usable": True,
    }) == FUNDAMENTAL_PIT_STRONG
    assert event_quality({"timestamp": "2024-11-03T20:20:00+05:30"}) == EVENT_TIMESTAMP_STRONG
    assert event_quality({"announced_date": "2024-11-03"}) == EVENT_DATE_ONLY


def test_feature002_ops_state_and_acceptance_do_not_fabricate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "research.feature002.health.SCAN_STATE", tmp_path / "no_scan.json",
    )
    st = operational_state(ledger_path=tmp_path / "empty.db")
    assert st["operational_state"] == NO_POST_ACTIVATION_SCAN
    assert UNTIL_MATURE in st["combined"]
    acc = evaluate_first_real_scan(ledger_path=tmp_path / "empty.db")
    assert acc["accepted"] is False
    assert acc["checks"]["primary_live_scan_rows"] is False


def test_feature002_collecting_state_separate_from_maturity():
    # Operational collecting + insufficient data is a valid pair (documented).
    assert HEALTHY_COLLECTING != UNTIL_MATURE
