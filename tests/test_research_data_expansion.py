"""Unit tests for research data expansion helpers (network-free)."""
from __future__ import annotations

from research.data_expansion.assess import (
    assess_fundamentals_events,
    assess_sector_history,
    future_research_families,
    low_vol_retest_readiness,
    research_power,
)
from research.data_expansion.classify import CLASSES, MIN_SESSIONS_CERT


def test_classes_cover_required_buckets():
    required = {
        "CERTIFIABLE",
        "PARTIAL",
        "BLOCKED_IDENTITY",
        "BLOCKED_CA",
        "BLOCKED_UNIVERSE",
        "INSUFFICIENT_HISTORY",
        "OTHER",
    }
    assert required == set(CLASSES)


def test_sector_history_not_research_ready():
    s = assess_sector_history()
    assert s["status"] == "NOT_RESEARCH_READY"
    assert s["pit_sector_history"] is False
    assert s["blocks_ohlcv_research"] is False
    assert "plain" in s


def test_fundamentals_require_available_at():
    f = assess_fundamentals_events()
    assert f["key_requirement"] == "AVAILABLE_AT"
    statuses = {r["dataset"]: r["status"] for r in f["datasets"]}
    assert statuses["reported_vs_available_timestamps"] == "MISSING"
    assert statuses["valuation_multiples"] == "OPERATIONAL_ONLY"


def test_research_power_gain_positive():
    p = research_power(
        n_securities=870,
        n_sessions=1900,
        security_sessions=1_497_079,
        date_start="2020-01-01",
        date_end="2026-08-11",
    )
    assert p["approx_sample_size_gain"] > 10
    assert p["independent_calendar_years"] == 7
    closed = [
        x for x in p["hypothesis_family_readiness"] if x["class"] == "CLOSED_REJECTED"
    ]
    assert len(closed) >= 5


def test_low_vol_retest_ready_on_expanded_surface():
    lv = low_vol_retest_readiness(n_securities=870, n_sessions=1900)
    assert lv["verdict"] == "LOW_VOL_RETEST_READY"
    assert lv["do_not_rerun_in_this_task"] is True
    assert lv["frozen_protocol_preserved"] is True
    assert lv["frozen"]["status"] == "INCONCLUSIVE"
    assert lv["frozen"]["next_action"] == "HOLD_NO_TUNING"


def test_low_vol_still_thin_when_shallow():
    lv = low_vol_retest_readiness(
        n_securities=29,
        n_sessions=200,
        prior29_post_ca_sessions=200,
    )
    assert lv["verdict"] == "LOW_VOL_STILL_TOO_THIN"


def test_future_families_rank_low_vol_first_when_ready():
    fams = future_research_families(
        n_certifiable=870, low_vol_verdict="LOW_VOL_RETEST_READY"
    )
    assert fams[0]["family"] == "low_volatility_retest"
    assert fams[0]["priority"] == 1


def test_cert_session_gate_is_strict():
    assert MIN_SESSIONS_CERT >= 1000
