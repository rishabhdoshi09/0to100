"""Tests for discontinuity classification + CA verify metric correction."""
from research.intelligence.data.discontinuity_audit import (
    classify_discontinuity,
    verification_trace,
)


def test_sparse_calendar_span_is_suspension_not_ca_failure():
    d = classify_discontinuity(
        symbol="SPARSE",
        d0="2024-01-01",
        d1="2024-06-01",
        cal_days=152,
        pct_raw=-50.0,
        pct_adj=-50.0,
        pre_raw=100.0,
        post_raw=50.0,
        pre_adj=100.0,
        post_adj=50.0,
        ca_near=[],
    )
    assert d.classification == "SUSPENSION_OR_RELISTING"


def test_verified_ca_restores_continuity():
    d = classify_discontinuity(
        symbol="RELIANCE",
        d0="2024-10-25",
        d1="2024-10-28",
        cal_days=3,
        pct_raw=-49.8,
        pct_adj=0.5,
        pre_raw=200.0,
        post_raw=100.4,
        pre_adj=100.0,
        post_adj=100.5,
        ca_near=[{"ex_date": "2024-10-28", "factor": 2.0, "type": "bonus"}],
    )
    assert d.classification == "SUPPORTED_CA"
    assert d.ca_status == "VERIFIED"
    trace = verification_trace(d)
    assert trace["verification_status"] == "VERIFIED"
    assert "corporate action" in trace["user_facing"]["layer1"]["explanation"].lower()


def test_missing_official_factor_is_unresolved_not_inferred():
    d = classify_discontinuity(
        symbol="KOTAKBANK",
        d0="2026-01-13",
        d1="2026-01-14",
        cal_days=1,
        pct_raw=-80.3,
        pct_adj=-80.3,
        pre_raw=500.0,
        post_raw=98.7,
        pre_adj=500.0,
        post_adj=98.7,
        ca_near=[],
    )
    assert d.classification == "UNRESOLVED"
    assert d.ca_status == "MISSING_SOURCE"
    assert d.ratio_hint is not None  # investigative
    assert "must not become an authoritative" in d.notes


def test_partial_when_ca_present_but_still_discontinuous():
    d = classify_discontinuity(
        symbol="BAJFINANCE",
        d0="2025-06-13",
        d1="2025-06-16",
        cal_days=3,
        pct_raw=-89.9,
        pct_adj=-49.7,
        pre_raw=1000.0,
        post_raw=100.5,
        pre_adj=200.0,
        post_adj=100.5,
        ca_near=[{"ex_date": "2025-06-16", "factor": 5.0, "type": "bonus"}],
    )
    assert d.classification == "UNRESOLVED"
    assert d.ca_status in {"PARTIAL", "CONFLICT"}
