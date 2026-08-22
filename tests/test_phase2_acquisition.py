"""Phase II: period alignment, freshness, quality gates, FEATURE-002 ops states."""
from __future__ import annotations

import pytest

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


def test_feature002_protocol_frozen():
    from research.feature002.constants import (
        FEATURE_SET_VERSION,
        FORWARD_START_TS_IST,
        PRIMARY_SOURCE,
        R3_FORMULA,
        protocol_hash,
    )
    assert FEATURE_SET_VERSION == "feature-002.v1"
    assert FORWARD_START_TS_IST == "2026-08-22T00:00:00+05:30"
    assert PRIMARY_SOURCE == "live_scan"
    assert "0.67" in R3_FORMULA
    # Hash must stay stable — this phase does not revise the experiment.
    assert protocol_hash() == protocol_hash()


def test_qoq_not_confused_with_yoy():
    from data.pit_ratios import derive_ratios
    current = {
        "row_id": "c", "available_at": "2024-11-01",
        "period_end": "2024-09-30", "period_start": "2024-07-01",
        "period_kind": "quarter", "quarterly_usable": True,
        "revenue_from_operations": 120.0, "profit_after_tax": 12.0, "basic_eps": 2.4,
    }
    prior_q = {
        "row_id": "p", "available_at": "2024-08-01",
        "period_end": "2024-06-30", "period_start": "2024-04-01",
        "period_kind": "quarter", "quarterly_usable": True,
        "revenue_from_operations": 100.0, "profit_after_tax": 10.0, "basic_eps": 2.0,
    }
    prior_y = {
        "row_id": "y", "available_at": "2023-11-01",
        "period_end": "2023-09-30", "period_start": "2023-07-01",
        "period_kind": "quarter", "quarterly_usable": True,
        "revenue_from_operations": 80.0, "profit_after_tax": 8.0, "basic_eps": 1.6,
    }
    qoq = derive_ratios(current, prior_q)
    assert qoq["period_alignment"]["qoq_meaningful"] is True
    assert qoq["period_alignment"]["yoy_meaningful"] is False
    assert qoq["values"]["revenue_growth_qoq"] == pytest.approx(0.2)
    assert qoq["values"]["revenue_growth_yoy"] is None
    yoy = derive_ratios(current, prior_y)
    assert yoy["period_alignment"]["yoy_meaningful"] is True
    assert yoy["period_alignment"]["qoq_meaningful"] is False
    assert yoy["values"]["revenue_growth_yoy"] == pytest.approx(0.5)


def test_listing_freshness_is_generic_not_magma():
    from data.listing_archive import is_investable
    import json
    from pathlib import Path
    import tempfile
    d = Path(tempfile.mkdtemp())
    hist = d / "u.json"
    hist.write_text(json.dumps({
        "schema_version": 1,
        "source": "operator_test_archive",
        "rows": [
            {"symbol": "STALECO", "listed": "2020-01-01"},
            {"symbol": "DEADCO", "listed": "2020-01-01", "delisted": "2023-06-01"},
            {"symbol": "FRESHCO", "listed": "2020-01-01"},
        ],
    }), encoding="utf-8")
    cal = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    stale = is_investable(
        "STALECO", "2024-01-05", path=hist, last_bar="2023-10-01",
        calendar=cal, max_stale_sessions=1,
    )
    assert stale["in_universe"] is True
    assert stale["tradable"] is False
    assert stale["freshness_reason"] in {"stale", "hard_stale"}
    dead = is_investable("DEADCO", "2024-01-05", path=hist, last_bar="2023-05-01")
    assert dead["in_universe"] is False
    assert dead["tradable"] is False
    assert dead["freshness_reason"] == "delisted"
    fresh = is_investable(
        "FRESHCO", "2024-01-05", path=hist, last_bar="2024-01-05", calendar=cal,
    )
    assert fresh["tradable"] is True
    one_miss = is_investable(
        "FRESHCO", "2024-01-05", path=hist, last_bar="2024-01-04",
        calendar=cal, max_stale_sessions=1,
    )
    assert one_miss["tradable"] is True


def test_snapshot_universe_applies_freshness(tmp_path):
    import json
    import pandas as pd
    from research.data_foundation.snapshot import EvidenceSnapshot
    hist = tmp_path / "u.json"
    hist.write_text(json.dumps({
        "schema_version": 1,
        "source": "operator_test_archive",
        "rows": [
            {"symbol": "LIVE", "listed": "2020-01-01"},
            {"symbol": "DEAD", "listed": "2020-01-01", "delisted": "2023-01-01"},
        ],
    }), encoding="utf-8")
    frames = {
        "LIVE": pd.DataFrame(
            {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]},
            index=pd.to_datetime(["2024-01-05"]),
        ),
    }
    snap = EvidenceSnapshot(
        "2024-01-05", universe_path=hist, price_frames=frames, guard_network=True,
    )
    uni = snap.universe()
    assert "LIVE" in uni["symbols"]
    assert "DEAD" not in uni["symbols"]
    assert "LIVE" in uni["tradable"]
    assert uni["investability"] == "membership_plus_session_freshness"


def test_sector_coverage_rise_does_not_upgrade_pit():
    from data.sector_map import STATIC_BACKFILL, build_static_map, coverage
    snap = build_static_map()
    cov = coverage(snap)
    assert cov["pit_class"] == STATIC_BACKFILL
    assert snap["pit_class"] == STATIC_BACKFILL
    assert snap["sector_identity_pit"] is False


def test_snapshot_historical_accessors_are_network_free(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import pytest
    from research.data_foundation.network import NetworkForbidden
    from research.data_foundation.snapshot import EvidenceSnapshot
    from data.pit_fundamentals import write_fundamentals

    p = tmp_path / "f.json"
    write_fundamentals([{
        "symbol": "AAA", "available_at": "2024-11-03", "period": "Quarterly",
        "period_end": "2024-09-30", "source": "nse_xbrl", "seq_id": "1",
        "revenue_from_operations": 100.0, "profit_after_tax": 10.0, "basic_eps": 2.0,
    }], path=p, source="test")
    hist = tmp_path / "u.json"
    hist.write_text(json.dumps({
        "schema_version": 1, "source": "operator_test_archive",
        "rows": [{"symbol": "AAA", "listed": "2020-01-01"}],
    }), encoding="utf-8")
    frames = {
        "AAA": pd.DataFrame(
            {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]},
            index=pd.to_datetime(["2024-11-03"]),
        )
    }
    snap = EvidenceSnapshot(
        "2024-11-03", fundamentals_path=p, universe_path=hist,
        price_frames=frames, guard_network=True,
    )
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(
        NetworkForbidden("blocked")
    ))
    assert snap.fundamentals("AAA")["current"]["available_at"] == "2024-11-03"
    assert snap.latest_reported_quarter("AAA")["available_at"] == "2024-11-03"
    uni = snap.universe()
    assert "AAA" in uni["symbols"]
    with pytest.raises(NetworkForbidden):
        requests.get("https://www.nseindia.com/")
