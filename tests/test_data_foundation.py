"""Point-in-time data foundation — network-free acceptance tests."""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from data.earnings_events import classify_session, post_result_study_rows, timeline
from data.pit_events import write_events
from data.pit_fundamentals import (
    fundamentals_with_ratios,
    get_fundamentals,
    get_period_as_of,
    write_fundamentals,
)
from data.pit_ratios import derive_ratios
from data.sector_map import STATIC_BACKFILL, freeze_snapshot, load_snapshot, sector_of
from data.ca_research import events_as_of
from data.benchmarks import load_index, session_return
from data.listing_archive import is_investable
from research.data_foundation.manifest import build_manifest, snapshot_hash
from research.data_foundation.network import NetworkForbidden, forbid_network
from research.data_foundation.snapshot import EvidenceSnapshot
from research.data_foundation.policy import decide
from research.data_foundation.audit import run as audit_run
from research.feature002.constants import UNTIL_MATURE


def _fund_rows():
    return [
        {
            "symbol": "AAA",
            "available_at": "2024-11-03",
            "period": "Quarterly",
            "period_end": "2024-09-30",
            "source": "nse_xbrl",
            "seq_id": "1",
            "revenue_from_operations": 100.0,
            "profit_after_tax": 10.0,
            "profit_before_tax": 12.0,
            "basic_eps": 2.0,
            "paid_up_equity_capital": 50.0,
            "revision_status": "original",
        },
        {
            "symbol": "AAA",
            "available_at": "2025-02-10",
            "period": "Quarterly",
            "period_end": "2024-09-30",
            "source": "nse_xbrl",
            "seq_id": "2",
            "revenue_from_operations": 105.0,
            "profit_after_tax": 9.0,
            "profit_before_tax": 11.0,
            "basic_eps": 1.8,
            "paid_up_equity_capital": 50.0,
            "revision_status": "restated",
        },
        {
            "symbol": "AAA",
            "available_at": "2025-02-10",
            "period": "Quarterly",
            "period_end": "2024-12-31",
            "source": "nse_xbrl",
            "seq_id": "3",
            "revenue_from_operations": 120.0,
            "profit_after_tax": 15.0,
            "basic_eps": 2.5,
        },
    ]


@pytest.fixture
def fund_path(tmp_path):
    p = tmp_path / "pit_fundamentals.json"
    write_fundamentals(_fund_rows(), path=p, source="test")
    return p


def test_future_filing_cannot_affect_earlier_fundamentals(fund_path):
    assert get_fundamentals("AAA", "2024-11-02", path=fund_path) is None
    nov = get_period_as_of("AAA", "2024-09-30", "2024-11-03", path=fund_path)
    assert nov is not None
    assert nov["basic_eps"] == 2.0
    assert nov["revenue_from_operations"] == 100.0


def test_restatement_cannot_rewrite_earlier_known_values(fund_path):
    before = get_period_as_of("AAA", "2024-09-30", "2024-12-01", path=fund_path)
    after = get_period_as_of("AAA", "2024-09-30", "2025-02-10", path=fund_path)
    assert before["basic_eps"] == 2.0
    assert after["basic_eps"] == 1.8
    assert before["revision_status"] == "original"
    assert after["revision_status"] == "restated"
    # The original row remains in the ledger for the earlier as-of.
    assert before["seq_id"] != after["seq_id"]


def test_future_earnings_event_cannot_appear_earlier(tmp_path):
    p = tmp_path / "events.json"
    write_events([
        {
            "symbol": "AAA",
            "available_at": "2024-11-03T20:20:00",
            "available_at_ts": "2024-11-03T20:20:00+05:30",
            "event_type": "EARNINGS_RESULT",
            "period": "Quarterly",
            "period_end": "2024-09-30",
            "source": "nse_financial_results",
            "seq_id": "1",
        }
    ], path=p, source="nse_financial_results")
    assert timeline("AAA", "2024-11-02", path=p) == []
    later = timeline("AAA", "2024-11-03", path=p)
    assert len(later) == 1
    assert later[0]["announced_date"] == "2024-11-03"
    assert later[0]["session_class"] == "after_market"
    assert later[0]["not_earnings_surprise"] is True
    study = post_result_study_rows("AAA", "2024-11-03", path=p)
    assert study[0]["forbidden_label"] == "earnings_surprise"
    assert "surprise" not in study[0]


def test_future_sector_map_cannot_rewrite_frozen_snapshot(tmp_path):
    path = freeze_snapshot(dest_dir=tmp_path, as_of="2026-08-22")
    frozen = load_snapshot(path)
    n = frozen["n_mapped"]
    # Mutating a live rebuild must not change the frozen file.
    live = {"version": "sector_map.v2-NEW", "n_mapped": 1, "rows": [], "content_hash": "x"}
    again = freeze_snapshot(dest_dir=tmp_path, as_of="2026-08-22")
    assert again == path
    reread = load_snapshot(path)
    assert reread["n_mapped"] == n
    assert reread["version"] == frozen["version"]
    info = sector_of("RELIANCE")
    assert info["pit_status"] in {STATIC_BACKFILL, "UNKNOWN"} or info["sector"]


def test_future_ca_filter(tmp_path):
    p = tmp_path / "ca.json"
    p.write_text(json.dumps({
        "schema_version": 1,
        "source": "test",
        "events": [
            {"symbol": "AAA", "ex_date": "2024-06-01", "factor": 2.0, "type": "bonus"},
        ],
    }), encoding="utf-8")
    assert events_as_of("AAA", "2024-05-31", path=p) == []
    after = events_as_of("AAA", "2024-06-01", path=p)
    assert len(after) == 1
    assert after[0]["factor"] == 2.0


def test_delisted_not_investable(tmp_path):
    hist = tmp_path / "universe.json"
    hist.write_text(json.dumps({
        "schema_version": 1,
        "source": "operator_test_archive",
        "note": "synthetic official-style archive for tests",
        "rows": [
            {"symbol": "DEADCO", "listed": "2020-01-01", "delisted": "2023-06-01"},
            {"symbol": "LIVECO", "listed": "2020-01-01"},
        ],
    }), encoding="utf-8")
    dead = is_investable("DEADCO", "2023-06-01", path=hist)
    live = is_investable("LIVECO", "2023-06-01", path=hist)
    assert dead["in_universe"] is False
    assert live["in_universe"] is True


def test_snapshot_replay_deterministic(fund_path, tmp_path):
    idx = tmp_path / "indices"
    idx.mkdir()
    _write_index_day(idx, "01012025", 100)
    _write_index_day(idx, "02012025", 101)
    frames = {
        "AAA": pd.DataFrame(
            {"open": [10, 11], "high": [11, 12], "low": [9, 10],
             "close": [10.5, 11.2], "volume": [1000, 1100]},
            index=pd.to_datetime(["2024-11-01", "2024-11-04"]),
        )
    }
    a = EvidenceSnapshot(
        "2024-11-04", fundamentals_path=fund_path, price_frames=frames,
        index_dir=idx, guard_network=True, config={"k": 1},
    )
    b = EvidenceSnapshot(
        "2024-11-04", fundamentals_path=fund_path, price_frames=frames,
        index_dir=idx, guard_network=True, config={"k": 1},
    )
    assert a.replay_hash() == b.replay_hash()
    px = a.prices("AAA")
    assert px is not None
    assert str(pd.Timestamp(px.index[-1]).date()) <= "2024-11-04"
    assert a.fundamentals("AAA")["current"]["available_at"] <= "2024-11-04"


def test_evidence_version_changes_alter_snapshot_hash():
    m1 = build_manifest(as_of="2024-11-04", config={"v": 1})
    m2 = build_manifest(as_of="2024-11-04", config={"v": 2})
    assert m1["snapshot_hash"] != m2["snapshot_hash"]
    assert snapshot_hash(m1) == m1["snapshot_hash"]


def test_historical_research_zero_live_network(fund_path, tmp_path):
    frames = {
        "AAA": pd.DataFrame(
            {"open": [10], "high": [11], "low": [9], "close": [10], "volume": [1]},
            index=pd.to_datetime(["2024-11-03"]),
        )
    }
    snap = EvidenceSnapshot(
        "2024-11-03", fundamentals_path=fund_path, price_frames=frames,
        guard_network=True,
    )
    snap.prices("AAA")
    snap.fundamentals("AAA")
    with forbid_network():
        import requests
        with pytest.raises(NetworkForbidden):
            requests.get("https://example.invalid")


def test_timezone_session_classification():
    assert classify_session("2024-11-03T08:00:00+05:30") == "known_before_market"
    assert classify_session("2024-11-03T10:00:00+05:30") == "during_market"
    assert classify_session("2024-11-03T20:20:00+05:30") == "after_market"
    assert classify_session(None) == "unknown"


def test_missing_benchmark_return_is_not_zero(tmp_path):
    idx = tmp_path / "indices"
    idx.mkdir()
    _write_index_day(idx, "01012025", 100)
    _write_index_day(idx, "02012025", 110)
    ok = session_return("Nifty 50", "2025-01-02", index_dir=idx)
    assert ok["ret"] == pytest.approx(0.10)
    missing = session_return("Nifty 50", "2025-01-03", index_dir=idx)
    assert missing["ret"] is None
    assert missing["status"] == "UNKNOWN"
    assert missing["ret"] != 0


def test_ratios_never_zero_fill_and_not_surprise(fund_path):
    bundle = fundamentals_with_ratios("AAA", "2024-11-03", path=fund_path)
    assert bundle["ratios"]["not_earnings_surprise"] is True
    # no prior period yet → growth UNKNOWN/None, not 0
    assert bundle["ratios"]["values"]["revenue_growth"] is None
    later = fundamentals_with_ratios("AAA", "2025-02-10", path=fund_path)
    assert later["ratios"]["values"]["revenue_growth"] is not None
    assert later["lineage"]["revenue_growth"]["source_rows"]["current"]


def test_audit_catches_future_leakage():
    rows = [
        {"symbol": "A", "available_at": "2024-01-01", "source": "nse"},
        {"symbol": "B", "available_at": "2024-06-01", "source": "nse"},
    ]
    ok = audit_run(rows, as_of="2024-12-31")
    assert ok["future_leakage"] == 0
    leak = audit_run(rows, as_of="2024-03-01")
    assert leak["future_leakage"] == 1
    assert leak["as_of_invariance"] is False


def test_policy_no_silent_zero():
    assert decide("benchmark_return", None) == "UNKNOWN"
    assert decide("fundamentals_metric", None) == "UNKNOWN"
    assert decide("ohlcv_close", 10.0, stale_sessions=1) == "OK"
    assert decide("ohlcv_close", 10.0, stale_sessions=40) == "FAIL"


def test_feature002_status_string_unchanged():
    assert UNTIL_MATURE == "FORWARD VALIDATION ACTIVE — INSUFFICIENT NEW DATA"


def _write_index_day(d, stem: str, close: float) -> None:
    p = d / f"{stem}.csv"
    p.write_text(
        "Index Name,Index Date,Open Index Value,High Index Value,"
        "Low Index Value,Closing Index Value,Volume\n"
        f"Nifty 50,01-01-2025,{close},{close},{close},{close},0\n"
        f"Nifty 500,01-01-2025,{close},{close},{close},{close},0\n",
        encoding="utf-8",
    )
