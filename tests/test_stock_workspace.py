import json

from datetime import datetime, timezone

import pandas as pd

from product.stock_workspace import (
    _hydrate_raw_fundamentals,
    build_stock_workspace,
)

_USABLE_FUND = {
    "roe": 18,
    "roce": 20,
    "pe": 24,
    "sales_growth_3y": 12,
    "profit_growth_3y": 14,
    "debt_to_equity": 0.4,
}


def _ohlcv_frame(end: str, periods: int = 280) -> pd.DataFrame:
    index = pd.date_range(end=end, periods=periods, freq="B")
    close = pd.Series([100 + i * 0.2 for i in range(periods)], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [100000 + i * 100 for i in range(periods)],
        },
        index=index,
    )


def test_stock_workspace_combines_technicals_fundamentals_and_sources():
    index = pd.date_range("2025-01-01", periods=280, freq="B")
    close = pd.Series([100 + i * 0.2 for i in range(280)], index=index)
    frame = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [100000 + i * 100 for i in range(280)],
        },
        index=index,
    )
    result = build_stock_workspace(
        "TEST",
        scan_payload={"scanned_at": "2026-01-28T10:00:00+00:00", "records": [{"symbol": "TEST", "company": "Test Ltd", "score": 80, "reasons": ["Momentum"]}]},
        long_term_payload={"scanned_at": "2026-01-27T10:00:00+00:00", "records": [{"symbol": "TEST", "sector": "Industrials", "fundamental_score": 74, "fundamentals": {"roe": 18, "roce": 20, "pe": 24, "sales_growth_3y": 12, "profit_growth_3y": 14, "debt_to_equity": 0.4}}]},
        raw_fundamentals={"available": True, "fetched_at": "2026-01-28T09:00:00+00:00", "data": {"about": "Makes industrial equipment"}, "section_as_of": {}},
        frame=frame,
        news=[{"headline": "New order", "published_at": "2026-01-28T08:00:00+00:00", "impact_score": 75}],
        fno_payload={"generated_at": "2026-01-28T06:00:00+00:00", "underlyings": [{"symbol": "TEST", "lot_size": 100}]},
        now=datetime(2026, 1, 28, 12, 0, tzinfo=timezone.utc),
    )
    assert result["state"] == "RESEARCH_READY"
    assert result["technical"]["trend"] in {"PRIMARY UPTREND", "UPTREND"}
    assert result["fundamentals"]["coverage_pct"] > 0
    assert result["company"] == "Test Ltd"
    assert result["sources"][0]["status"] == "FRESH"


def test_block_deals_use_institutional_flows_cache(monkeypatch):
    from data.block_deals import get_bulk_deals, get_significant_deals

    monkeypatch.setattr(
        "data.block_deals.get_flows",
        lambda max_age_s=10800: {
            "bulk_deals": [
                {"symbol": "RELIANCE", "client": "FUND", "side": "BUY", "qty": 100000, "price": 2500},
            ],
            "block_deals": [],
        },
    )
    bulk = get_bulk_deals()
    assert len(bulk) == 1
    assert bulk[0].symbol == "RELIANCE"
    sig = get_significant_deals(["RELIANCE"], min_value_cr=1.0)
    assert len(sig) == 1


def test_stock_workspace_key_ratios_pe_and_sector_peers():
    result = build_stock_workspace(
        "TEST",
        scan_payload={
            "records": [
                {"symbol": "TEST", "sector": "Banking", "score": 70, "status": "Watch", "company": "Test"},
                {"symbol": "AAA", "sector": "Banking", "score": 82, "status": "Ready", "company": "Alpha"},
            ],
        },
        long_term_payload={"records": []},
        raw_fundamentals={
            "available": True,
            "fetched_at": "2026-01-28T09:00:00+00:00",
            "data": {
                "key_ratios": [{"name": "Stock P/E", "value": "21.2"}],
                "peer_comparison": [{"": "Peer Ltd", "P/E": "19", "CMP": "400"}],
            },
            "section_as_of": {},
        },
        frame=[],
        news=[],
        fno_payload={},
        now=datetime(2026, 1, 28, tzinfo=timezone.utc),
    )
    pe = next(item for item in result["fundamentals"]["metrics"] if item["key"] == "pe")
    assert pe["value"] == 21.2
    assert result["fundamentals"]["key_ratios"][0]["name"] == "Stock P/E"
    assert result["peers"]["sector_peers"][0]["symbol"] == "AAA"
    assert result["peers"]["screener_table"][0]["P/E"] == "19"
    assert result["peers"]["average_pe"] == 19.0
    assert result["peers"]["peer_pe_sample_count"] == 1


def test_stock_workspace_serializes_when_volume_history_is_nan():
    index = pd.date_range("2025-01-01", periods=120, freq="B")
    close = pd.Series([100 + i * 0.1 for i in range(120)], index=index)
    frame = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [float("nan")] * 120,
        },
        index=index,
    )
    result = build_stock_workspace(
        "NANVOL",
        scan_payload={},
        long_term_payload={},
        raw_fundamentals={},
        frame=frame,
        news=[],
        fno_payload={},
        now=datetime(2026, 1, 28, tzinfo=timezone.utc),
    )
    json.dumps(result, allow_nan=False)
    vol_metric = next(m for m in result["technical"]["metrics"] if m["key"] == "volume_ratio")
    assert vol_metric["value"] is None
    assert result["technical"]["volume_ratio"] is None


def test_stock_workspace_stays_honest_when_data_is_missing():
    result = build_stock_workspace(
        "EMPTY",
        scan_payload={},
        long_term_payload={},
        raw_fundamentals={},
        frame=[],
        news=[],
        fno_payload={},
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert result["state"] == "DATA_INCOMPLETE"
    assert result["confidence_pct"] == 0
    assert result["gaps"]


def test_workspace_names_stale_filings_instead_of_incomplete_coverage():
    result = build_stock_workspace(
        "EIMCOELECO",
        scan_payload={
            "scanned_at": "2026-08-17T10:00:00+00:00",
            "records": [{"symbol": "EIMCOELECO", "company": "Eimco Elecon", "score": 70}],
        },
        long_term_payload={
            "scanned_at": "2026-08-16T10:00:00+00:00",
            "records": [{"symbol": "EIMCOELECO", "sector": "Capital Goods", "fundamentals": _USABLE_FUND}],
        },
        raw_fundamentals={
            "available": True,
            "fetched_at": "2026-08-17T09:00:00+00:00",
            "data": {
                "about": "Mining equipment",
                "quarterly_results": [{"": "Sales+", "Dec 2023": 48, "Mar 2024": 84}],
            },
            "section_as_of": {"financial_history": "2024-03-01"},
        },
        frame=_ohlcv_frame("2026-08-14"),
        news=[],
        fno_payload={},
        now=datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc),
    )
    assert result["fundamentals"]["coverage_pct"] >= 40
    assert result["state"] == "TECHNICAL_ONLY"
    assert "Mar 2024" in result["summary"]
    assert "incomplete or stale" not in result["summary"]
    deep = next(item for item in result["sources"] if item["name"] == "Deep fundamentals")
    assert deep["status"] == "STALE"
    assert deep["quarters_behind"] >= 1
    assert deep["as_of_label"] == "Mar 2024"


def test_workspace_marks_june_2026_filings_fresh_in_august():
    result = build_stock_workspace(
        "EIMCOELECO",
        scan_payload={
            "scanned_at": "2026-08-17T10:00:00+00:00",
            "records": [{"symbol": "EIMCOELECO", "company": "Eimco Elecon", "score": 70}],
        },
        long_term_payload={
            "scanned_at": "2026-08-16T10:00:00+00:00",
            "records": [{"symbol": "EIMCOELECO", "sector": "Capital Goods", "fundamentals": _USABLE_FUND}],
        },
        raw_fundamentals={
            "available": True,
            "fetched_at": "2026-08-17T09:00:00+00:00",
            "data": {
                "about": "Mining equipment",
                "quarterly_results": [{"": "Sales+", "Mar 2026": 70, "Jun 2026": 78}],
            },
            "section_as_of": {"financial_history": "2026-06-01"},
        },
        frame=_ohlcv_frame("2026-08-14"),
        news=[{"headline": "Results", "published_at": "2026-08-16T08:00:00+00:00", "impact_score": 70}],
        fno_payload={"generated_at": "2026-08-17T06:00:00+00:00", "underlyings": [{"symbol": "EIMCOELECO", "lot_size": 1}]},
        now=datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc),
    )
    deep = next(item for item in result["sources"] if item["name"] == "Deep fundamentals")
    assert deep["status"] == "FRESH"
    assert deep["quarters_behind"] == 0
    assert result["state"] == "RESEARCH_READY"
    assert "incomplete" not in result["summary"]


def test_workspace_does_not_blame_fundamentals_when_other_layers_are_the_gap():
    result = build_stock_workspace(
        "TEST",
        scan_payload={},
        long_term_payload={
            "scanned_at": "2025-01-01T10:00:00+00:00",
            "records": [{"symbol": "TEST", "sector": "Industrials", "fundamentals": _USABLE_FUND}],
        },
        raw_fundamentals={
            "available": True,
            "fetched_at": "2026-08-17T09:00:00+00:00",
            "data": {
                "about": "Makes industrial equipment",
                "quarterly_results": [{"": "Sales+", "Jun 2026": 78}],
            },
            "section_as_of": {"financial_history": "2026-06-01"},
        },
        frame=_ohlcv_frame("2026-08-14"),
        news=[],
        fno_payload={},
        now=datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc),
    )
    assert result["fundamentals"]["coverage_pct"] >= 40
    assert result["confidence_pct"] < 70
    assert result["state"] == "TECHNICAL_ONLY"
    assert "other research layers" in result["summary"]
    assert "incomplete" not in result["summary"]


def test_workspace_incomplete_coverage_is_not_called_stale():
    result = build_stock_workspace(
        "THIN",
        scan_payload={
            "scanned_at": "2026-08-17T10:00:00+00:00",
            "records": [{"symbol": "THIN", "company": "Thin Ltd"}],
        },
        long_term_payload={"records": []},
        raw_fundamentals={
            "available": True,
            "fetched_at": "2026-08-17T09:00:00+00:00",
            "data": {"about": "Thin coverage", "key_ratios": [{"name": "Stock P/E", "value": "21.2"}]},
            "section_as_of": {},
        },
        frame=_ohlcv_frame("2026-08-14"),
        news=[],
        fno_payload={},
        now=datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc),
    )
    assert result["fundamentals"]["coverage_pct"] < 40
    assert result["state"] == "TECHNICAL_ONLY"
    assert result["summary"] == "Price structure is available, but fundamental coverage is incomplete."


def test_hydrate_retries_frozen_pack_and_dates_section(monkeypatch):
    standalone = {
        "quarterly_results": [{"": "Sales+", "Jun 2026": 78}],
        "_qt_fetched_at": "2026-08-17T09:00:00+00:00",
    }
    monkeypatch.setattr(
        "fundamentals.lazy.ensure_deep_fundamentals",
        lambda symbol, force_refresh=False: standalone,
    )
    out = _hydrate_raw_fundamentals(
        "EIMCOELECO",
        {
            "available": True,
            "data": {"quarterly_results": [{"": "Sales+", "Mar 2024": 84}]},
            "section_as_of": {},
        },
    )
    assert out["section_as_of"]["financial_history"] == "2026-06-01"
    assert "Jun 2026" in str(out["data"].get("quarterly_results"))


def test_hydrate_fills_section_as_of_from_current_pack_without_fetch(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("must not refresh a current pack")

    monkeypatch.setattr("fundamentals.lazy.ensure_deep_fundamentals", _fail)
    out = _hydrate_raw_fundamentals(
        "EIMCOELECO",
        {
            "available": True,
            "data": {"quarterly_results": [{"": "Sales+", "Jun 2026": 78}]},
            "section_as_of": {},
        },
    )
    assert out["section_as_of"]["financial_history"] == "2026-06-01"
