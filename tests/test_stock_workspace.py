import json

from datetime import datetime, timezone

import pandas as pd

from product.stock_workspace import build_stock_workspace


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
