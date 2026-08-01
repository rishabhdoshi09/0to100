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
