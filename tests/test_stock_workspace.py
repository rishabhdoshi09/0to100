from datetime import datetime, timezone

import pandas as pd
import pytest

from product.stock_workspace import build_stock_workspace


@pytest.fixture(autouse=True)
def _isolate_case_db(tmp_path, monkeypatch):
    import product.case_memory as cm
    monkeypatch.setattr(cm, "CASES_DB", tmp_path / "cases.db")
    monkeypatch.setattr("product.stock_analyser.analyser_benchmark_frame", lambda: (None, "Nifty 50"))
    monkeypatch.setattr("product.monitor_context.nifty_frame", lambda: None)


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
    assert result["case"]["places_orders"] is False
    assert result["case"]["n_similar"] == 0
    assert "18" not in (result["case"].get("memory_line") or "")
    assert result["decision_memory"]["places_orders"] is False
    assert result["decision_memory"]["stance"] in {"YES", "NO", "WAIT"}
    assert result["analyser"]["available"] is True
    assert result["analyser"]["total"] == 7
    assert result["analyser"]["quote"]["close"]


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


def test_stock_workspace_attaches_acquired_option_chain(tmp_path, monkeypatch):
    monkeypatch.setattr("product.due_diligence.acquire.EVIDENCE_ROOT", tmp_path)
    from product.due_diligence.acquire import save_autonomy_facts

    save_autonomy_facts("TEST", {
        "acquired_at": "2026-01-28T11:00:00+00:00",
        "option_chain": {
            "available": True,
            "expiry": "28-Jan-2026",
            "pcr": 0.88,
            "max_pain": 100,
            "not_a_signal": True,
            "places_orders": False,
        },
    })
    result = build_stock_workspace(
        "TEST",
        scan_payload={},
        long_term_payload={},
        raw_fundamentals={},
        frame=[],
        news=[],
        fno_payload={"generated_at": "2026-01-28T06:00:00+00:00", "underlyings": [{"symbol": "TEST", "lot_size": 100, "future_symbol": "TEST26JANFUT"}]},
        now=datetime(2026, 1, 28, 12, 0, tzinfo=timezone.utc),
    )
    chain = result["fno"]["option_chain"]
    assert chain["available"] is True
    assert chain["pcr"] == 0.88
    assert chain["not_a_signal"] is True
    assert any(source["name"] == "Option-chain snapshot" for source in result["sources"])
    assert result["fno"]["lot_size"] == 100
