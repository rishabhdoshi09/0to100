from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reporting.pdf_renderer import render_basket_pdf, render_equity_pdf
from reporting.research_dossier import build_equity_dossier, build_long_term_basket


def _frame():
    dates = pd.date_range("2025-01-01", periods=280, freq="B")
    close = pd.Series([100 + index * 0.2 for index in range(len(dates))], index=dates)
    return pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1_000_000,
    })


def _long_term():
    return {
        "schema_version": 1,
        "scanned_at": "2026-08-01T00:00:00+00:00",
        "records": [{
            "symbol": "TEST",
            "sector": "Industrials",
            "classification": "QUALITY_COMPOUNDER",
            "technical_score": 72,
            "fundamental_score": 78,
            "combined_score": 75.3,
            "fundamental_coverage": 0.82,
            "timing": "TECHNICALLY_FAVORABLE",
            "quality_factors": ["ROCE 22.0%", "3y profit CAGR 18.0%"],
            "risk_flags": ["Rich valuation P/E 52.0"],
            "fundamentals": {
                "pe": 52,
                "roe": 20,
                "roce": 22,
                "sales_growth_3y": 16,
                "profit_growth_3y": 18,
                "debt_to_equity": 0.2,
                "promoter_holding": 58,
            },
        }],
    }


def test_dossier_exposes_missing_institutional_history():
    dossier = build_equity_dossier(
        "TEST",
        scan_payload={"schema_version": 1, "scanned_at": "x", "records": [{"symbol": "TEST", "company": "Test Ltd", "status": "Ready to trade", "reasons": ["Strong trend"], "score": 80}]},
        long_term_payload=_long_term(),
        frame=_frame(),
        market={"health": "Healthy", "trade_stance": "Selective risk-on"},
        news=[],
        fno_payload={"underlyings": []},
        generated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )

    assert dossier["company"] == "Test Ltd"
    assert dossier["coverage_pct"] == 80
    assert any("FII and DII" in item for item in dossier["open_items"])
    assert dossier["price"]["return_12m_pct"] is not None


def test_equity_pdf_is_generated(tmp_path: Path):
    dossier = build_equity_dossier(
        "TEST",
        scan_payload={"records": [{"symbol": "TEST", "company": "Test Ltd", "reasons": ["Strong trend"]}]},
        long_term_payload=_long_term(),
        frame=_frame(),
        market={"health": "Healthy", "trade_stance": "Selective"},
        news=[{"headline": "Test wins order", "published_at": "2026-07-31T00:00:00+00:00", "fetched_at": "2026-07-31T01:00:00+00:00", "source": "Exchange", "official": True, "event_type": "order_or_contract", "impact_score": 80, "why_it_matters": "Order improves revenue visibility."}],
        fno_payload={"underlyings": []},
    )
    path = render_equity_pdf(dossier, tmp_path / "test.pdf")
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 10_000


def test_basket_pdf_is_generated(tmp_path: Path, monkeypatch):
    import reporting.research_dossier as rd

    monkeypatch.setattr(rd, "build_equity_dossier", lambda symbol, **_kwargs: build_equity_dossier(
        symbol,
        scan_payload={"records": [{"symbol": symbol, "company": f"{symbol} Ltd", "reasons": ["Strong trend"]}]},
        long_term_payload={"records": [{**_long_term()["records"][0], "symbol": symbol}]},
        frame=_frame(),
        market={"health": "Healthy", "trade_stance": "Selective"},
        news=[],
        fno_payload={"underlyings": []},
    ))
    payload = {"records": [{"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "CCC"}]}
    basket = build_long_term_basket(limit=3, long_term_payload=payload)
    path = render_basket_pdf(basket, tmp_path / "basket.pdf")
    assert path.read_bytes().startswith(b"%PDF")
    assert len(path.read_bytes()) > 10_000
