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


def _raw_fundamentals():
    return {
        "available": True,
        "fetched_at": "2026-08-01T00:00:00+00:00",
        "freshness": "FRESH",
        "data": {
            "about": "Test Ltd manufactures industrial control systems.",
            "quarterly_results": [
                {"": "Sales", "Jun 2025": 100, "Sep 2025": 110, "Dec 2025": 125, "Mar 2026": 140},
                {"": "Net Profit", "Jun 2025": 10, "Sep 2025": 12, "Dec 2025": 14, "Mar 2026": 17},
            ],
            "profit_loss": [
                {"": "Sales", "Mar 2023": 300, "Mar 2024": 360, "Mar 2025": 430, "Mar 2026": 510},
            ],
            "shareholding": [
                {"": "FIIs", "Jun 2025": 5.0, "Sep 2025": 5.5, "Dec 2025": 6.0, "Mar 2026": 6.8},
                {"": "DIIs", "Jun 2025": 8.0, "Sep 2025": 8.2, "Dec 2025": 8.4, "Mar 2026": 8.9},
            ],
        },
    }


def test_dossier_exposes_strict_coverage_and_shareholding():
    dossier = build_equity_dossier(
        "TEST",
        scan_payload={"schema_version": 1, "scanned_at": "2026-08-01T00:00:00+00:00", "records": [{"symbol": "TEST", "company": "Test Ltd", "status": "Ready to trade", "reasons": ["Strong trend"], "score": 80}]},
        long_term_payload=_long_term(),
        raw_fundamentals=_raw_fundamentals(),
        frame=_frame(),
        market={"health": "Healthy", "trade_stance": "Selective risk-on"},
        news=[],
        fno_payload={"generated_at": "2026-08-01T00:00:00+00:00", "underlyings": []},
        generated_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )

    assert dossier["company"] == "Test Ltd"
    assert 0 <= dossier["coverage_pct"] <= 100
    assert len(dossier["section_coverage"]) >= 8
    assert dossier["company_about"].startswith("Test Ltd")
    assert dossier["fundamentals"]["fii_holding"] == 6.8
    assert dossier["fundamentals"]["dii_holding"] == 8.9
    assert dossier["price"]["return_12m_pct"] is not None
    assert dossier["evidence_requirements"]["symbol"] == "TEST"


def test_equity_pdf_is_generated(tmp_path: Path):
    dossier = build_equity_dossier(
        "TEST",
        scan_payload={"scanned_at": "2026-08-01T00:00:00+00:00", "records": [{"symbol": "TEST", "company": "Test Ltd", "reasons": ["Strong trend"]}]},
        long_term_payload=_long_term(),
        raw_fundamentals=_raw_fundamentals(),
        frame=_frame(),
        market={"health": "Healthy", "trade_stance": "Selective"},
        news=[{"headline": "Test wins order", "published_at": "2026-07-31T00:00:00+00:00", "fetched_at": "2026-07-31T01:00:00+00:00", "source": "Exchange", "official": True, "event_type": "order_or_contract", "impact_score": 80, "why_it_matters": "Order improves revenue visibility."}],
        fno_payload={"generated_at": "2026-08-01T00:00:00+00:00", "underlyings": []},
    )
    path = render_equity_pdf(dossier, tmp_path / "test.pdf")
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 10_000


def test_basket_pdf_is_generated(tmp_path: Path, monkeypatch):
    import reporting.research_dossier as rd

    monkeypatch.setattr(rd, "build_equity_dossier", lambda symbol, **_kwargs: build_equity_dossier(
        symbol,
        scan_payload={"scanned_at": "2026-08-01T00:00:00+00:00", "records": [{"symbol": symbol, "company": f"{symbol} Ltd", "reasons": ["Strong trend"]}]},
        long_term_payload={"scanned_at": "2026-08-01T00:00:00+00:00", "records": [{**_long_term()["records"][0], "symbol": symbol}]},
        raw_fundamentals=_raw_fundamentals(),
        frame=_frame(),
        market={"health": "Healthy", "trade_stance": "Selective"},
        news=[],
        fno_payload={"generated_at": "2026-08-01T00:00:00+00:00", "underlyings": []},
    ))
    payload = {"records": [{"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "CCC"}]}
    basket = build_long_term_basket(limit=3, long_term_payload=payload)
    path = render_basket_pdf(basket, tmp_path / "basket.pdf")
    assert path.read_bytes().startswith(b"%PDF")
    assert len(path.read_bytes()) > 10_000
