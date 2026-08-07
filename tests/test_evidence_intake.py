from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pytest

import reporting.evidence_intake as EI


def _sandbox(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(EI, "ROOT", tmp_path)
    monkeypatch.setattr(EI, "EVIDENCE_ROOT", tmp_path / "evidence")


def test_template_and_upload_round_trip(tmp_path: Path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)
    content = EI.template_csv("shareholding_history")
    assert b"quarter_end" in content
    item = EI.save_upload(
        "TEST",
        "shareholding_history",
        b"quarter_end,promoter_pct,fii_pct,dii_pct,public_pct,promoter_pledge_pct,source_url,as_of_date\n2026-06-30,55,7,9,29,0,https://example.com,2026-06-30\n",
        filename="shareholding.csv",
        as_of="2026-06-30",
        source_url="https://example.com",
    )
    assert item["structured"] is True
    assert item["extraction_status"] == "STRUCTURED_VALIDATED"
    rows = EI.structured_rows("TEST", "shareholding_history")
    fii = next(row for row in rows if row["row_label"] == "FIIs")
    assert fii["2026-06-30"] == "7"
    assert EI.upload_path("TEST", item["evidence_id"]).exists()


def test_requirements_show_dates_links_and_uploaded_status(tmp_path: Path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)
    monkeypatch.setattr(EI, "FUNDAMENTALS_DB", tmp_path / "missing.db")
    EI.save_upload(
        "TEST",
        "business_profile",
        b"as_of_date,business_summary,customers,demand_drivers,source_url\n2026-07-01,Industrial systems,Factories,Capex,https://example.com\n",
        filename="profile.csv",
        as_of="2026-07-01",
        source_url="https://example.com",
    )
    status = EI.evidence_requirements(
        "TEST",
        price_as_of="2026-08-01",
        scan_as_of="2026-08-01T00:00:00+00:00",
        long_term_as_of="2026-08-01T00:00:00+00:00",
    )
    profile = next(item for item in status["requirements"] if item["key"] == "business_profile")
    missing = next(item for item in status["requirements"] if item["key"] == "financial_history")
    assert profile["available"] is True
    assert profile["as_of"] == "2026-07-01"
    assert profile["links"]
    assert any("bseindia.com" in link["url"] for link in missing["links"])
    assert missing["status"] == "MISSING"
    assert missing["template_available"] is True


def test_unparsed_pdf_is_attached_but_not_analytical_data(tmp_path: Path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)
    monkeypatch.setattr(EI, "FUNDAMENTALS_DB", tmp_path / "missing.db")
    item = EI.save_upload(
        "TEST",
        "management_commentary",
        b"%PDF-1.4 source-only transcript",
        filename="transcript.pdf",
        as_of="2026-07-31",
        source_url="https://example.com/transcript.pdf",
    )
    assert item["extracted"] is False
    assert item["extraction_status"] == "SOURCE_ATTACHED_UNPARSED"
    status = EI.evidence_requirements("TEST")
    commentary = next(row for row in status["requirements"] if row["key"] == "management_commentary")
    assert commentary["source_attached"] is True
    assert commentary["available"] is False
    assert commentary["status"] == "SOURCE_ATTACHED_UNPARSED"
    assert EI.structured_rows("TEST", "management_commentary") == []


def test_malformed_structured_upload_is_rejected(tmp_path: Path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="missing required columns"):
        EI.save_upload(
            "TEST",
            "financial_history",
            b"period_end,revenue_cr\n2026-03-31,100\n",
            filename="financials.csv",
            as_of="2026-03-31",
            source_url="https://example.com/results",
        )
    with pytest.raises(ValueError, match="source_url"):
        EI.save_upload(
            "TEST",
            "annual_report",
            b"%PDF-1.4",
            filename="annual-report.pdf",
            as_of="2026-03-31",
            source_url="not-a-url",
        )


def test_raw_fundamentals_uses_disclosed_period_not_fetch_time(tmp_path: Path, monkeypatch):
    db = tmp_path / "fundamentals.db"
    connection = sqlite3.connect(str(db))
    try:
        connection.execute("CREATE TABLE fundamentals_cache (symbol TEXT PRIMARY KEY, data_json TEXT NOT NULL, fetched_at REAL NOT NULL)")
        payload = {
            "about": "A real company description",
            "quarterly_results": [{"": "Sales", "Jun 2025": 80, "Sep 2025": 90, "Dec 2025": 95, "Mar 2026": 100}],
            "profit_loss": [{"": "Sales", "Mar 2024": 70, "Mar 2025": 85, "Mar 2026": 100}],
            "shareholding": [{"": "FIIs", "Jun 2025": 5, "Sep 2025": 6, "Dec 2025": 6.5, "Mar 2026": 7}],
        }
        connection.execute(
            "INSERT INTO fundamentals_cache(symbol,data_json,fetched_at) VALUES(?,?,?)",
            ("TEST", json.dumps(payload), datetime(2026, 8, 1, tzinfo=timezone.utc).timestamp()),
        )
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(EI, "FUNDAMENTALS_DB", db)
    record = EI.load_raw_fundamentals("TEST")
    assert record["available"] is True
    assert record["data"]["about"] == "A real company description"
    assert record["fetched_at"].startswith("2026-08-01")
    assert record["section_as_of"]["financial_history"] == "2026-03-01"
    assert record["section_as_of"]["shareholding_history"] == "2026-03-01"
