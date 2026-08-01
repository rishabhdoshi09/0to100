from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import reporting.evidence_intake as EI


def test_template_and_upload_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(EI, "EVIDENCE_ROOT", tmp_path / "evidence")
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
    rows = EI.structured_rows("TEST", "shareholding_history")
    assert rows[0]["fii_pct"] == "7"
    assert EI.upload_path("TEST", item["evidence_id"]).exists()


def test_requirements_show_dates_links_and_uploaded_status(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(EI, "EVIDENCE_ROOT", tmp_path / "evidence")
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
    assert missing["status"] == "MISSING"
    assert missing["template_available"] is True


def test_raw_fundamentals_reads_full_cached_sections(tmp_path: Path, monkeypatch):
    db = tmp_path / "fundamentals.db"
    connection = sqlite3.connect(str(db))
    try:
        connection.execute("CREATE TABLE fundamentals_cache (symbol TEXT PRIMARY KEY, data_json TEXT NOT NULL, fetched_at REAL NOT NULL)")
        payload = {
            "about": "A real company description",
            "profit_loss": [{"": "Sales", "Mar 2026": 100}],
            "shareholding": [{"": "FIIs", "Mar 2026": 7}],
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
