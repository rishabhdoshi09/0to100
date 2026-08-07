"""Auto-fetch evidence from Screener cache / official links."""
from __future__ import annotations

from pathlib import Path

import reporting.evidence_autofetch as AF
import reporting.evidence_intake as EI


def _sandbox(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(EI, "ROOT", tmp_path)
    monkeypatch.setattr(EI, "EVIDENCE_ROOT", tmp_path / "evidence")
    monkeypatch.setattr(AF, "save_upload", EI.save_upload)
    monkeypatch.setattr(AF, "evidence_requirements", EI.evidence_requirements)
    monkeypatch.setattr(AF, "load_raw_fundamentals", lambda symbol, auto_fetch=False: {"data": {}})


def test_screener_financial_export_attaches(tmp_path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)
    raw = {
        "profit_loss": [
            {"row_label": "Sales", "Mar 2023": 100, "Mar 2024": 125, "Mar 2025": 160},
            {"row_label": "Net Profit", "Mar 2023": 10, "Mar 2024": 15, "Mar 2025": 22},
            {"row_label": "Operating Profit", "Mar 2023": 18, "Mar 2024": 25, "Mar 2025": 36},
        ],
        "about": "Forgings maker for auto OEMs.",
        "shareholding": [
            {"row_label": "Promoters", "Mar 2024": 55, "Mar 2025": 54},
            {"row_label": "FIIs", "Mar 2024": 8, "Mar 2025": 9},
            {"row_label": "DIIs", "Mar 2024": 10, "Mar 2025": 11},
            {"row_label": "Public", "Mar 2024": 27, "Mar 2025": 26},
        ],
    }

    fin = AF._autofetch_from_screener("RKFORGE", "financial_history", raw)
    assert fin and fin["ok"] is True
    assert fin["method"] == "screener_cache_export"
    rows = EI.structured_rows("RKFORGE", "financial_history")
    assert rows
    assert any(float(r.get("revenue_cr") or 0) == 160 for r in rows)

    profile = AF._autofetch_from_screener("RKFORGE", "business_profile", raw)
    assert profile and profile["ok"] is True

    holding = AF._autofetch_from_screener("RKFORGE", "shareholding_history", raw)
    assert holding and holding["ok"] is True


def test_autofetch_skips_google_and_reports_failures(tmp_path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)
    monkeypatch.setattr(
        AF,
        "resource_links",
        lambda symbol: {
            "management_commentary": [
                {
                    "label": "Google search",
                    "url": "https://www.google.com/search?q=RKFORGE+concall",
                    "official": "false",
                }
            ],
            "order_book_guidance": [],
            "annual_report": [],
            "financial_history": [],
            "business_profile": [],
            "shareholding_history": [],
        },
    )
    report = AF.autofetch_evidence(
        "RKFORGE",
        kinds=["management_commentary"],
        refresh_screener=False,
    )
    assert report["accepted"] is True
    assert report["attached_count"] == 0
    assert report["failed_count"] == 1
    assert "never invented" in report["honesty"].lower() or "never invent" in report["honesty"].lower()
    attempts = report["results"][0].get("attempts") or []
    assert attempts
    assert attempts[0].get("ok") is False
    assert "skipped" in str(attempts[0].get("error") or "").lower()


def test_direct_pdf_attach(tmp_path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)

    class FakeResp:
        def __init__(self, content, url, content_type):
            self.content = content
            self.url = url
            self.headers = {"Content-Type": content_type}

        def raise_for_status(self):
            return None

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, timeout=25, allow_redirects=True):
            return FakeResp(b"%PDF-1.4 fake", url, "application/pdf")

    monkeypatch.setattr(AF, "_session", lambda: FakeSession())
    monkeypatch.setattr(
        AF,
        "resource_links",
        lambda symbol: {
            "annual_report": [
                {
                    "label": "Company AR",
                    "url": "https://example.com/ar.pdf",
                    "official": "true",
                }
            ]
        },
    )
    report = AF.autofetch_evidence("TESTCO", kinds=["annual_report"], refresh_screener=False)
    assert report["attached_count"] == 1
    assert report["results"][0]["ok"] is True
    assert report["results"][0]["extraction_status"] == "SOURCE_ATTACHED_UNPARSED"


def test_autofetch_exports_via_public_entry(tmp_path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)
    raw = {
        "profit_loss": [
            {"row_label": "Sales", "Mar 2024": 100, "Mar 2025": 140},
            {"row_label": "Net Profit", "Mar 2024": 12, "Mar 2025": 18},
        ],
        "about": "Industrial systems company.",
        "shareholding": [
            {"row_label": "Promoters", "Mar 2025": 52},
            {"row_label": "FIIs", "Mar 2025": 8},
            {"row_label": "DIIs", "Mar 2025": 12},
            {"row_label": "Public", "Mar 2025": 28},
        ],
    }
    monkeypatch.setattr(
        AF,
        "load_raw_fundamentals",
        lambda symbol, auto_fetch=False: {"available": True, "data": raw},
    )
    monkeypatch.setattr(AF, "resource_links", lambda symbol: {k: [] for k in AF.DEFAULT_KINDS})
    report = AF.autofetch_evidence(
        "DEMO",
        kinds=["financial_history", "business_profile", "shareholding_history"],
        refresh_screener=False,
    )
    assert report["attached_count"] == 3
    assert report["failed_count"] == 0
