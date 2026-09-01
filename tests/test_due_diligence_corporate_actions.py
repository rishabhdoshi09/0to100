from __future__ import annotations

import product.corporate_actions as ca
from product.due_diligence import _attach_corporate_actions


def test_corporate_actions_project_to_overview_and_news(monkeypatch):
    monkeypatch.setattr(ca, "get_corporate_actions", lambda symbol: {
        "available": True,
        "delivery_state": "FRESH_OFFICIAL",
        "source": "NSE India",
        "source_tier": "official_exchange",
        "retrieved_at": "2026-09-01T10:00:00+00:00",
        "actions": [{
            "symbol": symbol,
            "action_type": "DIVIDEND",
            "subject": "Dividend - Rs 10 per share",
            "ex_date": "10-Sep-2026",
            "record_date": "10-Sep-2026",
            "announcement_date": "01-Sep-2026",
            "source": "NSE India",
            "source_tier": "official_exchange",
            "source_url": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
        }],
    })
    report = {
        "symbol": "TCS",
        "events": [],
        "sources": [],
        "first_screen": {"recent_material_events": []},
    }
    out = _attach_corporate_actions(report)
    assert out["corporate_actions"]["delivery_state"] == "FRESH_OFFICIAL"
    assert out["events"][0]["category"] == "corporate_action"
    assert out["events"][0]["official"] is True
    assert out["first_screen"]["recent_material_events"][0]["headline"] == "Dividend - Rs 10 per share"
    assert out["sources"][0]["source"] == "NSE India"
