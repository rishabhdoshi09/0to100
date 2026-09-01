from __future__ import annotations

from product.research_status import build_research_status
from product.system_health_contract import build_system_health_contract


def test_research_status_never_claims_ai_is_learning():
    payload = build_research_status(
        paper_learning={"available": True, "closed_trades": 0, "self_feed": {}, "summary": ""},
        decision_journal={
            "performance": {"sample_size": 0, "sufficient_sample": False, "sample_note": "No settled outcomes yet."},
            "entries": [],
            "counts": {"surfaced_history": 0, "latest_scan_decisions": 0},
        },
        autonomy={"learning_status": "WAITING_FOR_FRESH_EOD_DATA"},
    )
    blob = " ".join(payload["headlines"]).lower()
    assert "ai is learning" not in blob
    assert payload["production"]["ensemble"]["backtest_parity"] == "UNVERIFIED"
    assert payload["learning_status"] == "WAITING_FOR_FRESH_EOD_DATA"
    assert payload["paper"]["live_locked"] is True


def test_research_status_reports_insufficient_sample():
    payload = build_research_status(
        paper_learning={"available": True, "closed_trades": 12, "self_feed": {"taken": [], "skipped": []}},
        decision_journal={
            "performance": {
                "sample_size": 12,
                "hit_rate_pct": 50.0,
                "expectancy_pct": 0.1,
                "sufficient_sample": False,
            },
            "entries": [],
            "counts": {},
        },
        autonomy={"learning_status": "ACTIVE"},
    )
    assert any("confidence insufficient" in line.lower() for line in payload["headlines"])


def test_health_contract_has_no_collapsed_green_light():
    payload = build_system_health_contract(
        scan={"available": True, "scanned_at": "2026-09-01", "universe_size": 10, "coverage": {"requested": 12, "checked": 10, "qualified": 2}},
        data={"ready": True, "bhavcopy": {"ready": True, "sessions": 400, "symbols": 2000, "latest_date": "2026-08-29"}},
        news={"available": False, "stats": {"total": 0}, "latest_refresh": {}},
        operations={"running": True, "worker_pid": 1, "heartbeat": "now"},
        autonomy={"running": True, "state": "RUNNING", "plain_state": "ok", "learning_status": "WAITING_FOR_FRESH_EOD_DATA"},
        recommendations_available=True,
        market_report_as_of="",
        product_wired=True,
    )
    assert payload["collapsed_status"] is None
    by_key = {lane["key"]: lane for lane in payload["lanes"]}
    assert by_key["news_freshness"]["status"] == "MISSING"
    assert by_key["operations_worker"]["status"] == "HEALTHY"
    assert by_key["market_report_freshness"]["status"] == "MISSING"
    assert by_key["backtest_registry"]["status"] == "WAITING"
    assert "paper_execution" in by_key
    assert "autonomy_scheduler" in by_key
    assert "selection_authority" in by_key
    assert payload["collapsed_status"] is None
    assert payload["why_no_trade"]["available"] is False
    assert by_key["paper_execution"]["status"] != "HEALTHY"
    assert payload["counts"]["HEALTHY"] >= 1
    assert payload["counts"]["MISSING"] >= 1
