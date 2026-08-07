from datetime import datetime, timezone

from product.product_readiness import build_product_readiness


def test_readiness_is_partial_when_core_data_exists_but_context_missing():
    now = datetime(2026, 8, 1, 5, 0, tzinfo=timezone.utc)
    payload = build_product_readiness(
        market={"available": True, "summary": "Healthy breadth"},
        scan={"available": True, "scanned_at": "2026-08-01T03:00:00+00:00", "universe_size": 1800, "records": [{"symbol": "AAA"}]},
        long_term={"available": True, "scanned_at": "2026-07-31T03:00:00+00:00", "summary": {"coverage_pct": 60}, "records": [{"symbol": "AAA"}]},
        news={"available": False, "articles": [], "stats": {}},
        fno={"available": False, "mapped_underlyings": 0},
        data={"bhavcopy": {"ready": True, "sessions": 500, "symbols": 1800, "latest_date": "2026-07-31"}},
        operations={"running": True, "heartbeat": "2026-08-01T04:59:55+00:00", "active": []},
        now=now,
    )
    assert payload["state"] == "PARTIAL"
    assert 55 <= payload["score"] < 90
    assert any(item["key"] == "news" and item["status"] == "MISSING" for item in payload["lanes"])


def test_stale_scan_does_not_receive_full_credit():
    now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    payload = build_product_readiness(
        market={},
        scan={"available": True, "scanned_at": "2026-07-30T00:00:00+00:00", "records": [{"symbol": "AAA"}]},
        long_term={},
        news={},
        fno={},
        data={},
        operations={},
        now=now,
    )
    scan = next(item for item in payload["lanes"] if item["key"] == "scanner")
    assert scan["status"] == "STALE"
    assert scan["earned_weight"] < scan["weight"]
