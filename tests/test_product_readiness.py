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


def _all_fresh(now):
    return dict(
        market={"available": True, "summary": "ok", "as_of": now.isoformat(), "updated_at": now.isoformat()},
        scan={"available": True, "scanned_at": now.isoformat(), "universe_size": 2000, "records": [{"symbol": "AAA"}]},
        long_term={"available": True, "scanned_at": now.isoformat(), "summary": {"coverage_pct": 90}, "records": [{"symbol": "AAA"}]},
        news={"available": True, "articles": [{"published_at": now.isoformat()}]},
        fno={"available": True, "mapped_underlyings": 200, "updated_at": now.isoformat(), "as_of": now.isoformat()},
        data={"bhavcopy": {"ready": True, "sessions": 500, "symbols": 2000, "latest_date": now.date().isoformat()}},
        operations={"running": True, "heartbeat": now.isoformat(), "active": []},
    )


def test_previous_session_scan_is_not_fresh_however_recent_its_wall_clock():
    """Owner's Mac 2026-11-18: READY/100 while health said the scan was a session behind."""
    now = datetime(2026, 11, 18, 17, 0, tzinfo=timezone.utc)
    payloads = _all_fresh(now)
    baseline = build_product_readiness(**payloads, now=now)
    behind = build_product_readiness(
        **payloads, now=now,
        scan_verdict=("STALE", "Scan session 2026-11-17 behind expected 2026-11-18"),
    )
    scan = next(item for item in behind["lanes"] if item["key"] == "scanner")
    assert scan["status"] == "STALE"
    assert "behind expected 2026-11-18" in scan["details"]
    assert behind["state"] != "READY"
    assert any("Whole-market scanner" in b for b in behind["blockers"])
    assert behind["score"] < baseline["score"]


def test_current_session_verdict_keeps_fresh_scan_fresh():
    now = datetime(2026, 11, 18, 17, 0, tzinfo=timezone.utc)
    out = build_product_readiness(**_all_fresh(now), now=now, scan_verdict=("READY", "553 records"))
    scan = next(item for item in out["lanes"] if item["key"] == "scanner")
    assert scan["status"] == "FRESH"


def test_ready_is_never_reported_alongside_a_blocker():
    now = datetime(2026, 11, 18, 17, 0, tzinfo=timezone.utc)
    out = build_product_readiness(**_all_fresh(now), now=now, scan_verdict=("STALE", "behind"))
    assert not (out["state"] == "READY" and out["blockers"])


def test_session_current_scan_is_fresh_before_the_open_despite_wall_clock_age():
    """Owner's Mac 2026-11-19 08:08 IST: health READY, readiness 'scanner STALE' (22:45 EOD scan)."""
    now = datetime(2026, 11, 19, 2, 40, tzinfo=timezone.utc)  # 08:10 IST
    payloads = _all_fresh(now)
    payloads["scan"] = {**payloads["scan"], "scanned_at": "2026-11-18T17:15:00+00:00"}  # 9.4 h old
    stale_by_clock = build_product_readiness(**payloads, now=now)
    scan = next(item for item in stale_by_clock["lanes"] if item["key"] == "scanner")
    assert scan["status"] == "STALE"
    current = build_product_readiness(**payloads, now=now, scan_verdict=("READY", "449 records · session 2026-11-18"))
    scan = next(item for item in current["lanes"] if item["key"] == "scanner")
    assert scan["status"] == "FRESH"
    assert scan["earned_weight"] == scan["weight"]
    assert not any("Whole-market scanner" in b for b in current["blockers"])


def test_session_verdict_never_revives_a_missing_scan():
    now = datetime(2026, 11, 19, 2, 40, tzinfo=timezone.utc)
    payloads = _all_fresh(now)
    payloads["scan"] = {"available": False, "records": []}
    out = build_product_readiness(**payloads, now=now, scan_verdict=("READY", "x"))
    scan = next(item for item in out["lanes"] if item["key"] == "scanner")
    assert scan["status"] == "MISSING"
