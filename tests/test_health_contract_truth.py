"""Health lanes must state the business capability, not the presence of a file.

Audited defects:

  * ``news_freshness`` read HEALTHY off ``news["available"]``, a flag that is
    true as soon as the store and its source-health rows exist. The lane said
    HEALTHY with "0 articles on file" while the durable NEWS_REFRESH operation
    in the same payload was BLOCKED.
  * ``recommendations_freshness`` read HEALTHY off the projection file existing,
    with an empty ``as_of`` and no scan behind it, contradicting the
    ``scan_freshness`` lane beside it.
  * Lane timestamps mixed epoch floats, UTC ISO and naive Asia/Kolkata strings
    in one payload, so freshness arithmetic compared different clocks.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from product.system_health_contract import (
    IST,
    LANE_STATES,
    as_of_utc,
    build_system_health_contract,
)


def _lanes(**kwargs) -> dict[str, dict]:
    contract = build_system_health_contract(**kwargs)
    return {str(lane["key"]): lane for lane in contract["lanes"]}


def _news(*, status: str, articles: int, code: str = "", finished_at=1789143617.98) -> dict:
    return {
        "available": True,  # the store is reachable — deliberately not evidence
        "stats": {"total": articles},
        "latest_refresh": {
            "status": status,
            "error_code": code,
            "finished_at": finished_at,
        },
    }


# --------------------------------------------------------------------------
# News lane
# --------------------------------------------------------------------------

def test_news_blocked_acquisition_is_not_healthy():
    lane = _lanes(news=_news(status="BLOCKED", articles=0, code="NEWS_SOURCES_UNAVAILABLE"))
    assert lane["news_freshness"]["status"] == "BLOCKED"
    assert "NEWS_SOURCES_UNAVAILABLE" in lane["news_freshness"]["detail"]


def test_news_failed_acquisition_is_not_healthy():
    assert _lanes(news=_news(status="FAILED", articles=0, code="BOOM"))["news_freshness"]["status"] == "FAILED"


def test_news_with_zero_articles_is_never_healthy():
    """Even a SUCCEEDED refresh that delivered nothing is not a healthy lane."""
    assert _lanes(news=_news(status="SUCCEEDED", articles=0))["news_freshness"]["status"] == "MISSING"


def test_news_running_refresh_with_nothing_on_file_is_waiting():
    assert _lanes(news=_news(status="RUNNING", articles=0))["news_freshness"]["status"] == "WAITING"


def test_news_is_healthy_only_with_a_successful_refresh_and_articles():
    lane = _lanes(news=_news(status="SUCCEEDED", articles=42))["news_freshness"]
    assert lane["status"] == "HEALTHY"
    assert "42 articles" in lane["detail"]


def test_news_store_reachable_flag_alone_proves_nothing():
    """The exact shape of the audited bug: available=True, no outcome, no rows."""
    lane = _lanes(news={"available": True, "stats": {"total": 0}, "latest_refresh": {}})
    assert lane["news_freshness"]["status"] != "HEALTHY"


# --------------------------------------------------------------------------
# Recommendations lane
# --------------------------------------------------------------------------

def test_recommendations_without_a_scan_are_missing():
    lane = _lanes(scan={}, recommendations_available=True)["recommendations_freshness"]
    assert lane["status"] == "MISSING"
    assert lane["as_of"] == ""


def test_recommendations_healthy_requires_a_scan_timestamp():
    lane = _lanes(
        scan={"scanned_at": "2026-09-11T03:43:43+00:00", "available": True},
        recommendations_available=True,
    )["recommendations_freshness"]
    assert lane["status"] == "HEALTHY"
    assert lane["as_of"].endswith("+00:00")


def test_recommendations_wait_while_the_projection_rebuilds():
    lane = _lanes(
        scan={"scanned_at": "2026-09-11T03:43:43+00:00", "available": True},
        recommendations_available=False,
    )["recommendations_freshness"]
    assert lane["status"] == "WAITING"


def test_scan_and_recommendation_lanes_never_contradict():
    """If there is no scan, nothing downstream of the scan may claim health."""
    lanes = _lanes(scan={}, recommendations_available=True)
    assert lanes["scan_freshness"]["status"] == "MISSING"
    assert lanes["recommendations_freshness"]["status"] == "MISSING"
    assert lanes["recommendations"]["status"] == "MISSING"


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    ["1789143617.9807115", 1789143617.98, "2026-09-11T16:21:47+00:00", "2026-09-11T16:21:47Z"],
)
def test_every_lane_timestamp_shape_normalises_to_utc(value):
    out = as_of_utc(value)
    assert out.endswith("+00:00")
    assert datetime.fromisoformat(out).tzinfo == timezone.utc


def test_naive_ist_heartbeat_is_not_read_as_utc():
    """A naive IST stamp read as UTC lands 5h30m in the future."""
    ist_text = "2026-09-11T21:51:40.914872"
    as_ist = datetime.fromisoformat(as_of_utc(ist_text, naive_tz=IST))
    as_utc = datetime.fromisoformat(as_of_utc(ist_text))
    assert (as_utc - as_ist).total_seconds() == 5.5 * 3600
    assert as_ist.hour == 16 and as_ist.minute == 21


def test_auth_lane_heartbeat_is_converted_from_ist():
    lane = _lanes(
        autonomy={
            "running": True,
            "state": "OBSERVING",
            "heartbeat_ist": "2026-09-11T21:51:40.914872",
        }
    )["zerodha_auth"]
    assert lane["as_of"].startswith("2026-09-11T16:21:40")
    assert lane["as_of"].endswith("+00:00")


def test_unparseable_timestamp_is_surfaced_not_invented():
    assert as_of_utc("not-a-time") == "not-a-time"
    assert as_of_utc("") == ""
    assert as_of_utc(None) == ""


def test_all_lane_timestamps_in_one_payload_share_one_clock():
    contract = build_system_health_contract(
        scan={"scanned_at": "2026-09-11T03:43:43+00:00", "available": True},
        news=_news(status="SUCCEEDED", articles=5),
        operations={"running": True, "worker_pid": 4242, "heartbeat": "2026-09-11T16:21:47+0000"},
        autonomy={"running": True, "state": "OBSERVING", "heartbeat_ist": "2026-09-11T21:51:40.914872"},
        recommendations_available=True,
    )
    stamps = [str(lane["as_of"]) for lane in contract["lanes"] if lane["as_of"]]
    assert stamps
    for stamp in stamps:
        moment = datetime.fromisoformat(stamp)
        assert moment.tzinfo is not None, stamp
        assert moment.utcoffset().total_seconds() == 0, stamp


# --------------------------------------------------------------------------
# State vocabulary
# --------------------------------------------------------------------------

def test_blocked_and_failed_are_distinct_from_missing():
    assert {"BLOCKED", "FAILED", "MISSING"} <= LANE_STATES


def test_counts_cover_every_emitted_state():
    contract = build_system_health_contract(news=_news(status="BLOCKED", articles=0))
    emitted = {str(lane["status"]) for lane in contract["lanes"]}
    assert emitted <= set(contract["counts"])
    assert sum(contract["counts"].values()) == len(contract["lanes"])
