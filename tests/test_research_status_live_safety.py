from __future__ import annotations

from unittest.mock import patch

from product.research_status import build_research_status


def _build() -> dict:
    return build_research_status(
        paper_learning={"available": True, "closed_trades": 0, "self_feed": {}},
        decision_journal={"performance": {}, "entries": []},
        autonomy={},
    )


def test_research_status_never_turns_unverified_interlock_into_locked() -> None:
    safety = {
        "live_locked": None,
        "live_lock_verified": False,
        "live_execution_authorized": None,
        "live_lock_status": "UNVERIFIED",
        "live_lock_reason": "verification failed",
        "live_lock_source": "product.live_execution_interlock",
    }
    with patch("product.research_status.live_safety_projection", return_value=safety):
        payload = _build()

    paper = payload["paper"]
    assert paper["live_locked"] is None
    assert paper["live_lock_verified"] is False
    assert paper["live_execution_authorized"] is None
    assert any("UNVERIFIED" in line for line in payload["headlines"])
    assert not any("Live orders stay locked" in line for line in payload["headlines"])


def test_research_status_reports_locked_only_from_verified_canonical_state() -> None:
    safety = {
        "live_locked": True,
        "live_lock_verified": True,
        "live_execution_authorized": False,
        "live_lock_status": "LOCKED",
        "live_lock_reason": "paper-only deployment",
        "live_lock_source": "product.live_execution_interlock",
    }
    with patch("product.research_status.live_safety_projection", return_value=safety):
        payload = _build()

    paper = payload["paper"]
    assert paper["live_locked"] is True
    assert paper["live_lock_verified"] is True
    assert paper["live_execution_authorized"] is False
    assert any("VERIFIED / LOCKED" in line for line in payload["headlines"])
