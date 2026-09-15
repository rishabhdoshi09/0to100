"""Paper-cycle artifacts must never fabricate a positive live-money lock."""
from __future__ import annotations

from datetime import datetime, timezone

from product import live_safety
from product.paper_autopilot import run_reco_paper_cycle


def test_unverified_interlock_never_becomes_live_locked_true(monkeypatch):
    monkeypatch.setattr(
        live_safety,
        "live_safety_projection",
        lambda: {
            "live_locked": None,
            "live_lock_verified": False,
            "live_execution_authorized": None,
            "live_lock_status": "UNVERIFIED",
            "live_lock_reason": "test interlock unavailable",
            "live_lock_source": "product.live_execution_interlock",
        },
    )

    cycle = run_reco_paper_cycle(
        book=None,
        cards=[],
        as_of="2026-09-13",
        now=datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc),
        persist_journal=False,
    )

    assert cycle["live_locked"] is None
    assert cycle["live_lock_verified"] is False
    assert cycle["live_execution_authorized"] is None
    assert cycle["live_lock_status"] == "UNVERIFIED"
    assert cycle["live_lock_reason"] == "test interlock unavailable"
    assert cycle["live_lock_source"] == "product.live_execution_interlock"


def test_verified_locked_interlock_is_projected_without_reinterpretation(monkeypatch):
    monkeypatch.setattr(
        live_safety,
        "live_safety_projection",
        lambda: {
            "live_locked": True,
            "live_lock_verified": True,
            "live_execution_authorized": False,
            "live_lock_status": "LOCKED",
            "live_lock_reason": "deployment gate closed",
            "live_lock_source": "product.live_execution_interlock",
        },
    )

    cycle = run_reco_paper_cycle(
        book=None,
        cards=[],
        as_of="2026-09-13",
        now=datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc),
        persist_journal=False,
    )

    assert cycle["live_locked"] is True
    assert cycle["live_lock_verified"] is True
    assert cycle["live_execution_authorized"] is False
    assert cycle["live_lock_status"] == "LOCKED"
