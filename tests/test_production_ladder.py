"""Production ladder — paper is the path, live is earned."""
from __future__ import annotations

from product.production_ladder import (
    PAPER_E4_N,
    SUBSYSTEMS,
    build_production_ladder,
    live_arm_allowed,
    live_blockers,
)


def test_monitor_subsystem_never_orders():
    monitor = next(s for s in SUBSYSTEMS if s["id"] == "monitor")
    execution = next(s for s in SUBSYSTEMS if s["id"] == "execution")
    assert monitor["may_order"] is False
    assert execution["may_order"] is True
    assert execution["mode"] == "LIVE"


def test_live_stays_locked_without_evidence():
    ok, why = live_arm_allowed(
        env={},
        paper_closed=0,
        alpha_level=0,
        institutional_live_allowed=False,
    )
    assert ok is False
    assert "LIVE locked" in why
    blocks = live_blockers(
        env={"QT_LIVE_ENABLED": "1"},
        paper_closed=12,
        alpha_level=2,
        institutional_live_allowed=False,
    )
    assert len(blocks) >= 2
    assert any("12/300" in b or str(PAPER_E4_N) in b for b in blocks)


def test_qt_flag_alone_does_not_unlock_live():
    ok, _ = live_arm_allowed(
        env={"QT_LIVE_ENABLED": "1"},
        paper_closed=PAPER_E4_N,
        alpha_level=4,
        institutional_live_allowed=False,
    )
    assert ok is False


def test_all_gates_unlock_live():
    ok, why = live_arm_allowed(
        env={"QT_LIVE_ENABLED": "1"},
        paper_closed=PAPER_E4_N,
        alpha_level=4,
        institutional_live_allowed=True,
    )
    assert ok is True
    assert "unlocked" in why.lower()


def test_default_rung_is_not_live():
    payload = build_production_ladder(
        data={"bhavcopy": {"ready": True, "sessions": 400}},
        scan={"available": True, "scanned_at": "2026-08-19"},
        paper={"available": True},
        autonomy={"available": True},
        env={},
        paper_closed=4,
    )
    assert payload["live_unlocked"] is False
    assert payload["rung"]["id"] in {"PAPER", "RESEARCH", "OBSERVE", "TRANSITION"}
    assert payload["rung"]["id"] != "LIVE"
    assert payload["subsystems"][-2]["id"] == "execution"
    assert payload["subsystems"][-2]["status"] == "LOCKED"
    assert any(edge["from"] == "ladder" for edge in payload["handshake"])


def test_thirty_paper_trades_is_transition_not_live():
    payload = build_production_ladder(
        data={"bhavcopy": {"ready": True, "sessions": 400}},
        scan={"available": True},
        paper={"available": True},
        env={"QT_LIVE_ENABLED": "1"},
        paper_closed=30,
    )
    assert payload["rung"]["id"] == "TRANSITION"
    assert payload["live_unlocked"] is False
    assert payload["subsystems"][-2]["status"] == "LOCKED"
