from __future__ import annotations

import product.home_os as home_os
from product.backend_control_plane import build_check_system, _paper_lane


def _live_row(snapshot):
    return next(row for row in snapshot["lanes"] if row["id"] == "live_money")


def test_check_system_never_calls_unverified_live_money_locked():
    snapshot = build_check_system(
        {},
        live_locked=None,
        live_lock_verified=False,
        live_lock_reason="route probe failed",
    )

    row = _live_row(snapshot)
    assert snapshot["live_locked"] is None
    assert snapshot["live_lock_verified"] is False
    assert row["status"] == "Unverified"
    assert row["live_locked"] is None
    assert row["live_lock_verified"] is False
    assert "route probe failed" in row["detail"]


def test_check_system_calls_live_money_locked_only_with_verified_lock():
    snapshot = build_check_system(
        {},
        live_locked=True,
        live_lock_verified=True,
        live_lock_reason="all execution routes verified",
    )

    row = _live_row(snapshot)
    assert snapshot["live_locked"] is True
    assert snapshot["live_lock_verified"] is True
    assert row["status"] == "Locked"
    assert row["live_locked"] is True
    assert row["live_lock_verified"] is True


def test_paper_lane_marks_unknown_broker_boundary_as_problem():
    lane = _paper_lane(
        auto={},
        paper_d={"enabled": True},
        why_d={},
        paper_enabled=True,
        live_locked=None,
        live_lock_verified=False,
        taken=[],
        opens=[],
        closed=[],
        valid_no_trade=False,
        cycle_reasons=[],
        last_decision="",
        why_plain="",
        next_line="",
    )

    assert lane["status"] == "Problem"
    assert lane["status_code"] == "LIVE_LOCK_UNVERIFIED"
    assert lane["live_lock_verified"] is False
    assert lane.get("live_locked") is not True
    assert lane["technical"]["live_lock_verified"] is False
    assert lane["technical"].get("live_locked") is not True


def test_home_safety_projection_discards_positive_lock_without_verification(monkeypatch):
    import product.live_readiness as readiness

    monkeypatch.setattr(
        readiness,
        "evaluate_live_readiness",
        lambda: {
            "live_locked": True,
            "live_lock_verified": False,
            "live_execution_authorized": False,
            "live_lock_status": "UNVERIFIED",
            "live_lock_reason": "verification did not complete",
            "live_lock_source": "product.live_execution_interlock",
        },
    )

    projected = home_os._live_safety_projection()

    assert projected["live_locked"] is None
    assert projected["live_lock_verified"] is False
    assert projected["live_execution_authorized"] is None
    assert projected["live_lock_status"] == "UNVERIFIED"


def test_radar_home_fallback_reports_unverified_not_locked(monkeypatch):
    import product.observer_api as observer
    import product.home_os as home_module

    monkeypatch.setattr(observer.core, "_scan_payload", lambda: {})
    monkeypatch.setattr(observer.core, "_market_payload", lambda: {})
    monkeypatch.setattr(observer.core, "_long_term_payload", lambda: {})
    monkeypatch.setattr(
        home_module,
        "build_home_os",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("home projection failed")),
    )

    result = observer.radar_home_workspace()
    fallback = result["home_os"]

    assert fallback["state"] == "PROBLEM"
    assert fallback["live_locked"] is None
    assert fallback["live_lock_verified"] is False
    assert fallback["live_execution_authorized"] is None
    assert fallback["live_lock_status"] == "UNVERIFIED"
