"""The autonomy desk must show the broker session as of now (invariant 2).

Owner's Mac 2026-11-18: the Zerodha token had expired, yet /api/autonomy-desk
showed "SESSION_VALID" because desk_projection read the broker block from the
persisted 16 Nov loop summary.
"""
from __future__ import annotations

from product import autonomous_loop as AL
from product import readiness as RDY

_STALE_SUMMARY = {
    "session_date": "2026-11-16",
    "readiness": {"broker": {"state": "READY", "ready": True, "auth_status": "SESSION_VALID",
                             "login_required": False}},
}


def _patch(monkeypatch, broker_fn):
    monkeypatch.setattr(AL, "load_summary", lambda: dict(_STALE_SUMMARY))
    monkeypatch.setattr(AL, "project_events", lambda limit=24: [])
    monkeypatch.setattr(AL.CL, "state_counts", lambda session: {})
    monkeypatch.setattr(RDY, "broker_status", broker_fn)


def test_expired_session_is_shown_even_when_the_last_summary_said_valid(monkeypatch):
    live = {"state": "LOGIN_REQUIRED", "ready": False, "auth_status": "SESSION_EXPIRED",
            "login_required": True}
    _patch(monkeypatch, lambda: dict(live))
    out = AL.desk_projection()
    assert out["what_is_blocked"]["broker"]["auth_status"] == "SESSION_EXPIRED"
    assert out["what_is_blocked"]["broker"]["login_required"] is True
    assert out["what_needs_the_operator"]["state"] == "LOGIN_REQUIRED"


def test_a_failing_live_check_never_presents_the_old_state_as_ready(monkeypatch):
    def boom():
        raise RuntimeError("probe exploded")
    _patch(monkeypatch, boom)
    broker = AL.desk_projection()["what_is_blocked"]["broker"]
    assert broker["stale"] is True
    assert broker["ready"] is False and broker["execution_ready"] is False
    assert "RuntimeError" in broker["detail"]
