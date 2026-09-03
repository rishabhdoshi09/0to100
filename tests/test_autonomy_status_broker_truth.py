from __future__ import annotations

from product import autonomy_status as AS


def _raw(state: str = "OBSERVING") -> dict:
    return {
        "state": state,
        "supervisor_running": True,
        "active_failures": [],
        "jobs": {},
        "recent_transitions": [],
        "recent_dialogue": [],
        "owner_state": {},
        "last_cycle": {},
    }


def test_autonomy_status_always_carries_canonical_broker_projection(monkeypatch) -> None:
    monkeypatch.setattr(AS.H, "read_status", lambda **_kwargs: _raw())
    monkeypatch.setattr(
        AS,
        "_broker_status",
        lambda: {
            "state": "LOGIN_REQUIRED",
            "ready": False,
            "live_data_ready": False,
            "execution_ready": False,
            "login_required": True,
        },
    )

    status = AS.read_autonomy_status()

    assert status["running"] is True
    assert status["state"] == "OBSERVING"
    assert status["broker"]["state"] == "LOGIN_REQUIRED"
    assert status["broker"]["live_data_ready"] is False


def test_legacy_auth_state_copy_does_not_claim_whole_system_waits_for_broker(monkeypatch) -> None:
    monkeypatch.setattr(AS.H, "read_status", lambda **_kwargs: _raw("AUTH_REQUIRED"))
    monkeypatch.setattr(AS, "_broker_status", lambda: {"state": "LOGIN_REQUIRED", "ready": False})

    status = AS.read_autonomy_status()

    assert "non-broker autonomy can continue" in status["plain_state"]
