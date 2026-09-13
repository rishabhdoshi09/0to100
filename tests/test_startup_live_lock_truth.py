from __future__ import annotations

from types import SimpleNamespace

import product.live_execution_interlock as interlock
import product.startup_check as startup


def test_startup_unverified_interlock_never_projects_locked_true(monkeypatch):
    monkeypatch.setattr(
        interlock,
        "get_live_execution_state",
        lambda: SimpleNamespace(
            verified=False,
            locked=True,
            authorized=False,
            reason="verification incomplete",
            as_dict=lambda: {
                "verified": False,
                "locked": True,
                "authorized": False,
                "status": "UNVERIFIED",
                "reason": "verification incomplete",
                "source": "test",
            },
        ),
    )

    locked, verified, detail, payload = startup._live_lock_readiness()

    assert locked is None
    assert verified is False
    assert "unverified" in detail.lower()
    assert payload["verified"] is False


def test_startup_interlock_exception_is_unknown_not_locked(monkeypatch):
    def broken_probe():
        raise RuntimeError("route verification failed")

    monkeypatch.setattr(interlock, "get_live_execution_state", broken_probe)

    locked, verified, detail, payload = startup._live_lock_readiness()

    assert locked is None
    assert verified is False
    assert "route verification failed" in detail
    assert payload == {}


def test_operational_aggregate_fails_closed_on_unknown_lock():
    lanes = [
        {
            "name": "LIVE MONEY",
            "status": "UNVERIFIED",
            "required": True,
            "domain": startup.DOMAIN_OPERATIONAL,
        }
    ]

    result = startup._aggregate_operational(
        lanes,
        live_locked=None,
        live_lock_verified=False,
    )

    assert result["ready"] is False
    assert result["status"] == "FAILED"
    assert "LIVE MONEY" in result["blockers"]
