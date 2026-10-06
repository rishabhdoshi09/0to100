from __future__ import annotations

from scripts.verify_public_launch import Probe, evaluate


def _healthy():
    return {
        "live_lock_verified": True,
        "live_locked": True,
        "live_execution_authorized": False,
    }


def _access():
    return {
        "public_read_only": True,
        "mutation_policy": "READ_ONLY",
        "operator_token_configured": False,
    }


def test_public_launch_preflight_passes_only_on_read_only_and_live_lock():
    ok, lines = evaluate(
        _access(),
        _access(),
        _healthy(),
        Probe(403, {"code": "PUBLIC_READ_ONLY"}),
        Probe(403, {"code": "PUBLIC_READ_ONLY"}),
    )

    assert ok is True
    assert all(line.startswith("PASS") for line in lines)


def test_public_launch_preflight_fails_if_guard_falls_through_to_404():
    ok, lines = evaluate(
        _access(),
        _access(),
        _healthy(),
        Probe(404, {"detail": "Not Found"}),
        Probe(403, {"code": "PUBLIC_READ_ONLY"}),
    )

    assert ok is False
    assert any(line.startswith("FAIL") and "Main API rejects" in line for line in lines)


def test_public_launch_preflight_fails_if_live_lock_is_unverified():
    health = _healthy()
    health["live_lock_verified"] = False

    ok, lines = evaluate(
        _access(),
        _access(),
        health,
        Probe(403, {"code": "OPERATOR_AUTH_REQUIRED"}),
        Probe(403, {"code": "OPERATOR_AUTH_REQUIRED"}),
    )

    assert ok is False
    assert any(line.startswith("FAIL") and "interlock is verified" in line for line in lines)
