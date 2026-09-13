from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_product_acceptance",
    ROOT / "scripts" / "run_product_acceptance.py",
)
assert SPEC and SPEC.loader
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


def _healthy(**overrides):
    payload = {
        "ok": True,
        "live_locked": True,
        "live_lock_verified": True,
        "live_execution_authorized": False,
        "operational_ready": True,
        "evidence_ready": True,
        "lifecycle": "READY",
    }
    payload.update(overrides)
    return payload


def test_canonical_health_requires_verified_live_lock():
    result = acceptance.grade_canonical_health(_healthy(live_lock_verified=False))
    assert result["status"] == "FAIL"
    assert result["live_locked"] is True
    assert result["live_lock_verified"] is False


def test_canonical_health_requires_explicit_unauthorized_state():
    payload = _healthy()
    payload.pop("live_execution_authorized")
    result = acceptance.grade_canonical_health(payload)
    assert result["status"] == "FAIL"
    assert "missing live_execution_authorized" in result["blocker_reason"]

    result = acceptance.grade_canonical_health(_healthy(live_execution_authorized=True))
    assert result["status"] == "FAIL"
    assert "live_execution_authorized is not False" in result["blocker_reason"]


def test_canonical_health_valid_verified_locked_unauthorized_passes():
    result = acceptance.grade_canonical_health(_healthy())
    assert result["status"] == "PASS"
    assert result["live_locked"] is True
    assert result["live_lock_verified"] is True
    assert result["live_execution_authorized"] is False


def test_overall_verdict_cannot_pass_unverified_lock():
    rows = [{"feature": "Canonical stack / readiness", "status": "PASS"}]
    result = acceptance.product_acceptance_verdict(
        rows,
        live_locked=True,
        live_lock_verified=False,
    )
    assert result["exit_code"] == 1
    assert result["verdict"] == "PRODUCT ACCEPTANCE HOLD"
    assert "not verified" in result["reason"]


def _learning(**overrides):
    payload = {
        "schema_version": 1,
        "live_locked": True,
        "live_lock_verified": True,
        "live_execution_authorized": False,
        "policies": [],
        "counterfactuals": [],
    }
    payload.update(overrides)
    return payload


def test_learning_requires_verified_locked_unauthorized_truth():
    assert acceptance.grade_learning_dashboard(_learning())["status"] == "PASS"
    assert acceptance.grade_learning_dashboard(_learning(live_lock_verified=False))["status"] == "FAIL"
    assert acceptance.grade_learning_dashboard(_learning(live_locked=None))["status"] == "FAIL"
    assert acceptance.grade_learning_dashboard(_learning(live_execution_authorized=True))["status"] == "FAIL"

    missing = _learning()
    missing.pop("live_execution_authorized")
    assert acceptance.grade_learning_dashboard(missing)["status"] == "FAIL"


def _soak(**overrides):
    verification = {
        "lanes": {"taken": {"status": "READY"}},
        "live_locked": True,
        "live_lock_verified": True,
        "live_execution_authorized": False,
    }
    verification.update(overrides)
    return {"verification": verification}


def test_forward_soak_requires_verified_locked_unauthorized_truth():
    assert acceptance.grade_forward_soak(_soak())["status"] == "PASS"
    assert acceptance.grade_forward_soak(_soak(live_lock_verified=False))["status"] == "FAIL"
    assert acceptance.grade_forward_soak(_soak(live_locked=None))["status"] == "FAIL"
    assert acceptance.grade_forward_soak(_soak(live_execution_authorized=True))["status"] == "FAIL"

    payload = _soak()
    payload["verification"].pop("live_execution_authorized")
    assert acceptance.grade_forward_soak(payload)["status"] == "FAIL"


def _paper(**overrides):
    payload = {
        "available": True,
        "open_positions": [],
        "last_cycle": {},
    }
    payload.update(overrides)
    return payload


def test_paper_status_uses_canonical_safety_without_inventing_endpoint_claims():
    result = acceptance.grade_paper_status(
        _paper(),
        live_locked=True,
        live_lock_verified=True,
        live_execution_authorized=False,
    )
    assert result["status"] == "PASS"

    assert acceptance.grade_paper_status(
        _paper(), live_locked=True, live_lock_verified=False, live_execution_authorized=False
    )["status"] == "FAIL"
    assert acceptance.grade_paper_status(
        _paper(), live_locked=True, live_lock_verified=True, live_execution_authorized=True
    )["status"] == "FAIL"
