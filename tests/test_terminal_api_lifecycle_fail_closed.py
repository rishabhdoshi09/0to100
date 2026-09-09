from __future__ import annotations

import pytest

import terminal_api
from scripts.run_product_acceptance import grade_canonical_health, product_acceptance_verdict


def _assert_hold(payload: dict) -> None:
    graded = grade_canonical_health(payload)
    assert graded["status"] == "FAIL"
    verdict = product_acceptance_verdict(
        [{"feature": "Canonical stack / readiness", "status": graded["status"]}],
        live_locked=graded["live_locked"],
    )
    assert verdict["verdict"] == "PRODUCT ACCEPTANCE HOLD"


def test_health_does_not_invent_ready_when_runtime_omits_lifecycle(monkeypatch):
    monkeypatch.setattr(
        "product.runtime_lifecycle.inspect_runtime",
        lambda **_k: {
            "reason": "runtime state incomplete",
            "reasons": [],
            "components": [],
            "history": {},
            "resources": {},
            "operational_ready": True,
            "evidence_ready": True,
            "live_locked": True,
        },
    )
    payload = terminal_api.health()
    assert payload.get("lifecycle") != "READY"
    _assert_hold(payload)


@pytest.mark.parametrize("lifecycle", ["", None])
def test_health_does_not_invent_ready_for_blank_runtime_lifecycle(monkeypatch, lifecycle):
    monkeypatch.setattr(
        "product.runtime_lifecycle.inspect_runtime",
        lambda **_k: {
            "lifecycle": lifecycle,
            "reason": "runtime state incomplete",
            "reasons": [],
            "components": [],
            "history": {},
            "resources": {},
            "operational_ready": True,
            "evidence_ready": True,
            "live_locked": True,
        },
    )
    payload = terminal_api.health()
    assert payload.get("lifecycle") != "READY"
    _assert_hold(payload)


def test_health_preserves_explicit_ready_runtime(monkeypatch):
    monkeypatch.setattr(
        "product.runtime_lifecycle.inspect_runtime",
        lambda **_k: {
            "lifecycle": "READY",
            "reason": "Required services are alive and official history is current",
            "reasons": [],
            "components": [],
            "history": {},
            "resources": {},
            "operational_ready": True,
            "evidence_ready": True,
            "live_locked": True,
        },
    )
    payload = terminal_api.health()
    assert payload["lifecycle"] == "READY"
    graded = grade_canonical_health(payload)
    assert graded == {"status": "PASS", "blocker_reason": "", "live_locked": True}
