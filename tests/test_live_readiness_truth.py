from __future__ import annotations

from dataclasses import replace

from product import live_readiness
from product.live_execution_interlock import get_live_execution_state


def _ready_evidence_kwargs() -> dict:
    return {
        "settled_trades": 100,
        "trading_days": 40,
        "expectancy_R": 0.20,
        "max_drawdown_pct": 10.0,
        "distinct_regimes": 2,
        "stops_proven": True,
        "critical_lanes_broken": False,
        "rules_hash_stable": True,
    }


def test_evidence_ready_does_not_enable_live_execution():
    out = live_readiness.evaluate_live_readiness(**_ready_evidence_kwargs())

    assert out["contract_ready"] is True
    assert out["live_lock_verified"] is True
    assert out["live_locked"] is True
    assert out["live_execution_authorized"] is False
    assert out["live_enabled"] is False
    assert out["live_execution_source"] == "product.live_execution_interlock"


def test_unverified_interlock_never_becomes_positive_locked_claim(monkeypatch):
    import product.live_execution_interlock as interlock

    def broken_state():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(interlock, "get_live_execution_state", broken_state)
    out = live_readiness.evaluate_live_readiness()

    assert out["live_lock_verified"] is False
    assert out["live_locked"] is None
    assert out["live_execution_authorized"] is None
    assert out["live_execution_status"] == "UNVERIFIED"
    assert out["live_enabled"] is False
    assert "could not be verified" in out["live_execution_reason"]


def test_verified_unlocked_but_unauthorized_state_is_not_live_enabled(monkeypatch):
    import product.live_execution_interlock as interlock

    base = get_live_execution_state()
    monkeypatch.setattr(
        interlock,
        "get_live_execution_state",
        lambda: replace(base, locked=False, authorized=False, verified=True, status="UNLOCKED_UNAUTHORIZED"),
    )
    out = live_readiness.evaluate_live_readiness(**_ready_evidence_kwargs())

    assert out["live_lock_verified"] is True
    assert out["live_locked"] is False
    assert out["live_execution_authorized"] is False
    assert out["live_enabled"] is False
