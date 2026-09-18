from types import SimpleNamespace

import product.live_execution_interlock as interlock
from product.live_readiness import evaluate_live_readiness


def _ready_kwargs():
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


def _state(*, locked, authorized, verified=True, status=None, reason="test"):
    return SimpleNamespace(
        locked=locked,
        authorized=authorized,
        verified=verified,
        status=status or ("LOCKED" if locked else "UNLOCKED"),
        reason=reason,
        source="product.live_execution_interlock",
    )


def test_evidence_ready_does_not_override_verified_broker_lock(monkeypatch):
    monkeypatch.setattr(
        interlock,
        "get_live_execution_state",
        lambda: _state(locked=True, authorized=False),
    )

    result = evaluate_live_readiness(**_ready_kwargs())

    assert result["contract_ready"] is True
    assert result["live_lock_verified"] is True
    assert result["live_locked"] is True
    assert result["live_execution_authorized"] is False
    assert result["live_enabled"] is False


def test_live_requires_contract_and_verified_authorized_unlocked_boundary(monkeypatch):
    monkeypatch.setattr(
        interlock,
        "get_live_execution_state",
        lambda: _state(locked=False, authorized=True),
    )

    result = evaluate_live_readiness(**_ready_kwargs())

    assert result["contract_ready"] is True
    assert result["live_lock_verified"] is True
    assert result["live_locked"] is False
    assert result["live_execution_authorized"] is True
    assert result["live_enabled"] is True


def test_interlock_probe_failure_never_fabricates_positive_lock(monkeypatch):
    def broken_probe():
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(interlock, "get_live_execution_state", broken_probe)

    result = evaluate_live_readiness(**_ready_kwargs())

    assert result["contract_ready"] is True
    assert result["live_lock_verified"] is False
    assert result["live_locked"] is None
    assert result["live_execution_authorized"] is None
    assert result["live_lock_status"] == "UNVERIFIED"
    assert "probe unavailable" in result["live_lock_reason"]
    assert result["live_enabled"] is False


def test_execution_authorization_cannot_bypass_missing_evidence(monkeypatch):
    monkeypatch.setattr(
        interlock,
        "get_live_execution_state",
        lambda: _state(locked=False, authorized=True),
    )

    result = evaluate_live_readiness()

    assert result["contract_ready"] is False
    assert result["live_lock_verified"] is True
    assert result["live_locked"] is False
    assert result["live_execution_authorized"] is True
    assert result["live_enabled"] is False
