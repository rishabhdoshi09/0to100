"""Fail-closed live-money readiness contract.

Paper trading is the training ground. This module evaluates graduation evidence,
but it never authorizes live broker mutations. Operational lock truth comes only
from the canonical broker-boundary interlock.
"""
from __future__ import annotations

from typing import Any, Mapping

# Compatibility constant: live execution is intentionally unavailable in this
# production contract. Runtime status must still use the verified interlock
# below instead of treating this constant as proof.
LIVE_LOCKED = True

DEFAULT_FLOORS = {
    "min_settled_trades": 100,
    "min_trading_days": 40,
    "min_expectancy_R": 0.15,
    "max_drawdown_pct": 20.0,
    "min_distinct_regimes": 2,
    "require_stops_proven": True,
    "require_no_critical_lane": True,
    "require_stable_rules_hash": True,
}


def verified_live_execution_state() -> dict[str, Any]:
    """Return broker-boundary truth without converting verification failure to success."""
    try:
        from product.live_execution_interlock import get_live_execution_state
        return get_live_execution_state().as_dict()
    except Exception as exc:
        return {
            "schema_version": 1,
            "locked": None,
            "authorized": False,
            "verified": False,
            "status": "UNVERIFIED",
            "reason": f"Canonical live execution interlock could not be verified: {type(exc).__name__}: {exc}"[:240],
            "source": "product.live_execution_interlock",
        }


def evaluate_live_readiness(
    *,
    settled_trades: int = 0,
    trading_days: int = 0,
    expectancy_R: float | None = None,
    max_drawdown_pct: float | None = None,
    distinct_regimes: int = 0,
    stops_proven: bool = False,
    critical_lanes_broken: bool = True,
    rules_hash_stable: bool = False,
    floors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate paper evidence and attach independently verified execution-lock truth."""
    req = {**DEFAULT_FLOORS, **dict(floors or {})}
    unmet: list[str] = []
    if int(settled_trades) < int(req["min_settled_trades"]):
        unmet.append(f"settled_trades {settled_trades} < {req['min_settled_trades']}")
    if int(trading_days) < int(req["min_trading_days"]):
        unmet.append(f"trading_days {trading_days} < {req['min_trading_days']}")
    if expectancy_R is None or float(expectancy_R) < float(req["min_expectancy_R"]):
        unmet.append("expectancy missing or below floor (same-hash forward only)")
    if max_drawdown_pct is None or float(max_drawdown_pct) > float(req["max_drawdown_pct"]):
        unmet.append("drawdown missing or above floor")
    if int(distinct_regimes) < int(req["min_distinct_regimes"]):
        unmet.append("not observed across enough regimes")
    if req["require_stops_proven"] and not stops_proven:
        unmet.append("stop/target protection not proven")
    if req["require_no_critical_lane"] and critical_lanes_broken:
        unmet.append("a critical health lane is not healthy")
    if req["require_stable_rules_hash"] and not rules_hash_stable:
        unmet.append("rules hash not stable")

    state = verified_live_execution_state()
    contract_ready = not unmet
    return {
        # Evidence readiness never authorizes live execution.
        "live_enabled": False,
        "live_locked": state.get("locked") if state.get("verified") else None,
        "live_lock_verified": bool(state.get("verified")),
        "live_interlock_status": state.get("status"),
        "live_interlock_reason": state.get("reason"),
        "live_interlock_source": state.get("source"),
        "contract_ready": contract_ready,
        "unmet": unmet,
        "floors": dict(req),
        "note": (
            "Live money is fail-closed. Paper-evidence readiness and broker-boundary authorization "
            "are separate truths; this function never authorizes a live order."
        ),
    }
