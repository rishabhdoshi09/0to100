"""Evidence-readiness contract for future live-money graduation.

This module answers a statistical/evidence question only: has the paper system
collected enough same-rule forward evidence to be considered for graduation?
It does *not* authorize broker execution. The broker-boundary authority is
``product.live_execution_interlock`` and its verification state is surfaced
separately here so callers cannot mistake evidence readiness for a lock check.
"""
from __future__ import annotations

from typing import Any, Mapping


# These floors are minimum evidence, not a promise of profit or authorization.
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


def _execution_interlock_truth() -> dict[str, Any]:
    """Read canonical broker-boundary state; verification failure stays unknown."""
    try:
        from product.live_execution_interlock import get_live_execution_state

        state = get_live_execution_state()
        payload = state.as_dict()
        locked = bool(state.locked and not state.authorized) if state.verified else None
        return {
            "live_locked": locked,
            "live_lock_verified": bool(state.verified),
            "live_execution_authorized": bool(state.authorized) if state.verified else None,
            "live_execution_status": str(state.status or "UNKNOWN"),
            "live_execution_reason": str(state.reason or ""),
            "live_execution_source": str(state.source or "product.live_execution_interlock"),
            "live_interlock": payload,
        }
    except Exception as exc:
        return {
            "live_locked": None,
            "live_lock_verified": False,
            "live_execution_authorized": None,
            "live_execution_status": "UNVERIFIED",
            "live_execution_reason": f"Canonical live interlock could not be verified: {type(exc).__name__}: {exc}"[:240],
            "live_execution_source": "product.live_execution_interlock",
            "live_interlock": {},
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
    """Return evidence readiness plus the independently verified execution lock.

    ``contract_ready`` can become true when evidence floors are met. It never
    enables live orders. ``live_enabled`` is true only if the *canonical broker
    interlock* is verified, explicitly authorized, and unlocked.
    """
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

    contract_ready = not unmet
    execution = _execution_interlock_truth()
    lock_verified = bool(execution.get("live_lock_verified"))
    locked = execution.get("live_locked")
    authorized = execution.get("live_execution_authorized")
    live_enabled = bool(lock_verified and locked is False and authorized is True)

    return {
        "live_enabled": live_enabled,
        "contract_ready": contract_ready,
        "unmet": unmet,
        "floors": dict(req),
        **execution,
        "note": (
            "Evidence readiness does not authorize capital. Live execution is a separate "
            "broker-boundary fact and must be verified there."
        ),
    }
