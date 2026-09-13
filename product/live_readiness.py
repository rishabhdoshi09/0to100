"""Live-money readiness and execution-safety projection.

Paper trading is the training ground. Statistical readiness and broker-boundary
execution authorization are separate facts. This module never treats an
unverified interlock as proof that live money is locked, and it never enables
live orders unless the canonical execution boundary explicitly verifies both
unlocking and authorization.
"""
from __future__ import annotations

from typing import Any, Mapping

# Backward-compatible policy default only. Runtime truth comes from
# product.live_execution_interlock.get_live_execution_state().
LIVE_LOCKED = True

# These floors are justified as *minimum evidence*, not as a promise of profit.
# 100 settled paper trades ≈ a small but non-anecdotal forward sample.
# 40 distinct sessions ≈ two trading months of production conditions.
# Expectancy and drawdown are measured on the same rules_hash only.
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
    """Read canonical broker-boundary truth without inventing positive safety.

    Verification failure is fail-closed for *permission* (``live_enabled`` can
    never become true), but it is deliberately reported as ``live_locked=None``
    rather than fabricating proof that the lock was successfully verified.
    """
    try:
        from product.live_execution_interlock import get_live_execution_state

        state = get_live_execution_state()
        verified = bool(state.verified)
        locked = bool(state.locked) if verified else None
        authorized = bool(state.authorized) if verified else None
        return {
            "live_locked": locked,
            "live_lock_verified": verified,
            "live_execution_authorized": authorized,
            "live_lock_status": str(state.status or ("LOCKED" if locked else "UNLOCKED")),
            "live_lock_reason": str(state.reason or ""),
            "live_lock_source": str(state.source or "product.live_execution_interlock"),
        }
    except Exception as exc:
        return {
            "live_locked": None,
            "live_lock_verified": False,
            "live_execution_authorized": None,
            "live_lock_status": "UNVERIFIED",
            "live_lock_reason": f"Canonical live-execution interlock could not be verified: {exc}",
            "live_lock_source": "product.live_execution_interlock",
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
    """Return evidence readiness plus canonical live-execution safety truth."""
    req = {**DEFAULT_FLOORS, **dict(floors or {})}
    unmet: list[str] = []
    if int(settled_trades) < int(req["min_settled_trades"]):
        unmet.append(
            f"settled_trades {settled_trades} < {req['min_settled_trades']}"
        )
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
    lock_verified = execution["live_lock_verified"] is True
    locked = execution["live_locked"]
    authorized = execution["live_execution_authorized"]

    # Readiness evidence alone can never open the broker boundary. Live mode is
    # possible only when the canonical interlock itself verifies an authorized,
    # unlocked state. Under the current production contract this remains false.
    live_enabled = bool(
        contract_ready
        and lock_verified
        and locked is False
        and authorized is True
    )

    return {
        "live_enabled": live_enabled,
        "contract_ready": contract_ready,
        "unmet": unmet,
        "floors": dict(req),
        **execution,
        "note": (
            "Statistical readiness does not authorize live money. Broker execution "
            "is controlled only by the canonical live-execution interlock; missing "
            "verification is reported as UNVERIFIED rather than as a positive lock."
        ),
    }
