"""Fail-closed live-money readiness contract.

Paper trading is the training ground. This module never enables live orders.
Live mode stays locked until an owner explicitly flips a separate control after
the statistical contract is met — the bot cannot open that door.
"""
from __future__ import annotations

from typing import Any, Mapping

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
    """Return a fail-closed readiness verdict. Never sets live_enabled=True."""
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
    ready = not unmet
    return {
        "live_enabled": False,
        "live_locked": True,
        "contract_ready": ready,
        "unmet": unmet,
        "floors": dict(req),
        "note": (
            "Live money is fail-closed. Meeting this contract does not turn live "
            "trading on — an owner must enable a separate live adapter later. "
            "The decision object is identical for paper and future live adapters."
        ),
    }
