"""
🪜 Promotion ladder + allocation/retirement + live-readiness — mapped onto the EXISTING lifecycle.

This never creates a second lifecycle model: the ladder's states map onto
`research/strategy_studio/spec.py` constants, and the live door stays user-only exactly as
`spec._TRANSITIONS` enforces. Allocation may deploy/maintain/reduce/pause/retire but can never enlarge
a constitutional limit or authorise live capital. Live remains structurally refused this milestone.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.strategy_studio import spec as LC

# ladder → existing lifecycle constant
LADDER_TO_LIFECYCLE = {
    "PROPOSED": LC.GENERATED,
    "PREREGISTERED": LC.UNDER_REVIEW,
    "BACKTESTING": LC.UNDER_REVIEW,
    "REJECTED": LC.REJECTED,
    "INCONCLUSIVE": LC.UNDER_REVIEW,
    "HISTORICALLY_QUALIFIED": LC.PROMISING,
    "PAPER_NOMINATED": LC.APPROVED_FOR_PAPER,
    "PAPER_EVALUATION": LC.PAPER_EVALUATION,
    "PAPER_PROVEN": LC.PAPER_CONFIRMED,
    "DECAYED": LC.DECAYED,
    "RETIRED": LC.RETIRED,
}

COMMITTEE_TO_LADDER = {
    "REJECT": "REJECTED",
    "INCONCLUSIVE": "INCONCLUSIVE",
    "RETEST_WITH_MORE_DATA": "INCONCLUSIVE",
    "PAPER_NOMINATED": "PAPER_NOMINATED",
}

# live-readiness progression (owner-gated end state; live never reached by the system)
RESEARCH_ONLY = "RESEARCH_ONLY"
PAPER_AUTO = "PAPER_AUTO"
PAPER_PROVEN = "PAPER_PROVEN"
SHADOW_LIVE = "SHADOW_LIVE"
LIMITED_LIVE_ELIGIBLE = "LIMITED_LIVE_ELIGIBLE"
OWNER_ACTIVATED_LIMITED_LIVE = "OWNER_ACTIVATED_LIMITED_LIVE"

LIVE_EXECUTION_LOCKED = True          # this milestone never enables real orders


def map_to_lifecycle(ladder_state: str) -> str:
    return LADDER_TO_LIFECYCLE.get(ladder_state, LC.GENERATED)


def paper_proven(evidence: dict, *, min_forward_trades: int = 40, min_lower_bound_R: float = 0.05,
                 min_forward_to_backtest: float = 0.6) -> bool:
    """PAPER_PROVEN requires CONFIGURED forward evidence — never in-sample or optimistic language."""
    return (int(evidence.get("forward_trades", 0)) >= min_forward_trades
            and float(evidence.get("forward_lower_bound_R", -1.0)) >= min_lower_bound_R
            and float(evidence.get("forward_to_backtest", 0.0)) >= min_forward_to_backtest)


@dataclass(frozen=True)
class AllocationChange:
    strategy_id: str
    action: str                       # deploy | maintain | reduce | pause | retire
    before_weight: float
    after_weight: float
    reason: str


def allocation_action(strategy: dict, *, current_weight: float) -> AllocationChange:
    """Decide an allocation change from forward outcomes. Reduces/pauses/retires on decay; never
    increases beyond the constitutional per-strategy cap."""
    sid = str(strategy.get("strategy_id", ""))
    fwd_R = float(strategy.get("forward_expectancy_R", 0.0))
    decay = float(strategy.get("forward_to_backtest", 1.0))
    dd = float(strategy.get("current_drawdown_pct", 0.0))
    n = int(strategy.get("forward_trades", 0))
    cap = float(strategy.get("max_weight", 0.2))          # constitutional per-strategy cap (read-only)

    if n >= 20 and fwd_R <= -0.1:
        return AllocationChange(sid, "retire", current_weight, 0.0,
                                "forward expectancy decisively negative")
    if fwd_R < 0 or decay < 0.4 or dd >= 25.0:
        after = round(current_weight * 0.5, 4)
        return AllocationChange(sid, "reduce", current_weight, after,
                                "forward performance decayed vs backtest / drawdown pressure")
    if n < 10:
        return AllocationChange(sid, "maintain", current_weight, current_weight,
                                "insufficient forward sample — hold, do not add")
    target = min(cap, current_weight)                     # never exceed the cap autonomously
    return AllocationChange(sid, "maintain", current_weight, target, "forward performance holding")


@dataclass(frozen=True)
class LiveReadiness:
    state: str
    blockers: tuple
    owner_activation_present: bool = False


def live_readiness(pkg: dict) -> LiveReadiness:
    """Compute the highest readiness the EVIDENCE supports. The system can reach at most
    LIMITED_LIVE_ELIGIBLE; OWNER_ACTIVATED requires an explicit owner envelope that this milestone
    never grants. SHADOW_LIVE and live are out of scope and refused."""
    blockers = []
    if not pkg.get("paper_auto_operational"):
        return LiveReadiness(RESEARCH_ONLY, ("PAPER_AUTO not operational on genuine data",))
    if not pkg.get("paper_proven"):
        blockers.append("no PAPER_PROVEN strategy with configured forward evidence")
    for need, msg in [("broker_connected", "no genuine broker connection"),
                      ("reconciled", "broker/local state not reconciled"),
                      ("data_forward_eligible", "current data not forward-eligible"),
                      ("risk_governor_healthy", "Risk Governor unhealthy"),
                      ("protective_exits_proven", "protective exits not proven"),
                      ("restart_recovery_proven", "restart recovery not proven"),
                      ("order_persistence_healthy", "order persistence not proven"),
                      ("no_unresolved_critical_incident", "an unresolved critical incident exists")]:
        if not pkg.get(need):
            blockers.append(msg)
    if pkg.get("paper_proven") and not blockers:
        # eligible, but owner capital-envelope approval is deliberately still ABSENT
        return LiveReadiness(LIMITED_LIVE_ELIGIBLE, ("owner capital-envelope approval absent",),
                             owner_activation_present=False)
    state = PAPER_PROVEN if pkg.get("paper_proven") else PAPER_AUTO
    return LiveReadiness(state, tuple(blockers))
