"""
🛡️ Independent Live Risk Governor (Phases 9–11).

Structurally separate from Brain 2 and the EMS. It grants, reduces or denies LIVE risk and no
brain or the EMS may override it. It enforces the full risk hierarchy (order → position → symbol
→ strategy → family → sector → cluster → portfolio → account → day → drawdown) and the automatic
capital-protection state machine (daily loss / drawdown). Risk increases are gradual; reductions
are immediate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ems import schemas as SC


@dataclass
class RiskLimits:
    version: str = "v1"
    max_risk_per_trade_pct: float = 0.01
    max_symbol_risk_pct: float = 0.02
    max_strategy_risk_pct: float = 0.02
    max_family_risk_pct: float = 0.03
    max_sector_risk_pct: float = 0.04
    max_cluster_risk_pct: float = 0.03
    max_total_open_risk_pct: float = 0.05
    max_positions: int = 5
    max_daily_loss: float = 0.0            # ₹ (0 ⇒ from envelope)
    max_drawdown_pct: float = 0.10


def capital_protection_state(account: dict, limits: RiskLimits, envelope) -> str:
    """NORMAL → CAUTION → NO_NEW_ENTRIES → LIQUIDATE_ONLY → HALTED based on realized daily loss
    and drawdown. Deterministic; reductions bite immediately."""
    daily_loss_limit = limits.max_daily_loss or getattr(envelope, "daily_loss_limit", 0.0)
    realized = float(account.get("realized_pnl_today", 0.0))
    dd = float(account.get("drawdown_pct", 0.0))
    if daily_loss_limit and realized <= -daily_loss_limit:
        return SC.CAP_NO_NEW_ENTRIES
    dd_limit = getattr(envelope, "drawdown_limit_pct", limits.max_drawdown_pct)
    if dd >= dd_limit:
        return SC.CAP_LIQUIDATE_ONLY
    if dd >= 0.7 * dd_limit or (daily_loss_limit and realized <= -0.7 * daily_loss_limit):
        return SC.CAP_NO_NEW_ENTRIES
    if dd >= 0.5 * dd_limit:
        return SC.CAUTION
    return SC.NORMAL


class RiskGovernor:
    def __init__(self, limits: RiskLimits | None = None, *, healthy: bool = True):
        self.limits = limits or RiskLimits()
        self.healthy = healthy

    def evaluate(self, *, plan, envelope, portfolio: dict, account: dict,
                 trace: dict | None = None) -> SC.RiskDecision:
        L = self.limits
        trace = trace or {}
        req = int(plan.qty)

        def decision(d, qty, reason, code=""):
            return SC.RiskDecision(decision=d, approved_qty=qty, requested_qty=req,
                                   reason=reason, limit_code=code, limits_version=L.version,
                                   **{k: trace.get(k, "") for k in
                                      ("idempotency_key", "cycle_id", "snapshot_id",
                                       "strategy_id", "intent_id")})

        if not self.healthy:
            return decision(SC.REJECT, 0, "risk governor unhealthy", "RG_UNHEALTHY")
        if not envelope.is_user_approved():
            return decision(SC.REJECT, 0, "operating envelope not user-approved", "NO_ENVELOPE")

        state = capital_protection_state(account, L, envelope)
        if state in (SC.CAP_NO_NEW_ENTRIES, SC.CAP_LIQUIDATE_ONLY, SC.CAP_HALTED):
            code = {"NO_NEW_ENTRIES": "DAILY_LOSS", "LIQUIDATE_ONLY": "DRAWDOWN",
                    "HALTED": "HALT"}[state]
            return decision(SC.BLOCK_NEW_ENTRIES, 0,
                            f"capital-protection state {state} blocks new entries", code)

        fam = trace.get("family", "")
        sym = plan.symbol
        req_risk = float(plan.expected_risk)              # % of capital

        # portfolio caps (order/symbol/strategy/family/sector/cluster/total/positions)
        if portfolio.get("positions", 0) >= L.max_positions:
            return decision(SC.REJECT, 0, "max positions reached", "MAX_POSITIONS")
        for tag, cur_map, cap, code in [
            ("symbol", portfolio.get("symbol_risk", {}).get(sym, 0.0), L.max_symbol_risk_pct, "SYMBOL_CAP"),
            ("strategy", portfolio.get("strategy_risk", {}).get(trace.get("strategy_id", ""), 0.0),
             L.max_strategy_risk_pct, "STRATEGY_CAP"),
            ("family", portfolio.get("family_risk", {}).get(fam, 0.0), L.max_family_risk_pct, "FAMILY_CAP"),
            ("sector", portfolio.get("sector_risk", {}).get(trace.get("sector", ""), 0.0),
             L.max_sector_risk_pct, "SECTOR_CAP"),
            ("cluster", portfolio.get("cluster_risk", {}).get(trace.get("cluster", ""), 0.0),
             L.max_cluster_risk_pct, "CLUSTER_CAP"),
            ("total", portfolio.get("open_risk_pct", 0.0), L.max_total_open_risk_pct, "TOTAL_RISK_CAP"),
        ]:
            if cur_map + req_risk > cap + 1e-9:
                return decision(SC.REJECT, 0, f"{tag} risk cap {cap:.1%} would be breached", code)

        # per-trade risk: REDUCE quantity rather than reject outright
        if req_risk > L.max_risk_per_trade_pct + 1e-9 and req_risk > 0:
            scaled = max(1, int(req * (L.max_risk_per_trade_pct / req_risk)))
            if scaled < req:
                return decision(SC.APPROVE_REDUCED, scaled,
                                "per-trade risk reduced to the limit", "PER_TRADE_RISK")
        return decision(SC.APPROVE, req, "within all live risk limits")
