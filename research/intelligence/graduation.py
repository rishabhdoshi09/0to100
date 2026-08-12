"""
🎓 Graduation protocol — the evidence-gated path to a HUMAN live review.

A strategy that is genuinely confirmed forward may be NOMINATED for human review. This module
checks explicit, configurable criteria and emits a `LifecycleDecision`. It can nominate
(→ ELIGIBLE_FOR_HUMAN_LIVE_REVIEW) but it can NEVER emit USER_APPROVED — only a user may cross
that gate (enforced by the lifecycle). This is a review page, not an auto-buy.
"""
from __future__ import annotations

from dataclasses import dataclass

from research.intelligence import schemas as SC
from research.intelligence import evidence_brain as EB
from research.strategy_studio import spec as S


@dataclass(frozen=True)
class GraduationCriteria:
    min_forward_trades: int = 40
    min_lower_bound_R: float = 0.05          # positive uncertainty-adjusted expectancy
    min_deflated_sharpe: float = 0.90
    max_drawdown_R: float = 6.0
    min_forward_to_backtest: float = 0.5     # no severe backtest→forward collapse
    require_no_data_warnings: bool = True
    require_not_regime_dependent: bool = True


def evaluate(card: SC.StrategyEvidenceCard, *, criteria: GraduationCriteria | None = None
             ) -> SC.LifecycleDecision:
    """Check a card against the graduation criteria. Returns a LifecycleDecision that at MOST
    nominates for human review — never an approval."""
    c = criteria or GraduationCriteria()
    met, unmet = [], []

    def chk(ok: bool, label: str):
        (met if ok else unmet).append(label)

    chk(card.evidence_state == EB.CONFIRMED, "forward-confirmed evidence")
    chk(card.forward_trades >= c.min_forward_trades,
        f"≥{c.min_forward_trades} forward trades")
    chk(card.lower_bound_R >= c.min_lower_bound_R,
        f"lower-bound edge ≥ {c.min_lower_bound_R:+.2f}R")
    chk(card.deflated_sharpe >= c.min_deflated_sharpe,
        f"deflated Sharpe ≥ {c.min_deflated_sharpe}")
    chk(card.max_drawdown <= c.max_drawdown_R, f"drawdown ≤ {c.max_drawdown_R}R")
    chk(card.forward_to_backtest >= c.min_forward_to_backtest,
        "no severe backtest→forward collapse")
    if c.require_not_regime_dependent:
        chk(card.evidence_state != EB.REGIME_DEPENDENT, "not regime-dependent")
    if c.require_no_data_warnings:
        chk(len(card.data_quality_warnings) == 0, "no data-quality warnings")

    qualifies = not unmet
    to_state = S.ELIGIBLE_FOR_HUMAN_LIVE_REVIEW if qualifies else S.PAPER_EVALUATION
    return SC.LifecycleDecision(
        strategy_id=card.strategy_id, strategy_version=card.strategy_version,
        rules_hash=card.rules_hash, data_snapshot_id=card.data_snapshot_id,
        source="graduation", event_ts=card.event_ts, family=card.family,
        from_state=S.PAPER_CONFIRMED if qualifies else S.PAPER_EVALUATION,
        to_state=to_state, actor=S.PAPER_AUTOPILOT,
        met_criteria=tuple(met), unmet_criteria=tuple(unmet),
        user_gate_required=True)                 # the LIVE step always needs a human


def user_approve(strategy_id: str, *, actor: str, current_state: str) -> str:
    """The ONE user-owned live door. Raises unless a real user performs it — no brain or
    autopilot can. Returns the new state on success."""
    S.require_transition(current_state, S.USER_APPROVED, actor)   # user-only; else LifecycleError
    return S.USER_APPROVED
