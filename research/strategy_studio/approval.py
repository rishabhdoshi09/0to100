"""
✅ User approval + paper activation — the human-in-the-loop gate.

Research code can never approve a strategy. Only a USER may move a strategy from
AWAITING_USER_APPROVAL → APPROVED_FOR_PAPER, and only when the evidence gate is not red
and the evidence is real (not synthetic). Approval produces an IMMUTABLE, PAPER-ONLY
record bound to ONE frozen strategy version (config hash) — a later tweak (new hash)
does not inherit it. Paper activation is a SEPARATE, explicit confirmation. There is NO
live-approval path in this milestone.

Pure: no streamlit, no network, no order path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

from research.strategy_studio import spec as S
from research.strategy_studio.discovery import EvidenceReport


class ApprovalRefused(Exception):
    pass


@dataclass(frozen=True)
class ApprovalRecord:
    strategy_id: str
    version: int
    config_hash: str
    experiment_id: str
    dataset_snapshot: str
    evidence_verdict: str
    limitations: tuple
    approval_timestamp: str
    approving_user: str
    allowed_mode: str                 # always "PAPER"
    max_allocation: float
    max_open_risk_pct: float
    max_trades_per_day: int
    review_date: str
    paper_evaluation_requirements: tuple

    def as_dict(self):
        return asdict(self)


def approve_for_paper(spec: S.StrategySpec, ev: EvidenceReport, readiness: dict, *,
                      actor: str, approver: str, current_state: str,
                      max_allocation: float, max_open_risk_pct: float,
                      max_trades_per_day: int, review_date: str,
                      paper_requirements=(), experiment_id: str = "EXP-006") -> tuple:
    """Approve a strategy for PAPER only. Returns (ApprovalRecord, new_state).
    Guards (all must hold):
      • actor == 'user'  (research code cannot self-approve — enforced by lifecycle);
      • current_state == AWAITING_USER_APPROVAL;
      • readiness is not red AND evidence is real (not synthetic) AND verdict is a survivor.
    """
    # lifecycle: only a user may perform this transition
    S.require_transition(current_state, S.APPROVED_FOR_PAPER, actor)
    if not readiness or not readiness.get("can_run") or readiness.get("color") == "red":
        raise ApprovalRefused("evidence gate is red — cannot approve")
    if ev.is_synthetic:
        raise ApprovalRefused("evidence is synthetic (a software demo) — not market "
                              "evidence; cannot approve")
    if ev.verdict not in ("PROMOTE", "PASS"):
        raise ApprovalRefused(f"evidence verdict is '{ev.verdict}', not a survivor — "
                              "cannot approve")
    rec = ApprovalRecord(
        strategy_id=spec.strategy_id, version=spec.version, config_hash=spec.config_hash(),
        experiment_id=experiment_id, dataset_snapshot=spec.dataset_snapshot,
        evidence_verdict=ev.verdict, limitations=tuple(spec_limitations(readiness)),
        approval_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        approving_user=approver, allowed_mode="PAPER",
        max_allocation=float(max_allocation), max_open_risk_pct=float(max_open_risk_pct),
        max_trades_per_day=int(max_trades_per_day), review_date=review_date,
        paper_evaluation_requirements=tuple(paper_requirements))
    return rec, S.APPROVED_FOR_PAPER


def spec_limitations(readiness: dict) -> list:
    return list((readiness or {}).get("reasons", []))


def approval_valid_for(record: ApprovalRecord, spec: S.StrategySpec) -> bool:
    """An approval is bound to ONE frozen version. A tweak (new config hash / version)
    does not inherit it."""
    return (record.config_hash == spec.config_hash()
            and record.version == spec.version
            and record.strategy_id == spec.strategy_id)


@dataclass(frozen=True)
class PaperActivation:
    strategy_id: str
    version: int
    mode: str                         # always "PAPER"
    activated_by: str
    activation_timestamp: str
    max_allocation: float
    max_trades_per_day: int

    def as_dict(self):
        return asdict(self)


def activate_paper(record: ApprovalRecord, spec: S.StrategySpec, *, actor: str,
                   confirmed: bool) -> PaperActivation:
    """Register an APPROVED strategy for PAPER execution — a SEPARATE explicit step.
    Approval alone is NOT enough: `confirmed` must be True and the actor a user. This
    registers for paper only; it places NO order and NEVER enables live. Telegram stays
    paper-only regardless."""
    if actor != "user":
        raise ApprovalRefused("only a user may activate paper")
    if record.allowed_mode != "PAPER":
        raise ApprovalRefused("record is not PAPER-scoped")
    if not approval_valid_for(record, spec):
        raise ApprovalRefused("approval does not match this strategy version (it was "
                              "tweaked — approve the new version)")
    if not confirmed:
        raise ApprovalRefused("paper activation needs a separate explicit confirmation — "
                              "the approval click alone does not activate it")
    return PaperActivation(
        strategy_id=spec.strategy_id, version=spec.version, mode="PAPER",
        activated_by=actor, max_allocation=record.max_allocation,
        max_trades_per_day=record.max_trades_per_day,
        activation_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
