"""
⚔️ Adversarial research council — deterministic services that must independently clear a hypothesis.

Data Auditor, Sceptic, Reality Checker and Portfolio Examiner each return a typed verdict; the
Promotion Committee combines them into one decision. No role may approve its own hypothesis — the
committee refuses when the candidate's producer is the committee itself. LLMs may add commentary but
cannot change a verdict. Everything is a pure function of the candidate context, so it is deterministic
and network-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# committee decisions (map onto the promotion ladder)
REJECT = "REJECT"
INCONCLUSIVE = "INCONCLUSIVE"
RETEST_WITH_MORE_DATA = "RETEST_WITH_MORE_DATA"
PAPER_NOMINATED = "PAPER_NOMINATED"


@dataclass(frozen=True)
class Verdict:
    role: str
    passed: bool
    findings: tuple = ()
    blocking: bool = True             # a failed non-blocking check downgrades but does not reject


def _get(ctx, key, default=None):
    return ctx.get(key, default) if isinstance(ctx, dict) else getattr(ctx, key, default)


def data_auditor(ctx) -> Verdict:
    """Point-in-time integrity + data adequacy."""
    f = []
    if not _get(ctx, "forward_eligible", False):
        f.append("data tier is not forward-eligible")
    if _get(ctx, "requires_universe_history", False) and not _get(ctx, "universe_history_complete", False):
        f.append("survivorship-incomplete universe history for a history-dependent test")
    if _get(ctx, "requires_corporate_actions", False) and not _get(ctx, "corporate_actions_ok", False):
        f.append("corporate-action coverage insufficient")
    if _get(ctx, "leakage_detected", False):
        f.append("look-ahead leakage detected")
    if _get(ctx, "requires_benchmark", True) and not _get(ctx, "benchmark_available", False):
        f.append("benchmark unavailable")
    return Verdict("data_auditor", passed=not f, findings=tuple(f))


def sceptic(ctx) -> Verdict:
    """Overfitting / fragility red flags."""
    f = []
    if int(_get(ctx, "num_trials", 1)) > int(_get(ctx, "max_trials", 20)):
        f.append("excessive multiple testing")
    if int(_get(ctx, "parameter_count", 0)) > int(_get(ctx, "max_parameters", 8)):
        f.append("too many free parameters")
    if _get(ctx, "post_hoc_rule_change", False):
        f.append("post-hoc rule change (not preregistered)")
    if _get(ctx, "regime_cherry_picked", False):
        f.append("regime cherry-picking")
    if float(_get(ctx, "top_symbol_weight", 0.0)) > float(_get(ctx, "max_symbol_weight", 0.35)):
        f.append("single-symbol concentration")
    if float(_get(ctx, "turnover", 0.0)) > float(_get(ctx, "max_turnover", 3.0)):
        f.append("excessive turnover")
    return Verdict("sceptic", passed=not f, findings=tuple(f))


def reality_checker(ctx) -> Verdict:
    """Existing statistical evidence gate (deflated Sharpe / Reality-Check p / FDR)."""
    f = []
    n = int(_get(ctx, "n_trades", 0))
    if n < int(_get(ctx, "min_trades", 30)):
        return Verdict("reality_checker", passed=False, findings=("insufficient sample",), blocking=False)
    if float(_get(ctx, "net_expectancy_R", 0.0)) <= 0.0:
        f.append("non-positive net expectancy after costs")
    dsr = _get(ctx, "deflated_sharpe", None)
    if dsr is not None and float(dsr) < float(_get(ctx, "min_deflated_sharpe", 0.5)):
        f.append("deflated Sharpe below threshold")
    rc_p = _get(ctx, "reality_check_p", None)
    if rc_p is not None and float(rc_p) > float(_get(ctx, "max_reality_check_p", 0.05)):
        f.append("fails White's Reality Check")
    if _get(ctx, "fdr_reject", False):
        f.append("rejected by FDR control")
    dd = float(_get(ctx, "max_drawdown_pct", 0.0))
    if dd > float(_get(ctx, "max_allowed_drawdown_pct", 30.0)):
        f.append("drawdown exceeds limit")
    if not _get(ctx, "walk_forward_ok", True):
        f.append("weak out-of-sample / walk-forward")
    return Verdict("reality_checker", passed=not f, findings=tuple(f))


def portfolio_examiner(ctx) -> Verdict:
    """Does it add diversity rather than duplicate existing paper risk?"""
    corr = float(_get(ctx, "max_correlation_to_deployed", 0.0))
    if corr >= float(_get(ctx, "max_allowed_correlation", 0.8)):
        return Verdict("portfolio_examiner", passed=False,
                       findings=("duplicates an already-deployed strategy's risk",), blocking=False)
    return Verdict("portfolio_examiner", passed=True)


@dataclass(frozen=True)
class CommitteeDecision:
    decision: str
    verdicts: tuple
    rationale: str


def promotion_committee(ctx, *, producer: str, committee_actor: str = "promotion_committee") -> CommitteeDecision:
    """Combine the independent verdicts into one decision. No self-approval."""
    if producer == committee_actor:
        return CommitteeDecision(REJECT, (), "a role cannot approve its own hypothesis")
    verdicts = (data_auditor(ctx), sceptic(ctx), reality_checker(ctx), portfolio_examiner(ctx))
    hard_fail = [v for v in verdicts if not v.passed and v.blocking]
    soft_fail = [v for v in verdicts if not v.passed and not v.blocking]

    if any(v.role == "reality_checker" and not v.passed and not v.blocking for v in verdicts):
        return CommitteeDecision(RETEST_WITH_MORE_DATA, verdicts,
                                 "insufficient sample — retest when more forward data exists")
    if hard_fail:
        reasons = "; ".join(x for v in hard_fail for x in v.findings)
        return CommitteeDecision(REJECT, verdicts, f"blocked: {reasons}")
    if soft_fail:
        reasons = "; ".join(x for v in soft_fail for x in v.findings)
        return CommitteeDecision(INCONCLUSIVE, verdicts, f"not disqualifying but weak: {reasons}")
    return CommitteeDecision(PAPER_NOMINATED, verdicts, "cleared all independent challenges")
