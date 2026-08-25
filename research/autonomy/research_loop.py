"""Durable, constrained learning → hypothesis → challenge → paper-nomination loop.

This module never generates executable code.  It diagnoses canonical paper outcomes, creates a new
``StrategySpec`` version inside the approved grammar, preregisters it before evaluation, invokes the
existing evaluator, challenges the evidence, and either records negative knowledge or legally
nominates the successor for paper evaluation.  Actual trades remain the responsibility of Brain 2,
the portfolio gate and PaperBook.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable
import os

from research.autonomy import hypotheses as HYP
from research.autonomy import challenge as CH
from research.autonomy.dialogue import (
    DialogueLog, Record, OBSERVATION, EVIDENCE_GAP, HYPOTHESIS, CHALLENGE_REQUEST,
    CHALLENGE_REPORT, EXPERIMENT_REGISTRATION, EXPERIMENT_RESULT, PROMOTION_PROPOSAL,
    LEARNING_UPDATE,
)


def _append(dialogue, record):
    if dialogue is not None:
        return dialogue.append(record)
    return record


def _registered_specs(brain):
    registry = getattr(brain, "strategy_registry", None)
    if registry is None:
        return []
    try:
        return list(registry.deployable_specs())
    except Exception:
        return []


def derive_diagnostics(brain) -> list[dict]:
    """Derive measured, deterministic strategy diagnostics from the canonical PaperBook."""
    book = brain.intel_book
    diagnostics = []
    for spec in _registered_specs(brain):
        rs = book.r_stats(spec.strategy_id)
        st = book.stats(spec.strategy_id)
        n = int(rs.get("n", 0))
        mean = float(rs.get("mean_R", 0.0))
        lower = float(rs.get("lower_R", 0.0))
        base = {
            "strategy_id": spec.strategy_id, "strategy_version": spec.version,
            "family": spec.family, "n_trades": n, "forward_expectancy_R": mean,
            "forward_lower_R": lower, "current_drawdown_pct": float(st.get("max_drawdown_pct", 0.0)),
            "data_available": True,
        }
        if n < 10:
            diagnostics.append({**base, "kind": "insufficient_sample",
                "diagnosis": f"Only {n} resolved forward trades; no reliable adaptation claim yet.",
                "economic_impact": 0.25, "confidence": 1.0, "data_mining_risk": 0.8})
            continue
        if mean <= 0:
            diagnostics.append({**base, "kind": "negative_forward_expectancy",
                "diagnosis": f"Forward expectancy is {mean:+.3f}R over {n} trades.",
                "economic_impact": min(1.0, abs(mean) + 0.4), "confidence": min(1.0, n / 40),
                "data_mining_risk": 0.35})
        elif lower <= 0:
            diagnostics.append({**base, "kind": "poor_calibration",
                "diagnosis": f"Mean is positive ({mean:+.3f}R) but lower estimate is {lower:+.3f}R.",
                "economic_impact": 0.45, "confidence": min(1.0, n / 40), "data_mining_risk": 0.55})
        if float(st.get("max_drawdown_pct", 0.0)) >= 20.0:
            diagnostics.append({**base, "kind": "drawdown_pressure",
                "diagnosis": f"Paper drawdown reached {st.get('max_drawdown_pct', 0.0):.2f}%.",
                "economic_impact": 0.8, "confidence": 0.9, "data_mining_risk": 0.25})
        runtime_st = getattr(brain, "runtime_state", None)
        if runtime_st is not None:
            state = runtime_st.get(spec.strategy_id, spec.family)
            if state.unsupported_runtime:
                diagnostics.append({**base, "kind": "unsupported_runtime_family",
                    "diagnosis": f"Family {spec.family} has no supported runtime adapter.",
                    "economic_impact": 0.9, "confidence": 1.0, "data_mining_risk": 0.0,
                    "data_available": False})
    return diagnostics


def run_learning(brain, *, session_date: str, dialogue=None) -> dict:
    """Fold resolved paper outcomes into family knowledge and emit evidence gaps."""
    diagnostics = derive_diagnostics(brain)
    specs = {s.strategy_id: s for s in _registered_specs(brain)}
    for d in diagnostics:
        spec = specs.get(d["strategy_id"])
        _append(dialogue, Record(
            record_type=OBSERVATION, producer="learning_brain",
            consumer="research_planner", as_of=session_date,
            strategy_id=d["strategy_id"], strategy_version=d.get("strategy_version", 0),
            claim=d["diagnosis"], evidence=d, requested_action="DIAGNOSE"))
        if spec and d.get("n_trades", 0) >= 10:
            mean = float(d.get("forward_expectancy_R", 0.0))
            lower = float(d.get("forward_lower_R", 0.0))
            verdict = "CONFIRMED" if lower > 0 else ("WEAKER_POSITIVE" if mean > 0 else "DECAYED")
            brain.knowledge.remember_forward(spec.family, mean, verdict)
    try:
        brain.knowledge.save()
    except Exception:
        pass

    paper_memory: dict = {}
    try:
        from product.paper_learning import remember_paper_book
        book = getattr(brain, "intel_book", None)
        closed = list(getattr(book, "closed", []) or []) if book is not None else []
        paper_memory = remember_paper_book(closed, as_of=session_date)
    except Exception:
        paper_memory = {}
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from product.paper_self_feed import fold_latest_into_memory

            paper_memory = fold_latest_into_memory(paper_memory)
        except Exception:
            pass

    gaps = HYP.plan_gaps(diagnostics)
    for gap in gaps:
        _append(dialogue, Record(
            record_type=EVIDENCE_GAP, producer="research_planner", consumer="hypothesis_engine",
            as_of=session_date, strategy_id=gap.strategy_id, claim=gap.diagnosis,
            evidence=asdict(gap), requested_action=gap.recommended_action))
    paper_cooldown = len(paper_memory.get("cooldown") or [])
    paper_prefer = len(paper_memory.get("prefer") or [])
    paper_closed = int(paper_memory.get("closed_trades") or 0)
    _append(dialogue, Record(
        record_type=LEARNING_UPDATE, producer="learning_brain", consumer="supervisor",
        as_of=session_date, claim=f"Derived {len(diagnostics)} diagnostics and {len(gaps)} evidence gaps.",
        evidence={"diagnostics": len(diagnostics), "gaps": len(gaps),
                  "paper_cooldown": paper_cooldown, "paper_prefer": paper_prefer,
                  "paper_closed": paper_closed},
        decision="LEARNING_COMPLETE"))
    return {"session_date": session_date, "diagnostics": len(diagnostics), "gaps": len(gaps),
            "ranked_gaps": [asdict(g) for g in gaps],
            "paper_cooldown": paper_cooldown, "paper_prefer": paper_prefer,
            "paper_closed": paper_closed}


def _changes_for(parent, gap: HYP.EvidenceGap) -> dict:
    """One conservative, deterministic grammar change per diagnosed problem."""
    if gap.kind in ("negative_forward_expectancy", "poor_calibration"):
        hold = int(getattr(parent, "max_holding_days", 1))
        if hold > 2:
            return {"max_holding_days": max(1, int(round(hold * 0.75))),
                    "_why": "reduce exposure duration after weak forward outcomes"}
        return {"turnover_cap": max(0.1, float(getattr(parent, "turnover_cap", 1.0)) * 0.75),
                "_why": "reduce churn after weak forward outcomes"}
    if gap.kind == "drawdown_pressure":
        return {"max_positions": max(1, int(getattr(parent, "max_positions", 2)) - 1),
                "_why": "reduce concurrent exposure after drawdown pressure"}
    if gap.kind == "cost_drag":
        return {"max_holding_days": min(120, int(getattr(parent, "max_holding_days", 10)) * 2),
                "_why": "lower turnover and cost drag"}
    if gap.kind == "regime_specific_failure":
        current = tuple(getattr(parent, "regime_conditions", ()))
        return {"regime_conditions": tuple(sorted(set(current + ("RISK_ON",)))),
                "_why": "restrict deployment to the diagnosed supportive regime"}
    # Insufficient sample is not permission to mutate.  Keep tracking.
    return {}


def _evidence_context(report) -> dict:
    data = report.as_dict() if hasattr(report, "as_dict") else dict(report or {})
    # StrategyStudio EvidenceReport does not claim DSR/Reality-Check/walk-forward.  Do not fill
    # those fields with optimistic defaults; the committee will request more evidence.
    required = all(k in data for k in (
        "deflated_sharpe", "reality_check_p", "walk_forward_ok",
        "fdr_significant", "benchmark_available",
    ))
    return {
        "forward_eligible": not bool(data.get("invalid_data", False)) and not bool(data.get("is_synthetic", True)),
        "benchmark_available": bool(data.get("benchmark_available", False)),
        "n_trades": int(data.get("n_trades", 0)),
        "net_expectancy_R": float(data.get("net_expectancy_R", 0.0)),
        "deflated_sharpe": data.get("deflated_sharpe"),
        "reality_check_p": data.get("reality_check_p"),
        "walk_forward_ok": bool(data.get("walk_forward_ok", False)),
        "max_drawdown_pct": float(data.get("max_drawdown_pct", data.get("max_drawdown", 0.0))) *
                            (100.0 if float(data.get("max_drawdown_pct", 0.0) or 0.0) == 0.0 and
                             abs(float(data.get("max_drawdown", 0.0))) <= 1.0 else 1.0),
        "turnover": float(data.get("turnover", 0.0)),
        "top_symbol_weight": float(data.get("top_symbol_weight", data.get("max_symbol_share", 0.0))),
        "num_trials": int(data.get("num_trials", 1)),
        "parameter_count": int(data.get("parameter_count", 0)),
        "fdr_significant": bool(data.get("fdr_significant", False)),
        "required_evidence_complete": required,
        "raw": data,
    }


def _nominate_successor(brain, child) -> None:
    from research.strategy_studio import spec as LC
    # Verify every legal paper transition; the hard live door remains untouched.
    LC.require_transition(LC.GENERATED, LC.UNDER_REVIEW, "system")
    LC.require_transition(LC.UNDER_REVIEW, LC.PROMISING, "system")
    LC.require_transition(LC.PROMISING, LC.AWAITING_USER_APPROVAL, "system")
    LC.require_transition(LC.AWAITING_USER_APPROVAL, LC.APPROVED_FOR_PAPER, LC.PAPER_AUTOPILOT)
    brain.strategy_registry.replace_version(child)
    state = brain.runtime_state.get(child.strategy_id, child.family)
    state.lifecycle = LC.APPROVED_FOR_PAPER
    brain.runtime_state.save()


def execute_pipeline(brain, *, gap: HYP.EvidenceGap, parent, session_date: str,
                     dialogue=None, memory=None, experiment_runner: Callable | None = None) -> dict:
    """Execute one preregistered research proposal end-to-end."""
    if gap.recommended_action == "data_task":
        return {"decision": "DATA_TASK", "reason": gap.diagnosis}
    changes = _changes_for(parent, gap)
    if not changes:
        return {"decision": "RETEST_WITH_MORE_DATA", "reason": "no justified material mutation"}
    memory = memory or HYP.ResearchMemory()
    proposal, child_or_reason = HYP.propose_hypothesis(
        parent, gap, changes, memory=memory,
        expected_improvement="improve forward risk-adjusted expectancy without relaxing safety",
        failure_condition="net expectancy non-positive or required evidence incomplete",
        target_regime="diagnosed", dataset_requirements=("point_in_time_ohlcv", "benchmark"),
        research_budget=1)
    if proposal is None:
        return {"decision": "DUPLICATE_OR_INVALID", "reason": child_or_reason}
    child = child_or_reason
    hrec = _append(dialogue, Record(
        record_type=HYPOTHESIS, producer="hypothesis_engine", consumer="experiment_engine",
        as_of=session_date, strategy_id=child.strategy_id, strategy_version=child.version,
        claim=proposal.causal_explanation or proposal.observed_problem,
        evidence=asdict(proposal), requested_action="PREREGISTER"))
    prereg = _append(dialogue, Record(
        record_type=EXPERIMENT_REGISTRATION, producer="experiment_registry",
        consumer="experiment_engine", as_of=session_date,
        input_record_ids=(hrec.record_id,), strategy_id=child.strategy_id,
        strategy_version=child.version, claim="Experiment registered before results.",
        evidence={"hypothesis": asdict(proposal), "config_hash": child.config_hash()},
        decision="PREREGISTERED"))

    runner = experiment_runner
    if runner is None:
        runner = lambda spec: brain.evaluate_fn(spec, "validation") if brain.evaluate_fn else None
    report = runner(child)
    context = _evidence_context(report)
    result_rec = _append(dialogue, Record(
        record_type=EXPERIMENT_RESULT, producer="experiment_engine", consumer="research_council",
        as_of=session_date, input_record_ids=(prereg.record_id,), strategy_id=child.strategy_id,
        strategy_version=child.version, claim="Canonical experiment completed.",
        evidence=context["raw"], decision=str(context["raw"].get("verdict", "INCONCLUSIVE"))))
    _append(dialogue, Record(
        record_type=CHALLENGE_REQUEST, producer="experiment_engine", consumer="research_council",
        as_of=session_date, input_record_ids=(result_rec.record_id,), strategy_id=child.strategy_id,
        strategy_version=child.version, claim="Independently challenge the result."))
    decision = CH.promotion_committee(context, producer="hypothesis_engine")
    challenge_rec = _append(dialogue, Record(
        record_type=CHALLENGE_REPORT, producer="research_council", consumer="promotion_committee",
        as_of=session_date, input_record_ids=(result_rec.record_id,), strategy_id=child.strategy_id,
        strategy_version=child.version, claim=decision.rationale,
        evidence={"verdicts": [asdict(v) for v in decision.verdicts]}, decision=decision.decision))
    _append(dialogue, Record(
        record_type=PROMOTION_PROPOSAL, producer="promotion_committee", consumer="strategy_registry",
        as_of=session_date, input_record_ids=(challenge_rec.record_id,), strategy_id=child.strategy_id,
        strategy_version=child.version, claim=decision.rationale, decision=decision.decision))

    if decision.decision == CH.REJECT:
        memory.record_dead(proposal.semantic_hash, decision.rationale)
    elif decision.decision == CH.PAPER_NOMINATED:
        _nominate_successor(brain, child)
    return {"decision": decision.decision, "rationale": decision.rationale,
            "hypothesis_id": proposal.hypothesis_id, "strategy_id": child.strategy_id,
            "child_version": child.version, "config_hash": child.config_hash()}


def run_research_cycle(brain, *, session_date: str, dialogue=None) -> dict:
    diagnostics = derive_diagnostics(brain)
    gaps = HYP.plan_gaps(diagnostics)
    if not gaps:
        return {"decision": "NO_RESEARCH_GAP", "session_date": session_date}
    # Highest-priority gap only: bounded research budget, no indiscriminate parameter mining.
    gap = gaps[0]
    if gap.recommended_action == "data_task":
        return {"decision": "DATA_TASK", "reason": gap.diagnosis, "kind": gap.kind}
    parent = next((s for s in _registered_specs(brain) if s.strategy_id == gap.strategy_id), None)
    if parent is None:
        return {"decision": "NO_PARENT_STRATEGY", "strategy_id": gap.strategy_id}
    return execute_pipeline(brain, gap=gap, parent=parent, session_date=session_date,
                            dialogue=dialogue)
