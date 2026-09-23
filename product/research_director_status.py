"""Single truth-first Research Director projection for the QuantTerm operator.

This is a read-only composition layer. It does not start replay, train a model,
change a policy, or infer missing evidence. Every field comes from persisted
research/autonomy artifacts or deterministic summaries of those artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from product.live_safety import live_safety_projection

SCHEMA_VERSION = 1
IST = ZoneInfo("Asia/Kolkata")


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def _today_ist() -> str:
    return datetime.now(IST).date().isoformat()


def _event_is_today(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(IST).date().isoformat() == _today_ist()
    except Exception:
        return text[:10] == _today_ist()


def _request() -> dict[str, Any]:
    from research.autonomy.evidence_acquisition import load_request
    return dict(load_request() or {})


def _progress(request_id: str) -> dict[str, Any]:
    if not request_id:
        return {}
    from research.autonomy.evidence_progress import load_progress
    return dict(load_progress(request_id) or {})


def _historical_state() -> dict[str, Any]:
    from product.historical_paper_loop import load_state
    return dict(load_state() or {})


def _autonomy() -> dict[str, Any]:
    from product.autonomy_status import read_autonomy_status
    return dict(read_autonomy_status() or {})


def _calibration() -> dict[str, Any]:
    from scan.calibration_snapshot import load_current
    return dict(load_current() or {})


def _decision_calibration() -> dict[str, Any]:
    from product.decision_calibration import DecisionCalibrationEngine
    return dict(DecisionCalibrationEngine().dossier() or {})


def _signal_registry() -> dict[str, Any]:
    from scan.signal_registry import load_registry
    return dict(load_registry() or {})


def _learned_challenger() -> dict[str, Any]:
    from product.challenger_learning import load
    return dict(load() or {})


def _rule_challengers() -> dict[str, Any]:
    from product.champion_challenger import load_store
    return dict(load_store() or {})


def _research_overview() -> dict[str, Any]:
    from research.research_overview import overview, knowledge_growth
    out = dict(overview() or {})
    out["knowledge_growth_1d"] = dict(knowledge_growth(1) or {})
    return out


def _forward_soak() -> dict[str, Any]:
    from product.forward_soak import scoreboard
    return dict(scoreboard() or {})


def _activity_phase(activity: str, request: Mapping[str, Any], progress: Mapping[str, Any]) -> str:
    act = str(activity or "IDLE").upper()
    status = str(progress.get("status") or request.get("status") or "").upper()
    if status == "PLATEAUED":
        return "PLATEAUED"
    if status == "SATISFIED":
        return "SATISFIED"
    if status == "BLOCKED":
        return "BLOCKED"
    if act == "HISTORICAL_REPLAY":
        return "ACQUIRING_HISTORICAL_EVIDENCE"
    if act == "LEARNING":
        return "LEARNING"
    if act == "RESEARCH":
        return "RESEARCH_EVALUATION"
    if act == "FORWARD_PAPER":
        return "COLLECTING_FORWARD_EVIDENCE"
    return "IDLE" if not request else "WAITING_FOR_NEXT_EVIDENCE_STEP"


def _learning_delta(progress: Mapping[str, Any], overview: Mapping[str, Any]) -> dict[str, Any]:
    history = list(progress.get("history") or [])
    last = dict(history[-1]) if history else {}
    event_at = str(last.get("recorded_at") or progress.get("updated_at") or "")
    today = _event_is_today(event_at)
    kg = dict(overview.get("knowledge_growth_1d") or {})
    knowledge_changed = bool(
        int(kg.get("validated_in_window") or 0)
        or int(kg.get("retired_in_window") or 0)
    )
    evidence_added = int(last.get("samples_acquired") or progress.get("last_batch_yield") or 0)
    resolved = list(progress.get("resolved_metrics") or [])
    measurable = bool((today and (evidence_added > 0 or resolved)) or knowledge_changed)
    return {
        "measurable_change_today": measurable,
        "latest_evidence_event_at": event_at,
        "latest_event_is_today": today,
        "last_batch_evidence_added": evidence_added,
        "sample_count": int(progress.get("sample_count") or 0),
        "target_samples": int(progress.get("target_samples") or 0),
        "sample_deficit": int(progress.get("sample_deficit") or 0),
        "resolved_metrics": resolved,
        "unresolved_metrics": list(progress.get("unresolved_metrics") or []),
        "stagnant_batches": int(progress.get("stagnant_batches") or 0),
        "knowledge_validated_1d": int(kg.get("validated_in_window") or 0),
        "knowledge_retired_1d": int(kg.get("retired_in_window") or 0),
        "note": (
            "True only when persisted evidence/metric/knowledge state changed; "
            "cycle activity by itself does not count as learning."
        ),
    }


def build_research_director_status() -> dict[str, Any]:
    request = _safe(_request, {})
    request_id = str(request.get("request_id") or "")
    progress = _safe(lambda: _progress(request_id), {})
    historical = _safe(_historical_state, {})
    autonomy = _safe(_autonomy, {})
    calibration = _safe(_calibration, {})
    decision_calibration = _safe(_decision_calibration, {})
    registry = _safe(_signal_registry, {})
    learned = _safe(_learned_challenger, {})
    rules = _safe(_rule_challengers, {})
    overview = _safe(_research_overview, {})
    soak = _safe(_forward_soak, {})
    safety = live_safety_projection()

    activity_truth = dict(autonomy.get("activity_truth") or {})
    activity = str(
        autonomy.get("current_activity")
        or activity_truth.get("activity")
        or "UNKNOWN"
    )
    try:
        from research.autonomy.runtime_truth import transient_state_mismatch
        mismatch = transient_state_mismatch(
            str(autonomy.get("state") or ""),
            activity_truth,
        )
    except Exception:
        mismatch = {}

    selection = dict(historical.get("selection_details") or {})
    acquisition_tasks = list(request.get("acquisition_tasks") or [])
    current = dict(learned.get("current") or {})
    rule_challengers = list(rules.get("challengers") or [])
    rule_active = [
        row for row in rule_challengers
        if str(row.get("status") or "") not in {"REJECTED", "RETIRED", "PROMOTED"}
    ]

    question = str(
        request.get("diagnosis")
        or request.get("gap_kind")
        or ""
    )
    if not question:
        question = "No unresolved evidence request is currently persisted."

    next_action = str(
        progress.get("next_action")
        or (request.get("stop_conditions") or [""])[0]
        or ""
    )
    why_next = str(
        selection.get("selection_objective")
        or (acquisition_tasks[0] if acquisition_tasks else "")
        or "No targeted batch has been selected yet."
    )

    signal_summary = dict(registry.get("summary") or {})
    calibration_id = str(calibration.get("snapshot_id") or "")
    learning_delta = _learning_delta(progress, overview)

    blockers: list[str] = []
    if mismatch.get("mismatch"):
        blockers.append(str(mismatch.get("reason") or "runtime state/activity mismatch"))
    if request and str(request.get("status") or "").upper() == "PLATEAUED":
        blockers.append("Evidence request plateaued and needs a new research question.")
    if progress and str(progress.get("status") or "").upper() == "PLATEAUED":
        blockers.append(str(progress.get("reason") or "Evidence acquisition plateaued."))
    if not calibration_id:
        blockers.append("No immutable calibration snapshot is currently persisted.")
    if not registry:
        blockers.append("No persisted Signal Registry snapshot is available.")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_phase": _activity_phase(activity, request, progress),
        "current_activity": activity,
        "policy_state": str(autonomy.get("state") or "UNKNOWN"),
        "activity_truth": activity_truth,
        "state_truth": mismatch,
        "current_question": question,
        "evidence_request": {
            "request_id": request_id,
            "status": str(progress.get("status") or request.get("status") or "NONE"),
            "gap_kind": str(request.get("gap_kind") or ""),
            "evidence_origin": str(request.get("evidence_origin") or ""),
            "allowed_lanes": list(request.get("allowed_lanes") or []),
            "current_samples": int(request.get("current_samples") or 0),
            "target_samples": int(request.get("target_samples") or 0),
            "sample_deficit": int(
                progress.get("sample_deficit")
                if progress.get("sample_deficit") is not None
                else request.get("sample_deficit") or 0
            ),
            "missing_metrics": list(request.get("missing_metrics") or []),
            "acquisition_tasks": acquisition_tasks,
            "stop_conditions": list(request.get("stop_conditions") or []),
        },
        "evidence_progress": progress,
        "next_evidence_batch": {
            "batch_id": str(historical.get("current_batch_id") or ""),
            "phase": str(historical.get("phase") or "IDLE"),
            "sessions": list(historical.get("current_sessions") or []),
            "selection_policy": str(selection.get("selection_policy") or ""),
            "selection_objective": why_next,
            "outcome_blind_selection": selection.get("outcome_blind_selection"),
            "coverage_before": dict(selection.get("coverage_before") or {}),
            "coverage_after": dict(selection.get("coverage_after") or {}),
        },
        "next_action": next_action or "No explicit next action persisted.",
        "learning_delta": learning_delta,
        "calibration": {
            "snapshot_id": calibration_id,
            "identities": dict(calibration.get("identities") or {}),
            "immutable": calibration.get("immutable") is True,
        },
        "decision_calibration": {
            "settled_observations": int(decision_calibration.get("settled_observations") or 0),
            "explicit_probability_observations": int(
                decision_calibration.get("explicit_probability_observations") or 0
            ),
            "probability_coverage": float(decision_calibration.get("probability_coverage") or 0.0),
            "overall": dict(decision_calibration.get("overall") or {}),
            "buckets": dict(decision_calibration.get("buckets") or {}),
            "probability_drift": dict(decision_calibration.get("probability_drift") or {}),
            "affects_production": False,
            "live_locked": True,
            "truth_note": (
                "Hit-rate labels and explicit probability calibration are separate. "
                "Brier/miscalibration claims require their own explicit-probability sample floor."
            ),
        },
        "signals": {
            "registry_version": str(registry.get("registry_version") or ""),
            **signal_summary,
        },
        "challengers": {
            "learned": {
                "model_version": str(current.get("model_version") or ""),
                "status": str(current.get("status") or "NONE"),
                "trained_n": int(current.get("trained_n") or 0),
                "real_forward_n": int(current.get("real_forward_n") or 0),
                "promotion_dossier": dict(current.get("promotion_dossier") or {}),
            },
            "rule": {
                "under_evaluation": len(rule_active),
                "rows": rule_active[:8],
            },
        },
        "research_health": {
            "knowledge_growth": dict(overview.get("knowledge_growth") or {}),
            "edge_health": dict(overview.get("edge_health") or {}),
            "gate_scorecard": list(overview.get("gate_scorecard") or [])[:12],
            "data_health": dict(overview.get("data_health") or {}),
            "research_debt": dict(overview.get("research_debt") or {}),
        },
        "forward_evidence": {
            "status": str(soak.get("FORWARD_SOAK_STATUS") or "NOT_STARTED"),
            "real_forward_observations": int(soak.get("real_forward_observations") or 0),
            "settled_trades": int(soak.get("settled_trades") or 0),
            "rejected_candidates_settled": int(soak.get("rejected_candidates_settled") or 0),
            "missed_winners": int(soak.get("missed_winners") or 0),
            "avoided_losers": int(soak.get("avoided_losers") or 0),
        },
        "resource_governor": dict(
            activity_truth.get("resource_governor")
            or autonomy.get("resource_governor")
            or {}
        ),
        "blockers": blockers,
        **safety,
        "truth_note": (
            "Read-only composition of persisted evidence. Activity without a changed "
            "evidence/knowledge state is not counted as learning."
        ),
    }
