"""Runtime bridge from historical-paper execution to information-gain research.

The acquisition planner and stopping law are intentionally pure research
primitives. This module binds them to the closed-market historical-paper lane
without granting forward-paper or live-money authority.

Selection is recorded before replay. Realized gain is appended only after an
authoritative SUCCEEDED HISTORICAL_REPLAY run and is measured from settled
historical virtual-paper samples, never from PnL or winner/loser outcomes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.autonomy.acquisition_journal import records_for_request
from research.autonomy.acquisition_planner import plan_historical_acquisitions
from research.autonomy.replay_realized_gain import persist_realized_replay_gain
from research.autonomy.stopping_law import evaluate_realized_gain

EVIDENCE_ORIGIN = "HISTORICAL_REPLAY"
_REPLAY_RESEARCH_METRICS = frozenset({
    "deflated_sharpe",
    "reality_check_p",
    "walk_forward_ok",
    "fdr_significant",
})


def _stable_hash(payload: Mapping[str, Any]) -> str:
    material = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _canonical_identity(
    *,
    thesis_hash: str,
    universe_limit: int,
    calibration_snapshot: Mapping[str, Any] | None = None,
    data_version: str | None = None,
    policy_versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    thesis = str(thesis_hash or "").strip()
    if not thesis:
        raise ValueError("production thesis identity unavailable")

    snapshot = dict(calibration_snapshot or {})
    if not snapshot:
        from scan.calibration_snapshot import get_or_create_snapshot

        snapshot = dict(get_or_create_snapshot() or {})
    identities = dict(snapshot.get("identities") or {})

    data_id = str(data_version or "").strip()
    if not data_id:
        from product.pit_warehouse import warehouse_fingerprint

        data_id = str(warehouse_fingerprint() or "").strip()

    versions = dict(policy_versions or {})
    if not versions:
        from product.pit_versions import current_versions

        versions = dict(current_versions().as_dict())

    feature_version = str(identities.get("feature_version") or "").strip()
    model_version = str(identities.get("model_version") or "").strip()
    signal_registry_version = str(identities.get("signal_registry_version") or "").strip()
    calibration_snapshot_id = str(snapshot.get("snapshot_id") or "").strip()

    missing = [
        key
        for key, value in (
            ("data_version", data_id),
            ("feature_version", feature_version),
            ("model_version", model_version),
            ("signal_registry_version", signal_registry_version),
            ("calibration_snapshot_id", calibration_snapshot_id),
        )
        if not value
    ]
    if missing:
        raise ValueError("immutable replay identity unavailable: " + ",".join(missing))

    # Persist the exact canonical signal definitions referenced by the replay.
    from scan.signal_registry_versions import persist_canonical_version

    registry = dict(persist_canonical_version() or {})
    if str(registry.get("registry_version") or "") != signal_registry_version:
        raise ValueError("signal-registry version does not match calibration snapshot")

    universe_context = {
        "data_version": data_id,
        "policy_versions": versions,
        "universe_limit": int(universe_limit),
        "membership_rule": "official_bar_exactly_on_session",
    }
    return {
        "thesis_hash": thesis,
        "data_version": data_id,
        "feature_version": feature_version,
        "model_version": model_version,
        "signal_registry_version": signal_registry_version,
        "calibration_snapshot_id": calibration_snapshot_id,
        "policy_versions": versions,
        "universe_context_fingerprint": _stable_hash(universe_context),
    }


def _candidate_rows(
    request: Mapping[str, Any],
    sessions: Sequence[str],
    *,
    identity: Mapping[str, Any],
    curriculum: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    request_id = str(request.get("request_id") or "").strip()
    strategy_id = str(request.get("strategy_id") or "").strip()
    thesis_hash = str(identity.get("thesis_hash") or "").strip()
    if not request_id or not strategy_id or not thesis_hash:
        return []

    curriculum = dict(curriculum or {})
    states = dict(curriculum.get("session_states") or {})
    coverage = dict(curriculum.get("coverage_before") or {})
    missing = {
        str(value)
        for value in (request.get("missing_metrics") or ())
        if str(value) in _REPLAY_RESEARCH_METRICS
    }

    rows: list[dict[str, Any]] = []
    for value in sessions:
        day = str(value or "")[:10]
        if len(day) != 10:
            continue
        universe_snapshot_id = "unisnap_" + _stable_hash({
            "session_date": day,
            "universe_context_fingerprint": identity["universe_context_fingerprint"],
        })[:20]
        decision_fingerprint = "replayctx_" + _stable_hash({
            "request_id": request_id,
            "session_date": day,
            "strategy_id": strategy_id,
            "thesis_hash": thesis_hash,
            "universe_snapshot_id": universe_snapshot_id,
            "data_version": identity["data_version"],
            "feature_version": identity["feature_version"],
            "model_version": identity["model_version"],
            "signal_registry_version": identity["signal_registry_version"],
        })[:20]

        state = dict(states.get(day) or {})
        regime = str(state.get("regime") or "")
        regime_novelty = 0.0
        if state.get("available") and regime:
            regime_novelty = 1.0 / (1.0 + max(0, int(coverage.get(regime) or 0)))

        rows.append({
            "session_date": day,
            "strategy_id": strategy_id,
            "thesis_hash": thesis_hash,
            "universe_snapshot_id": universe_snapshot_id,
            "data_version": identity["data_version"],
            "feature_version": identity["feature_version"],
            "model_version": identity["model_version"],
            "signal_registry_version": identity["signal_registry_version"],
            # This is a pre-execution decision-context fingerprint. Actual
            # per-symbol decision fingerprints remain in the replay ledger.
            "decision_fingerprint": decision_fingerprint,
            # One settleable session is one ex-ante acquisition unit. Realized
            # useful samples are measured after paper-book settlement.
            "eligible_sample_count": 1,
            "metrics_available": sorted(missing),
            "regime_novelty": regime_novelty,
            "sector_novelty": 0.0,
            "decision_uncertainty": 0.0,
        })
    return rows


def plan_runtime_acquisitions(
    request: Mapping[str, Any],
    sessions: Sequence[str],
    *,
    thesis_hash: str,
    universe_limit: int,
    curriculum: Mapping[str, Any] | None = None,
    batch_size: int = 8,
    journal_path: str | Path | None = None,
    calibration_snapshot: Mapping[str, Any] | None = None,
    data_version: str | None = None,
    policy_versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan and journal an information-gain batch before replay starts."""
    req = dict(request or {})
    request_id = str(req.get("request_id") or "").strip()
    if not request_id:
        return {
            "sessions": [],
            "acquisitions": [],
            "selection_policy": "INFORMATION_GAIN",
            "reason": "missing_evidence_request_identity",
            "stop": False,
        }

    bound_thesis = str(thesis_hash or "").strip()
    requested_thesis = str(req.get("thesis_hash") or "").strip()
    if requested_thesis and requested_thesis != bound_thesis:
        return {
            "sessions": [],
            "acquisitions": [],
            "selection_policy": "INFORMATION_GAIN",
            "reason": "evidence_request_thesis_mismatch",
            "stop": False,
        }
    if not bound_thesis:
        return {
            "sessions": [],
            "acquisitions": [],
            "selection_policy": "INFORMATION_GAIN",
            "reason": "production_thesis_identity_unavailable",
            "stop": False,
        }
    # Older/open requests may predate thesis binding. Bind the execution copy
    # to the current production thesis; the durable request id is unchanged.
    req["thesis_hash"] = bound_thesis

    existing = records_for_request(request_id, path=journal_path)
    # A legacy request may predate explicit thesis binding. Never let realized
    # yield from an older production thesis stop a replay investigation under
    # the current thesis merely because the durable request id is unchanged.
    current_fingerprints = {
        str(row.get("acquisition_fingerprint") or "")
        for row in existing
        if str(row.get("event") or "").upper() == "SELECTED"
        and str(row.get("thesis_hash") or "") == bound_thesis
        and str(row.get("strategy_id") or "") == str(req.get("strategy_id") or "")
    }
    stopping_records = [
        row
        for row in existing
        if str(row.get("event") or "").upper() != "REALIZED"
        or str(row.get("acquisition_fingerprint") or "") in current_fingerprints
    ]
    stopping = evaluate_realized_gain(stopping_records, request_id=request_id)
    if stopping.stop:
        return {
            "sessions": [],
            "acquisitions": [],
            "selection_policy": "INFORMATION_GAIN",
            "evidence_origin": EVIDENCE_ORIGIN,
            "reason": "realized_information_gain_plateau",
            "stop": True,
            "stopping_reason": stopping.reason,
            "stopping_observations": stopping.observations,
            "recent_yield": stopping.recent_yield,
            "prior_yield": stopping.prior_yield,
            "zero_yield_streak": stopping.zero_yield_streak,
        }

    identity = _canonical_identity(
        thesis_hash=bound_thesis,
        universe_limit=universe_limit,
        calibration_snapshot=calibration_snapshot,
        data_version=data_version,
        policy_versions=policy_versions,
    )
    candidates = _candidate_rows(req, sessions, identity=identity, curriculum=curriculum)
    planned = plan_historical_acquisitions(
        req,
        candidates,
        batch_size=max(0, int(batch_size)),
        journal_path=journal_path,
    )
    return {
        **planned,
        "stop": False,
        "calibration_snapshot_id": identity["calibration_snapshot_id"],
        "data_version": identity["data_version"],
        "feature_version": identity["feature_version"],
        "model_version": identity["model_version"],
        "signal_registry_version": identity["signal_registry_version"],
        "request_thesis_inferred": not bool(requested_thesis),
    }


def persist_runtime_realized_gain(
    acquisitions: Sequence[Mapping[str, Any]],
    *,
    replay_report: Mapping[str, Any],
    trades: Sequence[Mapping[str, Any]],
    journal_path: str | Path | None = None,
) -> dict[str, Any]:
    """Close selected acquisitions from authoritative settled replay evidence."""
    selected = [dict(row) for row in acquisitions if isinstance(row, Mapping)]
    if not selected:
        return {"records": [], "reason": "no_journaled_acquisitions"}

    report = dict(replay_report or {})
    status = str(report.get("status") or "").upper()
    origin = str(report.get("provenance") or report.get("evidence_class") or "").upper()
    if status != "SUCCEEDED":
        return {"records": [], "reason": "replay_not_authoritative_success", "status": status}
    if origin != EVIDENCE_ORIGIN:
        raise ValueError("authoritative replay result must be HISTORICAL_REPLAY")

    by_session: dict[str, int] = {}
    for raw in trades:
        row = dict(raw or {})
        day = str(
            row.get("entry_date")
            or row.get("as_of")
            or row.get("decision_as_of")
            or ""
        )[:10]
        if len(day) == 10:
            by_session[day] = by_session.get(day, 0) + 1

    persisted = []
    for acquisition in selected:
        day = str(acquisition.get("session_date") or "")[:10]
        useful = max(0, int(by_session.get(day, 0)))
        realized = {
            **acquisition,
            "status": "SUCCEEDED",
            "evidence_origin": EVIDENCE_ORIGIN,
            "eligible_sample_count": useful,
            "metrics_produced": (
                ["historical_virtual_paper_sample"] if useful else []
            ),
        }
        persisted.append(
            persist_realized_replay_gain(
                acquisition,
                realized,
                journal_path=journal_path,
            )
        )
    return {
        "records": persisted,
        "reason": "persisted",
        "eligible_samples": sum(int(row.get("eligible_samples") or 0) for row in persisted),
    }
