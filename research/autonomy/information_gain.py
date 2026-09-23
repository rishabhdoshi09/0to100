"""Deterministic acquisition ranking for historical replay.

This module ranks already-eligible point-in-time historical sessions by the
research evidence gap they can reduce.  It does not create forward evidence,
change production theses, or grant execution authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
EVIDENCE_ORIGIN = "HISTORICAL_REPLAY"


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReplayAcquisition:
    session_date: str
    score: float
    rationale: tuple[str, ...]
    evidence_origin: str
    request_id: str
    strategy_id: str
    thesis_hash: str
    universe_snapshot_id: str
    data_version: str
    feature_version: str
    model_version: str
    signal_registry_version: str
    decision_fingerprint: str
    acquisition_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


def _candidate_value(candidate: Mapping[str, Any], key: str) -> float:
    try:
        return max(0.0, float(candidate.get(key) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def rank_historical_sessions(
    request: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank immutable PIT candidates for an open historical evidence request.

    Candidates are fail-closed: a session is rankable only when immutable
    universe identity, version identities, thesis identity and a decision
    fingerprint are present.  Scores are deterministic and intentionally use
    only acquisition metadata supplied by the historical replay layer.
    """
    if str(request.get("status") or "").upper() != "OPEN":
        return []
    allowed = {str(x).upper() for x in request.get("allowed_lanes") or ()}
    if EVIDENCE_ORIGIN not in allowed:
        return []

    deficit = max(0, int(request.get("sample_deficit") or 0))
    missing = {str(x) for x in request.get("missing_metrics") or ()}
    request_id = str(request.get("request_id") or "")
    strategy_id = str(request.get("strategy_id") or "")
    thesis_hash = str(request.get("thesis_hash") or "")
    ranked: list[ReplayAcquisition] = []

    identity_keys = (
        "session_date", "universe_snapshot_id", "data_version", "feature_version",
        "model_version", "signal_registry_version", "decision_fingerprint",
    )
    for candidate in candidates:
        if any(not str(candidate.get(key) or "").strip() for key in identity_keys):
            continue
        candidate_thesis = str(candidate.get("thesis_hash") or "")
        if thesis_hash and candidate_thesis != thesis_hash:
            continue
        candidate_strategy = str(candidate.get("strategy_id") or "")
        if strategy_id and candidate_strategy and candidate_strategy != strategy_id:
            continue

        rationale: list[str] = []
        score = 0.0
        if deficit > 0:
            sample_yield = min(float(deficit), _candidate_value(candidate, "eligible_sample_count"))
            if sample_yield > 0:
                score += sample_yield / float(max(1, deficit))
                rationale.append(f"sample_deficit:{int(sample_yield)}")
        supplied_metrics = {str(x) for x in candidate.get("metrics_available") or ()}
        metric_gain = len(missing.intersection(supplied_metrics))
        if metric_gain:
            score += float(metric_gain)
            rationale.append(f"missing_metrics:{metric_gain}")
        regime_novelty = min(1.0, _candidate_value(candidate, "regime_novelty"))
        if regime_novelty:
            score += 0.25 * regime_novelty
            rationale.append("regime_novelty")
        sector_novelty = min(1.0, _candidate_value(candidate, "sector_novelty"))
        if sector_novelty:
            score += 0.15 * sector_novelty
            rationale.append("sector_novelty")
        uncertainty = min(1.0, _candidate_value(candidate, "decision_uncertainty"))
        if uncertainty:
            score += 0.25 * uncertainty
            rationale.append("decision_uncertainty")
        if score <= 0.0:
            continue

        stable = {
            "request_id": request_id,
            "session_date": str(candidate["session_date"])[:10],
            "strategy_id": strategy_id,
            "thesis_hash": candidate_thesis or thesis_hash,
            "universe_snapshot_id": str(candidate["universe_snapshot_id"]),
            "data_version": str(candidate["data_version"]),
            "feature_version": str(candidate["feature_version"]),
            "model_version": str(candidate["model_version"]),
            "signal_registry_version": str(candidate["signal_registry_version"]),
            "decision_fingerprint": str(candidate["decision_fingerprint"]),
        }
        ranked.append(ReplayAcquisition(
            session_date=stable["session_date"],
            score=round(score, 6),
            rationale=tuple(rationale),
            evidence_origin=EVIDENCE_ORIGIN,
            request_id=request_id,
            strategy_id=strategy_id,
            thesis_hash=stable["thesis_hash"],
            universe_snapshot_id=stable["universe_snapshot_id"],
            data_version=stable["data_version"],
            feature_version=stable["feature_version"],
            model_version=stable["model_version"],
            signal_registry_version=stable["signal_registry_version"],
            decision_fingerprint=stable["decision_fingerprint"],
            acquisition_fingerprint="acq_" + _stable_hash(stable)[:20],
        ))

    ranked.sort(key=lambda item: (-item.score, item.session_date, item.acquisition_fingerprint))
    return [item.as_dict() for item in ranked]
