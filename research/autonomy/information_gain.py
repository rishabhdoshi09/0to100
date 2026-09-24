"""Deterministic acquisition ranking for historical replay.

Ranks already-eligible point-in-time historical sessions by the research
evidence gap they can reduce. It cannot create forward evidence or execution
authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
EVIDENCE_ORIGIN = "HISTORICAL_REPLAY"


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


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


def _value(candidate: Mapping[str, Any], key: str) -> float:
    try:
        return max(0.0, float(candidate.get(key) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def rank_historical_sessions(
    request: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if str(request.get("status") or "").upper() != "OPEN":
        return []
    if EVIDENCE_ORIGIN not in {
        str(x).upper() for x in request.get("allowed_lanes") or ()
    }:
        return []

    deficit = max(0, int(request.get("sample_deficit") or 0))
    missing = {str(x) for x in request.get("missing_metrics") or ()}
    request_id = str(request.get("request_id") or "")
    strategy_id = str(request.get("strategy_id") or "")
    thesis_hash = str(request.get("thesis_hash") or "")
    if not request_id or not strategy_id or not thesis_hash:
        return []

    ranked = []
    identity_keys = (
        "session_date",
        "strategy_id",
        "thesis_hash",
        "universe_snapshot_id",
        "data_version",
        "feature_version",
        "model_version",
        "signal_registry_version",
        "decision_fingerprint",
    )
    for candidate in candidates:
        if any(not str(candidate.get(key) or "").strip() for key in identity_keys):
            continue
        candidate_thesis = str(candidate["thesis_hash"])
        candidate_strategy = str(candidate["strategy_id"])
        if candidate_thesis != thesis_hash or candidate_strategy != strategy_id:
            continue

        rationale = []
        score = 0.0
        if deficit:
            sample_yield = min(float(deficit), _value(candidate, "eligible_sample_count"))
            if sample_yield:
                score += sample_yield / max(1, deficit)
                rationale.append(f"sample_deficit:{int(sample_yield)}")
        gain = len(missing & {str(x) for x in candidate.get("metrics_available") or ()})
        if gain:
            score += gain
            rationale.append(f"missing_metrics:{gain}")
        for key, weight in (
            ("regime_novelty", 0.25),
            ("sector_novelty", 0.15),
            ("decision_uncertainty", 0.25),
        ):
            value = min(1.0, _value(candidate, key))
            if value:
                score += weight * value
                rationale.append(key)
        if score <= 0:
            continue

        stable = {
            "request_id": request_id,
            "session_date": str(candidate["session_date"])[:10],
            "strategy_id": strategy_id,
            "thesis_hash": thesis_hash,
            "universe_snapshot_id": str(candidate["universe_snapshot_id"]),
            "data_version": str(candidate["data_version"]),
            "feature_version": str(candidate["feature_version"]),
            "model_version": str(candidate["model_version"]),
            "signal_registry_version": str(candidate["signal_registry_version"]),
            "decision_fingerprint": str(candidate["decision_fingerprint"]),
        }
        ranked.append(
            ReplayAcquisition(
                stable["session_date"],
                round(score, 6),
                tuple(rationale),
                EVIDENCE_ORIGIN,
                request_id,
                strategy_id,
                thesis_hash,
                stable["universe_snapshot_id"],
                stable["data_version"],
                stable["feature_version"],
                stable["model_version"],
                stable["signal_registry_version"],
                stable["decision_fingerprint"],
                "acq_" + _stable_hash(stable)[:20],
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.session_date, item.acquisition_fingerprint))
    return [item.as_dict() for item in ranked]
