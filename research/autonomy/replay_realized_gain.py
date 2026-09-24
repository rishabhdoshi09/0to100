"""Validate and persist realized evidence yield for historical replay only.

Selection and realization remain separate immutable events. This module never
creates forward evidence or execution authority; it only closes a previously
selected HISTORICAL_REPLAY acquisition after authoritative replay completion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from research.autonomy.acquisition_journal import record_realized_gain

EVIDENCE_ORIGIN = "HISTORICAL_REPLAY"


def _clean_metrics(values: Sequence[Any] | None) -> set[str]:
    return {str(value).strip() for value in (values or ()) if str(value).strip()}


def persist_realized_replay_gain(
    selection: Mapping[str, Any],
    replay_result: Mapping[str, Any],
    *,
    journal_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist outcome-blind evidence yield from one completed replay.

    The executor must echo every immutable acquisition identity. Any identity
    disagreement, forward evidence, or non-success result fails closed instead
    of silently attaching evidence to the wrong acquisition.
    """
    if str(selection.get("evidence_origin") or "").upper() != EVIDENCE_ORIGIN:
        raise ValueError("selection must be HISTORICAL_REPLAY")
    if str(replay_result.get("evidence_origin") or "").upper() != EVIDENCE_ORIGIN:
        raise ValueError("replay result must be HISTORICAL_REPLAY")
    if str(replay_result.get("status") or "").upper() not in {"COMPLETED", "SUCCEEDED"}:
        raise ValueError("replay result is not terminal-success")

    identity_keys = (
        "request_id", "session_date", "strategy_id", "thesis_hash",
        "universe_snapshot_id", "data_version", "feature_version",
        "model_version", "signal_registry_version", "decision_fingerprint",
        "acquisition_fingerprint",
    )
    for key in identity_keys:
        expected = str(selection.get(key) or "").strip()
        actual = str(replay_result.get(key) or "").strip()
        if not expected or not actual or expected != actual:
            raise ValueError(f"replay identity mismatch: {key}")

    produced = _clean_metrics(replay_result.get("metrics_produced"))
    try:
        eligible_samples = max(0, int(replay_result.get("eligible_sample_count") or 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("eligible_sample_count must be an integer") from exc

    return record_realized_gain(
        selection,
        eligible_samples=eligible_samples,
        metrics_realized=sorted(produced),
        path=journal_path,
    )
