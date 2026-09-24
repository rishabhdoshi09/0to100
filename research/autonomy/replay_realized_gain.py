"""Derive and persist realized information gain for historical replay only.

Selection and realization remain separate immutable events.  This module never
creates forward evidence or execution authority; it merely closes a previously
journaled HISTORICAL_REPLAY acquisition after the replay executor has produced
its authoritative result.
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

    The executor must echo the acquisition identity.  Identity disagreement,
    forward evidence, or incomplete execution fails closed instead of silently
    attaching a result to the wrong acquisition.
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

    requested = _clean_metrics(selection.get("requested_metrics") or replay_result.get("requested_metrics"))
    produced = _clean_metrics(replay_result.get("metrics_produced"))
    metrics_closed = sorted(requested & produced)
    try:
        eligible_samples = max(0, int(replay_result.get("eligible_sample_count") or 0))
    except (TypeError, ValueError):
        raise ValueError("eligible_sample_count must be an integer")

    # Realized gain is evidence acquisition yield, never trade P&L.  One unit
    # per requested metric closed plus bounded sample yield makes the value
    # deterministic and safe for plateau/stopping laws.
    requested_deficit = max(0, int(replay_result.get("requested_sample_deficit") or 0))
    sample_gain = min(eligible_samples, requested_deficit) if requested_deficit else 0
    realized_gain = float(len(metrics_closed) + sample_gain)

    return record_realized_gain(
        selection,
        realized_information_gain=realized_gain,
        eligible_sample_count=eligible_samples,
        metrics_produced=sorted(produced),
        path=journal_path,
    )
