"""Bounded, provenance-preserving planner for historical evidence acquisition.

This module is deliberately research-only. It ranks already eligible PIT replay
candidates against one open evidence request, records the selected acquisitions
before execution, and never creates forward/live authority.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from research.autonomy.acquisition_journal import record_selection
from research.autonomy.information_gain import rank_historical_sessions

EVIDENCE_ORIGIN = "HISTORICAL_REPLAY"


def plan_historical_acquisitions(
    request: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
    journal_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rank and durably record a bounded historical-replay acquisition batch.

    Empty/closed/ineligible requests return no work. Candidate identity and
    production-thesis matching remain delegated to the canonical ranker; the
    journal independently validates the immutable identity before persistence.
    """
    limit = max(0, int(batch_size))
    if limit == 0:
        return {
            "sessions": [],
            "acquisitions": [],
            "selection_policy": "INFORMATION_GAIN",
            "evidence_origin": EVIDENCE_ORIGIN,
            "reason": "batch_size_zero",
        }

    ranked = rank_historical_sessions(request, candidates)
    selected = ranked[:limit]
    persisted = [record_selection(row, path=journal_path) for row in selected]
    return {
        "sessions": [str(row["session_date"])[:10] for row in selected],
        "acquisitions": selected,
        "selection_policy": "INFORMATION_GAIN",
        "evidence_origin": EVIDENCE_ORIGIN,
        "outcome_blind_selection": True,
        "request_id": str(request.get("request_id") or ""),
        "ranked_candidates": len(ranked),
        "selected_count": len(selected),
        "selection_record_fingerprints": [row["record_fingerprint"] for row in persisted],
        "reason": "selected" if selected else "no_eligible_information_gain",
    }
