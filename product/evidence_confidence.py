"""Evidence confidence ladder: reproduced history -> trustworthy forward paper.

The score exposed here is an evidence-strength composite, not a win
probability. Positive forward confirmation is accepted only from taken-paper
sources that the learning policy marked selection-eligible. Conservative
negative gross evidence may still decay confidence.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _setup(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("setup")
        or candidate.get("setup_label")
        or candidate.get("primary_thesis")
        or ""
    ).strip()


def confidence_from_policies(
    candidate: Mapping[str, Any],
    policies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    setup = _setup(candidate)
    hist = next(
        (dict(p) for p in policies if str(p.get("policy_id") or "") == f"HIST_SETUP::{setup}"),
        {},
    )
    raw_forward = next(
        (dict(p) for p in policies if str(p.get("policy_id") or "") == f"SETUP::{setup}"),
        {},
    )

    hist_ready = bool(hist.get("historical_reproduced_positive"))
    hist_score = float(hist.get("historical_confidence_score") or 0.0)

    source = str(raw_forward.get("evidence_source") or "")
    is_taken_forward = source.startswith("paper_forward_taken")
    raw_n = int(raw_forward.get("sample_size") or 0)
    raw_edge = float(raw_forward.get("expectancy_difference_R") or 0.0)
    affects = raw_forward.get("affects_selection") is not False

    # Positive confidence is allowed only from actual taken-paper evidence whose
    # integrity contract permits it to affect selection. Gross-only positive
    # evidence stays observation-only. Negative gross evidence remains a
    # conservative upper bound and is allowed to reduce confidence.
    trusted_positive = bool(is_taken_forward and affects and raw_edge > 0)
    trusted_negative = bool(is_taken_forward and raw_edge < 0)
    forward_usable = bool(trusted_positive or trusted_negative)
    forward_n = raw_n if forward_usable else 0
    forward_edge = raw_edge if forward_usable else 0.0

    forward_sample = min(1.0, forward_n / 30.0)
    edge_component = math.tanh(forward_edge / 0.50) if forward_n else 0.0
    forward_score = max(
        0.0,
        min(100.0, 50.0 + 30.0 * edge_component + 20.0 * forward_sample),
    )

    if not hist_ready:
        combined = min(49.0, hist_score)
        stage = "HISTORICAL_UNPROVEN"
    elif forward_n <= 0:
        combined = min(79.0, hist_score)
        stage = (
            "FORWARD_EVIDENCE_UNTRUSTED"
            if raw_n and not forward_usable
            else "HISTORICAL_BASE"
        )
    else:
        forward_weight = min(0.70, 0.20 + 0.50 * forward_sample)
        combined = hist_score * (1.0 - forward_weight) + forward_score * forward_weight
        if forward_n < 8:
            stage = "FORWARD_EARLY"
        elif forward_edge <= -0.20:
            stage = "FORWARD_DECAYED"
        elif forward_n < 20:
            stage = "FORWARD_CALIBRATING"
        else:
            stage = "FORWARD_CONFIRMED" if forward_edge > 0 else "FORWARD_WEAK"

    combined = round(max(0.0, min(95.0, combined)), 1)
    paper_eligible = bool(hist_ready)
    if forward_n >= 5 and forward_edge <= -0.25:
        paper_eligible = False

    return {
        "setup": setup,
        "historical_ready": hist_ready,
        "historical_n": int(hist.get("sample_size") or 0),
        "historical_mean_R": hist.get("expectancy_R"),
        "historical_splits": int(hist.get("splits_tested") or 0),
        "historical_positive_splits": int(hist.get("positive_splits") or 0),
        "historical_confidence_score": round(hist_score, 1),
        "forward_n": forward_n,
        "forward_observed_n": raw_n,
        "forward_mean_R": raw_forward.get("expectancy_R"),
        "forward_source": source,
        "forward_trusted_positive": trusted_positive,
        "forward_trusted_negative": trusted_negative,
        "forward_confidence_score": round(forward_score, 1) if forward_n else 0.0,
        "evidence_confidence_score": combined,
        "confidence_stage": stage,
        "paper_eligible": paper_eligible,
        "is_win_probability": False,
        "live_locked": True,
    }
