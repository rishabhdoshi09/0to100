"""Safe research experiment queue.

Experiments arise from observed decision questions, not random strategy mining.
Each item is LEVEL 1/2 only. No p-hacking loop. Promotion stays in existing
walk-forward / Reality Check / DSR / PSR / promotion governance.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "logs" / "product" / "experiment_queue.jsonl"

SAFE_QUESTIONS = frozenset({
    "EXTENDED_THRESHOLD",
    "SECTOR_SUPPORT_EXPECTANCY",
    "EVIDENCE_COVERAGE_BUCKET",
    "FAMILY_AGREEMENT",
    "MISSED_REENTRY_WAKE",
    "RESEARCH_TYPE_VALUE",
})


def enqueue(
    *,
    hypothesis: str,
    question_kind: str,
    population: str,
    method: str = "walk_forward_observe",
    pit_required: bool = True,
    triggering_n: int = 0,
    path: Path | None = None,
) -> dict[str, Any]:
    kind = str(question_kind or "").upper()
    if kind not in SAFE_QUESTIONS:
        raise ValueError(f"refusing unconstrained experiment kind {kind}")
    row = {
        "at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": hypothesis,
        "question_kind": kind,
        "population": population,
        "method": method,
        "pit_required": bool(pit_required),
        "result": None,
        "sample_size": 0,
        "triggering_n": int(triggering_n),
        "statistical_evidence": None,
        "production_implication": "OBSERVE_ONLY",
        "promotion_state": "HYPOTHESIS",
        "p_hacking": False,
        "live_locked": True,
    }
    target = path or QUEUE
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def from_failures(attributions: list[Mapping[str, Any]], *, path: Path | None = None) -> list[dict[str, Any]]:
    """Turn repeated observed patterns into SAFE research questions only."""
    out: list[dict[str, Any]] = []
    missed = [a for a in attributions if str(a.get("wait_attribution") or "") == "MISSED_REENTRY"]
    if len(missed) >= 8:
        out.append(enqueue(
            hypothesis=(
                "Does the wake logic fail after a healthy contraction following WAIT_EXTENDED?"
            ),
            question_kind="MISSED_REENTRY_WAKE",
            population="WAIT_EXTENDED matured opportunities",
            triggering_n=len(missed),
            path=path,
        ))
    overstrict = [a for a in attributions if str(a.get("avoid_attribution") or "") == "OVERSTRICT_VETO"]
    if len(overstrict) >= 12:
        out.append(enqueue(
            hypothesis="A repeated veto fires on names that later offered a valid entry.",
            question_kind="FAMILY_AGREEMENT",
            population="AVOID rows labelled OVERSTRICT_VETO",
            triggering_n=len(overstrict),
            path=path,
        ))
    extended = [
        a for a in attributions
        if str(a.get("reason_code") or "") in {"ENTRY_TOO_EXTENDED", "EXTENDED"}
    ]
    if len(extended) >= 20:
        out.append(enqueue(
            hypothesis="The EXTENDED threshold may be excluding later-valid contractions.",
            question_kind="EXTENDED_THRESHOLD",
            population="WAIT/AVOID EXTENDED matured opportunities",
            triggering_n=len(extended),
            path=path,
        ))
    return out
