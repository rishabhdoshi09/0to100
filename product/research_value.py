"""Measure whether deep research changed a decision.

For each acquire job: decision before, missing evidence, research acquired,
decision after. Classify the delta. Never invent a research edge.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "logs" / "product" / "research_value.jsonl"

DECISION_CHANGED = "DECISION_CHANGED"
CONFIDENCE_CHANGED = "CONFIDENCE_CHANGED"
VETO_DISCOVERED = "VETO_DISCOVERED"
NO_MATERIAL_CHANGE = "NO_MATERIAL_CHANGE"


def classify_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    if str(before.get("decision") or "") != str(after.get("decision") or ""):
        return DECISION_CHANGED
    before_veto = {str(v.get("code") if isinstance(v, Mapping) else v) for v in (before.get("vetoes") or [])}
    after_veto = {str(v.get("code") if isinstance(v, Mapping) else v) for v in (after.get("vetoes") or [])}
    if after_veto - before_veto:
        return VETO_DISCOVERED
    if int(after.get("effective_confirmation_count") or 0) != int(before.get("effective_confirmation_count") or 0):
        return CONFIDENCE_CHANGED
    if str(after.get("information_value") or "") != str(before.get("information_value") or ""):
        return CONFIDENCE_CHANGED
    return NO_MATERIAL_CHANGE


def record_research_effect(
    *,
    symbol: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    missing_before: list[str] | None = None,
    research_type: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    kind = classify_delta(before, after)
    row = {
        "at": datetime.now(timezone.utc).isoformat(),
        "symbol": str(symbol).upper(),
        "decision_before": before.get("decision"),
        "decision_after": after.get("decision"),
        "missing_evidence": list(missing_before or before.get("missing_critical") or []),
        "research_type": research_type,
        "classification": kind,
        "before_confirmations": before.get("effective_confirmation_count"),
        "after_confirmations": after.get("effective_confirmation_count"),
        "updates_policy": False,
    }
    target = path or LEDGER
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def summary(path: Path | None = None) -> dict[str, Any]:
    target = path or LEDGER
    counts = {
        DECISION_CHANGED: 0,
        CONFIDENCE_CHANGED: 0,
        VETO_DISCOVERED: 0,
        NO_MATERIAL_CHANGE: 0,
    }
    n = 0
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            n += 1
            key = str(row.get("classification") or "")
            if key in counts:
                counts[key] += 1
    return {
        "n": n,
        "counts": counts,
        "sample_size": n,
        "note": "Which research types change decisions — observe only until n is large.",
    }
