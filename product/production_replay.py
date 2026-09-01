"""Integrity replay for the live recommendation ensemble.

This is NOT a performance backtest. It replays the family/thesis/tier decision
from point-in-time frozen expert outputs and verifies that today's ensemble code
reproduces what was recorded at capture time.

Why this exists:
- recommendation behaviour can drift while old UI cards still look plausible;
- full cross-sectional expert replay requires the original full enriched universe;
- the current replay tape freezes scan-wide expert outputs, so the honest scope is
  ``FROZEN_EXPERTS_TO_ENSEMBLE_DECISION``;
- outcome/performance attribution remains unavailable until a separate same-hash,
  point-in-time outcome artifact exists.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from product.reco_ensemble import attach_ensemble

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_PATH = ROOT / "logs" / "product" / "reco_replay.jsonl"

_COMPARE_FIELDS = (
    "reco_tier",
    "allows_recommend",
    "entry_state",
    "primary_thesis_id",
    "family_confirms",
)


def _current_hash() -> str:
    try:
        from product.strategy_catalog import current_rules_hash
        return current_rules_hash()
    except Exception:
        return ""


def load_replay_records(path: Path | None = None, *, limit: int = 200) -> list[dict[str, Any]]:
    target = path or DEFAULT_REPLAY_PATH
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    continue
                if isinstance(raw, dict):
                    rows.append(raw)
    except Exception:
        return []
    return rows[-max(1, int(limit)):]


def replay_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw_input = candidate.get("input")
    experts = candidate.get("expert_snapshot")
    expected = candidate.get("decision_at_capture")
    if not isinstance(raw_input, Mapping) or not isinstance(expected, Mapping):
        return {"replayable": False, "exact": False, "reason": "missing input/decision snapshot"}
    if not isinstance(experts, list) or not experts:
        return {"replayable": False, "exact": False, "reason": "missing frozen expert snapshot"}

    row = dict(raw_input)
    row["experts"] = [dict(x) for x in experts if isinstance(x, Mapping)]
    try:
        actual = attach_ensemble(row)
    except Exception as exc:
        return {"replayable": False, "exact": False, "reason": f"ensemble replay error: {exc}"}

    diffs: dict[str, dict[str, Any]] = {}
    for key in _COMPARE_FIELDS:
        before = expected.get(key)
        after = actual.get(key)
        if before != after:
            diffs[key] = {"captured": before, "replayed": after}
    return {
        "replayable": True,
        "exact": not diffs,
        "symbol": str(expected.get("symbol") or row.get("symbol") or "").upper(),
        "category_id": expected.get("category_id") or row.get("category_id"),
        "diffs": diffs,
        "captured": {k: expected.get(k) for k in _COMPARE_FIELDS},
        "replayed": {k: actual.get(k) for k in _COMPARE_FIELDS},
    }


def replay_record(record: Mapping[str, Any], *, expected_rules_hash: str | None = None) -> dict[str, Any]:
    current = expected_rules_hash if expected_rules_hash is not None else _current_hash()
    strategy = record.get("production_strategy") or {}
    captured_hash = str(strategy.get("rules_hash") or "") if isinstance(strategy, Mapping) else ""
    candidates = [c for c in (record.get("candidates") or []) if isinstance(c, Mapping)]
    results = [replay_candidate(c) for c in candidates]
    replayable = [r for r in results if r.get("replayable")]
    exact = [r for r in replayable if r.get("exact")]
    mismatches = [r for r in replayable if not r.get("exact")]
    same_hash = bool(current and captured_hash and captured_hash == current)
    return {
        "captured_at": record.get("captured_at"),
        "scan_scanned_at": record.get("scan_scanned_at"),
        "captured_rules_hash": captured_hash or None,
        "current_rules_hash": current or None,
        "same_rules_hash": same_hash,
        "captured_live": bool(record.get("captured_live")),
        "scope": record.get("replay_scope") or "UNKNOWN",
        "full_expert_replay_available": bool(record.get("full_expert_replay_available")),
        "candidates": len(candidates),
        "replayable": len(replayable),
        "exact": len(exact),
        "mismatched": len(mismatches),
        "exact_pct": round(len(exact) / len(replayable) * 100.0, 1) if replayable else None,
        "mismatch_examples": mismatches[:10],
        "integrity_pass": bool(same_hash and replayable and not mismatches),
    }


def replay_tape_status(
    path: Path | None = None,
    *,
    current_rules_hash: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Summarise same-hash decision integrity without calling it a backtest."""
    current = current_rules_hash if current_rules_hash is not None else _current_hash()
    records = load_replay_records(path, limit=limit)
    if not records:
        return {
            "available": False,
            "status": "NO_REPLAY_TAPE",
            "scope": "FROZEN_EXPERTS_TO_ENSEMBLE_DECISION",
            "current_rules_hash": current or None,
            "records": 0,
            "same_hash_records": 0,
            "candidates": 0,
            "replayable": 0,
            "exact": 0,
            "mismatched": 0,
            "integrity_pass": False,
            "performance_evidence": False,
            "detail": (
                "No production replay tape exists yet. Run/persist a fresh market scan + Recommendations build. "
                "This is synchronization evidence, not historical performance."
            ),
        }

    audits = [replay_record(r, expected_rules_hash=current) for r in records]
    same = [a for a in audits if a.get("same_rules_hash")]
    candidates = sum(int(a.get("candidates") or 0) for a in same)
    replayable = sum(int(a.get("replayable") or 0) for a in same)
    exact = sum(int(a.get("exact") or 0) for a in same)
    mismatched = sum(int(a.get("mismatched") or 0) for a in same)
    integrity = bool(same and replayable and mismatched == 0 and all(a.get("integrity_pass") for a in same))
    status = "EXACT" if integrity else ("MISMATCH" if mismatched else "NO_CURRENT_HASH_CAPTURE")
    return {
        "available": True,
        "status": status,
        "scope": "FROZEN_EXPERTS_TO_ENSEMBLE_DECISION",
        "current_rules_hash": current or None,
        "records": len(records),
        "same_hash_records": len(same),
        "candidates": candidates,
        "replayable": replayable,
        "exact": exact,
        "mismatched": mismatched,
        "exact_pct": round(exact / replayable * 100.0, 1) if replayable else None,
        "integrity_pass": integrity,
        "performance_evidence": False,
        "full_expert_replay_available": False,
        "detail": (
            f"{exact}/{replayable} current-hash candidate decisions replay exactly from frozen scan-wide expert outputs. "
            "This verifies ensemble synchronization only; it is not a return/backtest claim."
            if same else
            "Replay tape exists, but no capture uses the current executable hash. Persist a fresh scan before judging synchronization."
        ),
        "records_detail": audits[-10:],
    }
