"""Counterfactual outcomes for rejected / waited paper candidates.

Rejected trades are never booked as P&L. Forward bars classify the decision:

  CORRECT_REJECTION / MISSED_WINNER / AVOIDED_LOSER /
  RAN_AWAY_WITHOUT_ENTRY / GOOD_WAIT / FLAT
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from core.runtime_paths import logs_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = logs_dir() / "product" / "counterfactuals.jsonl"
SCHEMA_VERSION = 2

CORRECT_REJECTION = "CORRECT_REJECTION"
MISSED_WINNER = "MISSED_WINNER"
AVOIDED_LOSER = "AVOIDED_LOSER"
RAN_AWAY = "RAN_AWAY_WITHOUT_ENTRY"
GOOD_WAIT = "GOOD_WAIT"
FLAT = "FLAT"


def ledger_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_COUNTERFACTUALS")
    if override:
        return Path(override)
    return DEFAULT_PATH


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
    except Exception:
        return []
    return out


def _counterfactual_freeze_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    target = Path(path)
    return target.with_suffix(target.suffix + ".freeze.db")

def freeze_decision(
    *,
    symbol: str,
    reason_code: str,
    decision: str,
    entry: float | None,
    stop: float | None,
    target: float | None,
    as_of: str,
    evidence: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    evidence_map = dict(evidence or {})
    freeze_input = {
        "decision_id": str(evidence_map.get("decision_id") or ""),
        "symbol": str(symbol).upper(),
        "as_of": str(as_of or "")[:10],
        "decision": decision,
        "reason_code": reason_code,
        "entry": entry,
        "stop": stop,
        "target": target,
        "setup_label": evidence_map.get("setup_label") or evidence_map.get("setup"),
        "sector": evidence_map.get("sector"),
        "regime": evidence_map.get("regime"),
        "families": evidence_map.get("families"),
        "method_votes": evidence_map.get("method_votes"),
        "vetoes": evidence_map.get("vetoes"),
        "portfolio": evidence_map.get("portfolio"),
        "selection_score": evidence_map.get("selection_score"),
        "policy_effect": evidence_map.get("policy_effect"),
        "thesis_hash": evidence_map.get("thesis_hash"),
        "calibration_snapshot_id": evidence_map.get("calibration_snapshot_id"),
        "data_snapshot_id": evidence_map.get("data_snapshot_id"),
        "source_scan_id": evidence_map.get("source_scan_id") or evidence_map.get("scan_scanned_at"),
        "evidence_class": evidence_map.get("evidence_class") or "COUNTERFACTUAL",
        "versions": evidence_map.get("versions"),
    }
    from product.decision_freeze import freeze as freeze_canonical

    canonical = freeze_canonical(
        freeze_input,
        path=_counterfactual_freeze_path(path),
    )
    counterfactual_id = str(canonical.get("freeze_id") or "")
    row = {
        "schema_version": SCHEMA_VERSION,
        "counterfactual_id": counterfactual_id,
        "canonical_freeze_id": counterfactual_id,
        "decision_fingerprint": str(canonical.get("fingerprint") or ""),
        "fingerprint_schema_version": canonical.get("fingerprint_schema_version"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "symbol": str(symbol).upper(),
        "decision": decision,
        "reason_code": reason_code,
        "hypothetical_entry": entry,
        "hypothetical_stop": stop,
        "hypothetical_target": target,
        "as_of": as_of,
        "evidence": evidence_map,
        "rules_hash": str(evidence_map.get("rules_hash") or ""),
        "thesis_hash": str(evidence_map.get("thesis_hash") or ""),
        "calibration_snapshot_id": str(evidence_map.get("calibration_snapshot_id") or ""),
        "data_snapshot_id": str(evidence_map.get("data_snapshot_id") or ""),
        "regime": str(evidence_map.get("regime") or ""),
        "sector": str(evidence_map.get("sector") or ""),
        "setup": str(evidence_map.get("setup_label") or evidence_map.get("setup") or ""),
        "group": str(evidence_map.get("group") or ""),
        "evidence_class": str(evidence_map.get("evidence_class") or "COUNTERFACTUAL"),
        "outcome": None,
        "classification": None,
        "not_pnl": True,
    }
    target_path = ledger_path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    for existing in _read_ledger(target_path):
        if counterfactual_id and str(existing.get("counterfactual_id") or "") == counterfactual_id:
            return existing
    with target_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row

def classify_forward(
    *,
    entry: float | None,
    stop: float | None,
    target: float | None,
    forward_return_pct: float | None,
    later_entered: bool = False,
) -> str:
    if later_entered:
        return GOOD_WAIT
    if entry is None or forward_return_pct is None:
        return FLAT
    if target is not None and forward_return_pct >= 0 and (target - entry) != 0:
        # reached a meaningful positive move
        if forward_return_pct >= 5:
            return MISSED_WINNER
    if stop is not None and forward_return_pct <= -abs((entry - stop) / entry) * 100 * 0.8:
        return CORRECT_REJECTION if forward_return_pct < 0 else AVOIDED_LOSER
    if forward_return_pct <= -5:
        return AVOIDED_LOSER if forward_return_pct < 0 else CORRECT_REJECTION
    if forward_return_pct >= 8:
        return MISSED_WINNER
    if abs(forward_return_pct) < 1.5:
        return RAN_AWAY if later_entered is False else FLAT
    return FLAT if abs(forward_return_pct) < 3 else (
        MISSED_WINNER if forward_return_pct > 0 else CORRECT_REJECTION
    )


def settle(
    row: Mapping[str, Any],
    *,
    forward_return_pct: float | None,
    later_entered: bool = False,
) -> dict[str, Any]:
    out = dict(row)
    classification = classify_forward(
        entry=_f(row.get("hypothetical_entry")),
        stop=_f(row.get("hypothetical_stop")),
        target=_f(row.get("hypothetical_target")),
        forward_return_pct=forward_return_pct,
        later_entered=later_entered,
    )
    resolved_at = datetime.now(timezone.utc).isoformat()
    out["outcome"] = {
        "forward_return_pct": forward_return_pct,
        "later_entered": later_entered,
        "not_pnl": True,
        "resolved_at": resolved_at,
    }
    out["classification"] = classification

    # A rejected/waited decision still has a frozen entry and stop. Express its
    # hypothetical forward move in R so it can train selection without ever
    # being booked as P&L. Missing/invalid levels stay unlabelled.
    entry_f = _f(row.get("hypothetical_entry"))
    stop_f = _f(row.get("hypothetical_stop"))
    if entry_f is not None and stop_f is not None and forward_return_pct is not None:
        risk = abs(entry_f - stop_f)
        if risk > 0:
            move = entry_f * (float(forward_return_pct) / 100.0)
            out["counterfactual_R"] = round(move / risk, 6)
        else:
            out["counterfactual_R"] = None
    else:
        out["counterfactual_R"] = None
    return out


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out
