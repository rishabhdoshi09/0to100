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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "counterfactuals.jsonl"
SCHEMA_VERSION = 1

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
    row = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "symbol": str(symbol).upper(),
        "decision": decision,
        "reason_code": reason_code,
        "hypothetical_entry": entry,
        "hypothetical_stop": stop,
        "hypothetical_target": target,
        "as_of": as_of,
        "evidence": dict(evidence or {}),
        "outcome": None,
        "classification": None,
    }
    target_path = ledger_path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
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
    out["outcome"] = {
        "forward_return_pct": forward_return_pct,
        "later_entered": later_entered,
        "not_pnl": True,
    }
    out["classification"] = classification
    return out


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out
