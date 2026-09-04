"""Shadow execution hypotheses. Not paper fills.

When a READY BUY cannot be sent (broker auth missing, window closed,
portfolio wait), freeze the intended entry. Distinguish:

  PAPER_ENTERED          — paper book actually opened a position
  SHADOW_NOT_EXECUTED    — hypothesis only; no trade occurred
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "logs" / "product" / "shadow_intents.jsonl"

PAPER_ENTERED = "PAPER_ENTERED"
SHADOW_NOT_EXECUTED = "SHADOW_NOT_EXECUTED"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def freeze_shadow(rec: Mapping[str, Any], *, path: Path | None = None) -> dict[str, Any]:
    """Persist a hypothetical entry. Never claims a fill."""
    row = {
        "intent_id": str(rec.get("decision_id") or rec.get("symbol") or "") + ":shadow",
        "symbol": str(rec.get("symbol") or "").upper(),
        "decision_id": rec.get("decision_id"),
        "candidate_id": rec.get("candidate_id"),
        "decision": rec.get("decision"),
        "candidate_state": rec.get("candidate_state"),
        "entry_state": rec.get("entry_state"),
        "execution_state": rec.get("execution_state"),
        "status": SHADOW_NOT_EXECUTED,
        "not_a_trade": True,
        "paper_executed": False,
        "entry": rec.get("entry"),
        "stop": rec.get("stop"),
        "target": rec.get("target"),
        "method_votes": rec.get("methods_buy") or rec.get("method_votes"),
        "evidence_family_votes": rec.get("evidence_family_votes"),
        "effective_confirmation_count": rec.get("effective_confirmation_count"),
        "regime": (rec.get("references") or {}).get("regime") if isinstance(rec.get("references"), Mapping) else rec.get("regime"),
        "frozen_at": _now(),
        "live_locked": True,
    }
    target = path or LEDGER
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return row


def is_paper_fill(row: Mapping[str, Any]) -> bool:
    return str(row.get("status") or "") == PAPER_ENTERED and bool(row.get("paper_executed"))
