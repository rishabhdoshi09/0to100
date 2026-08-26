"""Append-only recommendation evidence ledger.

Stores enough to answer, months later: why did QuantTerm recommend this
name at that timestamp? Missing fields stay empty. Never invents outcomes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "logs" / "product" / "reco_ledger.jsonl"
LEDGER_VERSION = 1


def _compact_card(card: Mapping[str, Any]) -> dict[str, Any]:
    experts = []
    for item in card.get("experts") or []:
        experts.append({
            "id": item.get("id"),
            "status": item.get("status"),
            "eligible": item.get("eligible"),
            "score": item.get("score"),
            "rank": item.get("rank"),
        })
    families = [
        {"id": f.get("id"), "status": f.get("status")}
        for f in (card.get("families") or [])
    ]
    return {
        "symbol": str(card.get("symbol") or "").upper(),
        "tier": card.get("reco_tier"),
        "thesis": card.get("primary_thesis"),
        "horizon": card.get("thesis_horizon"),
        "entry_state": card.get("entry_state"),
        "family_confirms": card.get("family_confirms"),
        "families": families,
        "experts": experts,
        "timing": card.get("timing"),
        "stock_quality": card.get("stock_quality"),
        "conflicts": list(card.get("conflicts") or [])[:4],
        "scan_scanned_at": card.get("scan_scanned_at"),
        "category_id": card.get("category_id"),
        "action_badge": card.get("action_badge"),
    }


def append_recommendations(
    cards: Sequence[Mapping[str, Any]],
    *,
    scan_scanned_at: str = "",
    path: Path | None = None,
) -> Path | None:
    """Append high-conviction and good-setup cards. Watch-only days still write a heartbeat."""
    target = path or LEDGER_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        keep = [
            c for c in cards
            if str(c.get("reco_tier") or "") in {"high_conviction", "good_setup"}
        ]
        record = {
            "schema_version": LEDGER_VERSION,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "scan_scanned_at": scan_scanned_at,
            "n_recommend": len(keep),
            "n_seen": len(list(cards)),
            "cards": [_compact_card(c) for c in keep[:40]],
        }
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return target
    except Exception:
        return None
