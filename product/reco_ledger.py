"""Append-only recommendation evidence ledger.

Stores enough to answer, months later: why did QuantTerm surface this name at
that timestamp? Missing fields stay empty. Never invents outcomes. The evidence
score and production-strategy identity are frozen at decision time so later
method/data changes cannot rewrite what the system actually knew or which rules
created the decision.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "logs" / "product" / "reco_ledger.jsonl"
LEDGER_VERSION = 3


def _score_snapshot(card: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from product.evidence_authority import evidence_scorecard
        score = evidence_scorecard(card)
    except Exception:
        return {
            "score": None,
            "coverage_pct": None,
            "quality": "UNAVAILABLE",
            "passed": None,
            "failed": None,
            "unknown": None,
            "components": [],
        }
    components = []
    for row in score.get("components") or []:
        if not isinstance(row, Mapping):
            continue
        components.append({
            "id": row.get("id"),
            "label": row.get("label"),
            "score": row.get("score"),
            "coverage_pct": row.get("coverage_pct"),
            "passed": row.get("passed"),
            "failed": row.get("failed"),
            "unknown": row.get("unknown"),
        })
    return {
        "score": score.get("score"),
        "coverage_pct": score.get("coverage_pct"),
        "quality": score.get("quality"),
        "passed": score.get("passed"),
        "failed": score.get("failed"),
        "unknown": score.get("unknown"),
        "components": components,
    }


def _strategy_snapshot(card: Mapping[str, Any]) -> dict[str, Any]:
    category_id = str(card.get("category_id") or "")
    try:
        from product.strategy_contract import strategy_for_category
        strategy = strategy_for_category(category_id)
    except Exception:
        strategy = None
    if strategy is None:
        return {
            "strategy_id": None,
            "strategy_version": None,
            "rules_hash": None,
            "category_id": category_id or None,
            "status": "UNREGISTERED",
        }
    return {
        "strategy_id": strategy.strategy_id,
        "strategy_version": strategy.version,
        "rules_hash": strategy.rules_hash,
        "category_id": strategy.category_id,
        "status": strategy.status,
    }


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
        "entry": card.get("entry"),
        "stop": card.get("stop"),
        "target": card.get("target"),
        "cmp": card.get("cmp"),
        "strategy": _strategy_snapshot(card),
        "evidence_scorecard": _score_snapshot(card),
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