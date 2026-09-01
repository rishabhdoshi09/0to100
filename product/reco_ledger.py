"""Append-only recommendation evidence and production-replay ledgers.

Two different questions need two different immutable records:

1. ``reco_ledger.jsonl`` answers: what did QuantTerm surface and why?
2. ``reco_replay.jsonl`` answers: what point-in-time candidate inputs did the
   production ensemble actually see, so a later version can be replayed without
   smuggling future evidence into the test?

Missing fields stay missing. Outcomes are never invented here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "logs" / "product" / "reco_ledger.jsonl"
REPLAY_PATH = ROOT / "logs" / "product" / "reco_replay.jsonl"
LEDGER_VERSION = 3
REPLAY_VERSION = 1

_DERIVED_REPLAY_KEYS = frozenset({
    "methods", "experts", "families",
    "method_confirms", "method_fails", "method_known", "method_line",
    "family_confirms", "family_fails", "family_line",
    "quality_score", "primary_thesis", "primary_thesis_id",
    "thesis_horizon", "thesis_horizon_label", "reco_tier", "reco_tier_label",
    "entry_state", "stock_quality", "timing", "conflicts", "ensemble_why_now",
    "allows_recommend", "production_strategy", "backtest_parity",
    "backtest_parity_detail", "fundamental_disagreement", "evidence_scorecard",
    "authority", "action_badge",
})


def _score_snapshot(card: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from product.evidence_authority import evidence_scorecard
        score = evidence_scorecard(card)
    except Exception:
        return {
            "score": None, "coverage_pct": None, "quality": "UNAVAILABLE",
            "passed": None, "failed": None, "unknown": None, "components": [],
        }
    components = []
    for row in score.get("components") or []:
        if not isinstance(row, Mapping):
            continue
        components.append({
            "id": row.get("id"), "label": row.get("label"),
            "score": row.get("score"), "coverage_pct": row.get("coverage_pct"),
            "passed": row.get("passed"), "failed": row.get("failed"),
            "unknown": row.get("unknown"),
        })
    return {
        "score": score.get("score"), "coverage_pct": score.get("coverage_pct"),
        "quality": score.get("quality"), "passed": score.get("passed"),
        "failed": score.get("failed"), "unknown": score.get("unknown"),
        "components": components,
    }


def _strategy_snapshot() -> dict[str, Any]:
    try:
        from product.strategy_catalog import ensemble_identity
        ident = ensemble_identity()
        return {
            "strategy_id": ident.get("strategy_id"),
            "strategy_version": ident.get("strategy_version"),
            "rules_hash": ident.get("rules_hash"),
        }
    except Exception:
        return {
            "strategy_id": "QT_RECO_ENSEMBLE",
            "strategy_version": 1,
            "rules_hash": None,
        }


def _compact_card(card: Mapping[str, Any]) -> dict[str, Any]:
    experts = []
    for item in card.get("experts") or []:
        if not isinstance(item, Mapping):
            continue
        experts.append({
            "id": item.get("id"), "status": item.get("status"),
            "eligible": item.get("eligible"), "score": item.get("score"),
            "rank": item.get("rank"),
        })
    families = [
        {"id": f.get("id"), "status": f.get("status")}
        for f in (card.get("families") or []) if isinstance(f, Mapping)
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
        "entry": card.get("entry"), "stop": card.get("stop"),
        "target": card.get("target"), "cmp": card.get("cmp"),
        "evidence_scorecard": _score_snapshot(card),
        "production_strategy": _strategy_snapshot(),
    }


def _json_safe(value: Any) -> Any:
    """Freeze only values that survive an ordinary JSON round-trip."""
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return None


def _replay_input(card: Mapping[str, Any]) -> dict[str, Any]:
    """Remove outputs of the ensemble while preserving the evidence it consumed."""
    out: dict[str, Any] = {}
    for key, value in card.items():
        if key in _DERIVED_REPLAY_KEYS:
            continue
        safe = _json_safe(value)
        if safe is not None:
            out[str(key)] = safe
    # Symbol is mandatory for cross-sectional ranking and later outcome joining.
    out["symbol"] = str(card.get("symbol") or "").upper()
    return out


def _decision_snapshot(card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "symbol": str(card.get("symbol") or "").upper(),
        "reco_tier": card.get("reco_tier"),
        "allows_recommend": bool(card.get("allows_recommend")),
        "entry_state": card.get("entry_state"),
        "primary_thesis_id": card.get("primary_thesis_id"),
        "family_confirms": card.get("family_confirms"),
        "category_id": card.get("category_id"),
    }


def append_replay_snapshot(
    cards: Sequence[Mapping[str, Any]],
    *,
    scan_scanned_at: str = "",
    path: Path | None = None,
) -> Path | None:
    """Freeze every candidate seen by the production recommendation ensemble.

    These are live-captured point-in-time inputs, not reconstructed historical
    fundamentals. That makes them suitable for future leakage-safe decision replay.
    The function does not settle outcomes and does not call any network source.
    """
    target = path or REPLAY_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        captured_at = datetime.now(timezone.utc).isoformat()
        material = [c for c in cards if isinstance(c, Mapping) and str(c.get("symbol") or "").strip()]
        record = {
            "schema_version": REPLAY_VERSION,
            "captured_at": captured_at,
            "scan_scanned_at": scan_scanned_at,
            "captured_live": True,
            "production_strategy": _strategy_snapshot(),
            "n_candidates": len(material),
            "candidates": [
                {
                    "input": _replay_input(card),
                    "decision_at_capture": _decision_snapshot(card),
                }
                for card in material
            ],
        }
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")
        return target
    except Exception:
        return None


def append_recommendations(
    cards: Sequence[Mapping[str, Any]],
    *,
    scan_scanned_at: str = "",
    path: Path | None = None,
    replay_path: Path | None = None,
) -> Path | None:
    """Append surfaced decisions plus a separate all-candidate replay snapshot."""
    material = [c for c in cards if isinstance(c, Mapping)]
    target = path or LEDGER_PATH
    if replay_path is None:
        replay_target = REPLAY_PATH if path is None else target.with_name(target.stem + "_replay.jsonl")
    else:
        replay_target = replay_path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        keep = [
            c for c in material
            if str(c.get("reco_tier") or "") in {"high_conviction", "good_setup"}
        ]
        strategy = _strategy_snapshot()
        record = {
            "schema_version": LEDGER_VERSION,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "scan_scanned_at": scan_scanned_at,
            "production_strategy": strategy,
            "n_recommend": len(keep),
            "n_seen": len(material),
            "cards": [_compact_card(c) for c in keep[:40]],
        }
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        append_replay_snapshot(material, scan_scanned_at=scan_scanned_at, path=replay_target)
        return target
    except Exception:
        return None
