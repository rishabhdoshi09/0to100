"""The desk's decisions, as one canonical list the API and UI both read.

This is the seam that stops the pipeline's cards from leaking further into the
product. The saved recommendation workspace comes in; canonical
:class:`~product.decision.Decision` objects, ranked against measured evidence,
go out. Nothing downstream of here reads a card key.

Honesty rules carried over from the rest of the desk:

* No saved scan means NO_DECISIONS with the reason stated, not an empty list
  that a screen would render as "nothing looks good today".
* Ranking may demote on measured evidence and may never promote on it.
* Whatever the desk could not obtain is listed per decision as missing
  evidence, and summarised for the board so a data gap is visible without
  opening every name.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.decision import BUY, Decision
from product.decision_adapter import decisions_from_cards
from product.decision_explanation import explain, missing_sections
from product.decision_ranking import decision_context_key, rank, ranking_explanation
from product.evidence_class import PAPER_FORWARD

SCHEMA_VERSION = 1


def _cards(workspace: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for category in workspace.get("categories") or []:
        if not isinstance(category, Mapping):
            continue
        cid = str(category.get("id") or category.get("key") or "")
        for card in category.get("cards") or []:
            if isinstance(card, Mapping) and str(card.get("symbol") or "").strip():
                row = dict(card)
                row.setdefault("category_id", cid)
                out.append(row)
    return out


def decisions_from_workspace(
    workspace: Mapping[str, Any] | None,
    *,
    market_state: str = "",
    sector_state: str = "",
) -> list[Decision]:
    if not workspace:
        return []
    return decisions_from_cards(
        _cards(workspace),
        source_scan_id=str(workspace.get("scan_scanned_at") or ""),
        market_state=market_state,
        sector_state=sector_state,
        evidence_class=PAPER_FORWARD,
        decision_engine_version=str(workspace.get("engine_version") or ""),
        feature_schema_version=str(workspace.get("schema_version") or ""),
        strategy_version=str(workspace.get("strategy_version") or ""),
    )


def decision_board(
    *,
    workspace: Mapping[str, Any] | None = None,
    market_state: str = "",
    sector_state: str = "",
    limit: int = 40,
) -> dict[str, Any]:
    """The ranked board. Reads the saved scan; never runs one."""
    if workspace is None:
        try:
            from product.recommendations_store import load_recommendations

            workspace = load_recommendations()
        except Exception:
            workspace = None

    if not workspace:
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "state": "NO_DECISIONS",
            "reason": "No saved whole-market scan — nothing has been decided yet.",
            "scan_scanned_at": "",
            "decisions": [],
            "counts": {},
            "evidence_gaps": {},
        }

    decisions = decisions_from_workspace(
        workspace, market_state=market_state, sector_state=sector_state
    )
    ranked = rank(decisions)

    counts: dict[str, int] = {}
    gaps: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for row in ranked[: max(0, int(limit))]:
        decision = row.decision
        counts[decision.state] = counts.get(decision.state, 0) + 1
        for item in decision.missing_evidence:
            gaps[item.id] = gaps.get(item.id, 0) + 1
        payload = row.to_dict()
        payload["setup"] = decision.setup
        payload["entry"] = decision.entry
        payload["stop"] = decision.stop
        payload["target"] = decision.target
        payload["expected_R"] = decision.computed_expected_R
        payload["context_key"] = decision_context_key(decision)
        payload["evidence_counts"] = decision.evidence_counts()
        payload["why"] = ranking_explanation(row)
        rows.append(payload)

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "state": "DECIDED" if rows else "NO_CANDIDATES",
        "reason": "" if rows else "The scan completed and nothing qualified.",
        "scan_scanned_at": str(workspace.get("scan_scanned_at") or ""),
        "decisions": rows,
        "counts": counts,
        "actionable": sum(1 for r in ranked if r.decision.state == BUY),
        "evidence_gaps": dict(sorted(gaps.items(), key=lambda kv: -kv[1])),
    }


def decision_why(
    symbol: str,
    *,
    workspace: Mapping[str, Any] | None = None,
    market_state: str = "",
    sector_state: str = "",
) -> dict[str, Any]:
    """The WHY THIS DECISION payload for one name, or an honest absence."""
    wanted = str(symbol or "").strip().upper()
    if not wanted:
        raise ValueError("a symbol is required")

    if workspace is None:
        try:
            from product.recommendations_store import load_recommendations

            workspace = load_recommendations()
        except Exception:
            workspace = None

    decisions = decisions_from_workspace(
        workspace, market_state=market_state, sector_state=sector_state
    )
    match = next((d for d in decisions if d.symbol == wanted), None)
    if match is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "symbol": wanted,
            "reason": (
                "This name is not in the last saved scan, so the desk has not "
                "decided anything about it."
            ),
        }

    ranked = rank([match])[0]
    payload = explain(match)
    payload.update({
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "scan_scanned_at": str((workspace or {}).get("scan_scanned_at") or ""),
        "ranking": ranked.to_dict(),
        "ranking_explanation": ranking_explanation(ranked),
        "unfilled_sections": missing_sections(payload),
    })
    return payload
