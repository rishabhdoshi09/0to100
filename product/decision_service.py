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
    decisions = decisions_from_cards(
        _cards(workspace),
        source_scan_id=str(workspace.get("scan_scanned_at") or ""),
        market_state=market_state,
        sector_state=sector_state,
        evidence_class=PAPER_FORWARD,
        decision_engine_version=str(workspace.get("engine_version") or ""),
        feature_schema_version=str(workspace.get("schema_version") or ""),
        strategy_version=str(workspace.get("strategy_version") or ""),
        generated_at=str(workspace.get("scan_scanned_at") or ""),
    )
    # Evidence intelligence is an enrichment layer, never the source of the
    # decision state. Fail-open: a research-store issue must not erase the desk's
    # canonical decision or fabricate an alternative one.
    try:
        from product.evidence_intelligence import enrich
    except Exception:
        return decisions

    enriched: list[Decision] = []
    for decision in decisions:
        try:
            enriched.append(enrich(decision))
        except Exception:
            enriched.append(decision)
    return enriched



def _read_only_paper_book():
    """Restore the persisted PAPER_FORWARD book without starting the research brain."""
    try:
        import json
        from core.runtime_paths import logs_path
        from research.auto_research.costs import india_cash_costs
        from research.auto_research.paper_book import PaperBook

        book = PaperBook(slippage_bps=3.0, cost_model=india_cash_costs)
        target = logs_path("intelligence/intel_book.json")
        if target.exists():
            book.restore(json.loads(target.read_text(encoding="utf-8")))
        return book
    except Exception:
        return None


def _best_trades_from_production_thesis(
    workspace: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    market_state: str = "",
    limit: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return trades actionable under the same thesis + restored paper portfolio.

    Discovery is read-only. It disables only the circular historical-bootstrap
    prerequisite, then applies the same candidate gates, carried sector risk,
    portfolio authority and max-three-new-positions rule as PAPER_FORWARD.
    """
    from product.paper_autopilot import (
        ENTER_NOW,
        carried_sector_risk,
        evaluate_selection_candidate,
    )
    from product.portfolio_selection_authority import apply_portfolio_authority

    regime = str(
        market_state
        or workspace.get("regime")
        or workspace.get("market_regime")
        or "RISK_ON"
    )
    row_by_symbol: dict[str, dict[str, Any]] = {}
    for raw in rows:
        symbol = str(raw.get("symbol") or "").upper()
        if not symbol:
            continue
        prior = row_by_symbol.get(symbol)
        if prior is None or float(raw.get("ranking_score") or 0.0) > float(prior.get("ranking_score") or 0.0):
            row_by_symbol[symbol] = dict(raw)

    book = _read_only_paper_book()
    family_risk, cluster_risk = carried_sector_risk(book)
    eligible: dict[str, tuple[float, Any]] = {}
    errors: list[dict[str, str]] = []
    for card in _cards(workspace):
        symbol = str(card.get("symbol") or "").upper()
        if not symbol or symbol not in row_by_symbol:
            continue
        try:
            decision = evaluate_selection_candidate(
                card,
                book=book,
                workspace=workspace,
                entries_allowed=True,
                paper_enabled=True,
                regime=regime,
                enforce_history=False,
                family_risk=family_risk,
                cluster_risk=cluster_risk,
            )
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)[:180]})
            continue
        if str(decision.decision or "") != ENTER_NOW:
            continue
        score = float(decision.selection_score or 0.0)
        prior = eligible.get(symbol)
        if prior is None or score > prior[0]:
            eligible[symbol] = (score, decision)

    ranked = sorted(
        eligible.values(),
        key=lambda item: (-item[0], str(item[1].symbol or "")),
    )
    try:
        kept, _diverted = apply_portfolio_authority(
            ranked,
            book=book,
            max_new=min(3, max(0, int(limit))),
            regime=regime,
        )
    except Exception as exc:
        errors.append({"symbol": "", "error": f"portfolio authority: {str(exc)[:160]}"})
        kept = []

    best: list[dict[str, Any]] = []
    for score, decision in kept:
        symbol = str(decision.symbol or "").upper()
        base = row_by_symbol.get(symbol)
        if base is None:
            continue
        payload = dict(base)
        payload["production_selection_score"] = float(score)
        payload["production_reason_code"] = str(decision.reason_code or "")
        payload["production_policy_effect"] = str(decision.policy_effect or "NEUTRAL")
        payload["production_portfolio_authority"] = dict(decision.portfolio or {})
        payload["discovery_decision"] = ENTER_NOW
        payload["history_bootstrap_gate_applied"] = False
        payload["restored_paper_positions_considered"] = bool(
            book is not None and getattr(book, "open", {})
        )
        best.append(payload)

    return best[: max(0, int(limit))], errors


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
            "best_trades": [],
            "counts": {},
            "evidence_gaps": {},
            "thesis": {},
        }

    decisions = decisions_from_workspace(
        workspace, market_state=market_state, sector_state=sector_state
    )
    ranked = rank(decisions)

    counts: dict[str, int] = {}
    gaps: dict[str, int] = {}
    all_rows: list[dict[str, Any]] = []
    for row in ranked:
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
        payload["evidence_intelligence"] = dict(
            (decision.historical_evidence or {}).get("evidence_intelligence") or {}
        )
        try:
            from product.trading_thesis import decision_quality
            payload["trade_quality"] = decision_quality(
                decision,
                ranking_score=float(row.ranking_score),
            )
        except Exception:
            payload["trade_quality"] = {}
        all_rows.append(payload)
    rows = all_rows[: max(0, int(limit))]

    try:
        from product.trading_thesis import manifest as thesis_manifest
        thesis = thesis_manifest()
    except Exception:
        thesis = {}

    # Best trades are not inferred from tier labels alone. They are re-evaluated
    # through the exact non-executing production selection seam used by replay
    # and PAPER_FORWARD. Pre-approval discovery relaxes only the circular
    # history-bootstrap prerequisite; it does not relax strategy/risk gates.
    best_trades, best_trade_errors = _best_trades_from_production_thesis(
        workspace,
        all_rows,
        market_state=market_state,
        limit=5,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "state": "DECIDED" if rows else "NO_CANDIDATES",
        "reason": "" if rows else "The scan completed and nothing qualified.",
        "scan_scanned_at": str(workspace.get("scan_scanned_at") or ""),
        "decisions": rows,
        "best_trades": best_trades,
        "best_trades_mode": "PRODUCTION_THESIS_DISCOVERY",
        "best_trades_errors": best_trade_errors,
        "thesis": thesis,
        "counts": counts,
        "actionable": sum(1 for r in ranked if r.decision.state == BUY),
        "evidence_gaps": dict(sorted(gaps.items(), key=lambda kv: -kv[1])),
    }


def _scan_membership(symbol: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The whole-market scan record for a symbol, plus that scan's provenance."""
    try:
        from product.scan_store import load_scan

        payload = load_scan() or {}
    except Exception:
        return None, {}
    provenance = payload.get("provenance")
    provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    for row in payload.get("records") or []:
        if isinstance(row, Mapping) and str(row.get("symbol", "")).upper() == symbol:
            return dict(row), provenance
    return None, provenance


def _is_known_equity(symbol: str) -> bool | None:
    """True/False when the approved NSE universe can be read, else None."""
    try:
        from data.nse_universe import get_nse_universe

        return symbol in {str(s).upper() for s in (get_nse_universe() or [])}
    except Exception:
        return None


def _unshortlisted_view(symbol: str) -> dict[str, Any]:
    """Everything genuinely known about a name outside the shortlist.

    This is deliberately not a decision. It reports scan membership, the levels
    the scanner actually produced, and the session those prices came from, so
    the page can distinguish "evaluated and passed over" from "never looked at"
    from "not a tradable symbol" — three very different answers that all used
    to render as the same dead end.
    """
    row, provenance = _scan_membership(symbol)
    known = _is_known_equity(symbol)

    if row is not None:
        levels = {k: row.get(k) for k in
                  ("price", "entry", "stop", "target", "reward_risk",
                   "upside_pct", "downside_pct", "plan_complete", "plan_missing")}
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "symbol": symbol,
            "stance": "NOT_SHORTLISTED",
            "in_latest_scan": True,
            "known_equity": True,
            "reason": (
                "This name was evaluated in the latest whole-market scan but did "
                "not reach the shortlist."
            ),
            "scan_status": row.get("status") or "",
            "scan_verdict": row.get("verdict") or "",
            "score": row.get("score"),
            "signals": list(row.get("signals") or []),
            "scan_reasons": list(row.get("reasons") or []),
            "why": row.get("why") or "",
            "levels": levels,
            "provenance": provenance,
        }

    if known is False:
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "symbol": symbol,
            "stance": "NOT_A_TRADABLE_SYMBOL",
            "in_latest_scan": False,
            "known_equity": False,
            "reason": "This symbol is not in the approved NSE equity universe.",
            "provenance": provenance,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "symbol": symbol,
        "stance": "NOT_EVALUATED",
        "in_latest_scan": False,
        "known_equity": known,
        "reason": (
            "This name is in the approved universe but is absent from the latest "
            "whole-market scan, so nothing has been evaluated for it yet."
            if known
            else "The approved NSE universe could not be read, so membership is unknown."
        ),
        "provenance": provenance,
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
        # The recommendations workspace is a filtered shortlist. A name absent
        # from it may still have been fully evaluated by the whole-market scan,
        # in which case the desk knows its levels, score and signals. Reporting
        # "nothing decided" there discards real evidence the system holds.
        return _unshortlisted_view(wanted)

    ranked = rank([match])[0]
    payload = explain(match)
    payload.update({
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "scan_scanned_at": str((workspace or {}).get("scan_scanned_at") or ""),
        "ranking": ranked.to_dict(),
        "ranking_explanation": ranking_explanation(ranked),
        "evidence_intelligence": dict(
            (match.historical_evidence or {}).get("evidence_intelligence") or {}
        ),
        "unfilled_sections": missing_sections(payload),
    })
    return payload
