"""F&O evidence/learning summary -- whether QuantTerm's F&O ranking has
actually learned anything, and from which evidence class.

Read-only product projection, mirroring product.learning_impact's contract
for the equity desk: it never changes a decision and never authorizes live
money. It exists because two evidence classes exist for F&O and callers keep
conflating them if nothing keeps them visibly apart:

  - HISTORICAL (COUNTERFACTUAL): product.fno_historical_walkforward's
    point-in-time walk-forward replay over real historical equity OHLC
    (driven by product.fno_historical_loop). Establishes priors/hypotheses
    about a setup/regime/sector context. product.conditional_evidence
    .ranking_evidence() refuses any evidence class outside
    {PAPER_FORWARD, REAL_FORWARD} (product.evidence_class.MARKET_EVIDENCE),
    so nothing here can ever move F&O ranking -- by construction, not by
    convention.
  - FORWARD (PAPER_FORWARD): real settled F&O paper trades
    (product.fno_evidence.record_fno_settlement, fed by
    product.fo_paper_runtime.run_fo_paper_cycle). This is the only evidence
    class product.fno_ranking.rank_fno_candidates() ever reads, so it is the
    only one that can actually change which candidate ranks first.

The two are never pooled into one combined win rate anywhere in this
module's output -- each has its own block, its own evidence_class label, and
its own "can this affect ranking" flag.

``ranking_impact`` never claims a ranking change happened unless
product.fno_ranking already attached a nonzero, usable adjustment to a real
candidate this cycle -- it reads that field back rather than re-deriving or
guessing at an effect.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.evidence_class import COUNTERFACTUAL, PAPER_FORWARD


def _fno_cells(evidence_class: str) -> list[dict[str, Any]]:
    """F&O-tagged cells for one evidence class, read straight from the shared
    conditional-evidence store. A cell belongs to F&O iff its context key's
    setup component was built by product.fno_evidence.fno_context_key
    (prefixed "FNO_"); equity cells never match this prefix."""
    try:
        from product.conditional_evidence import load
        from product.evidence_class import normalise as normalise_evidence_class
    except Exception:
        return []
    try:
        store = load()
    except Exception:
        return []
    klass = normalise_evidence_class(evidence_class)
    out: list[dict[str, Any]] = []
    for key, cell in (store.get("cells") or {}).items():
        if not isinstance(key, str) or "::" not in key:
            continue
        cell_class, context = key.split("::", 1)
        if cell_class != klass or not context.startswith("setup=FNO_"):
            continue
        if isinstance(cell, Mapping):
            out.append(dict(cell))
    return out


def _historical_summary() -> dict[str, Any]:
    try:
        from product.fno_historical_loop import status as historical_status
        hist = dict(historical_status() or {})
    except Exception as exc:
        return {
            "available": False,
            "evidence_class": "HISTORICAL_COUNTERFACTUAL",
            "error": str(exc)[:240],
            "can_affect_ranking": False,
        }
    counts = dict(hist.get("classification_counts") or {})
    return {
        "available": bool(hist.get("available")),
        "evidence_class": "HISTORICAL_COUNTERFACTUAL",
        "last_run_at": hist.get("last_run_at", ""),
        "cursor_date": hist.get("cursor_date", ""),
        "coverage_complete": bool(hist.get("coverage_complete")),
        "historical_simulations_completed": int(hist.get("total_sessions_processed") or 0),
        "decisions_graded": int(hist.get("total_candidates_evaluated") or 0),
        "settled": int(hist.get("total_settled") or 0),
        "correct_rejects": int(counts.get("CORRECT_REJECTION") or 0),
        "missed_winners": int(counts.get("MISSED_WINNER") or 0),
        "avoided_losers": int(counts.get("AVOIDED_LOSER") or 0),
        "ran_away_without_entry": int(counts.get("RAN_AWAY_WITHOUT_ENTRY") or 0),
        "good_waits": int(counts.get("GOOD_WAIT") or 0),
        "flat": int(counts.get("FLAT") or 0),
        "evidence_cells": len(_fno_cells(COUNTERFACTUAL)),
        "can_affect_ranking": False,
        "note": (
            "Historical walk-forward replay establishes priors/hypotheses only. "
            "It can never move F&O ranking -- only a real settled forward paper "
            "trade can. It also grades the underlying call only: no historical "
            "NSE option-chain data source exists to simulate contract selection."
        ),
    }


def _forward_summary() -> dict[str, Any]:
    try:
        from product.conditional_evidence import MIN_SAMPLE
        from product.fo_paper_store import FoPaperStore

        with FoPaperStore() as store:
            paper_status = dict(store.status() or {})
            trades = [t for t in store.load_trades(limit=5000) if isinstance(t, Mapping)]
    except Exception as exc:
        return {
            "available": False,
            "evidence_class": "PAPER_FORWARD",
            "error": str(exc)[:240],
            "can_affect_ranking": False,
        }
    wins = sum(1 for t in trades if float(t.get("net_pnl") or 0.0) > 0)
    losses = sum(1 for t in trades if float(t.get("net_pnl") or 0.0) < 0)
    production_eligible = [t for t in trades if bool(t.get("production_evidence_eligible"))]
    cells = _fno_cells(PAPER_FORWARD)
    matured_cells = [c for c in cells if int(c.get("count") or 0) >= MIN_SAMPLE]
    return {
        "available": True,
        "evidence_class": "PAPER_FORWARD",
        "open_positions": int(paper_status.get("open_positions") or 0),
        "forward_paper_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "production_evidence_trades": len(production_eligible),
        "evidence_cells": len(cells),
        "matured_cells": len(matured_cells),
        "minimum_sample_for_ranking": MIN_SAMPLE,
        "can_affect_ranking": len(matured_cells) > 0,
        "note": (
            "Only trades with fully-modeled costs and a fully observed entry/exit "
            "path (production_evidence_eligible) count toward ranking-eligible "
            "evidence, and a context needs at least the minimum sample before it "
            "has any vote."
        ),
    }


def _ranking_impact(directional: Mapping[str, Any] | None) -> dict[str, Any]:
    """Whether ranking was ACTUALLY changed by validated evidence this cycle.

    Reads the ranking_adjustment/ranking_evidence fields
    product.fno_ranking.rank_fno_candidates already attached to each real
    candidate -- never re-derives, estimates, or assumes an effect.
    """
    candidates = list((directional or {}).get("candidates") or [])
    influenced: list[dict[str, Any]] = []
    for row in candidates:
        if not isinstance(row, Mapping):
            continue
        evidence = row.get("ranking_evidence") or {}
        try:
            adjustment = float(row.get("ranking_adjustment") or 0.0)
        except (TypeError, ValueError):
            adjustment = 0.0
        if isinstance(evidence, Mapping) and evidence.get("usable") and adjustment != 0.0:
            influenced.append({
                "symbol": row.get("symbol"),
                "direction": row.get("direction"),
                "base_score": row.get("base_score"),
                "ranking_score": row.get("ranking_score"),
                "adjustment": adjustment,
                "reason": evidence.get("reason"),
                "count": evidence.get("count"),
                "expectancy_R": evidence.get("expectancy_R"),
            })
    if influenced:
        return {
            "status": "RANKING_CHANGED_BY_VALIDATED_EVIDENCE",
            "plain": (
                f"Ranking demoted for {len(influenced)} candidate(s) after "
                "validated forward evidence of negative expectancy in this exact context."
            ),
            "influenced_count": len(influenced),
            "influenced": influenced[:8],
        }
    return {
        "status": "NO_RANKING_CHANGE_YET",
        "plain": (
            "Historical evidence may exist, but there is not yet enough validated "
            "forward evidence to alter F&O ranking."
        ),
        "influenced_count": 0,
        "influenced": [],
    }


def build_fno_learning_impact(directional: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "historical": _historical_summary(),
        "forward": _forward_summary(),
        "ranking_impact": _ranking_impact(directional),
        "contract": {
            "historical_and_forward_kept_separate": True,
            "historical_can_promote_ranking": False,
            "forward_required_to_change_ranking": True,
            "contract_selection_learning_requires_real_option_history": True,
            "live_money_affected": False,
        },
        "live_locked": True,
    }
