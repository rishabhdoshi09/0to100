"""F&O evidence/learning summary -- whether QuantTerm's F&O ranking has
actually learned anything, and from which evidence class.

Read-only product projection, mirroring product.learning_impact's contract
for the equity desk: it never changes a decision and never authorizes live
money. It exists because two evidence classes exist for F&O and callers keep
conflating them if nothing keeps them visibly apart:

  - HISTORICAL (COUNTERFACTUAL): product.fno_historical_walkforward's
    point-in-time walk-forward replay over real historical equity OHLC
    (driven by product.fno_historical_loop). Establishes priors/hypotheses
    about a setup/regime/sector context. product.fno_evidence_fusion lets a
    LARGE, STABLE historical cell act as a small, capped prior ONLY when
    forward evidence has nothing usable to say yet -- see that module's
    docstring for the full policy table. It can never come close to what
    real forward evidence can do, and it can never act at all once forward
    evidence exists and disagrees with it.
  - FORWARD (PAPER_FORWARD): real settled F&O paper trades
    (product.fno_evidence.record_fno_settlement, fed by
    product.fo_paper_runtime.run_fo_paper_cycle). This is always the
    stronger vote in product.fno_evidence_fusion, and the only evidence
    class that can promote (not merely demote) a candidate.

The two are never pooled into one combined win rate anywhere in this
module's output -- each has its own block, its own evidence_class label, and
its own richer per-cell breakdown (sample, win rate, expectancy, median R,
MFE/MAE, Wilson lower bound, and the regime/sector/volatility/extension/
confidence buckets already embedded in the context key).

``ranking_impact`` never claims a ranking change happened unless
product.fno_ranking already attached a nonzero, usable adjustment to a real
candidate this cycle -- it reads base_score/historical_prior/
forward_adjustment/ranking_score back from that row rather than re-deriving
or guessing at an effect, and it never describes a historical-only
adjustment as "forward evidence".
"""
from __future__ import annotations

from typing import Any, Mapping

from product.evidence_class import COUNTERFACTUAL, PAPER_FORWARD


def _fno_cell_rows(evidence_class: str) -> list[tuple[str, dict[str, Any]]]:
    """(context_key, cell) pairs for F&O UNDERLYING-setup cells of one
    evidence class, read straight from the shared conditional-evidence
    store. A cell belongs to F&O-underlying iff its context key's setup
    component was built by product.fno_evidence.fno_context_key (prefixed
    "setup=FNO_"); equity cells never match this prefix, and contract-shape
    cells (product.fno_contract_evidence, prefixed "CTXCTX_V1|") don't
    either, so the three never mix.
    """
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
    out: list[tuple[str, dict[str, Any]]] = []
    for key, cell in (store.get("cells") or {}).items():
        if not isinstance(key, str) or "::" not in key:
            continue
        cell_class, context = key.split("::", 1)
        if cell_class != klass or not context.startswith("setup=FNO_"):
            continue
        if isinstance(cell, Mapping):
            out.append((context, dict(cell)))
    return out


def _fno_cells(evidence_class: str) -> list[dict[str, Any]]:
    return [cell for _, cell in _fno_cell_rows(evidence_class)]


def _contract_cells(evidence_class: str) -> list[dict[str, Any]]:
    """Cells for F&O CONTRACT-shape evidence (product.fno_contract_evidence),
    kept in the same store but a disjoint key namespace from both equity and
    F&O-underlying cells (see contract_context_key's CTXCTX_V1 prefix)."""
    try:
        from product.conditional_evidence import load
        from product.evidence_class import normalise as normalise_evidence_class
        from product.fno_contract_evidence import CONTRACT_CONTEXT_SCHEMA_VERSION
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
        if cell_class != klass or not context.startswith(f"{CONTRACT_CONTEXT_SCHEMA_VERSION}|"):
            continue
        if isinstance(cell, Mapping):
            out.append(dict(cell))
    return out


def _cell_summary(context_key: str, cell: Mapping[str, Any]) -> dict[str, Any]:
    """One cell's stats plus its bucket breakdown, parsed back out of the key
    -- every field here was already computed by product.conditional_evidence,
    this just surfaces it instead of leaving it buried in a count.
    """
    from product.conditional_evidence import parse_context

    return {
        "context": parse_context(context_key),
        "count": int(cell.get("count") or 0),
        "win_rate": cell.get("win_rate"),
        "wilson_lower_bound": cell.get("wilson_lower_bound"),
        "expectancy_R": cell.get("expectancy_R"),
        "median_R": cell.get("median_R"),
        "mfe_R": cell.get("mfe_R"),
        "mae_R": cell.get("mae_R"),
    }


def _historical_summary() -> dict[str, Any]:
    from product.fno_evidence_fusion import (
        HISTORICAL_CAP,
        HISTORICAL_MIN_SAMPLE,
        _historical_component,
    )

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
    cell_rows = _fno_cell_rows(COUNTERFACTUAL)
    robust_cells = []
    for key, cell in cell_rows:
        component = _historical_component(cell)
        if not component.get("usable"):
            continue
        summary = _cell_summary(key, cell)
        summary["prior_direction"] = component.get("direction")
        summary["prior_adjustment"] = component.get("adjustment")
        robust_cells.append(summary)
    robust_cells.sort(key=lambda row: -int(row.get("count") or 0))
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
        "evidence_cells": len(cell_rows),
        "cells_large_enough_for_a_prior": len(robust_cells),
        "prior_min_sample": HISTORICAL_MIN_SAMPLE,
        "prior_cap": HISTORICAL_CAP,
        "richest_priors": robust_cells[:10],
        "can_affect_ranking": len(robust_cells) > 0,
        "note": (
            "Historical walk-forward replay establishes priors/hypotheses. A "
            "context with a large, stable historical sample can act as a small "
            "prior (capped far below what forward evidence can do) ONLY while "
            "no usable forward evidence exists yet for that same context -- see "
            "product.fno_evidence_fusion. It also grades the underlying call "
            "only: no historical NSE option-chain data source exists to "
            "simulate contract selection."
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
    contract_cells = _contract_cells(PAPER_FORWARD)
    matured_contract_cells = [c for c in contract_cells if int(c.get("count") or 0) >= MIN_SAMPLE]
    outcome_counts: dict[str, int] = {}
    for trade in trades:
        outcome = trade.get("contract_outcome")
        label = str((outcome or {}).get("classification") or "") if isinstance(outcome, Mapping) else ""
        if label:
            outcome_counts[label] = outcome_counts.get(label, 0) + 1
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
        "contract_selection": {
            "evidence_cells": len(contract_cells),
            "matured_cells": len(matured_contract_cells),
            "can_affect_contract_selection": len(matured_contract_cells) > 0,
            "outcome_classification_counts": outcome_counts,
            "note": (
                "Contract-shape evidence (delta/moneyness/DTE/IV/spread) is "
                "learned entirely separately from the underlying call above -- "
                "see product.fno_contract_evidence. It never uses fabricated "
                "historical option-chain data, only genuine settled trades."
            ),
        },
        "note": (
            "Only trades with fully-modeled costs and a fully observed entry/exit "
            "path (production_evidence_eligible) count toward ranking-eligible "
            "evidence, and a context needs at least the minimum sample before it "
            "has any vote."
        ),
    }


def _why_phrase(row: Mapping[str, Any], evidence: Mapping[str, Any]) -> str:
    """A short, honest sentence for the UI -- never claims forward evidence
    moved a ranking when it was actually a historical-only prior, or vice
    versa. Mirrors the exact branch fuse_fno_ranking_evidence took.
    """
    symbol = str(row.get("symbol") or "this setup")
    status = str(evidence.get("status") or "")
    historical_prior = float(evidence.get("historical_prior") or 0.0)
    forward_adjustment = float(evidence.get("forward_adjustment") or 0.0)
    count = evidence.get("count")
    expectancy = evidence.get("expectancy_R")
    if status == "FORWARD_DOMINATES":
        verb = "promoted" if forward_adjustment > 0 else "demoted"
        sentence = (
            f"{symbol}: {verb} by real settled forward paper evidence "
            f"({count} trades, expectancy {expectancy:+.2f}R)."
            if isinstance(expectancy, (int, float)) else f"{symbol}: {verb} by real settled forward paper evidence."
        )
        if historical_prior != 0.0:
            sentence += " Historical evidence agreed and added a small extra prior."
        return sentence
    if status == "HISTORICAL_PRIOR_ONLY":
        verb = "nudged up" if historical_prior > 0 else "nudged down"
        return (
            f"{symbol}: {verb} slightly by a large, stable historical walk-forward "
            f"prior ({count} simulated sessions) -- no real forward trade exists "
            "for this context yet, so the nudge stays small and bounded."
        )
    return f"{symbol}: no ranking change -- {str(evidence.get('reason') or 'insufficient evidence')}."


def _ranking_impact(directional: Mapping[str, Any] | None) -> dict[str, Any]:
    """Whether ranking was ACTUALLY changed by validated evidence this cycle.

    Reads base_score/historical_prior/forward_adjustment/ranking_score/
    ranking_evidence back from the row product.fno_ranking.rank_fno_candidates
    already computed -- never re-derives, estimates, or assumes an effect,
    and never claims an adjustment happened when historical_prior and
    forward_adjustment are both 0.0.
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
                "historical_prior": row.get("historical_prior"),
                "forward_adjustment": row.get("forward_adjustment"),
                "ranking_score": row.get("ranking_score"),
                "adjustment": adjustment,
                "status": evidence.get("status"),
                "reason": evidence.get("reason"),
                "count": evidence.get("count"),
                "expectancy_R": evidence.get("expectancy_R"),
                "why": _why_phrase(row, evidence),
            })
    forward_influenced = sum(1 for r in influenced if r["status"] == "FORWARD_DOMINATES")
    historical_influenced = sum(1 for r in influenced if r["status"] == "HISTORICAL_PRIOR_ONLY")
    if influenced:
        return {
            "status": "RANKING_CHANGED_BY_VALIDATED_EVIDENCE",
            "plain": (
                f"Ranking adjusted for {len(influenced)} candidate(s): "
                f"{forward_influenced} by real forward paper evidence, "
                f"{historical_influenced} by a bounded historical prior only."
            ),
            "influenced_count": len(influenced),
            "forward_influenced_count": forward_influenced,
            "historical_influenced_count": historical_influenced,
            "influenced": influenced[:8],
        }
    return {
        "status": "NO_RANKING_CHANGE_YET",
        "plain": (
            "No candidate this cycle carries enough validated evidence (forward or "
            "a large, stable historical prior) to move F&O ranking."
        ),
        "influenced_count": 0,
        "forward_influenced_count": 0,
        "historical_influenced_count": 0,
        "influenced": [],
    }


def _threshold_review() -> dict[str, Any]:
    """Recommendation-only view of the min-score entry gate (req: rejected/
    missed-trade learning must also inform future thresholds, without ever
    autonomously changing one). See product.fno_threshold_review.
    """
    try:
        from product.fno_threshold_review import review_min_score_threshold
        return review_min_score_threshold()
    except Exception as exc:
        return {
            "recommendation": "UNAVAILABLE",
            "autonomous_change_applied": False,
            "error": str(exc)[:240],
        }


def _exposure_summary() -> dict[str, Any]:
    try:
        from product.fno_exposure_tracking import exposure_report
        return exposure_report()
    except Exception as exc:
        return {"contexts_tracked": 0, "error": str(exc)[:240]}


def build_fno_learning_impact(directional: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "historical": _historical_summary(),
        "forward": _forward_summary(),
        "ranking_impact": _ranking_impact(directional),
        "threshold_review": _threshold_review(),
        "exposure": _exposure_summary(),
        "policy": {
            "historical_and_forward_kept_in_separate_cells": True,
            "historical_alone_can_move_ranking": True,
            "historical_prior_is_small_and_bounded": True,
            "forward_is_always_the_stronger_vote": True,
            "forward_required_to_promote_past_the_historical_cap": True,
            "contract_selection_learned_separately_from_underlying_call": True,
            "contract_selection_requires_genuine_forward_option_trades": True,
            "no_fabricated_historical_option_chain_data": True,
            "live_money_affected": False,
        },
        "live_locked": True,
    }
