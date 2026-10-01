"""Tournament orchestrator: Champion's real decision + bounded Challenger shadows.

Consumes the Champion's ALREADY-COMPUTED real decisions (from
product.paper_autopilot.run_reco_paper_cycle's production evaluation) -- this
module never re-runs the scanner and never calls any execution/broker/book-
mutation path itself. Its job is split into two independent phases, on
purpose (see product.paper_autopilot.run_reco_paper_cycle for how the real
caller straddles a PAPER mutation between them):

  1. freeze_premutation_bundle() -- freeze the immutable pre-mutation Market
     Twin snapshot and the Champion's real verdict against it. Cost is
     proportional to candidates seen, never to Challenger count, so this
     phase alone is safe to run on the execution-critical path.
  2. evaluate_challengers_from_bundle() -- evaluate every active Challenger
     against that EXACT frozen bundle and compute per-symbol consensus. This
     is the phase that can take arbitrarily long (N Challengers, each an
     evaluator call) and MUST run only after the real Champion PAPER
     mutation has already happened -- never inline before it, or a single
     slow/hung Challenger can delay real order placement no matter how
     tight the budget below is (a per-cycle time check between Challengers
     cannot interrupt one that is already blocking).

run_tournament_cycle() below runs both phases back-to-back for callers that
have no mutation to straddle (off-hours/batch/historical contexts, most
tests).

Challenger failures are isolated per policy (section 32): one broken
Challenger is logged and skipped, never raised up to block any sibling
Challenger. A challenger budget (max_challengers) bounds the work per cycle
so N policies never turn into an O(policies x full market scan) cost --
every policy here evaluates the ALREADY-COMPUTED snapshot, never re-fetches
data.
"""
from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from logger import get_logger
from product.evolution import policy_eval, policy_registry, shadow_decisions, snapshot

log = get_logger(__name__)

DEFAULT_MAX_NEW = 3
DEFAULT_MAX_CHALLENGERS = 12
# A SECOND, independent bound on top of max_challengers: once the cumulative
# wall-clock time spent evaluating Challengers exceeds this, remaining
# Challengers are skipped (never raised, never blocking). This bounds total
# RESEARCH cost per cycle (section 6) -- it does NOT, by itself, protect
# Champion execution latency, since the check only runs BETWEEN Challengers
# and cannot interrupt one that is already hung. That protection comes from
# evaluate_challengers_from_bundle() never running until after the real
# Champion PAPER mutation has already completed (see module docstring).
DEFAULT_MAX_SECONDS = float(os.environ.get("QT_EVOLUTION_TOURNAMENT_MAX_SECONDS") or 8.0)


def compute_consensus(qualified_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Section 16: policy agreement as an uncertainty signal. Only policies
    that actually ran successfully this cycle count toward the denominator --
    a retired/broken/unqualified policy is never silently counted either way."""
    rows = list(qualified_rows)
    n = len(rows)
    selecting = [r for r in rows if r.get("decision") == "ENTER_NOW"]
    dissent = Counter(
        str(r.get("reason_code") or "") for r in rows if r.get("decision") != "ENTER_NOW"
    )
    main_dissent = dissent.most_common(1)[0][0] if dissent else None
    return {
        "qualified_count": n,
        "selecting_count": len(selecting),
        "consensus_pct": round(100.0 * len(selecting) / n, 1) if n else 0.0,
        "main_dissent_reason": main_dissent,
        "dissent_breakdown": dict(dissent),
    }


def _champion_verdict(
    symbol: str, snap: Mapping[str, Any], real_decision: Mapping[str, Any],
    *, champion_policy_id: str,
) -> dict[str, Any]:
    ctx = snap.get("context") or {}
    took = str(real_decision.get("decision") or "") == "ENTER_NOW"
    return {
        "policy_id": champion_policy_id,
        "market_snapshot_id": snap["market_snapshot_id"],
        "domain": snap["domain"],
        "symbol": symbol,
        "decision": "ENTER_NOW" if took else "REJECT",
        "reason_code": real_decision.get("reason_code") or "NOT_EVALUATED",
        "adjusted_score": real_decision.get("selection_score"),
        "breakdown": real_decision.get("breakdown") or {},
        "entry": ctx.get("entry"),
        "stop": ctx.get("stop"),
        "target": ctx.get("target"),
        "sector": ctx.get("sector"),
        "setup_label": ctx.get("setup_label"),
        "is_champion_decision": True,
    }


def _rank_and_cap(
    verdicts: list[dict[str, Any]], *, max_new: int,
) -> list[dict[str, Any]]:
    """Mirror product.portfolio_selection_authority's shape at a policy level:
    a candidate eligible on its own merits may still not be SELECTED once
    ranked against every other eligible candidate the same policy saw this
    cycle. Demote-only -- never promotes a hard-ineligible candidate."""
    eligible = [v for v in verdicts if v.get("decision") == "ENTER_NOW"]
    eligible.sort(key=lambda v: -float(v.get("adjusted_score") or 0.0))
    selected_symbols = {v["symbol"] for v in eligible[:max_new]}
    out = []
    for v in verdicts:
        v = dict(v)
        if v.get("decision") == "ENTER_NOW" and v["symbol"] not in selected_symbols:
            v["decision"] = "REJECT"
            v["reason_code"] = "NOT_TOP_RANKED_THIS_CYCLE"
        out.append(v)
    return out


def freeze_premutation_bundle(
    card_list: Sequence[Mapping[str, Any]],
    champion_decisions_by_symbol: Mapping[str, Mapping[str, Any]],
    *,
    champion_policy_id: str,
    domain: str = policy_registry.EQUITY,
    book: Any = None,
    regime: str = "",
    as_of: str = "",
    snapshot_path: str | Path | None = None,
    shadow_path: str | Path | None = None,
) -> dict[str, Any]:
    """Phase 1 (execution-critical path): freeze the immutable pre-mutation
    Market Twin snapshot for every candidate and the Champion's ALREADY-
    DECIDED real verdict against it. This is the ONLY part of the tournament
    that may run before real PAPER mutation -- it is proportional to
    len(card_list), never to the number of Challengers, so it cannot turn
    into an unbounded delay the way per-Challenger evaluation can.

    The caller MUST proceed with the real Champion PAPER mutation immediately
    after this returns, then evaluate Challengers separately (see
    evaluate_challengers_from_bundle) from the bundle this returns -- never
    by re-reading live book/market state, which would no longer be the same
    pre-mutation information the Champion decided against."""
    snapshots: dict[str, dict[str, Any]] = {}
    for card in card_list:
        symbol = str(card.get("symbol") or "").upper()
        if not symbol or symbol in snapshots:
            continue
        snapshots[symbol] = snapshot.build_snapshot(
            card, book=book, regime=regime, domain=domain, as_of=as_of, path=snapshot_path,
        )

    champion_rows: dict[str, dict[str, Any]] = {}
    for symbol, snap in snapshots.items():
        real = champion_decisions_by_symbol.get(symbol) or {}
        verdict = _champion_verdict(symbol, snap, real, champion_policy_id=champion_policy_id)
        champion_rows[symbol] = shadow_decisions.freeze_shadow_decision(snap, verdict, path=shadow_path)

    return {
        "domain": domain,
        "as_of": as_of,
        "champion_policy_id": champion_policy_id,
        "snapshots": snapshots,
        "champion_rows": champion_rows,
    }


def evaluate_challengers_from_bundle(
    bundle: Mapping[str, Any],
    *,
    max_new: int = DEFAULT_MAX_NEW,
    max_challengers: int | None = DEFAULT_MAX_CHALLENGERS,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    registry_path: str | Path | None = None,
    shadow_path: str | Path | None = None,
    challenger_policies: Sequence[Mapping[str, Any]] | None = None,
    challenger_batch_evaluator=None,
) -> dict[str, Any]:
    """Phase 2 (OUTSIDE the execution-critical path): evaluate every active
    Challenger against the EXACT bundle freeze_premutation_bundle() produced
    -- the same immutable snapshots the Champion's real decision was already
    made against. Must be called only AFTER the real Champion PAPER mutation
    has already happened; nothing here can delay or affect that mutation,
    because the mutation is already done by the time this runs.

    A hung/slow Challenger can still only block THIS function's return, never
    the Champion's PAPER entry -- that is the whole point of the split. The
    wall-clock budget below remains as a bound on total research cost per
    cycle (section 6), not as the mechanism that protects Champion latency."""
    domain = str(bundle.get("domain") or policy_registry.EQUITY)
    champion_policy_id = str(bundle.get("champion_policy_id") or "")
    as_of = str(bundle.get("as_of") or "")
    snapshots: dict[str, dict[str, Any]] = dict(bundle.get("snapshots") or {})
    champion_rows: dict[str, dict[str, Any]] = dict(bundle.get("champion_rows") or {})

    challengers = (
        [dict(p) for p in challenger_policies]
        if challenger_policies is not None
        else policy_registry.active_challengers(domain, path=registry_path)
    )
    if max_challengers is not None:
        challengers = challengers[:max_challengers]

    challenger_rows: dict[str, dict[str, dict[str, Any]]] = {}
    budget_start = time.monotonic()
    budget_exhausted = False
    for policy in challengers:
        policy_id = policy["policy_id"]
        if max_seconds is not None and (time.monotonic() - budget_start) > max_seconds:
            if not budget_exhausted:
                log.warning(
                    "evolution_tournament_budget_exhausted",
                    max_seconds=max_seconds,
                    evaluated=len(challenger_rows),
                    remaining=len(challengers) - len(challenger_rows),
                )
                budget_exhausted = True
            continue
        try:
            if challenger_batch_evaluator is not None:
                verdicts = list(
                    challenger_batch_evaluator(list(snapshots.values()), policy) or []
                )
                normalized_verdicts: list[dict[str, Any]] = []
                for verdict in verdicts:
                    row = dict(verdict or {})
                    symbol = str(row.get("symbol") or "").upper()
                    snap = snapshots.get(symbol)
                    if snap is None:
                        raise ValueError(
                            f"Challenger {policy['policy_id']} returned unknown symbol {symbol!r}"
                        )
                    row["policy_id"] = str(policy["policy_id"])
                    row["market_snapshot_id"] = str(snap["market_snapshot_id"])
                    row["domain"] = domain
                    normalized_verdicts.append(row)
                verdicts = normalized_verdicts
            else:
                verdicts = [
                    policy_eval.evaluate_snapshot(snap, policy) for snap in snapshots.values()
                ]
                verdicts = _rank_and_cap(verdicts, max_new=max_new)
            rows: dict[str, dict[str, Any]] = {}
            for raw_verdict in verdicts:
                v = dict(raw_verdict)
                symbol = str(v.get("symbol") or "").upper()
                if symbol not in snapshots:
                    continue
                snap = snapshots[symbol]
                # Canonical batch evaluators return the real decision object
                # shape; tournament-owned provenance is attached here so every
                # Challenger freeze is attributable to this exact policy and
                # immutable Market Twin snapshot.
                v.setdefault("policy_id", policy_id)
                v.setdefault("market_snapshot_id", snap["market_snapshot_id"])
                v.setdefault("domain", domain)
                row = shadow_decisions.freeze_shadow_decision(snap, v, path=shadow_path)
                rows[symbol] = row
            challenger_rows[policy_id] = rows
        except Exception as exc:
            # Failure isolation (section 32): one broken Challenger must never
            # interrupt the Champion's real execution path or any sibling
            # Challenger's evaluation.
            log.warning("evolution_challenger_failed", policy_id=policy_id, error=str(exc))
            continue

    per_symbol: list[dict[str, Any]] = []
    for symbol, snap in snapshots.items():
        qualified = [champion_rows[symbol]] + [
            challenger_rows[p["policy_id"]][symbol]
            for p in challengers
            if p["policy_id"] in challenger_rows and symbol in challenger_rows[p["policy_id"]]
        ]
        per_symbol.append({
            "market_snapshot_id": snap["market_snapshot_id"],
            "symbol": symbol,
            "champion": champion_rows[symbol],
            "challengers": {
                p["policy_id"]: challenger_rows[p["policy_id"]][symbol]
                for p in challengers
                if p["policy_id"] in challenger_rows and symbol in challenger_rows[p["policy_id"]]
            },
            "consensus": compute_consensus(qualified),
        })

    return {
        "domain": domain,
        "as_of": as_of,
        "champion_policy_id": champion_policy_id,
        "challengers_evaluated": list(challenger_rows.keys()),
        "challengers_skipped": [
            p["policy_id"] for p in challengers if p["policy_id"] not in challenger_rows
        ],
        "tournament_elapsed_seconds": round(time.monotonic() - budget_start, 3),
        "tournament_budget_exhausted": budget_exhausted,
        "results": per_symbol,
    }


def run_tournament_cycle(
    card_list: Sequence[Mapping[str, Any]],
    champion_decisions_by_symbol: Mapping[str, Mapping[str, Any]],
    *,
    champion_policy_id: str,
    domain: str = policy_registry.EQUITY,
    book: Any = None,
    regime: str = "",
    as_of: str = "",
    max_new: int = DEFAULT_MAX_NEW,
    max_challengers: int | None = DEFAULT_MAX_CHALLENGERS,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    registry_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    shadow_path: str | Path | None = None,
    challenger_batch_evaluator=None,
) -> dict[str, Any]:
    """Convenience wrapper combining both phases in sequence -- freeze then
    evaluate -- for callers where the two do NOT need to straddle a real
    PAPER mutation (off-hours/batch/historical contexts, and most tests).

    product.paper_autopilot's real cycle does NOT use this wrapper: it calls
    freeze_premutation_bundle() before mutation and
    evaluate_challengers_from_bundle() after, specifically so a slow/hung
    Challenger (bounded only by evaluate_challengers_from_bundle's budget,
    never by anything here) cannot delay the real Champion PAPER entry that
    sits between the two phases in that caller."""
    bundle = freeze_premutation_bundle(
        card_list, champion_decisions_by_symbol,
        champion_policy_id=champion_policy_id, domain=domain, book=book,
        regime=regime, as_of=as_of, snapshot_path=snapshot_path, shadow_path=shadow_path,
    )
    return evaluate_challengers_from_bundle(
        bundle, max_new=max_new, max_challengers=max_challengers, max_seconds=max_seconds,
        registry_path=registry_path, shadow_path=shadow_path,
        challenger_batch_evaluator=challenger_batch_evaluator,
    )
