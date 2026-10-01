"""Tournament orchestrator: Champion's real decision + bounded Challenger shadows.

Consumes the Champion's ALREADY-COMPUTED real decisions (from
product.paper_autopilot.run_reco_paper_cycle's production evaluation) -- this
module never re-runs the scanner and never calls any execution/broker/book-
mutation path itself. Its only job is: freeze the Champion's real decision for
fair comparison, evaluate every active Challenger against the exact same
immutable snapshot, and compute per-symbol policy consensus.

Challenger failures are isolated per policy (section 32): one broken
Challenger is logged and skipped, never raised up to block the Champion's
real execution or any other Challenger. A challenger budget (max_challengers)
bounds the work per cycle so N policies never turn into an
O(policies x full market scan) cost -- every policy here evaluates the
ALREADY-COMPUTED snapshot, never re-fetches data.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from logger import get_logger
from product.evolution import policy_eval, policy_registry, shadow_decisions, snapshot

log = get_logger(__name__)

DEFAULT_MAX_NEW = 3
DEFAULT_MAX_CHALLENGERS = 12


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
    registry_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    shadow_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze the Champion's real decisions and every active Challenger's
    shadow decisions against one shared immutable snapshot per candidate,
    then compute per-symbol consensus. Returns a summary; every frozen row
    is independently durable (see product.evolution.shadow_decisions).

    registry_path/snapshot_path/shadow_path override three INDEPENDENT
    stores -- never collapsed into one generic path, so a caller can never
    accidentally point two different stores at the same file."""
    challengers = policy_registry.active_challengers(domain, path=registry_path)
    if max_challengers is not None:
        challengers = challengers[:max_challengers]

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

    challenger_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in challengers:
        policy_id = policy["policy_id"]
        try:
            verdicts = [
                policy_eval.evaluate_snapshot(snap, policy) for snap in snapshots.values()
            ]
            verdicts = _rank_and_cap(verdicts, max_new=max_new)
            rows: dict[str, dict[str, Any]] = {}
            for v in verdicts:
                row = shadow_decisions.freeze_shadow_decision(snapshots[v["symbol"]], v, path=shadow_path)
                rows[v["symbol"]] = row
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
        "results": per_symbol,
    }
