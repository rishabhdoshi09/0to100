"""F&O adapters for the generic Evolution tournament.

Underlying and contract policies are deliberately separate domains. Every
adjustment is bounded and is applied only AFTER the existing F&O hard gates:
Evolution may re-rank a tradable underlying or an already-eligible contract,
but can never resurrect a rejected setup/contract or invent option history.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.evolution import policy_registry as PR
from product.evolution import shadow_decisions as SD
from product.evolution import snapshot as SNAP


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def _mult(policy: Mapping[str, Any] | None, key: str) -> float:
    try:
        value = float((policy or {}).get("weights", {}).get(key, 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(3.0, value))


def underlying_policy_adjustment(
    candidate: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bounded rank-only adjustment for an ALREADY-tradable F&O setup."""
    setup = candidate.get("setup") if isinstance(candidate.get("setup"), Mapping) else {}
    if not bool(setup.get("tradable")):
        return {"adjustment": 0.0, "eligible": False, "reason": "HARD_SETUP_REJECTED"}
    components = setup.get("components") if isinstance(setup.get("components"), Mapping) else {}
    parts: dict[str, float] = {}

    oi = _f(components.get("futures_oi"))
    parts["oi"] = ( _mult(policy, "oi_confirmation_mult") - 1.0) * oi

    macro = _f(components.get("sector_strength")) + _f(components.get("nifty_alignment"))
    parts["sector_nifty"] = (_mult(policy, "sector_confirmation_mult") - 1.0) * macro

    distance = abs(_f(setup.get("breakout_distance_pct")))
    ext_mult = _mult(policy, "extension_penalty_mult")
    # Only penalise a move already materially beyond the trigger. The hard
    # EXTENDED_CHASE blocker remains authoritative upstream.
    parts["extension"] = -max(0.0, distance - 0.75) * 2.0 * max(0.0, ext_mult - 1.0)

    raw = sum(parts.values())
    adjustment = max(-8.0, min(8.0, raw))
    return {
        "adjustment": round(adjustment, 4),
        "eligible": True,
        "parts": {k: round(v, 4) for k, v in parts.items()},
        "policy_id": str((policy or {}).get("policy_id") or ""),
        "policy_fingerprint": PR.policy_manifest_fingerprint(policy or {}) if policy else "",
    }


def contract_policy_adjustment(
    contract: Mapping[str, Any],
    policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bounded rank-only adjustment for an ALREADY-eligible option contract."""
    if not bool(contract.get("eligible")):
        return {"adjustment": 0.0, "eligible": False, "reason": "HARD_CONTRACT_REJECTED"}
    components = contract.get("components") if isinstance(contract.get("components"), Mapping) else {}
    parts: dict[str, float] = {}
    parts["delta_fit"] = (
        (_mult(policy, "delta_preference_mult") - 1.0) * _f(components.get("delta_fit"))
    )
    parts["liquidity"] = (
        (_mult(policy, "liquidity_mult") - 1.0) * _f(components.get("liquidity"))
    )
    # theta component is a fit score: above midpoint is good, below is costly.
    theta_fit = _f(components.get("theta"))
    parts["theta"] = (
        (_mult(policy, "theta_penalty_mult") - 1.0) * (theta_fit - 6.0)
    )
    raw = sum(parts.values())
    adjustment = max(-8.0, min(8.0, raw))
    return {
        "adjustment": round(adjustment, 4),
        "eligible": True,
        "parts": {k: round(v, 4) for k, v in parts.items()},
        "policy_id": str((policy or {}).get("policy_id") or ""),
        "policy_fingerprint": PR.policy_manifest_fingerprint(policy or {}) if policy else "",
    }


def run_underlying_tournament(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected: Mapping[str, Any] | None,
    as_of: str,
) -> dict[str, Any]:
    population = PR.ensure_seed_population(PR.FNO_UNDERLYING)
    champion = population["champion"]
    challengers = PR.active_challengers(PR.FNO_UNDERLYING)
    rows = [dict(row) for row in candidates if isinstance(row, Mapping)]
    if not rows:
        return {"domain": PR.FNO_UNDERLYING, "results": [], "challengers_evaluated": []}

    snapshots: dict[str, dict[str, Any]] = {}
    for row in rows:
        setup = dict(row.get("setup") or {})
        plan = dict(setup.get("underlying_trade_plan") or {})
        direction = str(row.get("direction") or setup.get("direction") or "")
        symbol = str(row.get("symbol") or "")
        context = {
            "entry": plan.get("entry"),
            "stop": plan.get("stop"),
            "target": plan.get("target"),
            "setup_label": f"FNO_{direction}_{setup.get('futures_oi_state') or 'NEUTRAL'}",
            "regime": str((row.get("ranking_evidence") or {}).get("market_regime") or ""),
            "selection_score": row.get("pre_evolution_ranking_score", row.get("ranking_score")),
            "direction": direction,
            "futures_oi_state": setup.get("futures_oi_state"),
        }
        snapshots[direction] = SNAP.build_domain_snapshot(
            symbol=symbol,
            domain=PR.FNO_UNDERLYING,
            as_of=as_of,
            context=context,
            card=row,
            identity_suffix=f"UNDERLYING:{direction}",
        )

    selected_direction = str((selected or {}).get("direction") or "")
    policy_rows: dict[str, dict[str, dict[str, Any]]] = {}

    def freeze_policy(policy: Mapping[str, Any], *, is_champion: bool) -> dict[str, dict[str, Any]]:
        scored = []
        for row in rows:
            direction = str(row.get("direction") or "")
            base = _f(row.get("pre_evolution_ranking_score", row.get("ranking_score")))
            adj = underlying_policy_adjustment(row, policy)
            scored.append((base + _f(adj.get("adjustment")), direction, row, adj))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen = selected_direction if is_champion else (scored[0][1] if scored else "")
        out: dict[str, dict[str, Any]] = {}
        for score, direction, row, adj in scored:
            setup = dict(row.get("setup") or {})
            plan = dict(setup.get("underlying_trade_plan") or {})
            verdict = {
                "policy_id": policy["policy_id"],
                "market_snapshot_id": snapshots[direction]["market_snapshot_id"],
                "domain": PR.FNO_UNDERLYING,
                "symbol": row.get("symbol"),
                "direction": direction,
                "decision": "ENTER_NOW" if direction == chosen else "REJECT",
                "reason_code": "SELECTED_DIRECTION" if direction == chosen else "NOT_TOP_RANKED_DIRECTION",
                "adjusted_score": score,
                "breakdown": {"evolution": adj},
                "entry": plan.get("entry"),
                "stop": plan.get("stop"),
                "target": plan.get("target"),
                "setup_label": f"FNO_{direction}",
                "grading_mode": "UNDERLYING_FORWARD",
                "is_champion_decision": is_champion,
            }
            out[direction] = SD.freeze_shadow_decision(snapshots[direction], verdict)
        return out

    policy_rows[champion["policy_id"]] = freeze_policy(champion, is_champion=True)
    for policy in challengers:
        try:
            policy_rows[policy["policy_id"]] = freeze_policy(policy, is_champion=False)
        except Exception:
            continue
    return {
        "domain": PR.FNO_UNDERLYING,
        "champion_policy_id": champion["policy_id"],
        "challengers_evaluated": [
            p["policy_id"] for p in challengers if p["policy_id"] in policy_rows
        ],
        "results": policy_rows,
    }


def run_contract_tournament(
    *,
    underlying_symbol: str,
    direction: str,
    setup: Mapping[str, Any],
    eligible_contracts: Sequence[Mapping[str, Any]],
    selected_contract: Mapping[str, Any] | None,
    as_of: str,
) -> dict[str, Any]:
    """Freeze Champion/Challenger contract choices. Generic historical
    grading is disabled; rows require genuine PAPER_FORWARD option outcomes."""
    population = PR.ensure_seed_population(PR.FNO_CONTRACT)
    champion = population["champion"]
    challengers = PR.active_challengers(PR.FNO_CONTRACT)
    contracts = [dict(row) for row in eligible_contracts if bool(row.get("eligible"))]
    if not contracts:
        return {"domain": PR.FNO_CONTRACT, "results": [], "challengers_evaluated": []}

    snapshot = SNAP.build_domain_snapshot(
        symbol=underlying_symbol,
        domain=PR.FNO_CONTRACT,
        as_of=as_of,
        context={
            "setup_label": f"FNO_CONTRACT_{direction}",
            "direction": direction,
            "grading_mode": "PAPER_FORWARD_CONTRACT_ONLY",
            "eligible_contract_count": len(contracts),
        },
        card={"setup": dict(setup), "eligible_contracts": contracts},
        identity_suffix=f"CONTRACT:{direction}",
    )
    snapshot["grading_mode"] = "PAPER_FORWARD_CONTRACT_ONLY"

    champion_symbol = str((selected_contract or {}).get("symbol") or "")

    def choose(policy: Mapping[str, Any], *, is_champion: bool) -> dict[str, Any]:
        scored = []
        for row in contracts:
            base = _f(row.get("learned_contract_score", row.get("score")))
            adj = contract_policy_adjustment(row, policy)
            scored.append((base + _f(adj.get("adjustment")), row, adj))
        scored.sort(key=lambda item: item[0], reverse=True)
        chosen_row = None
        if is_champion and champion_symbol:
            chosen_row = next((row for _, row, _ in scored if str(row.get("symbol") or "") == champion_symbol), None)
        if chosen_row is None:
            chosen_row = scored[0][1]
        chosen_score, _, chosen_adj = next(
            item for item in scored if str(item[1].get("symbol") or "") == str(chosen_row.get("symbol") or "")
        )
        plan = dict(chosen_row.get("trade_plan") or {})
        from product.fno_contract_evidence import contract_context_key
        verdict = {
            "policy_id": policy["policy_id"],
            "market_snapshot_id": snapshot["market_snapshot_id"],
            "domain": PR.FNO_CONTRACT,
            "symbol": underlying_symbol,
            "direction": direction,
            "decision": "ENTER_NOW",
            "reason_code": f"SELECTED_CONTRACT:{chosen_row.get('symbol') or ''}",
            "adjusted_score": chosen_score,
            "breakdown": {"evolution": chosen_adj},
            "entry": plan.get("entry"),
            "stop": plan.get("stop"),
            "target": plan.get("target"),
            "setup_label": f"FNO_CONTRACT_{direction}",
            "grading_mode": "PAPER_FORWARD_CONTRACT_ONLY",
            "selected_contract": chosen_row,
            "contract_symbol": chosen_row.get("symbol"),
            "contract_context_key": contract_context_key(chosen_row),
            "is_champion_decision": is_champion,
        }
        return SD.freeze_shadow_decision(snapshot, verdict)

    out = {champion["policy_id"]: choose(champion, is_champion=True)}
    for policy in challengers:
        try:
            out[policy["policy_id"]] = choose(policy, is_champion=False)
        except Exception:
            continue
    return {
        "domain": PR.FNO_CONTRACT,
        "market_snapshot_id": snapshot["market_snapshot_id"],
        "champion_policy_id": champion["policy_id"],
        "challengers_evaluated": [
            p["policy_id"] for p in challengers if p["policy_id"] in out
        ],
        "results": out,
    }
