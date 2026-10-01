"""Historical Market-Twin diagnostics for Evolution policies.

This lane is deliberately separate from product.evolution.shadow_decisions:
historical replay is a weak, point-in-time research prior only. It may explain
which bounded policy variants would have ranked the already-eligible names
better in past sessions, but it can never qualify a Champion promotion.

Inputs must be matured rows produced by product.historical_replay. That engine
already freezes selection_card at T and resolves the later outcome separately.
We never query price history or current research stores here, so appending
future data after T cannot change a policy's historical decision.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.runtime_paths import logs_dir
from product import decision_context
from product.evolution import policy_eval
from product.evolution import policy_registry as PR

SCHEMA_VERSION = 1
EVIDENCE_CLASS = "HISTORICAL_REPLAY"


def historical_evidence_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_EVOLUTION_HISTORICAL_EVIDENCE")
    if override:
        return Path(override)
    return logs_dir() / "product" / "evolution_historical_policy_evidence.jsonl"


def _read(path: str | Path | None = None) -> list[dict[str, Any]]:
    target = historical_evidence_path(path)
    if not target.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
    except Exception:
        return []
    return out


def _write(rows: Sequence[Mapping[str, Any]], path: str | Path | None = None) -> None:
    target = historical_evidence_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
    tmp.replace(target)


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _decision_key(row: Mapping[str, Any]) -> str:
    return str(
        row.get("canonical_decision_id")
        or row.get("freeze_id")
        or row.get("decision_id")
        or f"{str(row.get('symbol') or '').upper()}:{str(row.get('as_of') or '')[:10]}"
    )


def _is_point_in_time_safe(row: Mapping[str, Any]) -> bool:
    pit = row.get("pit") if isinstance(row.get("pit"), Mapping) else {}
    if bool(pit.get("future_evidence_used")):
        return False
    if str(row.get("outcome_status") or "") != "MATURED":
        return False
    if not isinstance(row.get("selection_card"), Mapping):
        return False
    if _f(row.get("r_multiple")) is None:
        return False
    return True


def evaluate_historical_policies(
    decisions: Sequence[Mapping[str, Any]],
    *,
    domain: str = PR.EQUITY,
    max_new_per_session: int = 3,
    path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate bounded policies over frozen historical decision cards.

    Policy variation may only reorder rows that the canonical historical
    production path already classified BUY. WAIT/AVOID/REJECT remain
    ineligible under every policy; Evolution weights cannot retrospectively
    rescue a hard gate.

    The returned/persisted rows always carry not_promotion_evidence=True.
    promotion.py does not read this store.
    """
    population = PR.ensure_seed_population(domain, path=registry_path)
    champion = population["champion"]
    policies = [champion, *PR.active_challengers(domain, path=registry_path)]

    valid = [dict(row) for row in decisions if _is_point_in_time_safe(row)]
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_session[str(row.get("as_of") or "")[:10]].append(row)

    generated: list[dict[str, Any]] = []
    for policy in policies:
        policy_id = str(policy.get("policy_id") or "")
        for session, rows in sorted(by_session.items()):
            session_rows: list[dict[str, Any]] = []
            selectable: list[tuple[float, str]] = []
            for source in rows:
                card = dict(source.get("selection_card") or {})
                regime = str(source.get("regime") or "UNKNOWN")
                ctx = decision_context.snapshot(card, book=None, regime=regime)
                base = decision_context.score_breakdown(card, policy=None, context=ctx)
                weighted = policy_eval.apply_policy_weights(base, ctx, policy)
                score = float(weighted.get("selection_rank") or 0.0)
                canonical = str(source.get("decision") or "").upper()
                raw = str(source.get("raw_decision") or "").upper()
                canonical_eligible = canonical == "BUY" or raw == "ENTER_NOW"
                key = _decision_key(source)
                if canonical_eligible:
                    selectable.append((score, key))
                session_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "policy_id": policy_id,
                    "policy_manifest_fingerprint": PR.policy_manifest_fingerprint(policy),
                    "domain": domain,
                    "historical_decision_id": key,
                    "symbol": str(source.get("symbol") or "").upper(),
                    "as_of": session,
                    "regime": regime,
                    "pit_grade": source.get("pit_grade"),
                    "canonical_decision": canonical,
                    "canonical_reason_code": str(source.get("reason_code") or ""),
                    "canonical_eligible": canonical_eligible,
                    "policy_score": round(score, 4),
                    "policy_selected": False,
                    "realized_R": _f(source.get("r_multiple")),
                    "classification": str(source.get("classification") or ""),
                    "evidence_class": EVIDENCE_CLASS,
                    "not_promotion_evidence": True,
                    "not_real_pnl": True,
                    "future_evidence_used": False,
                })

            selectable.sort(key=lambda pair: (-pair[0], pair[1]))
            selected_ids = {
                key for _score, key in selectable[: max(0, int(max_new_per_session))]
            }
            for record in session_rows:
                record["policy_selected"] = (
                    bool(record["canonical_eligible"])
                    and record["historical_decision_id"] in selected_ids
                )
                generated.append(record)

    existing = _read(path)
    keyed: dict[tuple[str, str], dict[str, Any]] = {
        (str(row.get("policy_id") or ""), str(row.get("historical_decision_id") or "")): row
        for row in existing
    }
    for row in generated:
        keyed[(row["policy_id"], row["historical_decision_id"])] = row
    merged = sorted(
        keyed.values(),
        key=lambda row: (
            str(row.get("as_of") or ""),
            str(row.get("policy_id") or ""),
            str(row.get("historical_decision_id") or ""),
        ),
    )
    _write(merged, path)

    return {
        "domain": domain,
        "champion_policy_id": str(champion.get("policy_id") or ""),
        "policies_evaluated": len(policies),
        "historical_rows_evaluated": len(valid),
        "evidence_rows_written": len(generated),
        "not_promotion_evidence": True,
        "evidence_class": EVIDENCE_CLASS,
        "policy_summaries": [
            historical_scorecard(str(policy.get("policy_id") or ""), domain=domain, path=path)
            for policy in policies
        ],
    }


def historical_rows(
    policy_id: str,
    *,
    domain: str | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows = [row for row in _read(path) if row.get("policy_id") == policy_id]
    if domain is not None:
        rows = [row for row in rows if row.get("domain") == domain]
    return rows


def historical_scorecard(
    policy_id: str,
    *,
    domain: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    rows = historical_rows(policy_id, domain=domain, path=path)
    selected = [row for row in rows if row.get("policy_selected")]
    selected_rs = [
        float(row["realized_R"]) for row in selected if _f(row.get("realized_R")) is not None
    ]
    not_selected = [row for row in rows if not row.get("policy_selected")]
    missed = sum(
        1 for row in not_selected
        if str(row.get("classification") or "") == "MISSED_WINNER"
    )
    avoided = sum(
        1 for row in not_selected
        if str(row.get("classification") or "") in {"AVOIDED_LOSER", "CORRECT_REJECTION"}
    )
    return {
        "policy_id": policy_id,
        "domain": domain,
        "observations": len(rows),
        "selected": len(selected),
        "selected_expectancy_R": (
            round(sum(selected_rs) / len(selected_rs), 4) if selected_rs else None
        ),
        "missed_winners": missed,
        "avoided_losers_or_correct_rejections": avoided,
        "evidence_class": EVIDENCE_CLASS,
        "not_promotion_evidence": True,
    }



def evaluate_fno_underlying_historical(
    candidates: Sequence[Mapping[str, Any]],
    *,
    max_new_per_session: int = 3,
    path: str | Path | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Underlying-only F&O historical policy diagnostics.

    The source walk-forward has real point-in-time underlying OHLC/volume and
    later underlying closes, but no historical futures OI, sector feed or
    option chain. Missing dimensions therefore remain neutral (zero) and no
    FNO_CONTRACT historical evidence is ever produced.
    """
    from product.evolution.fno_adapter import underlying_policy_adjustment

    domain = PR.FNO_UNDERLYING
    population = PR.ensure_seed_population(domain, path=registry_path)
    champion = population["champion"]
    policies = [champion, *PR.active_challengers(domain, path=registry_path)]

    valid = [
        dict(row) for row in candidates
        if not bool(row.get("future_evidence_used"))
        and _f(row.get("realized_R")) is not None
        and isinstance(row.get("setup"), Mapping)
    ]
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_session[str(row.get("as_of") or "")[:10]].append(row)

    generated: list[dict[str, Any]] = []
    for policy in policies:
        policy_id = str(policy.get("policy_id") or "")
        for session, rows in sorted(by_session.items()):
            session_rows: list[dict[str, Any]] = []
            selectable: list[tuple[float, str]] = []
            for source in rows:
                setup = dict(source.get("setup") or {})
                eligible = bool(source.get("canonical_eligible"))
                # The adapter's hard-gate contract remains authoritative.
                setup["tradable"] = eligible
                candidate = {
                    "symbol": str(source.get("symbol") or "").upper(),
                    "direction": str(source.get("direction") or setup.get("direction") or ""),
                    "setup": setup,
                }
                base_score = float(source.get("base_score") or setup.get("score") or 0.0)
                adjustment = underlying_policy_adjustment(candidate, policy)
                final_score = base_score + float(adjustment.get("adjustment") or 0.0)
                key = str(
                    source.get("historical_decision_id")
                    or f"FNO:{candidate['symbol']}:{session}:{candidate['direction']}"
                )
                if eligible:
                    selectable.append((final_score, key))
                session_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "policy_id": policy_id,
                    "policy_manifest_fingerprint": PR.policy_manifest_fingerprint(policy),
                    "domain": domain,
                    "historical_decision_id": key,
                    "symbol": candidate["symbol"],
                    "as_of": session,
                    "direction": candidate["direction"],
                    "regime": str(source.get("regime") or "UNKNOWN"),
                    "canonical_decision": "TAKE" if eligible else "REJECT",
                    "canonical_eligible": eligible,
                    "policy_score": round(final_score, 4),
                    "policy_selected": False,
                    "realized_R": _f(source.get("realized_R")),
                    "classification": str(source.get("classification") or ""),
                    "available_historical_dimensions": [
                        "underlying_ohlcv",
                        "breakout_distance",
                    ],
                    "unavailable_historical_dimensions": [
                        "futures_oi",
                        "live_sector_strength",
                        "option_chain",
                        "option_greeks",
                        "option_iv",
                        "option_spread",
                    ],
                    "evidence_class": EVIDENCE_CLASS,
                    "option_evidence_status": "UNDERLYING_ONLY_COUNTERFACTUAL",
                    "not_promotion_evidence": True,
                    "not_real_pnl": True,
                    "future_evidence_used": False,
                })

            selectable.sort(key=lambda pair: (-pair[0], pair[1]))
            selected_ids = {
                key for _score, key in selectable[: max(0, int(max_new_per_session))]
            }
            for record in session_rows:
                record["policy_selected"] = (
                    bool(record["canonical_eligible"])
                    and record["historical_decision_id"] in selected_ids
                )
                generated.append(record)

    existing = _read(path)
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {
        (
            str(row.get("domain") or ""),
            str(row.get("policy_id") or ""),
            str(row.get("historical_decision_id") or ""),
        ): row
        for row in existing
    }
    for row in generated:
        keyed[(row["domain"], row["policy_id"], row["historical_decision_id"])] = row
    merged = sorted(
        keyed.values(),
        key=lambda row: (
            str(row.get("as_of") or ""),
            str(row.get("domain") or ""),
            str(row.get("policy_id") or ""),
            str(row.get("historical_decision_id") or ""),
        ),
    )
    _write(merged, path)

    return {
        "domain": domain,
        "champion_policy_id": str(champion.get("policy_id") or ""),
        "policies_evaluated": len(policies),
        "historical_rows_evaluated": len(valid),
        "evidence_rows_written": len(generated),
        "option_evidence_status": "UNDERLYING_ONLY_COUNTERFACTUAL",
        "not_promotion_evidence": True,
        "evidence_class": EVIDENCE_CLASS,
        "policy_summaries": [
            historical_scorecard(
                str(policy.get("policy_id") or ""), domain=domain, path=path,
            )
            for policy in policies
        ],
    }
