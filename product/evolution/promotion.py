"""Scientific Champion promotion gate.

Build the full promotion machinery now; keep automatic Champion replacement
disabled until the mechanism has been proven on real forward evidence.
AUTO_PROMOTION_ENABLED gates ONLY the automatic path -- an explicit,
deliberate call to promote_to_champion() always works (an operator reviewing
a PROMOTION_ELIGIBLE policy is exactly the "single deliberate setting" the
product brief asks for), and it still enforces every guardrail below. There
is no path from this module to live execution: it imports nothing from
execution/* and only ever mutates product.evolution.policy_registry
(a PAPER-only policy's lifecycle status).

A challenger must clear ALL of:
  - minimum paired forward sample
  - research.harness.evaluate()'s PROMOTE verdict on the INCREMENTAL diff
    series (challenger_R - champion_R per shared snapshot) -- this is the
    "alpha, not beta" framing for free: testing the paired difference
    directly controls for whatever the shared market did, so a challenger
    cannot look good merely by trading in a strong tape (section 15).
  - Benjamini-Hochberg FDR correction across every challenger tested in the
    SAME cycle (section 14) -- a raw PROMOTE is not enough if it does not
    survive correction for the number of simultaneous hypotheses.
  - no worse max drawdown than the current Champion over the same paired
    window.
  - adequate regime breadth (the incremental edge isn't a one-regime fluke).
No single metric may trigger promotion alone.

Two independent stores are involved (the policy registry and the shadow-
decision ledger), each with its own path convention -- registry_path and
ledger_path are passed through separately rather than collapsed into one
generic override, so a caller can never accidentally point both at the
same file.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logger import get_logger
from core.runtime_paths import logs_dir
from product.evolution import policy_registry, scorecard

log = get_logger(__name__)

# ── the single deliberate setting (section 41) ──────────────────────────────
AUTO_PROMOTION_ENABLED = False

MIN_PAIRED_SAMPLE = 30
MIN_INCREMENTAL_EXPECTANCY_R = 0.05
FDR_ALPHA = 0.05
MIN_REGIME_BREADTH_OBSERVED = 2  # need at least this many distinct regimes to judge breadth at all
MIN_REGIME_BREADTH_ACCEPTABLE = 2  # of the observed regimes, at least this many must be non-negative
MIN_CHAMPION_TENURE_DAYS = 3
REVERSAL_INCREMENTAL_MARGIN_R = 0.10
RETIRE_MIN_PAIRED_SAMPLE = 60
RETIRE_MAX_INCREMENTAL_EXPECTANCY_R = -0.15

PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"
NOT_ELIGIBLE = "NOT_ELIGIBLE"
RETIREMENT_ELIGIBLE = "RETIREMENT_ELIGIBLE"


def promotion_proof_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_EVOLUTION_PROMOTION_PROOFS")
    if override:
        return Path(override)
    return logs_dir() / "product" / "evolution_promotion_proofs.jsonl"


def _read_proofs(path: str | Path | None = None) -> list[dict[str, Any]]:
    target = promotion_proof_path(path)
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


def _persist_proof(row: dict[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    target = promotion_proof_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **dict(row),
        "proof_created_at": datetime.now(timezone.utc).isoformat(),
        "build_sha": str(os.environ.get("QT_BUILD_SHA") or os.environ.get("GITHUB_SHA") or ""),
    }
    rows = _read_proofs(path)
    rows.append(record)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for item in rows[-5000:]:
            fh.write(json.dumps(item, default=str, sort_keys=True) + "\n")
    tmp.replace(target)
    return record


def latest_promotion_proof(
    policy_id: str, *, path: str | Path | None = None,
) -> dict[str, Any] | None:
    rows = [row for row in _read_proofs(path) if row.get("policy_id") == policy_id]
    return rows[-1] if rows else None


def _dt(value: Any):
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _regime_breadth(
    champion_policy_id: str, challenger_policy_id: str,
    *, domain: str | None = None, ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Segment the SAME paired snapshots by regime; a real edge should not
    live in exactly one regime (section 12/21's regime-stability requirement)."""
    champ = {r["market_snapshot_id"]: r for r in scorecard.graded_rows(champion_policy_id, domain=domain, path=ledger_path)}
    chal = {r["market_snapshot_id"]: r for r in scorecard.graded_rows(challenger_policy_id, domain=domain, path=ledger_path)}
    shared = set(champ) & set(chal)
    by_regime: dict[str, list[float]] = {}
    for sid in shared:
        c, h = champ[sid], chal[sid]
        c_r, h_r = _f(c.get("counterfactual_R")), _f(h.get("counterfactual_R"))
        if c_r is None or h_r is None:
            continue
        regime = str(h.get("regime") or c.get("regime") or "UNKNOWN")
        by_regime.setdefault(regime, []).append(h_r - c_r)

    regime_means = {
        regime: round(sum(diffs) / len(diffs), 4)
        for regime, diffs in by_regime.items() if diffs
    }
    acceptable = sum(1 for mean in regime_means.values() if mean >= 0)
    return {
        "regimes_observed": len(regime_means),
        "regime_means": regime_means,
        "regimes_acceptable": acceptable,
        "breadth_ok": (
            len(regime_means) >= MIN_REGIME_BREADTH_OBSERVED
            and acceptable >= MIN_REGIME_BREADTH_ACCEPTABLE
        ),
    }


def evaluate_promotion(
    domain: str, challenger_policy_id: str,
    *, n_simultaneous_challengers: int = 1, fdr_rejected: bool | None = None,
    registry_path: str | Path | None = None, ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run every promotion gate for ONE challenger against the domain's
    current Champion. Does not mutate anything -- a pure read-and-decide
    function so it can be called as often as desired (e.g. every off-hours
    cycle) without side effects.

    `fdr_rejected`: pass the Benjamini-Hochberg significance verdict for
    this policy from evaluate_promotion_batch() (below), which corrects
    across every challenger tested in the SAME cycle. A lone call to this
    function with fdr_rejected=None skips that specific cross-policy gate
    (there is nothing to correct against) but still requires every other
    guardrail -- callers that test multiple challengers together MUST use
    evaluate_promotion_batch() instead, precisely to avoid the uncorrected-
    multiple-testing trap section 14 warns about.
    """
    champion = policy_registry.current_champion(domain, path=registry_path)
    if champion is None:
        return {"policy_id": challenger_policy_id, "status": NOT_ELIGIBLE, "reason": "no current champion for domain"}
    champion_id = champion["policy_id"]
    if champion_id == challenger_policy_id:
        return {"policy_id": challenger_policy_id, "status": NOT_ELIGIBLE, "reason": "already champion"}

    paired = scorecard.paired_comparison(champion_id, challenger_policy_id, domain=domain, path=ledger_path)
    n = paired["paired_snapshots"]
    diffs = paired["incremental_diffs"]

    result: dict[str, Any] = {
        "policy_id": challenger_policy_id,
        "champion_policy_id": champion_id,
        "domain": domain,
        "paired_snapshots": n,
        "incremental_expectancy_R": paired["incremental_expectancy_R"],
        "min_paired_sample": MIN_PAIRED_SAMPLE,
    }

    if n < MIN_PAIRED_SAMPLE:
        return {**result, "status": NOT_ELIGIBLE, "reason": f"only {n} paired forward observations, need {MIN_PAIRED_SAMPLE}"}
    if float(paired["incremental_expectancy_R"] or 0.0) < MIN_INCREMENTAL_EXPECTANCY_R:
        return {
            **result,
            "status": NOT_ELIGIBLE,
            "reason": (
                f"incremental expectancy {float(paired['incremental_expectancy_R'] or 0.0):+.4f}R "
                f"is below required margin {MIN_INCREMENTAL_EXPECTANCY_R:+.4f}R"
            ),
        }

    breadth = _regime_breadth(
        champion_id, challenger_policy_id, domain=domain, ledger_path=ledger_path,
    )
    result["regime_breadth"] = breadth
    if not breadth["breadth_ok"]:
        return {
            **result,
            "status": NOT_ELIGIBLE,
            "reason": (
                f"insufficient regime breadth: observed={breadth['regimes_observed']} "
                f"acceptable={breadth['regimes_acceptable']} "
                f"(need >= {MIN_REGIME_BREADTH_OBSERVED} observed and "
                f">= {MIN_REGIME_BREADTH_ACCEPTABLE} acceptable)"
            ),
        }

    from research.harness import evaluate as harness_evaluate

    verdict = harness_evaluate(diffs, n_trials=max(1, n_simultaneous_challengers))
    result["harness_verdict"] = verdict.verdict
    result["harness_insight"] = verdict.insight
    result["harness_stats"] = {
        "mean_r": verdict.mean_r, "sharpe": verdict.sharpe, "p_value": verdict.p_value,
        "psr": verdict.psr, "dsr": verdict.dsr, "n_trials": verdict.n_trials,
    }

    if verdict.verdict != "PROMOTE":
        return {**result, "status": NOT_ELIGIBLE, "reason": f"harness verdict {verdict.verdict}: {verdict.insight}"}

    if fdr_rejected is False:
        return {**result, "status": NOT_ELIGIBLE, "reason": "fails Benjamini-Hochberg FDR correction across simultaneously-tested challengers"}

    champ_card = scorecard.scorecard(champion_id, domain=domain, path=ledger_path)
    chal_card = scorecard.scorecard(challenger_policy_id, domain=domain, path=ledger_path)
    champ_dd = champ_card.get("max_drawdown_R")
    chal_dd = chal_card.get("max_drawdown_R")
    result["champion_max_drawdown_R"] = champ_dd
    result["challenger_max_drawdown_R"] = chal_dd
    if champ_dd is not None and chal_dd is not None and chal_dd < champ_dd:
        return {**result, "status": NOT_ELIGIBLE, "reason": f"worse tail risk: drawdown {chal_dd}R vs champion {champ_dd}R"}

    result["primary_improvement"] = chal_card
    return {**result, "status": PROMOTION_ELIGIBLE, "reason": "cleared all promotion gates"}


def evaluate_promotion_batch(
    domain: str, *, registry_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    proof_path: str | Path | None = None,
    persist_proofs: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate EVERY active challenger for a domain together, applying
    Benjamini-Hochberg FDR correction across the whole family (section 14) --
    never judge one challenger's significance in isolation when several are
    being tested in the same cycle, or one will eventually look good by
    chance alone."""
    challengers = policy_registry.active_challengers(domain, path=registry_path)
    champion = policy_registry.current_champion(domain, path=registry_path)
    if champion is None or not challengers:
        return []

    raw: list[dict[str, Any]] = []
    for policy in challengers:
        r = evaluate_promotion(
            domain, policy["policy_id"],
            n_simultaneous_challengers=len(challengers),
            registry_path=registry_path, ledger_path=ledger_path,
        )
        raw.append(r)

    for row in raw:
        row.setdefault("fdr_rejected", None)
        row.setdefault("fdr_n_tested", 0)
    tested = [r for r in raw if "harness_stats" in r]
    if tested:
        from research.harness import benjamini_hochberg

        pvalues = [float(r["harness_stats"]["p_value"]) for r in tested]
        fdr = benjamini_hochberg(pvalues, alpha=FDR_ALPHA)
        for r, rejected in zip(tested, fdr["rejected"]):
            r["fdr_rejected"] = bool(rejected)
            r["fdr_n_tested"] = len(tested)
            if r["status"] == PROMOTION_ELIGIBLE and not rejected:
                r["status"] = NOT_ELIGIBLE
                r["reason"] = "fails Benjamini-Hochberg FDR correction across simultaneously-tested challengers"

    if persist_proofs:
        for row in raw:
            _persist_proof(
                {
                    **row,
                    "policy_manifest_fingerprint": (
                        policy_registry.policy_manifest_fingerprint(
                            policy_registry.get_policy(
                                str(row.get("policy_id") or ""), path=registry_path
                            ) or {}
                        )
                        if row.get("policy_id")
                        else ""
                    ),
                    "champion_manifest_fingerprint": (
                        policy_registry.policy_manifest_fingerprint(champion)
                        if champion else ""
                    ),
                },
                path=proof_path,
            )
    return raw


def evaluate_retirement_batch(
    domain: str, *, registry_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Conservative research-only retirement eligibility.

    A policy is eligible for retirement only after a large paired sample and
    materially negative incremental expectancy. This never affects the current
    Champion and never touches PAPER/live execution state.
    """
    champion = policy_registry.current_champion(domain, path=registry_path)
    if champion is None:
        return []
    out: list[dict[str, Any]] = []
    for policy in policy_registry.active_challengers(domain, path=registry_path):
        lifecycle = list(policy.get("lifecycle_history") or [])
        rollback_protected = (
            policy.get("status") == policy_registry.PROBATION
            or any(
                event.get("status") == policy_registry.CHAMPION
                for event in lifecycle
                if isinstance(event, dict)
            )
        )
        if rollback_protected:
            out.append({
                "policy_id": policy["policy_id"],
                "domain": domain,
                "paired_snapshots": 0,
                "incremental_expectancy_R": None,
                "status": NOT_ELIGIBLE,
                "rollback_protected": True,
                "reason": (
                    "former/probation Champion is rollback-protected and cannot "
                    "be auto-retired"
                ),
            })
            continue
        paired = scorecard.paired_comparison(
            champion["policy_id"], policy["policy_id"], domain=domain, path=ledger_path,
        )
        n = int(paired.get("paired_snapshots") or 0)
        edge = _f(paired.get("incremental_expectancy_R"))
        eligible = (
            n >= RETIRE_MIN_PAIRED_SAMPLE
            and edge is not None
            and edge <= RETIRE_MAX_INCREMENTAL_EXPECTANCY_R
        )
        out.append({
            "policy_id": policy["policy_id"],
            "domain": domain,
            "paired_snapshots": n,
            "incremental_expectancy_R": edge,
            "status": RETIREMENT_ELIGIBLE if eligible else NOT_ELIGIBLE,
            "reason": (
                "sufficient paired evidence shows materially negative incremental value"
                if eligible
                else "retirement evidence threshold not met"
            ),
        })
    return out


def retire_qualified_challengers(
    domain: str, *, registry_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    retired: list[dict[str, Any]] = []
    for row in evaluate_retirement_batch(
        domain, registry_path=registry_path, ledger_path=ledger_path,
    ):
        if row["status"] != RETIREMENT_ELIGIBLE:
            continue
        retired.append(
            policy_registry.set_status(
                row["policy_id"], policy_registry.RETIRED,
                reason=(
                    f"automatic research retirement: {row['paired_snapshots']} paired, "
                    f"incremental expectancy {float(row['incremental_expectancy_R']):+.3f}R"
                ),
                path=registry_path,
            )
        )
    return retired


def promote_to_champion(
    domain: str, new_champion_policy_id: str,
    *, actor: str, reason: str, allow_auto: bool = False,
    registry_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    proof_path: str | Path | None = None,
) -> dict[str, Any]:
    """Explicitly, deliberately replace the domain's Champion. Never called
    automatically unless AUTO_PROMOTION_ENABLED is True AND allow_auto=True
    is also passed by the scheduled caller -- a single flag flip can never
    silently enable automatic promotion by itself.

    Rollback-safe by construction: the OLD champion is demoted to PROBATION
    (never RETIRED/REJECTED by this function), so it remains fully available
    for immediate rollback() and keeps all of its own evidence."""
    if allow_auto and not AUTO_PROMOTION_ENABLED:
        raise RuntimeError(
            "AUTO_PROMOTION_ENABLED is False -- automatic promotion is disabled "
            "until the tournament mechanism is proven on real forward evidence"
        )
    current = policy_registry.current_champion(domain, path=registry_path)
    policy = policy_registry.get_policy(new_champion_policy_id, path=registry_path)
    if policy is None:
        raise KeyError(f"no such policy {new_champion_policy_id!r}")

    # Human approval authorizes the transition; it does NOT bypass science.
    # Re-evaluate the full simultaneous Challenger family against the exact
    # current evidence state so an arbitrary internal/API caller cannot
    # promote an unqualified policy by calling this mutation directly.
    batch = evaluate_promotion_batch(
        domain,
        registry_path=registry_path,
        ledger_path=ledger_path,
        proof_path=proof_path,
        persist_proofs=True,
    )
    eligibility = next(
        (row for row in batch if row.get("policy_id") == new_champion_policy_id),
        None,
    )
    if not eligibility or eligibility.get("status") != PROMOTION_ELIGIBLE:
        why = (eligibility or {}).get("reason") or "policy has no current scientific eligibility proof"
        raise RuntimeError(
            f"{new_champion_policy_id} is not scientifically PROMOTION_ELIGIBLE: {why}"
        )

    previous_champion_id = current["policy_id"] if current else None

    # Anti-flip-flop hysteresis. The very first baseline replacement is
    # intentionally exempt; once a Champion was promoted, it earns a minimum
    # PAPER tenure before another routine replacement may occur. Emergency
    # rollback remains separately available and immediate.
    if current is not None and list(current.get("promotion_history") or []):
        last_promotion = _dt((current.get("promotion_history") or [])[-1].get("at"))
        now = datetime.now(timezone.utc)
        if last_promotion is not None:
            tenure_days = (now - last_promotion).total_seconds() / 86400.0
            if tenure_days < MIN_CHAMPION_TENURE_DAYS:
                raise RuntimeError(
                    f"Champion hysteresis: only {tenure_days:.2f}d tenure; "
                    f"need {MIN_CHAMPION_TENURE_DAYS}d before routine replacement"
                )

    # Re-promoting the Champion that was just displaced requires a material
    # extra margin, not merely the ordinary promotion floor.
    if current is not None:
        prior = list(current.get("promotion_history") or [])
        if prior and str(prior[-1].get("previous_champion_policy_id") or "") == new_champion_policy_id:
            required = MIN_INCREMENTAL_EXPECTANCY_R + REVERSAL_INCREMENTAL_MARGIN_R
            if float(eligibility.get("incremental_expectancy_R") or 0.0) < required:
                raise RuntimeError(
                    f"reversal hysteresis requires incremental expectancy >= {required:+.3f}R"
                )

    previous_champion_id = current["policy_id"] if current else None
    if current is not None:
        policy_registry.set_status(
            current["policy_id"], policy_registry.PROBATION,
            reason=f"demoted: promoted {new_champion_policy_id} ({reason})",
            path=registry_path,
        )
    promoted = policy_registry.set_status(
        new_champion_policy_id, policy_registry.CHAMPION,
        reason=f"promoted by {actor}: {reason}",
        path=registry_path,
    )
    store = policy_registry.load_registry(registry_path)
    record = dict(store["policies"][new_champion_policy_id])
    record["promotion_history"] = list(record.get("promotion_history") or []) + [{
        "at": promoted["lifecycle_history"][-1]["at"],
        "actor": actor,
        "reason": reason,
        "previous_champion_policy_id": previous_champion_id,
        "eligibility_proof": eligibility,
        "policy_manifest_fingerprint": policy_registry.policy_manifest_fingerprint(policy),
    }]
    store["policies"][new_champion_policy_id] = record
    policy_registry.save_registry(store, registry_path)
    log.info("evolution_champion_promoted", domain=domain, policy_id=new_champion_policy_id,
             previous_champion_policy_id=previous_champion_id, actor=actor)
    return record


def rollback(
    domain: str, *, actor: str, reason: str, registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Demote the current Champion and restore the most recently demoted
    PROBATION policy that was itself once CHAMPION. No destructive overwrite:
    every policy's evidence and lifecycle history stays intact regardless."""
    current = policy_registry.current_champion(domain, path=registry_path)
    if current is None:
        raise RuntimeError(f"no current champion for domain {domain} to roll back")

    candidates = [
        p for p in policy_registry.list_policies(domain=domain, status=policy_registry.PROBATION, path=registry_path)
        if any(h.get("status") == policy_registry.CHAMPION for h in p.get("lifecycle_history") or [])
    ]
    if not candidates:
        raise RuntimeError(f"no prior CHAMPION available in PROBATION to roll back to for domain {domain}")
    restore_to = max(candidates, key=lambda p: str(p.get("created_at") or ""))

    policy_registry.set_status(
        current["policy_id"], policy_registry.PROBATION,
        reason=f"rolled back by {actor}: {reason}", path=registry_path,
    )
    restored = policy_registry.set_status(
        restore_to["policy_id"], policy_registry.CHAMPION,
        reason=f"restored by {actor} on rollback: {reason}", path=registry_path,
    )
    log.warning("evolution_champion_rollback", domain=domain, from_policy_id=current["policy_id"],
                restored_policy_id=restore_to["policy_id"], actor=actor, reason=reason)
    return restored
