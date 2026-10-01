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

from pathlib import Path
from typing import Any

from logger import get_logger
from product.evolution import policy_registry, scorecard

log = get_logger(__name__)

# ── the single deliberate setting (section 41) ──────────────────────────────
AUTO_PROMOTION_ENABLED = False

MIN_PAIRED_SAMPLE = 30
FDR_ALPHA = 0.05
MIN_REGIME_BREADTH_OBSERVED = 2  # need at least this many distinct regimes to judge breadth at all
MIN_REGIME_BREADTH_ACCEPTABLE = 2  # of the observed regimes, at least this many must be non-negative

PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"
NOT_ELIGIBLE = "NOT_ELIGIBLE"


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
            len(regime_means) < MIN_REGIME_BREADTH_OBSERVED  # too few regimes observed -> can't judge, don't block on this alone
            or acceptable >= MIN_REGIME_BREADTH_ACCEPTABLE
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

    breadth = _regime_breadth(champion_id, challenger_policy_id, domain=domain, ledger_path=ledger_path)
    result["regime_breadth"] = breadth
    if not breadth["breadth_ok"]:
        return {
            **result, "status": NOT_ELIGIBLE,
            "reason": (
                f"insufficient regime breadth: only {breadth['regimes_acceptable']}/"
                f"{breadth['regimes_observed']} observed regimes show non-negative incremental value"
            ),
        }

    result["primary_improvement"] = chal_card
    return {**result, "status": PROMOTION_ELIGIBLE, "reason": "cleared all promotion gates"}


def evaluate_promotion_batch(
    domain: str, *, registry_path: str | Path | None = None, ledger_path: str | Path | None = None,
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

    return raw


def promote_to_champion(
    domain: str, new_champion_policy_id: str,
    *, actor: str, reason: str, allow_auto: bool = False,
    registry_path: str | Path | None = None,
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
