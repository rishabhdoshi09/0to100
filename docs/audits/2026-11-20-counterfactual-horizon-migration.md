# Counterfactual horizon: finding, fix, and a reversible production migration plan

20 Nov 2026. This is development work. **Nothing here has been applied to production.** The migration needs the owner's separate approval ("apply counterfactual migration"), and so does the learning-policy repair ("apply repair").

## 1. What the code actually did (verified)

| Evidence source | Horizon and rule | Feeds |
|---|---|---|
| Walk-forward backtest | 10 sessions, `simulate_trade` (buy-stop fill, gap-aware stop and target, breakeven trail), net of costs | BUY evidence gate |
| signal_log (UNIFIED_BUY, EDGE_GATED_SHADOW) | 10 sessions, `resolve_trade` → `simulate_trade` | Report Card, scorecard tracked BUYs |
| Paper book (taken trades) | `HOLD_DAYS = HORIZON_SESSIONS` (10), the same exits | `ingest_closed_trade` → SETUP:: and related policies; forward ledger `gross_R` |
| Decision journal | 10 sessions (`DECISION_HORIZON_SESSIONS`), close-to-close | decision report |
| Challenger shadow | 10 official sessions, open → close | shadow IC |
| **Forward-ledger rejections**: `forward_soak.settle_pending_from_market` (`SETTLE_HORIZON = 5`) and `autonomous_loop.settle_official_outcomes` (`horizon=5`, twice) | **5 sessions, close-to-close, stop ignored**, classified by fixed % thresholds | `ingest_counterfactual` → REJECT:: (±0.40), SETUP:: (+0.40 on an EVIDENCE_POLICY_BLOCK missed winner), ENTRY_QUALITY::GOOD_WAIT; feature-store labels (`counterfactual_R`); learning memory; scorecard and scoreboard rejection gates |
| decision_simulator, historical_replay | 5 sessions, close-to-close | simulation approval statistics; historical HIST_REJECT:: hypotheses (historical-only, never ACTIVE) |
| outcome_integrity re-probe | 5 sessions | resolver-error detection only, no labels |

**Intent.** `core/evaluation_spec.py` (since 2026-11-15) states that every evidence source reads its horizon and trade management from that module and resolves trades with `simulate_trade`. The forward-ledger settlement dates from 2026-09-01 and was missed by that unification. The 5-session setting is legacy, not a deliberate choice.

## 2. Two defects

1. **Incompatible rejection labels.** The labels came from a different horizon and ignored the stop. Three consequences:
   - A rejection stopped out on day 2 that recovered by day 5 was a MISSED_WINNER.
   - A buy-stop that never traded counted as a winner if the price drifted up.
   - A target hit on day 8 counted as FLAT.

   These labels were pooled with taken-trade outcomes. SETUP:: policies mixed real paper R with +0.40 counterfactual pseudo-R, and missed-winner rates were compared with taken win rates on a different clock.
2. **Taken trades blocked from their real result.** `settle_official_outcomes` did not skip entered rows. A taken paper trade still open 5 sessions after entry was settled as a counterfactual first. `attach_closed_trades` only attaches to rows whose `later_outcome` is empty, so the trade's real `gross_R` and `execution_adjusted_R` could never reach the forward ledger.

## 3. Fix (development, tested)

- `core.outcome_resolver.counterfactual_outcome` judges a rejection exactly as a taken trade is judged: `resolve_trade` (the signal_log resolver), the same costs (`core.costs.cost_in_r`), and the rulebook's `eval_spec` (`core.evaluation_spec.spec_tag()`, today `sim:h10:be2:gap`). It returns None until the horizon is published. Invalid levels give NO_LEVELS, never a guess.
- `counterfactual_learning.classify_rulebook` applies one rule to both sides: positive net R is MISSED_WINNER for a rejection and WINNER_TAKEN for a taken trade. A stop-out is AVOIDED_LOSER, a buy-stop that never traded is NOT_TRIGGERED, and invalid levels are UNMEASURED.
- Both settlement loops use it, and every settled row carries `eval_spec` and `counterfactual_R` (rulebook net R). **`settle_official_outcomes` never settles an entered row.**
- **Contamination guard.** `ingest_counterfactual` refuses any forward-outcome label whose `eval_spec` is not the current rulebook, and refuses UNMEASURED. Such a label updates no policy, no feature-store label and no dedupe claim. Learning memory, the scorecard rejection gates, the scoreboard and `learning_policy_repair` all count only comparable rows, and report the legacy count separately.
- **`learning_policy_repair` mirrors the guard.** A counterfactual-owned policy left with no comparable evidence is reset to n = 0 (OBSERVING) rather than kept on legacy values. **Its earlier dry run (7 policies, 0 status changes) is superseded and must be re-run before any "apply repair".**
- Adversarial tests:
  - `tests/test_counterfactual_horizon.py` (20 tests): stopped-out-then-recovered, never-triggered, winner after day 5, identical R and grade for rejected and taken, pending until published, invalid levels, legacy and foreign specs refused, dedupe, entered rows untouched by the outcome job (and the real close still attaches), pooling of rulebook rows only, and learning memory;
  - `tests/test_counterfactual_reconcile.py` (4 tests);
  - the legacy-only policy reset in `tests/test_learning_observation_dedupe.py`.

**Not changed:** the policy floors (`MIN_SESSIONS_ELIGIBLE` = 20, ACTIVE = 40). Their comment justified them as 4 and 8 independent 5-session horizons; under the 10-session rulebook they are only 2 and 4. Making them `4 × horizon` and `8 × horizon` (40 and 80) would be **stricter**, and no policy clears the current floors today. It is still a change in learning behaviour, so it is left as an owner decision. Decision_simulator and historical_replay still use 5 sessions. They are historical-only lanes that cannot promote a policy to ACTIVE, and are listed for a follow-up.

## 4. Reversible production migration (requires the owner's approval)

The production desk runs `125880a6`, which has none of this. The migration only makes sense **after** a deploy that contains the fix, because otherwise the old code keeps writing legacy rows.

1. **Measure first (read-only, offline, already allowed).** Copy `logs/product/forward_evidence.jsonl`, `logs/product/counterfactuals.jsonl` and `logs/product/learning_policies.json` to the SSD after market close. Then run:
   ```
   python -m research.counterfactual_reconcile --ledger <copy>/forward_evidence.jsonl \
       --counterfactuals <copy>/counterfactuals.jsonl --store-root <SSD store copy> \
       --out docs/research/counterfactual_reconcile_<date>.json --stage <copy>/staged
   ```
   It reports the legacy rows, the label transitions, the per-gate missed-winner rates before and after, and every blocked taken row. It writes migrated copies **only** into `--stage`.
2. **Owner review.** Review the transitions and the blocked taken rows. The decision: approve or reject the swap.
3. **Apply (owner-approved, outside market hours, desk stopped between paper slots):**
   1. Back up the three files plus `learning_observations` and the feature store under `runtime/logs/backup/<stamp>_pre_horizon_migration/`, and verify the backup checksums.
   2. Swap in the staged copies atomically (`os.replace`). In those copies, every legacy value stays under `legacy_later_outcome` / `legacy_counterfactual_classification` / `legacy_outcome`, and nothing is deleted.
   3. For each blocked taken row, the next EOD `attach_closed_trades` attaches the real paper close (its `later_outcome` is now empty). Check the count afterwards.
   4. Run `python -m product.learning_policy_repair` (dry run), review it, then run `--apply` only after a separate "apply repair". This rebuilds the counterfactual-owned policies from comparable units only.
   5. Verify the learning scorecard: `rejections_legacy_excluded` falls to the number still pending, the gates are recomputed, and no policy changes status unexpectedly.
4. **Rollback:** stop the desk, restore the three files and the observation ledger from the backup directory (the backup copy from step 3.1), and restart. Because the swap only adds `legacy_*` keys and new labels, rolling back loses no original value.

## 5. Not authorized here

Running step 3, the learning-policy repair, any deploy, and any change to the policy floors.
