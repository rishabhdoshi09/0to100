# Cycle 2 engineering and research checkpoint (21 Nov 2026)

This is a factual journal of the autonomous cycle that followed `f92a3e80`/`d0b668fe`. It holds no secrets.

- **Live trading:** HARD-LOCKED throughout.
- **Production:** nothing deployed or restarted (`125880a6`).
- **Data:**
  - No production learning record was modified.
  - Learning files were COPIED read-only (sequential copy; SQLite through `?mode=ro` backup) to the session scratchpad for the learning report.
  - Research ran on the offline SSD copy only.

## Start state

- **Branch:** `claude/settlement-batch-20261118`, HEAD `d0b668fe` (docs only), equal to origin.
- **Tested candidate:** `f92a3e80`. Its 5 workflows were green and 503 of 503 local files passed. It was reported in #333 comment 7050466612.
- **Relay:** polling; the last delivery was acked; nothing new on #333 at start.

## 1. The remaining verification findings

| Finding | Verdict | Fix | Tests (fail on `d0b668fe`) |
|---|---|---|---|
| QT-06 research-outcome identity | **Confirmed in production data.** 0 of 6,594 forward rows carry a feature id, and 0 forward counterfactual labels exist; all 5,949 feature labels are HISTORICAL_REPLAY. | A paper decision carries the id of the feature observation it froze (`feature_decision_id`), separate from the forward-row id. A settled outcome labels it only through `verified_feature_link`: the observation exists, has the same symbol, is a PAPER_FORWARD decision and comes from the same scan. Conflicts and legacy rows are reported in the outbox result (`applied;feature=<status>`) and never matched. Verified end to end: the board and the paper cycle derive the same id. | 5 |
| QT-07 GOOD_WAIT recovery | **Confirmed.** The fill was dropped while the archive was empty. | An unverifiable fill is queued on the WAIT (`later_entry_pending`) and re-verified before any price-path settlement. It is credited only when it verifies LIVE, and it is dropped (never credited) when EXPIRED or still unverifiable after 45 days, measured against the settled session. | 3 |
| QT-08 sampling stability | **Resolved by experiment** (`docs/research/qt08_sampling_stability_2026-11-21.md`). Point-in-time correctness and unbiasedness held. Reproducibility and certification stability failed. An unexpected second source, the even-stride SYMBOL sample, was worse than the decision grid. | Session-anchored decision grid plus hash-rank symbol sample. `SAMPLING = hash_rank_session_anchored_v1` is a new gate program, so earlier attestations fail closed and the supervisor reruns. Production-realistic result: 11% → 99.9% of decisions kept on append; the phase-only shift fell from a median 0.84 SE (max 1.97) to 0.00 (max 0.13). | 4 |

## 2. Broker and account-risk audit (independent GPT, read-only, whole tree at `d0b668fe`)

**Verified correct by the auditor:**

- The canonical live lock is hard-coded. Environment flags, arming, credentials and missing settings cannot unlock it.
- Every named Kite order and GTT method checks the lock.
- Product API routes have no direct order route.
- Telegram taps are paper-only under the account lock.
- The US autopilot is paper-only, in a separate USD pool.

| Finding | Severity | Status | Notes |
|---|---|---|---|
| B05 installed paper composer ignored BLOCKED_RECOVERY and an unreadable permission | P0 | **Fixed** | This is production code. It now refuses exactly as the base method does. |
| B02 legacy and manual paper entries over 1% risk and 10% notional | P0 | **Fixed** | Hard caps at `account_exposure_gate` against account capital (or the modern book's realized-equity risk capital). `place_trade` writes every INR paper journal entry through the gate under a now re-entrant lock. The legacy autopilot sizes within the caps. |
| B03 F&O did not share the 5 slots and the one position per underlying | P0 | **Fixed** | One account-wide count; the F&O book refuses an underlying already open in equity. |
| B04 a stale F&O cycle could erase another lane's position | P0 | **Fixed** | One cross-process cycle lock over read → modify → commit. |
| B01 the Kite proxy exposed the authenticated SDK (`__self__`/`__wrapped__`); AlgoLab ran typed code beside it | P0 | **Fixed (defence in depth)** | Plain-function wrappers. The archived live runner no longer executes typed code. **In-process isolation is impossible**, so a separate broker process is a live-readiness gate (below). |
| B06 legacy reconciliation closed LIVE rows on a price cross | P0 | **Fixed** | A cross is now an incident. A sell closes a row only when it is provably after its entry. Durable fill identity is still needed before live (gate). |
| B07 the lock consumed pending GTT recovery; a partial fill was marked ENTRY_FAILED | P1 | **Fixed** | Refused or unresolved protection stays PENDING_GTT; filled quantity is protected. |
| B10 the kill switch had a 60-second cache, an unreadable file read as clean, and writes were swallowed | P1 | **Fixed** | The durable halt is read on every decision, an unreadable file is a halt, and writes are atomic and confirmed. |
| B08 the legacy Zerodha adapter has no idempotency after an accepted timeout | P1 | **Open: live-readiness blocker** | Dormant; the real wrapper blocks every submission today. |
| B09 the EMS reports CLOSED after a protection failure without a broker exit | P1 | **Fixed (state); exit primitive open** | It is CLOSED only when the broker is flat. Otherwise RECONCILIATION_REQUIRED, with a critical incident and entries blocked. The adapter has no exit/sell primitive, which is a live-readiness blocker. No production adapter uses this EMS. |

**Behaviour changes (documented):**

- Conviction sizing and the FAST-BOOK cap relax can no longer exceed 1% risk or 10% per name.
- Tests that seeded over-limit books through `place_trade` now seed through the journal, or use house-sized trades.

## 3. Why there is no edge (descriptive, discovery data only)

`docs/research/loss_attribution_2026-11-21.md` covers the production walk-forward on the SSD copy, 400 symbols, STRICT tier, discovery decisions only. It has 7,394 trades and 12,199 detections over 78 independent blocks; the holdout year is counted, never measured.

- **No stock-selection alpha.** Taken over EVERY detection, the 10-session excess over the equal-weight market is **−0.17% [−0.49, +0.15]%**. The +1.8% "edge" of filled trades is look-ahead: a buy-stop fills only after a rise, and unfilled detections show −3.2%.
- **Gross ≈ 0, costs decide.** Gross is −0.025R [−0.08, +0.03]. Costs are **0.112R per trade**, because stops sit about 3% from entry. Net is **−0.137R [−0.21, −0.07]**. Against the market, beta is 0.55 and alpha −0.99% per trade.
- **Where the R goes.**
  - Stops: 31% of trades at −0.80R.
  - Breakeven scratches: 34%.
  - Time exits: 28%.
  - Targets: only 7.7% (+1.78R).
  - Trades of 1–3 sessions: −0.20R.
- **No pre-declared exit geometry, regime, breadth state or signal is positive.** More confirmation is worse. Quality concentration cannot be measured, because the ledger's `confidence` is never populated.
- **Conclusion.** Tuning this family's geometry would be curve fitting on a zero-alpha selection. It is not done.

**Pre-registered hypothesis tested this cycle: PR-2026-11-21-LOWVOL** (frozen `660ccc89` before any look; spec SHA `e1e21ebb…`).

| Step | Commit | Result |
|---|---|---|
| Discovery | `0d60d8f2` | Alpha **+0.43%/month**, p = 0.049 (PASS), beta 0.59, Sharpe 1.65 vs 1.35, max drawdown −15.5% |
| Single held-out look | `c2931cad` | Alpha **−0.30%/month**, p = 0.70: **NOT_SIGNIFICANT**. The pass did not replicate, and no paper experiment is proposed. |

Across this week's two portfolio pre-registrations (momentum and low volatility), Bonferroni at 0.025 was never met. **Scientific standing: no supported edge. Further evidence is necessary.** The next clean source is the frozen prospective 52-week-high test (owner approval to wire into the shadow ledger).

## 4. The GPT checkpoint over this cycle (`d0b668fe..dead3b67`, one read-only pass)

The pass reported 10 findings: 7 P0 and 3 P1. Each was verified by a test that fails on `dead3b67`. All 10 are fixed:

| Finding | Fix |
|---|---|
| C2-01: a rolling-grid run could attest as production | A non-production grid is research-only: it never reserves a look and is never saved. |
| C2-02: a hash-RANK sample let a future listing evict an incumbent's whole history | Bernoulli hash-THRESHOLD sample (`HASH_SAMPLE_FRACTION` 0.2, ~694 names). Membership is per symbol. A rank cut applies only past the 800 cap. `SAMPLING = hash_threshold_session_anchored_v1`. |
| C2-03: a configured 10 slots allowed a sixth position; the 5% cap used a rounded percentage | The slots are clamped at the house 5 (account-wide). Open risk is compared in unrounded rupees. |
| C2-04: legacy reconciliation erased a cancelled partial fill | A positive fill is never ENTRY_FAILED. |
| C2-05: the proxy exported the SDK through `__getstate__`, pickle and copy | Those escape hatches are refused. In-process isolation is still impossible, which stays a live gate. |
| C2-06: the feature link checked neither the decision nor its levels; a storage outage dropped a label for good | The link now requires the observation's own `decision_id`, equal entry/stop/target, an explicit PAPER_FORWARD class and the same scan. `get_observation_strict` raises on an outage, which keeps the event retryable. The replay comparison binds `feature_decision_id`. |
| C2-07: a symbol boolean could credit GOOD_WAIT | A later entry must be a verified fill session applied to a WAIT whose ordinal verifies LIVE. Ingestion refuses a GOOD_WAIT without `later_entry_verified`. |
| C2-08: one failed GOOD_WAIT ingest stalled every queued row | Each row is handled independently; errors are reported. |
| C2-09: the learning count counted SUPPORT on an unranked WAIT and simulated rows | Forward provenance is filtered before dedupe. A score effect counts only on a ranked decision. The unrecorded challenger adjustment is stated as a limit. |
| C2-10: stricter account settings made the modern engine fail instead of size down | Sizing is clamped inside the account caps. |

As agreed, there was no further review loop this cycle.

## 5. Learning effectiveness (copy of production learning data, 20 Nov 23:40 IST)

`research/learning_effectiveness.py` (read-only, refuses `/Volumes`):

| Measure | Value |
|---|---|
| Forward ledger rows | 6,594 (5.24 cycle copies per decision) |
| Independent forward decisions | 1,259 (1,060 REJECTED, 198 AVOID, 1 TAKEN) |
| Settled / pending | 1,003 / 256 |
| **Settled counterfactuals that ever reached learning** | **188 of 1,002 (18.8%)**; 814 never did (the bypassing second settler L-01 and at-most-once crash loss in the deployed build) |
| Outbox | absent (deployed build predates A-3) |
| Decisions whose policy effect is recorded BLOCK | 1,059, of which only **37** had the history-first SYSTEM gate as the binding reason; **1,022 were rejected by another gate** |
| **Decisions changed by a learned policy** | **0** |
| Policies that selection can consult | 0 of 11 (7 REJECTED, 2 OBSERVING, 2 historical-only / affects_selection=false) |
| Policy sample sizes | inflated by cycle copies: `REJECT::LOW_QUALITY_SETUP` n = 3,083 against 94 independent units claimed (what the owner-approval `apply repair` rebuilds) |
| Feature labels by lane | 5,949 HISTORICAL_REPLAY; 0 forward counterfactual (the QT-06 gap) |
| ML challenger | 21 of 25 features constant zero; 0 real forward observations |
| **Feature corpus** | **18 of 30 decision features are constant** in the latest 5,000 observations: `regime`, `vix`, `breadth_pct_above_50dma`, `rel_strength`, `rs_percentile`, `sector_rank_percentile`, `sector_strength`, `volatility_percentile`, `index_trend`, `correlation_regime`, `delivery_pct`, `usdinr`, `crude_usd`, `pct_from_pivot`, `base_days`, `base_depth_pct`, `contraction_ratio`, `accum_score`. `decision_adapter` reads them from recommendation cards that never carry them, so the learning corpus has no market context. Next cycle: fill them point-in-time at the scan → card boundary (with a PIT test), then re-measure. |
| Champion/challenger | 0 challengers with forward evidence |

**Metric fix.** `learning_scorecard.learning_influence` now counts a decision as changed only when a learned policy was its binding reason or moved its score. History-first blocks, recorded-but-not-binding effects and the exploration lane are reported separately. The A-10 reader fix alone would report about 1,061 "changed" decisions on this data, where the truth is 0.

**Conclusion.** Learning cannot be promoted, and the reason is measurable. Most settled outcomes never reached it. The counts it did receive are inflated duplicates. No policy is consulted. The one forward paper trade cannot calibrate anything. The deployed build would keep losing about 80% of outcomes; the development build fixes delivery (A-3, L-01..L-09, QT-02). Whether the 814 lost outcomes are replayed into learning is an owner decision.

## 6. Readiness gates (independent; each must hold for its scope)

| Gate | Status | Evidence / blocker |
|---|---|---|
| Source correctness | improving | 16 + 5 + 10 independently found defects fixed this week, each with a base-failing test |
| Five GitHub CI workflows | see the cycle report on #333 | exact-SHA results are posted there, for the head that carries this journal |
| Local regression coverage | see the cycle report on #333 | guarded full suite on that same SHA |
| Historical data integrity | partial | CA coverage protocol 2 and conflict carry-forward; the SSD copy's coverage is legacy UNBOUND; no symbol rename map |
| Point-in-time validity | holds | future-poisoning tests; QT-08 shared-decision outcomes identical |
| Statistical evidence | **fails** | no signal or combination has a proven edge; see §3 |
| Risk-control correctness | improved | house limits now hard at the one gate (B02/B03), F&O serialised (B04) |
| Learning correctness | delivery fixed in dev; effect = 0 | §4 |
| Paper execution reliability | in development | 1 taken paper trade; deploy pending owner |
| Realistic execution costs | modelled | 0.32% round trip; costs are about 0.11R per trade at the typical stop width (§3) |
| Mac host operational verification | owner | deploy and storage behaviour (`docs/audits/2026-11-21-overnight-checkpoint.md`) |
| Backup and rollback | documented | runbook in audit §10.3 / §11.7 |
| Forward-paper performance | **insufficient** | 0 settled paper trades |
| Real-money execution authorization | **LOCKED** | also requires a separate broker process (B01/C2-05), durable order/fill identity (B06/B08) and an EMS exit primitive (B09) |

## Open / owner decisions

- `deploy <sha>`;
- `apply repair`;
- whether to replay the 814 settled-but-unlearned outcomes;
- resolving any carried CA conflict;
- C-5;
- the 52WH shadow test;
- a full-history run.
