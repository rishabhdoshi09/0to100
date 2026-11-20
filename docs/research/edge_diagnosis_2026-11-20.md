# Why the signals show no edge: point-in-time diagnosis, one pre-registered test, readiness matrix

20 Nov 2026. Development and offline research only. Nothing here was deployed. No production ledger was touched. The whole-universe full-history job was not run.

## 1. Negative-result diagnosis (P0)

### Method

`research/edge_diagnosis.py` runs the **production** walk-forward (`scan.signal_backtest.run_backtest`) on an offline SSD copy. It uses:

- the production scanner, with `evidence_gate=False`;
- the one rulebook, `core.evaluation_spec`;
- the cost model in `core.costs` (0.22% taxes and fees plus 0.10% slippage, round trip).

Hooks capture every detection. Each one is classified as a risk skip, a no-fill, or a filled trade with its decision date.

**Point-in-time rules:**

- The scanner sees bars up to the decision session only.
- A trade exists only when the buy-stop traded.
- Outcomes are exactly 10 official sessions.
- A trade is excluded, and counted, when its symbol was not listed, or was already delisted, on the decision date (`universe_history.json`).
- Corporate actions are verified only from 2021-08-28 (`ca_coverage.json`). Trades are tiered:
  - **STRICT:** the decision is at least 252 sessions after coverage starts, so the scanner's whole 52-week lookback is adjusted.
  - **ENTRY_ONLY:** the decision is after coverage starts, but the lookback is partly uncovered.
  - **UNCOVERED:** before coverage. A sensitivity check only, never evidence.

**Benchmark:** an equal-weight index of every stock in the copy that traded that day, delisted names included. Daily returns are winsorised at ±20%, and a day needs at least 50 names. The window runs from the decision close to the exit close. Nifty history in the copy starts only in December 2025, so it cannot cover 2021–2025.

**Statistics:**

- Nominal 95% intervals per signal from `research.overlap_inference`: the offset-subsample bound, widened for lag-10 residual dependence. There is no multiplicity correction; this is a diagnosis, not an admission test.
- The minimum detectable effect (MDE) is at 80% power, one-sided 5%.

**Verdicts on net R:**

| Verdict | Meaning |
|---|---|
| NEGATIVE_EDGE | The interval is below zero: measurably harmful, not a power problem. |
| NO_MATERIAL_EDGE | The interval straddles zero, but its top is below the +0.05R BUY floor. |
| UNDERPOWERED | The interval straddles zero and reaches the floor: the data cannot tell. |
| POSITIVE_IN_SAMPLE | A hypothesis for a prospective test, never a BUY. |
| INSUFFICIENT | Fewer than 30 trades or fewer than 20 independent blocks. |

**Drivers named per signal:**

- COSTS_FLIP_SIGN: gross positive, net not.
- MOST_ENTRIES_NEVER_TRIGGER: fill rate below 50%.
- MARKET_DRIFT_NOT_SELECTION: the benchmark fell but the excess return did not.
- UNDERPERFORMS_EQUAL_WEIGHT_MARKET.

**Safety:**

- No report is written.
- The walk-forward's alpha look is never reserved, because a discarded report spends none (`ae55b791`). A throwaway ledger is kept as defence in depth.
- Every hook and environment variable is restored.
- There is no network access.
- A copy under `/Volumes` (the live runtime) is refused.

### Reproduce

```
# 1. once, before market hours: copy the live runtime to the internal SSD (sequential, read-only)
#    logs/bhav/store_cache.pkl, logs/ca_events.json, logs/ca_coverage.json, logs/universe_history.json
# 2. offline, with the desk idle:
nice -n 19 venv/bin/python -m research.edge_diagnosis --root <SSD copy root> --symbols 200 \
    --out docs/research/edge_diagnosis_2026-11-20.json
```

### Results on the realistic sampled copy

Run on 20 Nov 2026, after the cross-review fixes:

- Data: an offline SSD copy, 1,761 sessions (2019-08-23 to 2026-11-19), 3,469 symbols, 424 with corporate-action events.
- Sample: 200 symbols spread evenly across the store, decisions every 5th session, `lookback_sessions=1691`.
- Fates: 12,701 detections → 7,502 filled trades, 5,192 untriggered, 7 risk skips. 476 windows were skipped because they were not exactly 10 official sessions.
- Corporate-action coverage: 2021-08-28 to 2026-11-19.
- Survivorship: 41 sampled symbols stopped trading before the copy's last session, losing about 82 sample points.

**STRICT tier (all signals pooled):**

| Measure | Value |
|---|---|
| Trades | 5,239 |
| Independent 10-session blocks | 103 |
| Gross mean | −0.013R, CI [−0.103, +0.076] |
| Net mean | **−0.092R/trade, 95% CI [−0.186, +0.001]**: NO_MATERIAL_EDGE |
| Excess return vs the equal-weight market | −1.38% per trade, CI [−1.92, −0.84]%: measurably worse than the market |
| Mean round-trip cost | 0.079R |
| False breakouts (stopped out within 3 sessions) | 18.4% |

**Net R by cost multiple:**

| Cost multiple | Net R |
|---|---|
| 0× | −0.013 |
| 0.5× | −0.053 |
| 1× | −0.092 |
| 2× | −0.171 |
| 1× + 0.10% slippage each side | −0.142 |

**Per signal (STRICT tier):**

| Signal | Trades | Net R mean [95% CI] | Gross R | Fill rate | Verdict |
|---|---|---|---|---|---|
| CUP_HANDLE | 1,570 | −0.042 [−0.195, +0.111] | +0.004 | 0.79 | UNDERPOWERED (costs flip the sign) |
| POCKET_PIVOT | 1,318 | −0.028 [−0.183, +0.127] | +0.022 | 0.79 | UNDERPOWERED (costs flip the sign) |
| ASC_TRIANGLE | 1,286 | −0.177 [−0.395, +0.040] | −0.003 | 0.70 | NO_MATERIAL_EDGE |
| PRE_BREAKOUT | 1,281 | −0.089 [−0.281, +0.103] | +0.042 | 0.83 | UNDERPOWERED (costs flip the sign) |
| MOMENTUM | 1,121 | −0.078 [−0.246, +0.091] | −0.034 | 0.65 | UNDERPOWERED |
| PULLBACK_SUPPORT | 985 | −0.086 [−0.248, +0.076] | −0.039 | 0.45 | UNDERPOWERED (most entries never trigger) |
| FLAT_BASE | 650 | −0.250 [−0.604, +0.104] | +0.004 | 0.62 | UNDERPOWERED |
| VCP | 103 | −0.358 [−3.50, +2.78] | +0.045 | 0.61 | UNDERPOWERED |
| BREAKOUT_52W | 28 | +0.168 [−0.62, +0.95] | +0.218 | 0.93 | INSUFFICIENT (< 30 trades) |

The other signals all show a negative or zero net mean and are UNDERPOWERED. Their rows are in the JSON next to this file.

**What the numbers say:**

- **No signal shows an edge.** Every signal with ≥ 30 trades has a net mean ≤ 0.
- **Costs are a real driver, but not the only one.** Even at zero cost no signal clears the +0.05R floor. Gross means sit at about zero, and the pooled excess return against the equal-weight market is negative with a CI that excludes zero. That points to selection, not drift.
- **More confirmation does not help.** Mean net R falls with more simultaneous signals: −0.068 for one, −0.105 for two, −0.110 for three or more.
- **The verdict holds on every slice.** The ENTRY_ONLY tier (792 trades, −0.008R) and the UNCOVERED sensitivity tier (1,471 trades, −0.012R) are UNDERPOWERED and centred near zero. Removing the 152 windows with a phantom corporate-action gap leaves the verdict unchanged.
- **The data could resolve +0.05R, and found nothing.** The pooled minimum detectable effect at 80% power is +0.118R, so an edge at the +0.05R floor would need about 572 blocks. The data rule out an edge larger than about +0.12R, and find no sign of a smaller one.

**This is an in-sample diagnosis only.** It does not test or claim profitability, and no multiplicity correction was applied. The full output is `docs/research/edge_diagnosis_2026-11-20.json`.

### Hermetic evidence

`tests/test_edge_diagnosis.py` (20 tests) covers:

- the verdict boundaries and the MDE formula;
- the coverage chain and the tiers;
- point-in-time membership;
- the equal-weight benchmark (including delisted names);
- duplicated evidence gaining no precision;
- cost monotonicity and a cost-flipped edge;
- a **future-poisoning** check: multiplying every bar after a cut by 3 changes no earlier decision, fill or outcome;
- the no-write, no-alpha and hook-restoration guarantees.

## 2. One pre-registered prospective test (P1)

`research/preregistered/pr_2026_11_20_52wh.json` is frozen. Its SHA-256 is pinned in `research/prereg_52wh.py` and checked by CI.

| Item | Frozen value |
|---|---|
| Thesis | 52-week-high proximity, close ÷ 252-session high. The literature mechanism is anchoring (George and Hwang, 2004). India-specific evidence is not assumed. |
| Universe | Point-in-time EQ members with ≥ 260 sessions, price ≥ ₹20, and a 20-day median traded value ≥ ₹5 crore |
| Schedule | Every 10th official session from 2026-11-23. Frozen before the next 09:15 IST, and a missed session is never backfilled. |
| Entry and exit | Open of the next session to the close of the 10th session |
| Primary outcome | Spearman IC per non-overlapping block; one-sided t-test, α = 0.05, m = 1 (outside the walk-forward family) |
| Power | IC 0.03 at an SD of 0.10 per block needs **69 blocks (~2.7 years)**. At an SD of 0.12 it needs 99; at 0.08, 44. |
| Stopping | One final analysis at 69 blocks; a non-binding futility look at 35; no early stop for efficacy |
| On PASS | Only a proposal for an owner-approved paper experiment. Never a BUY, and never a gate or risk change. |
| Disclosure | The author had seen in-sample event-signal diagnostics. This thesis is a different object, taken from the literature, and no historical return of its score was computed before freezing. |

Nothing is scheduled. Wiring the thesis into the shadow ledger is a deployment and needs the owner's approval. The honest headline is that a literature-sized edge needs **years** of prospective blocks. No backtest can shortcut that without becoming the selection bias this test exists to avoid.

## 3. Prospective shadow and forward settlement (P1)

The current code was verified, and no defect needed a new test.

- **Rejections and missed winners.** `product/challenger_shadow.build_predictions` freezes **every** scan record with its champion verdict, not only BUYs. In a probe, BUY, REJECTED and WATCH rows were frozen together. After the exact 10-session window, 6 non-BUY rows had positive returns, so missed winners can be measured.
- **Null-candidate sessions.** A zero-candidate scan is recorded as a durable session marker (probe: `PROSPECTIVE 0`; tests `test_a_zero_candidate_session_still_gets_a_durable_marker` and `..._missed_gaps`).
- **Duplicates.** Re-recording a session returns `already_recorded`. Symbols are deduplicated within a session, and one scan_id can never be frozen for two sessions. Signal-log rows are unique per symbol, day and type.
- **No label leakage.** Predictions are frozen before the next 09:15 IST, future bars do not change them, and a backfilled or edited row is never evidence. All of these have existing tests.
- **Exits.** Signal-log settlement (UNIFIED_BUY and EDGE_GATED_SHADOW rows) uses `core.evaluation_spec.simulate_trade`: gap-aware stop and target and the breakeven trail. The shadow ledger measures a fixed 10-session window by design.
- **Counterfactual horizon: independent cross-review (20 Nov).** The GPT cross-review of `9636b030` found the same 5-session or legacy-path grading in three more places, all now on the rulebook: historical replay (classification and training R), the decision simulator, and evolution equity shadow grading. It also found that settlement windows could stretch across a missing session or a live overlay. Those now require exactly the next official sessions (`core/session_window.py`). See `docs/audits/2026-11-20-gpt-crossreview.md`.
- **Counterfactual horizon (P0, confirmed and fixed in development).** The two production settlement loops judged rejections by a 5-session close-to-close move that ignored the stop, while taken trades use the 10-session rulebook. The outcome job could also stamp that label on a **taken** trade, which blocked its real paper result for good. Rejections are now judged exactly as taken trades are, and every row carries `eval_spec`. Legacy labels never reach a learning score, and taken rows are never settled as counterfactuals. Details, tests and the reversible production migration are in `docs/audits/2026-11-20-counterfactual-horizon-migration.md`; the measured impact on a copy of the production ledger is in §1.

## 4. Desk statuses and alpha-ledger integrity (P1)

**Distinct no-BUY statuses (fixed).** `product/buy_admission.classify` is shared by Home, Opportunities and Telegram:

| Code | Meaning | Action shown |
|---|---|---|
| SCAN_NOT_CURRENT | The data, or the saved scan, is older than the current session. Home withholds any row from it. | Refresh and rescan; log in to Kite when the session is missing |
| EVIDENCE_UNAVAILABLE | The walk-forward evidence cannot certify anything (stale, unprovable, wrong rulebook, budget) | Wait for the weekly walk-forward to regenerate it |
| NO_PROVEN_EDGE | Setups were found and withheld as WATCH | None; WATCH is correct |
| NO_SETUP | Nothing qualified | None |
| ELIGIBLE_BUY | At least one BUY cleared every gate | — |

Before the fix:

- Home said "Nothing to buy today" in all four cases, and explained an old scan as today's.
- Telegram's EOD message called every detection "Qualified setups", WATCH included.

It now reports "BUY (proven edge)" separately from "Setups detected (research, WATCH included)". Paper EXPLORATION stays a separately labelled lane, outside the eligible tiers.

**Alpha-ledger crash and restart (one fix).** `run_backtest` reserved a look **before** deciding whether to save the report. A discarded thin run (too few symbols, for example a fresh process before the store loads, or no signal) therefore burned one of the year's 52 looks. The save decision now comes first, from counts only. New tests:

- a discarded run reserves nothing and cannot certify;
- a crash between reservation and the test spends the look, and the rerun spends the next one;
- a hard kill (`os._exit`) mid-migration leaves the legacy ledger intact, because SQLite rolls the hot journal back, and the next open finishes the migration with every look still spent.

**Remaining availability risk (fail-closed):** with exactly 52 looks per year and a weekly walk-forward, any extra saved run (a forced rerun, or a schema-bump rerun) exhausts the year early. The later reports then get α = 0 and admit nothing, so no unsafe BUY can result. Changing the allocation would change alpha thresholds and is out of scope.

## 5. Readiness decision matrix (P2)

| Dimension | Status now | Evidence | Needed before requesting a paper-only deploy | Needed before the learning-ledger repair |
|---|---|---|---|---|
| GitHub CI | Green on the exact SHA (5 workflows) | issue #333 reports | Necessary, **never sufficient** | Necessary, never sufficient |
| Mac host deployment | Production `125880a6`; candidate not deployed | runbook audit §10.3 and §11.7; backup drill passed 19 Nov | Owner "deploy <sha>"; a fresh storage backup; a deploy outside market hours and between paper slots; rollback SHA recorded; afterwards health READY, a first FRESH scan, the schema-5 walk-forward rerun, and the BUY section showing an honest reason | Owner "apply repair" after reading the dry-run diff (7 policies, 0 status changes); backup `learning_policies_20261119_092827_pre_repair_review.json` |
| Risk-control safety | Live HARD-LOCKED; 1% / 10% / 5% limits enforced | `test_money_paths`; `live_lock_verified` in health | After deploy, `live_lock_verified` true and Telegram taps paper-only | Unchanged by the repair (policies only) |
| Market-data freshness | Health READY this morning; Kite needs a daily login | `/api/health` | Kite login and a scan FRESH for the current session before judging any day | Not dependent |
| Forward-paper maturity | **Insufficient**: 1 paper trade taken, 0 settled; tracked BUYs over 26 sessions at −0.127R, CI [−0.31, +0.06] | learning scorecard (19 Nov snapshot) | Not a deploy blocker for paper-only, but no claim of maturity until ≥ 20 settled paper sessions | The repair must not count copies as evidence (fixed in `524a11cd`) |
| Independent code review | GPT cross-review of `9636b030`: 33 findings. Every confirmed defect is fixed or listed open with a reason | `docs/audits/2026-11-20-gpt-crossreview.md`; `tests/test_crossreview_20261120.py` | The open P1 items (B-6 delisting truncation, A-3 at-most-once ingest, the rest of C-8) known and accepted | A-4: mixed legacy `SETUP::` aggregates must be rebuilt by the repair, not kept |
| Statistical strategy edge | **None established** | walk-forward: every signal with ≥ 30 trades is negative; §1 diagnosis; tracked BUYs not distinguishable from zero | Not required for a paper-only deploy, but NO_ELIGIBLE_BUY must stay the expected outcome | Not required |

Do not approve a deployment or the repair on CI alone. The bars above are the evidence a request should cite.

## 6. Open items for the owner

1. `deploy <sha>`, paper-only, with the checklist in §5.
2. `apply repair` (learning policies), separately.
3. The counterfactual-horizon migration of production ledgers (plan in `docs/audits/2026-11-20-counterfactual-horizon-migration.md`). The earlier repair dry run is superseded and must be re-run. Separately: whether the policy floors become `4 × horizon` / `8 × horizon` (stricter).
4. Whether to start the pre-registered shadow test (needs wiring at deploy time; about 2.7 years to its single analysis).
5. The full-history walk-forward run (still not approved).
6. **Admission rule change (C-5).** A setup without tested evidence for its exact combination is now judged on its WEAKEST evidenced signal, not the average. This is strictly tighter. The owner may instead require tested evidence for the exact combination; that would be stricter still.
