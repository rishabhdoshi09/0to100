# QuantTerm — learning integrity, evidence and graduation audit (2026-11-18)

Branch: `claude/settlement-batch-20261118`. The host was deployed at `4e0c5d81`
(22:23 IST on 2026-11-18, previous build `6031824d`). The commits listed in §3
are **pushed but not deployed**; deploying them needs the owner's explicit
"deploy". Live trading is **LOCKED** and nothing here changes that.

Method: every number below was measured on **copies** of the production
runtime files (taken 22:54–23:10 IST, read on the internal SSD so the USB
runtime disk was not loaded), or on the running desk's read-only APIs.
Nothing was written to production state.

## 1. Verdicts

| Category | Verdict | Exact unmet conditions |
|---|---|---|
| RESEARCH READY | **NOT READY** | Tracked BUY net edge is INSUFFICIENT EVIDENCE: 16 sessions, −0.098 R/session, 95% CI [−0.25, +0.18]. Point-in-time data controls are 0/5 certified (institutional `data` = PARTIAL). The setup score has no useful discrimination (AUC 0.511). |
| FORWARD PAPER READY | **NOT READY** | 0 settled paper trades (need ≥100 over ≥40 sessions, then CI lower bound > 0). Execution-adjusted coverage is unmeasured. Stage 1 is not met. |
| OPERATIONALLY READY | **NOT READY** | The institutional operations, reconciliation, execution, protection and risk domains are BLOCKED or PARTIAL. 395 BUY outcomes are stuck (§2.4). The host soak, fault-injection and restore drill are **BLOCKED**: they need host evidence, and CI cannot provide it. |
| SUPERVISED LIVE ELIGIBLE | **NOT ELIGIBLE** | Stages 1–3 are not met. It needs a separate owner authorization. The interlock is LOCKED by design. |
| AUTONOMOUS LIVE ELIGIBLE | **NOT ELIGIBLE** | Supervised micro-capital stage not done. No real-fill analysis. Needs a second owner authorization. |

**Current graduation stage: 0 of 5.** The system has no demonstrated edge
yet. Its decision machinery is now measured honestly, and §4 shows that this
is far from guaranteed.

## 2. What was broken or underperforming, and why

### 2.1 Learning counted copies as evidence (fixed in `524a11cd`)

Every paper cycle re-freezes the names it rejects. Each copy was then settled
and ingested into the policy ladder as its own observation. On 11 Nov, 36
cycles froze 175 symbols as 3,744 ledger rows. Overall, 4,258 settled rows
were **814 independent decisions**:

| Policy | Stored n | Independent n | Sessions |
|---|---|---|---|
| REJECT::LOW_QUALITY_SETUP | 2,989 | 298 | 6 |
| REJECT::HISTORICAL_EVIDENCE_PENDING | 436 | 38 | 2 |
| REJECT::DD_GATE_FAILED | 289 | 21 | 2 |
| REJECT::EMPIRICAL_GATE_FAILED | 224 | 137 | 6 |

A policy at n ≥ 20 with an edge of at least 0.25 R SUPPORTs or PENALIZEs paper
selection, and at n ≥ 30 it may BLOCK. So copies of one day's names could have
moved decisions with almost no shrinkage. **None did**, because every stored
edge was below 0.25 R.

The fix:

- A durable SQLite observation ledger (`product/learning_observations.py`).
- One observation per (decision day, symbol, reason). A duplicate changes
  neither the policy ladder nor the calibration record.
- A claim is written before the increment, so a crash can only under-count.
- A **session-diversity floor**: ELIGIBLE needs ≥ 20 distinct decision
  sessions and ACTIVE needs ≥ 40. With a 5-session outcome horizon, that is
  about 4 and about 8 non-overlapping windows. Unknown coverage counts as zero.
- The scoreboard and learning memory now count decisions, not rows. The
  scoreboard had labelled itself "MEASURED" from 5,782 rows while 0 trades
  had settled.
- The taken-trade dedupe kept the alphabetically-last 2,000 keys rather than
  the newest, which is a latent double-count. All keys are now kept.
- Three tests reached their sample floors by ingesting one trade five times.
  They now use independent trades.
- Repair tool: `python -m product.learning_policy_repair`, dry-run by default
  (§6).

### 2.2 The settlement backlog blocked every EOD job for an hour (fixed in `4e0c5d81`, deployed)

`attach_settlement` rewrote the whole 11 MB forward ledger for every row.
After the 16–18 Nov outage, 3,784 rows took 63 minutes on the USB HDD
(about 1 row/s). Settlement is now batched: one rewrite per 250 or 500 rows,
plus a final flush. Counterfactuals are written to the ledger before the
learning ingest.

### 2.3 Product readiness said READY over a stale scan (fixed in `4b40a7bc`)

`/api/product-readiness` judged the scan by wall-clock age. At 22:42 IST it
reported score 100 / READY while `/api/health` was DEGRADED with "Scan
session 2026-11-17 behind expected 2026-11-18". It now uses health's
session verdict, and READY is impossible while any weighted lane is a blocker.

### 2.4 Tracked-BUY outcomes have not settled since 30 July (made observable in `4349bfd6`)

`update_outcomes()` hid everything behind `except Exception: pass`. The
outcome job reported SUCCEEDED every night while no `signal_log` row had been
written by it since **2026-07-30 23:22**. 395 BUYs with complete entry, stop
and target from late May and 21–30 Jul were never evaluated. The 107 November
BUYs are queued behind the same path.

The resolver itself is sound:

- On the API's real bars for three stuck rows, it returns FLAT / FLAT /
  NO_FILL.
- On a copy of the production database with synthetic bars, it settles 503
  rows.

So the failure is specific to the production run, and I have not yet
identified the cause.

2026-11-19 07:39 IST reproduction: the same code ran on a copy of the
production database, in a separate process, with each symbol's real bars
taken from the running API's `/api/chart/{symbol}`. It settled **527 of 607**
candidates in 29.5 s: 423 resolved, 104 never filled, 78 still inside their
horizon and 2 stuck (AMIRCHAND, too few bars), with no errors. That rules
out the code and the data. Only the nightly autonomy process's environment
remains.

**Root cause (found 2026-11-19 08:33 IST).** `install_outcome_liveness()`
(`98bb38ba`, 2026-09-06) replaces `jobs.run_outcome_resolution` with
`_run_outcome_resolution_light`, and the light handler never called
`update_outcomes`. The only other caller was the legacy `auto_scan` loop,
which the canonical stack does not start, so the last write was
2026-07-30. The 200-day catch-up (`0bc1f6a3`, 2026-11-11) and the
diagnostics (`4349bfd6`) both landed in the replaced handler, which never
runs, and both summaries read "outcomes resolved". Variant (A) reproduces
the symptom exactly: with an empty store in the process, 0 rows are
written and 500 are reported as "no bars".

Fix: `jobs.settle_signal_log()` is now called by **both** handlers. It
loads the official store first, and an empty store reports
`OFFICIAL_STORE_NOT_LOADED` instead of "every BUY still open". A test pins
the handler that production installs. The 527 stuck outcomes were
backfilled at 08:20 IST on owner approval (backup:
`backup/signal_outcomes_20261119_081833_pre_backfill.db`). `update_outcomes` now returns and persists
`logs/product/signal_outcome_resolution.json`, with per-row fates, stuck rows
and their reason, row errors, write errors and fatal errors. The outcome job
carries this summary. **The first nightly run after deploy will name the
root cause.** This is the top open item.

### 2.5 Outage sessions were never marked against open paper positions (fixed in this release)

The EOD settlement marked one session, so sessions skipped by an outage were
never checked against stops or targets. A stop touched only on a missed day
would have been silently survived: phantom P&L that an exchange-side GTT
would not allow.

RPEL (entered 11 Nov) missed 16 and 17 Nov. Its lows those days (1,926 and
1,930) stayed above its 1,744.5 stop, so its only damage is `bars_held=3`
instead of 5, which delays the 20-session max-hold by two sessions.

The fix: settlement now replays every missed weekday oldest-first with the
same exact-session official bars, and holidays are skipped. `PaperBook.mark`
now refuses any session at or before a position's last mark.

### 2.6 Findings recorded but not yet fixed

- **The setup score is not a probability and barely ranks.** `quality_score`
  AUC is 0.511. Read as P(win), its Brier score is 0.424 against a base-rate
  Brier of 0.159 (skill −1.67). The top quintile implies 91% and wins 19.9%.
- **The ML challenger's features had no producer (fixed in this release).**
  The schema names 25 features. Recommendation cards and saved scan records
  carry only rsi, volume_ratio and quality_score, with reward_risk derived.
  Every frozen DECISION observation therefore stored the other 21 as
  missing, and they became constant zero after imputation (out-of-fold
  Brier 0.237, about base-rate).
  - `product/pit_features.py` now computes 8 of them from official bars
    that existed at decision time: atr_pct, adr_pct, dist_from_high_pct,
    clv, volume_z, liquidity_cr, dist_20ema_pct and dist_50dma_pct. It
    uses bars up to the decision date if the decision came after 15:30
    IST, otherwise up to the previous session. Look-ahead is tested.
  - These features fill only the frozen learning record and the
    challenger's scoring, so training and serving see the same input. The
    live nearest-neighbour analog query is unchanged, so current rankings
    do not move.
  - Still unproduced: the cross-sectional features (rel_strength, the
    rs/sector/volatility percentiles), delivery_pct, breadth, VIX and base
    geometry.
- **Regime labels** are missing for every tracked BUY before 16 Jul (69% of
  settled rows). Paper decision context records `regime=UNKNOWN` and
  `sector=UNKNOWN` (RPEL). Learning cannot be regime-conditional yet.
- **The July evidence is weak and was inflated.** It ran at 300–600 "BUYs" a
  day, including 94 liquid/money-market ETFs with ~0.1% stops. Those ETFs are
  no longer in the scanner, as today's scan shows.
- **Counterfactual "R"** is a classification score (±0.40 for
  missed-winner / correct-reject), not a measured return. It is labelled
  `not_pnl` and is never used as P&L.
- **Two parallel paper engines** (legacy trades.db and PaperBook) still
  cross-check rather than share one store (CLAUDE.md).

## 3. Commits (branch `claude/settlement-batch-20261118`)

| Commit | Change | Deployed |
|---|---|---|
| `4e0c5d81` | batched forward-ledger settlement | yes (22:23) |
| `524a11cd` | independent-decision learning, session floor, repair tool | no |
| `4b40a7bc` | readiness uses the session verdict; no READY over a blocker | no |
| `4349bfd6` | signal-outcome settlement reports every row's fate | no |
| `2e860b7d` | Learning Effectiveness Scorecard + graduation verdicts | no |
| `19c5616f` | Home: SCORECARD & GRADUATION section | no |
| this release | missed-session paper catch-up + out-of-order mark guard; this report | no |

## 4. Learning Effectiveness Scorecard (production copies, 2026-11-18)

Sources: `GET /api/learning-scorecard` and `python -m product.learning_scorecard`.

**Unit of evidence.** Signals logged on the same session share one market
path, so each session's mean net R is one observation. The time-ordered
session series is bootstrapped with `research.harness.block_bootstrap_mean_ci`.
Costs come from `core.costs` (delivery taxes, fees and slippage of about
0.32% round trip). R is defined only for stops of at least 1% of entry.

1. **Matured independent decisions:**
   - 2,937 filled, settled tracked BUYs. 2,843 have defined R, from only
     **16 sessions** (6–25 Jul 2026).
   - 1,378 BUYs never filled, a 68% fill rate.
   - 0 settled paper trades.
   - 814 settled independent counterfactual rejections.
2. **Accuracy and calibration:**
   - Hit rate 18.4% on the 2,843 R-defined trades (Wilson 95% CI [17.0%,
     19.8%]). The Wilson interval treats trades as independent, so it is too
     narrow given the same-session clustering.
   - `quality_score` has NO_USEFUL_DISCRIMINATION (AUC 0.511).
3. **Strategy and regime:** every archetype is INSUFFICIENT EVIDENCE, with
   4–15 sessions each, and point estimates are negative for most. Regime
   coverage is 31%.
4. **Challenger vs champion, forward:** INSUFFICIENT EVIDENCE. There are no
   registered challengers, and the ML challenger has 0 forward outcomes.
5. **Net of costs:**
   - Per session, −0.098 R, 95% CI [−0.25, +0.18].
   - Per trade (naive), −0.23 R net and −0.18 R gross.
   - Net % on all filled trades: −1.03% per session, CI [−1.58%, −0.26%].
   - Profit factor (net) 0.44.
6. **Statistically credible?** No, at 16 sessions. The minimum detectable
   edge is 0.34 R. Detecting a +0.10 R edge needs about **188 sessions**.
7. **Has learning changed decisions?** **No.** 0 policies influence
   selection, and 0 of 1,092 independent decisions were changed by learning.
8. **Improving or deteriorating?** INSUFFICIENT EVIDENCE. This needs ≥ 40
   sessions, so that two independent halves can be compared.

**Selection versus rejection (5-session, counterfactual):**

| Gate | Missed-winner rate (95% CI) | Mean forward 5-session return | n | Sessions |
|---|---|---|---|---|
| LOW_QUALITY_SETUP | 19.5% [15.4%, 24.3%] | −0.23% | 298 | 6 |
| EMPIRICAL_GATE_FAILED | 29.9% | +1.16% | 137 | 6 |
| DD_GATE_FAILED | 42.9% | +3.39% | 21 | 2 |

All of these are INSUFFICIENT EVIDENCE. The selected tracked BUYs use a
different horizon (a 10-session path), so no selected-versus-rejected
difference is claimed.

## 5. Safety evidence

- **Live boundary.** `product/live_execution_interlock.py` is hard-wired
  LOCKED, and no flag or environment variable is authorization. The raw Kite
  SDK is wrapped deny-by-default (read allowlist only). A single
  `KiteConnect(` exists in `data/kite_client.py`. Every order call site
  (`execution/*`, `ems`, `ui/live_runner.py`) reaches the guarded proxy or
  `KiteClient.place_order`. Existing tests cover this exhaustively:
  `test_live_execution_interlock.py`, `test_kite_broker_boundary.py` and
  `test_broker_mutation_boundary_exhaustive.py`.
- **Secrets.** This is a public repo. Tracked files and the full history
  were scanned for Telegram bot tokens, `sk-` keys and assigned
  Kite/DeepSeek values. Only fake test tokens and documentation placeholders
  were found. `.env`, `logs/` and `*.db` are ignored. One tracked pickle,
  `models/lgb_trading_model.pkl`, should only ever be loaded from this repo.
  This scan is pattern-based and not a guarantee.
- **Paper P&L integrity:**
  - Missed sessions are now marked (§2.5).
  - Out-of-order marks are refused.
  - Counterfactual settlement can't double-ingest.
  - Batch flushes survive mid-loop exceptions.

## 6. Tests run (actual results)

| Scope | Result |
|---|---|
| Learning, forward-evidence, policy, paper-autopilot and autonomy test files, per file on the Mac with host commands stubbed and blank credentials | 92 files: 91 clean at first. `test_learning_policy_source_guard` was updated for the session floor (it now states its coverage), plus a fail-closed sibling test. |
| New tests | `test_learning_observation_dedupe.py` (20), `test_forward_settlement_batch.py` (8), `test_learning_scorecard.py` (19), money-path outcome-tracker tests (3), product readiness (3), outcome liveness catch-up (3) |
| Frontend | `tsc --noEmit` clean. vitest 35 files, 181 tests passing, including `homeScorecard.test.ts`. |
| CI | Green on `4e0c5d81` and `4349bfd6`. `524a11cd`/`4b40a7bc` failed one test (the source-guard test, fixed in `4349bfd6`). Later commits were still in flight when this was written. |
| Mutation checks | The counterfactual crash test fails against the re-flush ordering. The settlement batching tests pin the write counts. |
| Not run | The full 473-file suite on the Mac; it overloads the 8 GB host while the desk runs. CI runs it. There was no host soak, fault injection or restore drill. |

## 7. Highest-priority remaining work

1. **Deploy, then read the first `signal_outcome_resolution.json`.** Fix the
   real cause of the 4-month BUY-outcome stall. Until then the scorecard's
   evidence ends on 25 Jul.
2. **Run `product.learning_policy_repair --apply`** after deploy, between
   outcome jobs, to correct the inflated counterfactual policies.
3. **ML features.** 8 point-in-time features are now produced for new
   decisions. Next, produce the cross-sectional features, let replay batches
   accumulate rows with the real features, and re-validate the challenger
   out-of-sample before it may rank anything.
4. **Settle real paper trades.** Forward evidence is at 0 settled. Keep paper
   enabled. The learning lane needs gated setups to explore.
5. **Regime and sector labels on every decision context.**
6. **Host power.** The lid-close sleep missed the 15:42 deploy and the
   18:56 EOD refresh. AC power plus `pmset disablesleep` is the owner's call.
7. **Certify the institutional domains** (PIT data, OMS, reconciliation,
   protection, independent risk) before any live discussion.

## 8. Owner commands

Deploy (on the Mac, outside 09:15–15:30 IST, with no paper cycle or
outcome_resolution RUNNING):

```bash
cd /Users/apple/0to100
date                                   # confirm IST time first
venv/bin/python -m product.launchd_control stop
launchctl list | grep quantterm        # wait until empty
git fetch origin claude/settlement-batch-20261118
git merge --ff-only FETCH_HEAD         # host branch host/mac-release-20261110
venv/bin/python -m product.host_install install \
  --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --env-file .env --manager launchd
curl -s http://127.0.0.1:8765/api/health
```

Rollback: `git reset --hard 4e0c5d81` on the host branch, then the same
`host_install install`.

After deploy, with the desk idle between jobs:

```bash
venv/bin/python -m product.learning_policy_repair          # dry run: review the table
venv/bin/python -m product.learning_policy_repair --apply  # backs up, then rewrites
curl -s 'http://127.0.0.1:8765/api/learning-scorecard?refresh=true' | head -c 2000
```

## 9. Continuation, 19 Nov 2026 (after the GitHub unblock)

Authority: autonomous development and paper-only work. **Not deployed**:
production is still on `125880a6`. No production learning ledger was
modified, and live trading stays LOCKED.

### 9.1 Release state

| Item | State |
|---|---|
| GitHub branch head | the commit that adds this section (see the final report) |
| Deployed | `125880a6` (08:22 IST, 19 Nov). Rollback target for the next deploy. |
| Pushed since then | `63238457` (test fixes), `85ea49f1` (Home learning-impact counts), `741eab96` (production-chain acceptance test), `fbd6101b` (exit experiment, maturity rule), `a4c8e1b7` (bars_held reconciliation), `134e02d1` (pre-registered feature confirmation), `e19ceed7` (report schema 3) |
| CI | All 5 workflows green on `85ea49f1`. Superseded runs were cancelled. The final SHA's result is in the final report. |

### 9.2 Root causes, with proof

- **BUY outcomes were never settled by the installed handler.** Production installs `outcome_integrity.strict_outcome_resolution` over `outcome_liveness._run_outcome_resolution_light`. Only the replaced `jobs.run_outcome_resolution` called `update_outcomes`.
  - `jobs.settle_signal_log()` is now called by both handlers.
  - `research.autonomy.install_runtime_handlers()` is the exact production install sequence.
  - `tests/test_buy_outcome_acceptance.py` runs that sequence in a fresh interpreter with real frames in `data.bhavcopy_store._store`. A BUY logged on 1 Sep settles to +6% after its target touch, a BUY inside its horizon stays open, one with no bars is reported stuck, a rerun settles nothing twice, and an empty store reports `OFFICIAL_STORE_NOT_LOADED`.
- **Home LEARNING IMPACT counted copies** (740 missed winners for about 185 decisions). `autonomous_learning._evidence_counts` now counts decision units. There is a regression test.
- **`bars_held` counted marks, not sessions.** RPEL shows 3 after 5 sessions. `outcome_liveness.reconcile_bars_held` rebuilds the count from official bars at every EOD settlement, so RPEL becomes 5 at the first settlement after deploy. No hand edit is needed.
- **The scorecard had a maturation bias.** November rows that resolved early were counted while their peers were still open. A signal now counts only 16 or more calendar days after it was logged.

### 9.3 Learning-policy repair: reconciliation (dry run, not applied)

- **Backup:** `logs/backup/learning_policies_20261119_092827_pre_repair_review.json`, SHA-256 `58dfd70f…dd695080`, identical to the live file.
- **Evidence:** 4,258 forward-ledger rows and 4,101 counterfactual-ledger rows are **815 independent decisions** (706 appear in both ledgers).
- **Effect:** seven policies get corrected sample sizes and means, and **none changes status**. All stay REJECTED or OBSERVING, so none affects selection before or after.
- **Gap:** `ENTRY_QUALITY::GOOD_WAIT` (n=18, OBSERVING) cannot be rebuilt because no settled ledger row carries GOOD_WAIT.
- **Duplicates cannot come back:** the observation ledger claims each unit before it is counted, and the tests pin that. The production claims store is created by the first deduplicated ingest, which is the first EOD after deploy.

### 9.4 Why the system has no edge: diagnosis

The forward tracked sample: 4,628 matured signals with defined R, 24 sessions. Statistics are per session.

| Question | Result | Verdict |
|---|---|---|
| Selection: excess 10-session return vs the same-window universe | +0.15%, CI [−0.80, +1.52], t = 0.28 | no detectable skill |
| Ranking: `quality_score` per-session IC | +0.036 to +0.040, t ≈ 0.9 | no skill |
| Entry: share of signals whose next open is already above the trigger | 19%; −0.16R vs −0.19R otherwise | not the main problem |
| Exit: production (breakeven trail, 10 sessions) | −0.086R, CI [−0.21, +0.07]; 29% scratches | — |
| Exit: no trail, 20 sessions (pre-declared V3) | +0.102R, CI [−0.10, +0.37] | not significant after BH (q = 0.28) |

The 2-year walk-forward (`logs/signal_backtest.json`, about 800 symbols, about 5,000 trades) shows **every setup family negative**: CUP_HANDLE −0.12R, POCKET_PIVOT −0.13R, PRE_BREAKOUT −0.17R, ASC_TRIANGLE −0.34R, all with 110–290 session clusters. A +2% target is reached 76% of the time, yet expectancy is −0.13R. The signals select almost nothing, and the stop/trail construction turns drift into losses.

**Discovery: three point-in-time features rank excess returns** (per-session IC over 20 sessions, BH across 9 candidates):

- lower ATR %: +0.078, t 4.76;
- closer to the 52-week high: +0.089, t 3.36;
- less extended above the 50-DMA: oriented +0.060, t 2.96.

RS 20/60, sector strength, volume ratio and liquidity show no skill. These results come from one regime and 20 sessions, so they are **hypotheses, not findings**.

### 9.5 Baseline versus challenger

| Candidate | Evidence today | Verdict |
|---|---|---|
| Baseline: `quality_score` ranking with production exits | net −0.086 to −0.127 R/session; AUC 0.509; Brier skill −1.47 | rejected as an edge (no proof of skill) |
| Exit challengers V1–V3 | forward point estimates −0.06 to +0.10 R, none BH-significant | **INSUFFICIENT EVIDENCE**; pre-registered in the walk-forward (`exit_variants`) |
| Feature challenger (low ATR, near high, not extended, composite) | in-sample IC t 3–5 | **INSUFFICIENT EVIDENCE** (in-sample only); pre-registered out-of-sample `feature_confirmation`, discovery window excluded |
| ML challenger | 21 of 25 features had no producer; 8 PIT features added (`bded0395`) | INSUFFICIENT EVIDENCE (0 forward outcomes) |

**Did learning improve decisions?** No. Zero decisions have changed, and no challenger has passed a gate. The schema-3 walk-forward reruns on the first closed-market night after deploy and returns the out-of-sample verdicts. A CONFIRMED feature must still pass champion/challenger before it reaches PAPER.

### 9.6 End-to-end loop status

| Boundary | Status |
|---|---|
| Kite data → scan → ranking | runs daily; the evidence gate blocks unproven BUYs (0 BUY today) |
| Frozen decision → paper execute or reject | runs; 1 open position (RPEL), 0 closed |
| Outcome settlement: paper book | runs nightly; outage catch-up and `bars_held` reconciliation (from the next deploy) |
| Outcome settlement: BUY signals | **fixed in code, not deployed**. Backlog backfilled; the acceptance test passes on the production chain. |
| Learning evidence | deduplicated; repair awaiting approval |
| Challenger evaluation → policy update | gates in place; no candidate qualifies |
| Future decision changed by learning | none yet |

**Forward paper evidence:** 0 settled trades. The 29 closed legacy paper trades in `trades.db` (13 Jul–9 Nov, legacy rules, only 4 with a usable stop) are excluded. The legacy engine is dormant in the canonical stack.

### 9.7 Graduation

RESEARCH, FORWARD PAPER and OPERATIONAL are **NOT READY**. SUPERVISED LIVE and AUTONOMOUS LIVE are **NOT ELIGIBLE**. The stage is 0 of 5.

On the operational side, the OMS, reconciliation, protection and risk-governor modules exist but are uncertified. `risk/iron_lock.py` (a legacy kill switch) is unused by the canonical path. The paper kill switch is the owner halt and pause control.

### 9.8 Owner approvals needed

1. **"deploy <final SHA>"** activates BUY settlement, the Home count fix, `bars_held` reconciliation, the maturity rule and the schema-3 rerun. Rollback is to `125880a6`.
2. **"apply repair"** rewrites the 7 inflated counterfactual policies to independent counts. No status changes; the backup is verified.

## 10. Validation fixes and the paper-only release plan (19 Nov, after owner review)

### 10.1 Scientific-validation fixes

- **`2d023f6d`: confirmation and exit-variant inference.**
  - Spearman IC now uses tie-corrected ranks (equal to `scipy.spearmanr`). A constant feature or outcome yields no IC.
  - Daily values are averaged in non-overlapping 10-session NSE blocks, then tested with a Student t (df = blocks − 1) and BH across candidates.
  - Any verdict beyond INSUFFICIENT EVIDENCE needs ≥ 20 independent blocks and non-zero dispersion. A zero-dispersion sample is `DEGENERATE_ZERO_DISPERSION`, never decisive.
  - A missing or empty holiday calendar returns `EXCHANGE_CALENDAR_UNAVAILABLE`.
  - Adversarial tests cover identical perfect ICs, 19 blocks, 300 days packed into 5 blocks, pure noise, ties, constants, and a missing calendar.
- **`f0e80fe4`: scorecard maturity.**
  - A signal is MATURED only when its first 10 NSE sessions (holidays included) have completed **and** each has a verified official bhavcopy on disk.
  - It is IMMATURE before that. It is NOT_YET_VERIFIED when bars are missing or the calendar or verified-dates source is unavailable; that is never assumed matured.
  - Production check (read-only): 4,815 MATURED, 123 IMMATURE, 0 NOT_YET_VERIFIED. Tracked BUYs: −0.129 R/session over 24 sessions, CI [−0.28, +0.02], not distinguishable from zero.
- **`3ae98541`: prospective shadow test** (`product/challenger_shadow.py`). Each night the learning cycle freezes the challenger composite's and the champion's ranks of that session's scan before outcomes exist, and settles only MATURED sessions with the same block inference. SHADOW_ONLY: no orders, no ranking change, no policy write.

### 10.2 Production effects of deploying (125880a6 → final SHA)

| Change | Effect | Risk |
|---|---|---|
| BUY settlement in the installed handler | Nightly writes to `signal_log`; the store is loaded first; an empty store is reported | Low; idempotent |
| `install_runtime_handlers` | Identical install calls in identical order (verified by diff) | None |
| `reconcile_bars_held` | RPEL goes from 3 to 5 at the first EOD settlement, so max-hold fires 2 sessions earlier (correct) | Low; paper only |
| Observation-claims store | `learning_policies.observations.sqlite3` is created on the first deduplicated ingest | Low; additive |
| Report schema 3 | The walk-forward reruns on the first closed-market night (about 10 min, background) and refreshes calibration a few days before the weekly run | Low; the same job was healthy on 16 Nov |
| Challenger shadow | One SQLite transaction per session in `logs/product/challenger_shadow.db` (§11; replaces the JSONL) | Low; additive |
| Scorecard maturity, Home counts | Display only | None |

**Rollback** is `git reset --hard 125880a6` on `host/mac-release-20261110`, then `host_install install`. Every new artefact is additive. The old code treats a schema-3 report as stale and rebuilds its own, ignores the claims store and the shadow ledger, and runs normally with RPEL's corrected count.

### 10.3 Paper-only deploy runbook (requires the owner's explicit "deploy <SHA>")

Read-only preflight, verified at 10:16 IST on 19 Nov:

- AC power at 100%;
- image `QuantTermStorage_WORKING_20261107` mounted;
- 103 GB free on the runtime volume and 15 GB on the SSD;
- checkout clean on `host/mac-release-20261110` at `125880a6`;
- a fast-forward to the release is possible;
- no RUNNING jobs; health READY.

Run all of it after 15:30 IST and before the EOD outcome job:

```bash
cd /Users/apple/0to100 && date                          # outside 09:15-15:30 IST
sqlite3 "file:/Volumes/QuantTermStorage/QuantTerm/runtime/logs/autonomy/jobs.db?mode=ro" \
  "select job_type from jobs where status='RUNNING'"    # must be empty
R=/Volumes/QuantTermStorage/QuantTerm/runtime/logs; TS=$(date +%Y%m%d_%H%M%S)
sqlite3 $R/signal_outcomes.db ".backup '$R/backup/signal_outcomes_${TS}_pre_deploy.db'"
cp $R/intelligence/intel_book.json $R/backup/intel_book_${TS}_pre_deploy.json
cp $R/product/learning_policies.json $R/backup/learning_policies_${TS}_pre_deploy.json
shasum -a 256 $R/intelligence/intel_book.json $R/backup/intel_book_${TS}_pre_deploy.json \
  $R/product/learning_policies.json $R/backup/learning_policies_${TS}_pre_deploy.json   # each pair must match
sqlite3 $R/backup/signal_outcomes_${TS}_pre_deploy.db "pragma integrity_check"         # must print ok
venv/bin/python -m product.launchd_control stop && launchctl list | grep quantterm   # wait until gone
git fetch origin claude/settlement-batch-20261118 && git merge --ff-only <SHA>
venv/bin/python -m product.host_install install --runtime-root /Volumes/QuantTermStorage/QuantTerm/runtime --env-file .env --manager launchd
curl -s http://127.0.0.1:8765/api/health
```

**Verify the next morning:**

- the outcome job summary contains "signal_log N settled";
- `logs/product/signal_outcome_resolution.json` is fresh;
- `bars_held_reconciled` lists RPEL 3 → 5;
- `logs/signal_backtest.json` has `report_schema: 3` with `exit_variants` and `feature_confirmation`;
- `logs/product/challenger_shadow.db` has the session recorded (`sessions` row FROZEN, or MISSED with its reason). §11 replaced the JSONL.

The learning-policy repair is **not** part of this deploy. It needs a separate "apply repair".

## 11. Scientific-integrity pass (19 Nov, owner directive after `71166a1b`)

Branch `claude/settlement-batch-20261118`, on top of `71166a1b`. Live trading stays HARD-LOCKED. Nothing was deployed and no production ledger was modified.

### 11.1 Residual dependence between 10-session outcome windows (finding)

The 2026-11-19 method averaged daily values in non-overlapping 10-session blocks and treated the block means as independent. They are not:

- The last sessions of a block share nine of ten outcome sessions with the first sessions of the next block.
- Slow regimes add dependence beyond the overlap.

Monte Carlo on null data (no true effect, two-sided 5% nominal, 2,000 simulations):

| Null structure | Superseded block t | Day-clustered error | `research/overlap_inference.py` |
|---|---|---|---|
| Overlapping 10-session windows | 10.2% | 52.7% | 3.0% |
| + slow regime (AR 0.8) | 15.4% | 63.2% | 5.6% |
| + very slow regime (AR 0.9) | 25.6% | 70.5% | 5.2% |
| A 10-session block copied into the next window (400 simulations) | 28.2% | 66.7% | 1.3% |

The BUY gate's one-sided bound (day-clustered, z = 1.645) marked 45 of 120 null setups "proven". The new bound marks 8.

On the real tracked BUYs (SSD copy of `signal_outcomes.db`):

- 3,203 settled trades come from only 24 sessions, which is 4 independent 10-session blocks.
- The session means are autocorrelated: +0.28 at lag 1 and +0.12 at lag 5.
- The honest variance is 5.4 × the day-clustered one. The CI moves from [−0.31, +0.05] R to [−0.81, +0.55] R, and the verdict is INSUFFICIENT EVIDENCE (4 of 20 blocks).
- 16 `(session, symbol)` rows in `signal_log` are duplicated decisions.

### 11.2 Method (`research/overlap_inference.py`)

1. **Session cells.** Values are keyed by NSE session ordinal; rows on one session form one cluster, so duplicating rows changes nothing.
2. **Offset subsamples.** The cells are split by `ordinal mod 10`. Within a subsample no two windows overlap. The standard error is the weighted sum of the subsample errors, an upper bound by Minkowski's inequality whatever their cross-correlation. It is never below the day-clustered error.
3. **Residual dependence.** The lag-10 residual autocorrelation widens the variance with an AR(1) factor (bias-corrected). A Bartlett Newey–West term at lags 10 and 20 is the floor.
4. **Degrees of freedom and floor.** Student t uses df = independent blocks − 1. A verdict needs ≥ 20 blocks and non-zero dispersion.

The method is used by:

- the walk-forward BUY-gate bound (`_evidence_fields`; the realised official-bar session index; worst case √10 × the clustered error if no index);
- the exit variants and the feature confirmation;
- the scorecard (`session_series_verdict`, `trend`);
- the challenger shadow.

Regression tests: `tests/test_overlap_inference.py`, `TestOneRulebook` in `tests/test_money_paths.py`.

### 11.3 Exchange calendar fails closed

`data/nse_holidays.json` lists **2026 only**, and there is no runtime amendment file on the host.

- `nse_calendar.covered_years` counts a year as covered only with ≥ 8 listed holidays.
- `calendar_gap`, `forward_sessions` and `strict_session_ordinals` return "not covered" for any year outside that set.
- `maturity_checker` returns NOT_YET_VERIFIED, with reasons (`EXCHANGE_CALENDAR_UNAVAILABLE`, `…NOT_COVERED:<year>`), for an empty, stray-entry or uncovered calendar. It never assumes weekdays. A window still in the future by a weekdays-only count stays IMMATURE.
- **Operational consequence:** signals from 17 Dec 2026 onwards, whose 10-session window reaches into 2027, are NOT_YET_VERIFIED until NSE's 2027 holiday list is added to `data/nse_holidays.json`. This is by design. Freezing still works until 31 Dec, because the shadow gate needs the calendar only through the next session.

### 11.4 Prospective challenger shadow (`product/challenger_shadow.py`, schema 2)

- **What may be frozen.** Only the live session-close scan of the latest completed NSE session, before the first eligible forward entry (the next session's 09:15 IST open).
  - The scan must carry an identity (`scan_id`) and current input (`data_current`, expected session).
  - It must have completed after the 15:30 close.
  - The session's official bar must be published.
  - Replays (`pit`), stale or anonymous scans, and late writes freeze nothing.
- **What is frozen** in one SQLite transaction (`logs/product/challenger_shadow.db`):
  - per symbol: the champion's actual score, rank and verdict, and the challenger's composite score and rank;
  - per session: the model version (hypothesis, challenger spec hash, `QT_BUILD_SHA`, horizon), the instrument universe (approved and scanned counts, candidate-list SHA-256), the exclusions, and a payload hash.
  - Zero candidates still write the session marker. A crash rolls back the marker and the predictions together. The primary key makes writes idempotent.
- **What is never evidence.** A deadline that passed unfrozen becomes MISSED. Evaluation also refuses rows that were:
  - frozen after the deadline;
  - inserted after a later session's row (a backfill);
  - edited after freezing (payload hash).
- **How outcomes are measured.**
  - Over exactly the 10 official NSE sessions after the signal: the open of the 1st to the close of the 10th.
  - Every bar must exist for the symbol. A missing bar excludes that symbol and is counted; more than 20% missing excludes the session.
- **How the decision is made.**
  - It is the paired IC improvement (challenger minus champion, same candidates, same window), tested with the overlap-robust error.
  - It has a top-quintile spread as the second member of a Holm-corrected family of two.
  - Two separately positive ICs are not evidence; a test shows that.
- Seven mutation tests (each guard removed in turn) all make the suite fail.

### 11.5 Review relay (development-only)

`scripts/quantterm_relay.py` is installed as the launchd agent `com.quantterm.relay` with state in `~/Library/Application Support/QuantTermRelay`.

- It polls issue #333 every 5 minutes.
- It accepts only owner comments whose first line is `<!-- quantterm-relay-v1 sha=<40 hex> -->`, for a SHA on the engineering branch with all five workflows green.
- It delivers once per SHA with `claude -p --cloud <session>`. The text arrives as a peer message, never as owner approval.
- Delivery is at-most-once (SENDING → UNCERTAIN on a crash), with acknowledgements, a cap of 8 deliveries per day, and a self-disable if the CLI ever reports another session.
- It runs only while the Mac is awake and online.

### 11.6 Unchanged

The BUY-settlement chain, `bars_held` reconciliation, the observation-claims dedupe, the scorecard maturity rules, the paper accounting and the live interlock are unchanged. The money-path suite passes in full.

### 11.7 Rollback

As in §10.2, `git reset --hard 125880a6` on `host/mac-release-20261110`, then `host_install install`. The new artefacts are additive:

- `challenger_shadow.db` is ignored by the old code.
- The schema-3 report's stricter `lower_r` is read by the old gate unchanged and only withholds more.

The relay is independent of the desk. To stop it: `launchctl bootout gui/501/com.quantterm.relay`, or create the `DISABLED` file.

## 12. Fail-closed admission pass (20 Nov; ChatGPT review of `de061791`, delivered by the relay)

The review arrived while this session was idle and started a turn by itself: posted 18:43:49Z, delivered 18:45:57Z, acknowledged 18:46:03Z. Nothing was deployed and no production data was changed. Live trading is HARD-LOCKED.

### 12.1 BUY gate: every legacy exception traced and closed

Production path: walk-forward report → `backtest_edges` → calibration snapshot → `UnifiedScanner._signal_edges` → `_evidence_verdict` / `setup_evidence`.

Three paths let a BUY through without proof:

- **H1.** A bare float, or a row without `n`, got BUY on its mean alone, and an infinite mean or lower bound passed as "proven".
- **H2.** If evidence was loaded but none of the setup's signals had ≥ 30 trades, the setup got BUY on score alone. That violated "< 30 trades = no claim".
- **H3.** With the gate on but no report loaded, the legacy score rule applied.

A BUY now needs `unproven_reason(ev) == ""`:

- a finite mean ≥ +0.05R;
- ≥ 30 trades;
- a finite one-sided 95% lower bound above 0, from the `OVERLAP_ROBUST` inference;
- ≥ 20 independent blocks.

Legacy, pre-rerun, mixed, partial and non-finite evidence, and setups with no evidenced signal, are WATCH, and the card shows the reason.

- The backtest and historical replay construct the scanner with the gate off and keep their raw verdicts. A test checks this.
- **Production effect:** the schema-2 report on the host predates the dependence-aware bound, so every BUY is withheld until the schema-3 walk-forward reruns. That happens on the first closed-market night after a deploy. Today there are zero BUYs either way.

### 12.2 Exchange calendar: provenance, not plausibility

`data/nse_calendar_manifest.json` records 2026 as VERIFIED_COMPLETE against NSE's own list (`api/holiday-master?type=trading`, segment CM). The 16 weekday holidays match exactly.

- **Covered year:** only a manifest entry that is VERIFIED_COMPLETE and whose weekday-holiday SHA-256 matches the loaded holidays.
- **Fails closed:** a partial list (even 8 real holidays), a stray list, an unverified amendment (a newly announced holiday), or a missing manifest.
- **Special sessions:** the Diwali Laxmi Pujan* entry (Sunday 8 Nov, Muhurat) is recorded and is not a regular session.
- **Refresh tool:** `scripts/verify_nse_calendar.py` writes nothing unless the lists match exactly. 2027 is not published yet, so the tool refuses it, as designed.

### 12.3 Stress study of `research/overlap_inference.py`

The study is reproducible: `python -m research.overlap_stress --sims 2000` (seeds in the table). A false confirmation is a two-sided p < 0.05 with ≥ 20 blocks, under a true null.

Under strong regime persistence the point-estimate widening gave 5.9% [5.0%, 7.1%]. The widening now uses the larger of rho's one-sided 95% upper bound and its bias-corrected value, giving 5.3%.

| Null scenario | seed | robust (Wilson 95%) | superseded block t | same SE, normal reference |
|---|---|---|---|---|
| overlapping windows | 101 | 3.0% [2.3%, 3.8%] | 11.2% | 3.5% |
| heavy tails t3 | 102 | 0.2% [0.1%, 0.5%] | 8.6% | 0.3% |
| regime persistence ar0.9 | 103 | 5.3% [4.5%, 6.4%] | 24.4% | 6.2% |
| cross sectional concentration | 104 | 0.1% [0.1%, 0.4%] | 9.2% | 0.1% |
| missing sessions 40pct plus month gap | 105 | 0.6% [0.3%, 1.1%] | 10.8% | 0.8% |
| duplicated outcomes lag1to3 | 106 | 3.4% [2.7%, 4.3%] | 21.4% | 3.7% |
| overlapping event windows | 107 | 3.4% [2.7%, 4.3%] | 11.1% | 4.0% |
| exactly 20 blocks | 108 | 2.1% [1.6%, 2.9%] | 10.0% | 2.9% |

Selection: the best of 20 correlated null features (seed 201, 500 simulations) has p < 0.05 in 47% of runs, but BH across the family rejects in only 3.2% [2.0%, 5.1%].

Student-t (df = blocks − 1) is more conservative than a normal reference in every scenario. Full output: `docs/research/overlap_stress_2026-11-20.json`.

### 12.4 Challenger shadow audit

Three concrete holes were fixed:

1. **Clock skew.** A freeze must land at least 30 minutes before the 09:15 IST entry deadline, checked at write time and again at evaluation.
2. **Naive clock.** A naive (timezone-less) reading is refused.
3. **Scan-ID reuse.** A `scan_id` already frozen for another session is refused.

Unchanged and checked:

- IST has no DST and every deadline is a fixed +05:30 time.
- Late and stale scans, SQLite rollback, missing symbol bars, and pairing on the same frozen candidates and window.
- A schema mismatch raises instead of writing.

Eleven guard-removal mutations are all caught. The paired lower bound turned out to be redundant with Holm; a worse-challenger test now pins it.

### 12.5 Tests

- The 50 scanner-dependent files pass.
- The 39 calendar, inference, shadow, scorecard, gate and relay files pass: 903 tests.
- All of these ran on the Mac under the guard, one interpreter per file.

## 13. End-to-end admission audit (20 Nov; ChatGPT review of `b60df3b2`)

The review was delivered at 20:47:32Z and acknowledged at 20:48:09Z, with no human in the loop. Nothing was deployed and no production data was changed. Live trading is HARD-LOCKED.

### 13.1 Every route to a BUY label or an entry (P0)

The only producer of verdict BUY and status "Ready to trade" is the live scanner's gate (`_evidence_verdict` → `scan_store._record`). The routes that consume it:

| Route | Keys on | Covered by the gate |
|---|---|---|
| Recommendation tiers (`reco_ensemble.entry_state` → `decide_tier`) | READY needs "Ready to trade", or grade A/B with verdict BUY | yes |
| Paper autopilot (`evaluate_candidate`) | `reco_tier` ∈ {high, good} | yes |
| Home BUY-only, Telegram "qualified setups" | verdict BUY | yes |
| Paper learning lane (`_exploration_pick`) | edge-gated cards, explicitly; 1/day, 0.5% risk, labelled EXPLORATION; `QT_PAPER_EXPLORATION=0` disables | deliberate research exception |
| Telegram live-tick "SNIPER BREAKOUT CONFIRMED" | any sniper-watch row, **including gated WATCH rows, with a paper plan** | **fixed:** an unproven row now reads "WATCH — no proven edge; research alert, not a recommendation" |
| F&O paper (`fo_runtime` → `fo_paper_runtime`) | its own directional breakout scan and risk gates; **no proven-edge gate** (no historical option-chain data exists) | paper research lane, labelled "PAPER_OPTION_CANDIDATE"/"Candidate only"; Telegram now adds "No proven edge" |
| `breakout_sniper` ticks, `auto_scan`, the Telegram tap-to-paper listener | — | not started by the canonical stack (legacy_app/auto_scan only) |

Also fixed:

- **Stale evidence.** Snapshots now carry `backtest_generated_at`. Undated, unreadable, future-dated or older-than-21-days evidence is WATCH with its reason.
- **Tests.** `tests/test_buy_admission_end_to_end.py` runs scanner → record → tier → paper decision, plus Home/Telegram, for each case: stale or undated snapshot, no report, empty evidence, mixed legacy rows, NaN/Inf, low n, too few blocks, a pre-rerun bound, a failed family correction, no market data, and an unavailable calendar. Three mutations (score-only BUY, accepting stale evidence, the old sniper wording) are each caught.

### 13.2 Calendar freshness (P1)

A verified digest proves consistency with the list retrieved on its date. It does not prove that NSE has not amended the calendar since.

- **Live freshness is BLOCKED offline**, and the scorecard says so in `calendar_provenance`.
- **What is checkable offline:** completed windows are cross-checked against the official bhavcopy dates.
  - A missing bar (an emergency closure or a new holiday) makes the window NOT_YET_VERIFIED, as before.
  - A bar on a calendar-closed day that the manifest does not document (an unrecorded special session such as an exceptional Saturday) now does too: `CALENDAR_CONTRADICTED_BY_OFFICIAL_BARS`.
  - The documented Muhurat session is not a contradiction.
- **Point-in-time session counting.** What is frozen at freeze time is the definition (the first 10 official sessions after the signal). The realised sessions come from the verified calendar plus the official bars at evaluation, so a holiday declared after the freeze is counted correctly.
- **Freeze deadline.** It is now the next calendar day's 09:15 IST, independent of the calendar. An unannounced special session can never make a late freeze look prospective.

### 13.3 Shadow freeze TOCTOU (P1)

- **Atomic scan-ID uniqueness.** It is enforced at the database boundary: a partial unique index `one_session_per_scan_id`, plus the check inside the same `BEGIN IMMEDIATE` transaction as the insert.
  - 8 racing threads: exactly one write. 4 separate OS processes: exactly one.
  - Same-session writes under a race stay idempotent: 1 write and 5 already_recorded.
- **Crash and restart.** A simulated I/O error mid-insert leaves nothing, and a retry writes once.
- **Scan completion is checked on its own.** Evaluation re-checks that the *scan* completed after the session close, before the cutoff, and no later than 5 minutes after its own `frozen_at`. The margin can no longer certify a scan completed after the entry, even with a backdated `frozen_at`.

### 13.4 Inference connections and calibration (P1)

- **Effective df.** The Student-t df is now (blocks − 1) / residual widening; the ≥ 20 raw-block floor is unchanged.
  - Under AR(0.9) regimes: seed 103 gives 4.45% [3.63%, 5.44%] and seed 1103 gives 3.10% [2.43%, 3.95%]. Pooled over 4,000 simulations: 3.77% [3.23%, 4.41%].
  - Nominal control is therefore supported in the pooled estimate, but not shown at the single-seed upper bound.
- **Selection hole in the BUY gate, now closed.** The gate judged about 15 signals and 29 combinations at a one-sided 5% each with no multiplicity correction. `apply_family_correction` now runs BH across every row the gate can use. Untestable rows count with p = 1, and proof needs `bh_significant`.
- **The pieces are now connected:**
  - research verdicts (confirmation, exit variants): two-sided p with BH;
  - BUY gate: one-sided lower bound, plus BH across the gate family;
  - shadow: one-sided lower bound, plus Holm across a fixed family of two.

Adversarial nulls (`python -m research.overlap_stress --sims 2000`; a false confirmation is a two-sided p < 0.05 with ≥ 20 blocks):

| Null scenario | seed | robust (Wilson 95%) | superseded block t |
|---|---|---|---|
| overlapping windows | 101 | 3.00% [2.34%, 3.84%] | 11.2% |
| heavy tails t3 | 102 | 0.15% [0.05%, 0.44%] | 8.6% |
| regime persistence ar0.9 | 103 | 4.45% [3.63%, 5.44%] | 24.4% |
| cross sectional concentration | 104 | 0.15% [0.05%, 0.44%] | 9.2% |
| missing sessions 40pct plus month gap | 105 | 0.60% [0.34%, 1.05%] | 10.8% |
| duplicated outcomes lag1to3 | 106 | 2.65% [2.03%, 3.45%] | 21.4% |
| overlapping event windows | 107 | 3.30% [2.60%, 4.18%] | 11.1% |
| exactly 20 blocks | 108 | 2.15% [1.60%, 2.88%] | 10.0% |
| volatility regime switches | 109 | 2.35% [1.77%, 3.11%] | 10.0% |
| nonstationary variance | 110 | 2.60% [1.99%, 3.39%] | 10.2% |
| symbol dependence rank ic | 111 | 2.65% [2.03%, 3.45%] | 9.2% |

Selection and tuning:

- Best of 20 correlated null features: 46% raw, 3.0% after BH.
- Best of 5 horizon definitions on one null: 4.8% raw, 0.8% with Holm, 2.8% with BH.

Power (300 sessions; the true mean shift is in per-session SD units):

| Scenario | 0.1 SD | 0.2 SD | 0.3 SD | 0.5 SD |
|---|---|---|---|---|
| overlapping windows | 4% | 14% | 26% | 68% |
| regime persistence ar0.9 | 3% | 8% | 11% | 22% |

**Limits.** These nulls are synthetic. Under strong persistence, the power to see even a large edge in about 300 sessions is low. That is the honest price of not confirming noise, and it is why evidence graduation needs real forward sessions.

### 13.5 Production effects if deployed

- No BUY is possible until the schema-3 walk-forward reruns on the first closed-market night. The rerun report carries the overlap-robust bounds, the family correction and the timestamp. A BUY after that needs a BH survivor.
- More cards are edge-gated: they become EDGE_GATED shadow rows and feed the paper learning lane.
- The shadow ledger freezes a Friday session before Saturday 09:15.
- F&O and sniper Telegram messages are labelled as research.

## 14. Admission-proof challenge (20 Nov; ChatGPT review of `845ad743`)

The review was delivered at 21:58:24Z and acknowledged. Nothing was deployed and no production data was changed. Live trading is HARD-LOCKED.

### 14.1 Multiple testing for individual BUY admission (P0)

The 845ad743 gate used Benjamini–Hochberg. The review was right that this is the wrong guarantee for this decision:

- BH controls the expected *share* of false discoveries.
- It assumes positive dependence.
- It says nothing about whether one admitted BUY is real.

A BUY admission is an individual decision, so the predeclared procedure is now **Holm-Bonferroni (family-wise)**. Holm bounds P(any false admission) at 5% per report under **arbitrary** dependence. The choice was made on theory, not tuned on the runs below.

`research/admission_stress.py` builds a reproducible null of the real gate's structure (seeds 501–504; 100 simulations × 13 weekly looks on a rolling 250-session window):

- 12 signals and their pairwise combinations, loading on shared market and style factors;
- overlapping 10-session outcome windows, with firing that co-moves with a slow regime;
- a trade floor of 30.

Two families were tested: a production-like family (every row with ≥ 30 trades), and an adversarial one that picks combinations by in-sample mean and then tests them on the same data. The admission rule is mean ≥ +0.05R plus a one-sided p below the procedure's threshold.

| Family | Procedure | False admission per look | Any false admission in 13 looks |
|---|---|---|---|
| production family | none | 1.00% [0.59%, 1.70%] | 1.0% [0.2%, 5.5%] |
| production family | bh | 0.00% [0.00%, 0.29%] | 0.0% [0.0%, 3.7%] |
| production family | by | 0.00% [0.00%, 0.29%] | 0.0% [0.0%, 3.7%] |
| production family | holm | 0.00% [0.00%, 0.29%] | 0.0% [0.0%, 3.7%] |
| production family | bonferroni | 0.00% [0.00%, 0.29%] | 0.0% [0.0%, 3.7%] |
| mean-selected family | none | 0.00% [0.00%, 0.29%] | 0.0% [0.0%, 3.7%] |
| mean-selected family | bh | 0.00% [0.00%, 0.29%] | 0.0% [0.0%, 3.7%] |
| mean-selected family | by | 0.00% [0.00%, 0.29%] | 0.0% [0.0%, 3.7%] |
| mean-selected family | holm | 0.00% [0.00%, 0.29%] | 0.0% [0.0%, 3.7%] |
| mean-selected family | bonferroni | 0.00% [0.00%, 0.29%] | 0.0% [0.0%, 3.7%] |

| True per-trade edge in one signal | none | BH | Holm |
|---|---|---|---|
| +0.15R | 0.008 | 0.000 | 0.000 |
| +0.30R | 0.027 | 0.000 | 0.000 |

Reading:

- Under factor-driven dependence the overlap-robust per-row test is already so conservative that no procedure produced a false admission in 1,300 looks. Only the uncorrected test did, at 1%.
- The same conservatism leaves **almost no power**. A large +0.30R edge is admitted in 2.7% of looks uncorrected, and in **0%** after any correction, within one 250-session window.
- The simulations therefore cannot tell BH from Holm. Theory decides, and Holm costs no measurable power here.
- Not covered: weekly re-tests across a full year (52 looks). The measured 13-look rate is 0 of 100 runs (Wilson upper bound 3.7%).
- These are synthetic stress tests, not a guarantee for Indian market returns.

### 14.2 Decision-time provenance (P0)

- **A fresh timestamp alone certifies nothing.** `evidence_provenance_note` requires all of:
  - report schema ≥ 4;
  - the current rulebook's `eval_spec` and horizon;
  - the predeclared family method;
  - the generating code SHA (`QT_BUILD_SHA`, or else git HEAD);
  - a date ≤ 21 days old.

  The report writes `evidence_provenance`, and the calibration snapshot freezes it as `backtest_provenance`.
- **One immutable binding per scan.** The scanner deep-copies its evidence, so a refresh mid-scan does not change that scan's verdicts.
- **Each row carries its certificate.** Every result carries `edge_evidence_id`, and the payload names its `evidence_identity`.
- **Every consumer re-checks the certificate.** `scan_store.bind_evidence` runs in `load_scan()` and in the recommendation workspace for in-memory payloads. A BUY row whose id is missing or different reads as WATCH, with the reason. That covers:
  - a legacy artifact (no identity);
  - a rolled-back or mixed one.

  The paper autopilot, Home and Telegram all read through these, and historical replays (`pit`) are exempt.
- **Other cases tested:** a duplicated signal no longer doubles its weight; a truncated or corrupt report file is no evidence; a restart gives the same answer.
- **Fixture changes.** Three recommendation tests built synthetic BUY payloads with no evidence identity. They now carry one, as a real gated scan does. The tests' subject (quality ranking, breakout grading, watchlist order) is unchanged.

### 14.3 Shadow and calendar integrity (P1)

- **Migration.** A ledger from before the unique index, with duplicate `scan_id` rows, is migrated on open, not broken. The earliest row stays FROZEN and the later ones become MISSED, `QUARANTINED_DUPLICATE_SCAN_ID`. Nothing is deleted.
- **kill -9 mid-transaction.** A real separate process killed inside `BEGIN IMMEDIATE` leaves no row, and the lock is released.
- **A corrupt ledger** raises and the nightly job reports `LEDGER_UNREADABLE`. It is never recreated, so the record of what was frozen is never erased.
- **Exceptional sessions.** An unrecorded Saturday bar inside a shadow window makes the session NOT_YET_VERIFIED instead of stretching or shortening it.
- **Exact horizons in the walk-forward.** It measured outcomes over the next 10 *bars* of each symbol, which a suspension or a missing bar silently stretches. `window_is_exact` now requires the 10 consecutive official sessions after the signal bar, from the bhavcopy dates on disk. Other trades are skipped and counted (`horizon_gap_windows_skipped`).
- **Calendar.** Live amendment freshness remains BLOCKED offline. Historical completeness is the verified manifest plus the official-bars cross-check (§13.2).

### 14.4 Reproducing the statistics (P1)

- The seeded `research.overlap_stress` run (2,000 simulations per null) was repeated from scratch and compared with the committed JSON. Every scenario, the selection and tuning arms, and power came out **identical** (numpy 2.4.4, scipy 1.17.1). The output now records those versions. A different numpy random stream would change the numbers; the pinned-fixture test below does not depend on it.
- `test_the_conservative_estimator_is_pinned` pins the exact SE, df and lower bound on three RNG-free fixtures. Any change to the estimator fails, and a less conservative change is named as such.
- Synthetic nulls remain a stress test, not a guarantee.

## 15. Trust boundary, lifetime error and power (20 Nov; ChatGPT review of `68514c59`)

The review was delivered at 23:54:43Z and acknowledged at 23:56:27Z. Nothing was deployed and no production data was changed. Live trading is HARD-LOCKED.

### 15.1 The evidence certificate's actual authority (P0)

The review was right. `bind_evidence` compared a row's id with an id stored in the *same payload*, so a stale but self-consistent artifact passed. Tracing the chain:

- **The authority.** It is the active calibration snapshot. Its id is a content hash over the whole walk-forward report (`backtest_evidence_hash` in the identities), plus the data, thesis, feature and model versions.
- **Defect 1, the pointer.** `get_or_create_snapshot` did not move the `current.json` pointer on a cache hit. After a rollback to an older report, scanners used snapshot X while the pointer still named Y. Fixed: the pointer always names the snapshot built from the current inputs.
- **Defect 2, the boundary.** `load_scan()` is the boundary between the producing scan and every later consumer, across processes and time. It now **re-derives** each BUY row's evidence from the authoritative snapshot: the snapshot's own edges for the row's signals, `unproven_reason`, and the provenance check.

  These now fail:
  - a stale self-consistent artifact;
  - a verdict promoted after the scan;
  - a report replaced between the scan and the read;
  - stale authoritative provenance;
  - a missing authority (nothing is certified).

  Replays have their own explicit point-in-time route, so the forgeable `pit` exemption is removed from this path. The in-memory path straight from the scanner keeps the self-consistency check, because it was certified by the same snapshot moments earlier.
- **Data binding.** The report now records the data version (first and last official sessions read), the universe (SHA-256 of the sampled symbols), the family size and the family alpha. They are hashed into the snapshot id.
- **Holes found and fixed:**
  - A zero-dispersion sample had `lower_r = mean` and showed a "proven" bound. It now has no bound.
  - Family p-values were recomputed from 4-decimal values. They now use the unrounded p, coherent with the lower bound, and any non-finite p becomes 1.
- **Legacy path found.** `scan/conviction.build_conviction` re-derives BUY / STRONG BUY from news-buzz and earnings "conviction", bypassing the gate. Only the legacy `auto_scan` calls it, and the canonical stack never starts that.
- **`code_sha` is audit metadata, not an authority.** The authority for "same rules" is the report schema, eval spec, horizon and family method, which the running build checks against its own constants. This deliberately does not defend against an adversary who can write the runtime (and could equally edit the code). It defends against staleness, rollback, mixing, races, partial writes and promotion bugs.

### 15.2 Per-report versus lifetime family-wise error (P0)

Holm bounds false admissions *per report*. Weekly reports re-test overlapping data about 52 times a year. The predeclared rule is now Holm at **α / 52 per weekly report**. By the union bound, P(any false admission in a calendar year) ≤ 5% under any dependence, and the bound grows by 5% for each further year. The rule was chosen on theory.

Lifetime null (seeds 601–602; weak dependence so the simulation can discriminate; 6 signals plus pairs; 52 weekly looks; 45 simulations):

- Every procedure made 0 false admissions over 52 looks. With raw tests the Wilson upper bound is 7.9%, so this run cannot separate the procedures empirically.
- A true +0.30R edge was admitted in 6.1% of looks uncorrected and 0% after any correction.
- Still open: alpha spending across years, e-values, and a preregistered forward holdout (§15.4).

### 15.3 Power: 20 blocks is a floor, not a sufficient sample (P1)

`python -m research.power_study --sims 300` gives the power of one row's test (status OK, mean > 0, one-sided p below the threshold) under a true edge measured in per-session SD units. The thresholds are: one test at 5%, the worst-placed row of a 44-row Holm family, and the annual budget.

| Dependence | Independent blocks | 0.3 SD: single / Holm / annual | 0.5 SD: single / Holm / annual |
|---|---|---|---|
| overlap | 20 (200 sessions) | 0.28 / 0.01 / 0.00 | 0.57 / 0.06 / 0.00 |
| overlap | 40 (400 sessions) | 0.50 / 0.06 / 0.00 | 0.93 / 0.34 / 0.04 |
| overlap | 80 (800 sessions) | 0.82 / 0.26 / 0.06 | 1.00 / 0.80 / 0.37 |
| overlap | 160 (1600 sessions) | 0.98 / 0.73 / 0.27 | 1.00 / 1.00 / 0.96 |
| overlap | 320 (3200 sessions) | 1.00 / 0.99 / 0.91 | 1.00 / 1.00 / 1.00 |
| overlap + ar0.5 | 20 (200 sessions) | 0.26 / 0.01 / 0.00 | 0.55 / 0.06 / 0.01 |
| overlap + ar0.5 | 40 (400 sessions) | 0.46 / 0.07 / 0.00 | 0.88 / 0.36 / 0.08 |
| overlap + ar0.5 | 80 (800 sessions) | 0.72 / 0.21 / 0.04 | 0.98 / 0.76 / 0.31 |
| overlap + ar0.5 | 160 (1600 sessions) | 0.98 / 0.67 / 0.27 | 1.00 / 1.00 / 0.93 |
| overlap + ar0.5 | 320 (3200 sessions) | 1.00 / 0.96 / 0.77 | 1.00 / 1.00 / 1.00 |
| overlap + ar0.9 | 20 (200 sessions) | 0.19 / 0.02 / 0.00 | 0.26 / 0.03 / 0.00 |
| overlap + ar0.9 | 40 (400 sessions) | 0.24 / 0.02 / 0.00 | 0.51 / 0.08 / 0.00 |
| overlap + ar0.9 | 80 (800 sessions) | 0.34 / 0.03 / 0.00 | 0.75 / 0.18 / 0.03 |
| overlap + ar0.9 | 160 (1600 sessions) | 0.68 / 0.10 / 0.00 | 0.95 / 0.53 / 0.13 |
| overlap + ar0.9 | 320 (3200 sessions) | 0.90 / 0.36 / 0.07 | 1.00 / 0.94 / 0.69 |

What this means:

- **At the verdict floor of 20 blocks, the annual budget has about 0 power even for 0.5 SD.** The production walk-forward uses a 250-session lookback (25 blocks), so under the current rules a BUY from the walk-forward alone is **unattainable**. That is a justified blocked outcome, not something to tune away.
- **The attainable budget is larger.** The bhavcopy store holds about 1,760 sessions (~170 blocks). A walk-forward over the full history would give about 90% power at 0.3 SD under the annual budget with plain overlap. It would also make the weekly host job several times slower, so it is **an owner decision** and was not changed here.
- **Evidence a claim would need.** Preregister ONE primary forward hypothesis, a single test with no family penalty, and confirm it on new forward sessions only. That needs about 40 blocks (400 sessions, ~1.6 years) for a 0.5 SD edge, or 80 blocks (~3 years) for 0.3 SD, at 80% power under plain overlap. Persistence raises it to 320+ blocks.
- **The current real evidence** is 24 sessions, which is 4 blocks.

### 15.4 Robustness (P1)

- **Shadow ledger migration:**
  - it keeps every session row and every prediction;
  - a second open changes nothing;
  - a simulated crash halfway through rolls back as a whole, and a rerun completes.
- **Earlier guarantees re-checked:** exact 10-session windows, the calendar-contradiction check, the corrupt-ledger refusal and kill -9 recovery are all still covered by their tests.

## 16. Alpha accounting, the full-history decision, honest UX (20 Nov; ChatGPT review of `f89c60af`)

The review was delivered at 01:11:28Z and acknowledged at 01:12:21Z. Nothing was deployed, no production data was changed, and no long backtest was run. Live trading is HARD-LOCKED.

### 16.1 The annual budget is now enforced, not just declared (P0)

`0.05/52` was a constant: forced reruns, restarts, extra report dates or a changed family could all spend budget unseen. `research/alpha_ledger.py` is a durable, research-only SQLite ledger (`logs/research/alpha_ledger.db`, new, not a production learning ledger).

- **Programs.** Each program is the hash of the predeclared family method, rulebook, horizon, sampling, membership rule and report schema. A program change opens a new budget, and old attestations do not verify under the current program.
- **Looks.** Each walk-forward report reserves one numbered look **before** its family correction runs, in one `BEGIN IMMEDIATE` transaction:
  - a forced rerun, a crash after reservation, or the same data under a new report SHA all spend budget;
  - a write retry of the same report does not;
  - nothing is refunded.
- **Budget.** Budgets are per IST calendar year. Look 53 gets α = 0 and admits nothing. Changing family membership cannot reset the budget.
- **Attestation.** Report schema 5 carries `alpha_accounting` (program, year, look index, look id, α). The gate (`evidence_provenance_note`) re-reads the ledger and refuses a missing, unknown, tampered or exhausted look, or one from another program. It does the same at read time in `load_scan`.
- **Tests:** the 53rd look, a forced rerun versus a retry, membership change, program change, rollback, five tampering cases, missing or corrupt ledger, the year boundary, 10 concurrent reservers, and an exhausted family correction.

### 16.2 Full-history walk-forward: plan only (P1)

`docs/research/full_history_backtest_plan.md`:

- **Cost.** About 70 minutes of one core on the i5-5250U, measured from the production 609.5 s run and an evaluation cost that is flat in history length.
- **Data validity.** It lists the checks needed first (corporate-action PASS on an SSD copy, survivorship of older years, exact-horizon skip rates).
- **Options compared:** one off-hours run, incremental recomputation, a partitioned resumable run, symbol sampling, a preregistered single hypothesis, and prospective paper evidence.
- **What the current report shows.** Every signal with ≥ 30 trades is measured **negative** (−0.02R to −0.88R), so a longer history buys precision, not an edge.
- **Not executed;** the owner decides.

### 16.3 Honest UX while no BUY is eligible (P1)

Home's BUY-only section:

- now reports `admission_state = NO_ELIGIBLE_BUY` and, when the evidence itself is the blocker (stale, unprovable budget, wrong rulebook), says so;
- states the actual admission standard and that "a BUY is not a win probability";
- has an evidence cache that turns over when the evidence authority changes (snapshot pointer or alpha ledger), not only when the scan file does.

Unchanged and checked:

- Telegram sends only WATCH lists when nothing is eligible; paper EXPLORATION stays labelled and separate.
- Legacy and missing-data paths were covered by earlier end-to-end tests.

### 16.4 Legacy bypass and pointer concurrency (P2)

- **Conviction.** `scan/conviction.build_conviction` can no longer **promote**: a row the evidence gate held at WATCH stays WATCH whatever its news or earnings "conviction".
  - A regression fails if any canonical module calls `build_conviction(` or imports `scan.conviction`.
  - The canonical `product.conviction.build_conviction_shortlist` inherits the gate: HIGH_CONVICTION needs a re-verified "Ready to trade" row.
- **Snapshot pointer.** Writes happen under an exclusive lock, and the report on disk is re-read inside the lock. A snapshot built from a report replaced in the meantime never moves the pointer. Writes are atomic (temp file + rename), so a crash leaves the previous pointer.

