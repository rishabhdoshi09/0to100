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

