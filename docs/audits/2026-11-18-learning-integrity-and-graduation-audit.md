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
identified the cause. `update_outcomes` now returns and persists
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
- **The ML challenger's features are mostly dead.** 21 of its 25 features are
  constant zero, which points to a feature-store population gap. Its
  out-of-fold Brier of 0.237 is about base-rate. It has 0 forward outcomes.
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
3. **Populate the ML feature store.** 21 of 25 features are zero. Re-validate
   the challenger out-of-sample before it is allowed to rank anything.
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
