# Independent GPT cross-review of `9636b030` (20 Nov 2026)

Target: branch `claude/settlement-batch-20261118`, commit `9636b03055961e5f53b5dfb7cfe5cfede03ba13f`. That commit was green on all five CI workflows.

**This is a development record. It approves nothing.**

- Nothing was deployed or restarted.
- No production ledger was read for writing or modified.
- Live broker execution stays HARD-LOCKED.
- A green suite does not establish trading profitability. The offline diagnosis below still finds no edge.

## Protocol

1. **Reviewer.** GPT ran through the OpenAI Codex CLI, signed in to the owner's ChatGPT account, in a `read-only` sandbox.
2. **Isolation.** GPT read its own frozen, detached checkout of `9636b030`, not the working tree being edited.
3. **Inputs.**
   - GPT was given the source and tests, the commit list, and the diffs `125880a6..9636b030` and `dd4e9c34..9636b030`.
   - It was not given conclusions: developer docs and commit messages were labelled claims to verify.
   - The prompts are in `crossreview-20261120/prompt_{A,B,C}.md`.
4. **Scope.** Three parallel runs:
   - **A:** the pipeline, the 5- vs 10-session counterfactual horizon, and whether feedback steers decisions.
   - **B:** point-in-time integrity, look-ahead, survivorship and corporate actions.
   - **C:** the alpha budget, provenance, BUY admission and risk-control bypass.
5. **Output format.** GPT classified each finding P0/P1/P2 and marked it CONFIRMED or HYPOTHESIS, with `file:line`, a counterexample and a pytest sketch. The raw reports are in `crossreview-20261120/gpt_run_{A,B,C}.md`.
6. **Independent pass.** While GPT ran, Claude reviewed the same scope separately and wrote its own findings down before reading GPT's.
7. **Verification.** Every GPT claim was run as a deterministic test against `9636b030`.
   - A finding counts as CONFIRMED only if its test failed there.
   - All 26 regressions in `tests/test_crossreview_20261120.py` fail on `9636b030` and pass on the fix.
   - 8 of the 9 money-path regressions in `tests/test_money_paths.py::TestCrossReview20261120` fail on `9636b030`. The ninth is a deliberate control that must pass on both.
8. **Secrets and security.** No secrets were shared. Codex was told not to read `.env`, home configs, `/Volumes` or the production checkout. No network, broker or process control was allowed.

## Results: 33 GPT findings, 5 Claude findings

GPT flagged most of its findings P0. Our severity column re-grades them for this host, where live trading is locked and the evidence gate currently admits zero BUYs:

- **P0** means paper risk limits or a production paper fill can be bypassed today, or an invalid BUY can be certified once evidence exists.
- **P1** means the statistics, learning or labelling are wrong.
- **P2** means robustness or display.

| ID | GPT | Ours | Verified on 9636b030 | Resolution |
|---|---|---|---|---|
| A-1 | P0 | P0 | yes (probe + `test_a_withdrawn_buy_never_fills_as_a_production_entry`) | **Fixed.** `paper_autopilot._current_scan_certificate`: when the recommendations' own scan is current, every production-tier card must still be a BUY in `load_scan()`; otherwise it is withdrawn to a gated WATCH card, explorable only in the labelled 0.5% lane |
| A-2 | P0 | P1 | yes | **Fixed.** A close joins its entry by `paper_decision_id` (legacy rows by entry session, only when unambiguous), once (`closed_trade_key`); `_lookup_taken` joins the same way |
| A-3 | P0 | P1 | not run | **Open, agreed.** At-most-once ingest: a crash between the ledger write and the policy ingest loses the observation (by design until a dedupe existed). Needs a replayable learning-event log; no money path |
| A-4 | P0 | P1 | yes (challenger part) | **Fixed for the challenger:** `outcome_meta.eval_spec` is recorded, and simulated labels without the current rulebook do not train it. **Open:** mixed legacy `SETUP::` aggregates need the owner-approved repair rebuild |
| A-5 | P0 | P1 | yes | **Fixed.** Staging keys each freeze by its own id. This is a research migration tool and was never run on production |
| A-6 | P0 | P1 | yes | **Fixed.** BACKTEST / WALK_FORWARD rows never settle into forward learning, the official loop or the repair |
| A-7 | P0 | P1 | yes | **Fixed.** `core.session_window`: outcome bars must be exactly the next official sessions; a missing session ends the verified window |
| A-8 | P0 | P1 | yes | **Fixed.** Equity evolution shadows are graded by `counterfactual_outcome` (rulebook gap exits, costs, `eval_spec`) |
| A-9 | P1 | P1 | yes | **Fixed.** A signal_log write or fatal error (or an unloaded store) makes the installed outcome job RETRYABLE_FAILED |
| A-10 | P1 | P2 | yes | **Fixed.** The scorecard reads the persisted `policy_result` |
| B-1 | P0 | P1 | yes | **Fixed for the ₹20 floor.** `adjust_frame` carries its events in `attrs`; the floor uses the traded price. The bias was conservative: future splitters, typically past winners, were dropped. Residual: tick rounding of adjusted order levels |
| B-2 / C-7 | P0 | P1 | yes | **Fixed.** Replay uses predeclared neutral weights (`NEUTRAL_REPLAY_CALIBRATION`), never today's snapshot |
| B-3 | P0 | P0 | yes | **Fixed.** In `load_scan`, a BUY's `edge_evidence_id` (and the scan's id) must be the current authority's snapshot id; a later report does not retro-certify |
| B-4 | P0 | P1 | yes | **Fixed for covered years.** The calendar's sessions fill a market-wide missing archive, and uncovered future years are never verified. Residual: uncovered past years rely on archive dates |
| B-5 | P0 | P1 | yes | **Fixed.** A live intraday overlay row is dropped before settlement; late-target windows fail the exact-session check |
| B-6 | P0 | P1 | not run | **Open, agreed in principle.** A signal near a delisted symbol's last bar is never evaluated, so an observable early stop-out is lost (bias up). Needs a delisting-aware schedule; the next cycle |
| B-7 | P0 | P1 | yes; **found independently by Claude** | **Fixed.** Replay classification and training R are the rulebook's; a trade that never triggered has no R |
| B-8 | P0 | P1 | yes | **Fixed.** A non-finite bar neither fills nor exits; a fill at or above the target is a 0R SCRATCH at the fill. **Open (P2):** `sweep_targets` (research display) still ignores opens |
| B-9 | P1 | P1 | yes | **Fixed.** CA coverage carries its end; a window past it is UNCOVERED; a coverage file without events covers nothing |
| B-10 | P1 | P2 | yes | **Fixed.** A parseable but empty or invalid membership table is `survivorship_complete=False` |
| C-1 | P0 | P1 / disagree | counting: yes | **Fixed:** blocks are maximal disjoint windows, not occupied bins (stricter df and min-blocks gate). **Disagree that the common-shock null is a code defect:** one sign shared by the whole history is a single observation that no test on one realised history can detect. The assumption (dependence decays within ~2 horizons) is now stated in `overlap_inference` and the alpha-ledger guarantee |
| C-2 | P0 | P1 | yes (id and cache parts) | **Fixed:** foreign-id BUYs (as B-3) and an authority cache keyed on the snapshot target. **Open:** a result digest finalised per look. Editing report rows needs write access to the runtime, which is out of scope (as stated in the f89c60af cycle) |
| C-3 | P0 | P1 | yes | **Fixed:** an overridden `QT_ALPHA_LEDGER` never certifies outside pytest; a lost ledger with prior looks opens no fresh budget; the guarantee is stated per *reservation* year |
| C-4 | P0 | P1 | yes | **Fixed:** the run configuration is part of the gate program, so only the production configuration certifies; every catalogue signal is in the family (untested at p = 1); the save threshold counts usable symbols. Disclosed: combinations still join the family at ≥ 30 trades |
| C-5 | P0 | P1 / partial | yes | **Fixed:** averaging is replaced by the weakest evidenced signal (never looser). **Disclosed, owner decision:** the exact intersection is only tested at ≥ 30 trades, so a marginally proven pair can still lose as a subgroup |
| C-6 | P0 | P0 | yes | **Fixed.** Provenance must carry `exact_horizon_check=True` and data no older than 21 days |
| C-8 | P0 | P0 / P1 | yes (paper + Home) | **Fixed:** the production paper fill (A-1) and Home BUY-only require the read-time certificate. **Open (P1):** the decision-board API, the Telegram desk alert and `track_published_buys` still consume the publication-time verdict |
| C-9 | P0 | P0 | yes | **Fixed.** A Telegram tap refuses a zero size (no forced share) and re-checks the shared account gate under the account lock; callback levels keep paise |
| C-10 | P0 | P2 | not run | **Open.** The legacy sniper and legacy autopilot are not started by the canonical stack (`legacy_app.py` only) |
| C-11 | P0 | P0 | yes | **Fixed.** The installed paper-cycle wrapper passes `durable_book_save`, so the open is persisted before the account lock is released |
| C-12 | P0 | P0 | yes | **Fixed.** The exploration slot is re-checked and reserved under the lock before the fill; an unreadable or unwritable quota blocks the probe |
| C-13 | P1 | P2 | yes | **Fixed.** A read-time authority outage reads EVIDENCE_UNAVAILABLE, not NO_PROVEN_EDGE |

Claude's own findings, written before reading GPT's:

1. The replay 5-session grading, the same as B-7.
2. `decision_simulator._classify` still graded on a 5-session close move (P2, display). **Fixed.**
3. `outcome_integrity` re-probed with the retired resolver and included taken rows (P2). **Fixed.**
4. The silent exploration-quota write, the same as C-12.
5. **New, open (P2):** `note_later_entry` stamps GOOD_WAIT on every pending rejection of a symbol at each paper entry, however old.

## Second pass: GPT verified the fixes (`9636b030..01dec27c`)

There was one verification pass, not a loop. The scope is in `crossreview-20261120/prompt_verify.md`. GPT found 14 gaps in the first-round fixes: 9 P0, 4 P1 and 1 P2. Each was reproduced as a failing test on `01dec27c`: the `QT*` tests in `tests/test_crossreview_20261120.py` and four tests in `TestCrossReview20261120`.

| ID | Gap in the first fix | Resolution |
|---|---|---|
| QT-01 | The paper certificate failed OPEN on an unreadable scan or another scan generation | Fails closed: no matching certified scan, no production entry |
| QT-02 | The lost-ledger guard overwrote the last spent report, so the next run reopened the budget | The loss is latched: such a run is not saved, and a `.provisioned` sentinel or the snapshot's provenance also detect earlier spending |
| QT-03 | A tap followed looser 2% / 20% settings | A tap is hard-capped at 1% risk and 10% of capital per name |
| QT-04 | A failed in-lock save released the lock with in-memory exposure | Retried under the lock. If it still fails, a plain PaperBook is rolled back before release; any other book raises inside the lock (fail-closed job boundary) |
| QT-05 | Home's cache missed the snapshot target and evidence expiry | The authority key includes the target file identity and an hourly bucket |
| QT-06 | Store bars with no archive listing settled positionally | Unresolved when the store is loaded but the official session list is unavailable |
| QT-07 | A missing or incoherent open let a bar fill below its own low | When opens are part of the series, a bar without a finite, in-range open is unusable |
| QT-08 | The date fallback borrowed another position's context; a legacy close could settle a second row of its session | No conflicting ids, unambiguous date matches only; legacy rows must be the session's ONLY entered row |
| QT-09 | Evolution promotion still read legacy equity grades | `scorecard.graded_rows` admits only rulebook-graded equity rows |
| QT-10 | A replayed NO_FILL kept a close-path return that scorecards fell back to | Its top-level return is None (the close path stays under `path`) |
| QT-11 | An unusable frame counted toward the save threshold and the data span | A symbol counts only once it contributes an analysed window |
| QT-12 | Coverage written before events could vouch for an older events file | **Open (P1, research tool):** needs a writer-side events digest in the coverage file |
| QT-13 | A `listed: null` row counted as valid membership; unknown membership stayed in the diagnosis | Validated rows only; unknown membership is excluded from every tier |
| QT-14 | The async replay cache keyed the default run as uncalibrated | The same neutral default for lookup and run |

GPT's views on the two disagreements:

- **C-1.** GPT agrees that the greedy count is exact and that a common shock across the whole history cannot be identified from one history. The inference is conditional on the stated assumption, as now written.
- **C-5.** GPT holds that "under the stated BUY invariant, an untested intersection remains WATCH". We keep weakest-evidenced-link plus disclosure. Requiring tested evidence for the exact combination (or for every component signal) is a stricter admission policy and is left to the owner.

## Answers to the six questions

1. **Pipeline.** The canonical chain is wired (GPT and Claude agree on the trace). The settlement → learning links had the defects above (A-2, A-6, A-7, A-9, B-5), now fixed.
2. **5- vs 10-session horizon.** `19a98b2f` fixed the two forward-ledger loops correctly. The same defect survived in three more places, now fixed:
   - historical replay (B-7);
   - the decision simulator;
   - evolution equity grading (A-8, which used a first-touch path, not 5 sessions).

   No equity learning label path still grades on a 5-session close.
3. **Point-in-time.** Five leaks or optimisms were real and are now fixed:
   - B-1: a future split changing the ₹20 floor;
   - B-2: replay scored with a future calibration;
   - B-4 and B-5: windows stretched across a missing session or an overlay;
   - B-8: impossible fill-bar executions.

   B-6 (delisting truncation) remains open.
4. **Alpha budget.** One global 52-look budget per reservation year holds for one intact canonical ledger. It now refuses:
   - an overridden or lost ledger;
   - a non-production configuration;
   - an unverified window check.

   The family is fixed before outcomes. Its validity rests on the stated decay assumption (C-1).
5. **Do outcomes steer decisions?** Yes, through four routes. None can promote a setup to BUY, which only the walk-forward gate can certify:
   - setup and reason policies, and conditional evidence, which act on paper rank and veto;
   - scanner demote-only live-edge multipliers;
   - the challenger once PAPER_ACTIVE;
   - exploration probes of gated setups.

   Written but unread: `learning_memory.json` and `learning_observations.jsonl` (GPT A, verified-correct section).
6. **Bypass paths.**
   - **Closed:** the production paper fill on a withdrawn BUY (A-1), Home (C-8), Telegram taps (C-9), the lock race (C-11) and the exploration quota (C-12).
   - **Remaining, listed above:** the board API and desk alerts (C-8 rest) and the legacy-only paths (C-10).
   - **Broker:** no path can reach the broker. Both reviewers traced the live interlock.

## Consequences when this is deployed (owner decision, not done)

- **BUYs.** No BUY is possible until the walk-forward reruns: the gate program now includes the run configuration and the fixed family, so every earlier attestation stops verifying. This is fail-closed and the same as after every evidence-standard change.
- **Replay cache.** Historical replay caches re-key, because neutral calibration changes their identity, and rerun on host time.
- **Suspended names.** Some forward rows of suspended names stay pending, because the exact-session check never fabricates their window.
- **Outcome job.** A signal_log failure now retries the outcome job instead of reporting SUCCEEDED.

## Reproduce

```
# deterministic regressions (network-free)
python -m pytest tests/test_crossreview_20261120.py tests/test_money_paths.py -k "CrossReview or crossreview"
# re-run the review itself (read-only; see scripts/gpt_crossreview.sh)
bash scripts/gpt_crossreview.sh <base_sha> <head_sha>
```
