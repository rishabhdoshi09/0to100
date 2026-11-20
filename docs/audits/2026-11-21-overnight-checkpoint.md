# Overnight engineering checkpoint (20–21 Nov 2026)

This is a factual journal of the autonomous overnight cycle, updated in the commits it describes. It holds no secrets.

- **Live trading:** HARD-LOCKED throughout.
- **Production:** nothing deployed or restarted. No production learning ledger was touched, and the learning-policy repair was not applied.

## Start state

- **Branch:** `claude/settlement-batch-20261118`, starting at `f6081b26`. All 5 workflows are green there, all 498 test files pass locally, and the GPT cross-review was done (`docs/audits/2026-11-20-gpt-crossreview.md`).
- **Production:** `125880a6`.
- **Host at 23:33 IST:** the desk was DOWN.
  - The WORKING runtime image was attached but stale after this morning's power loss, so the API and UI were not answering.
  - The owner wrote "detach and remount". A plain `hdiutil detach` of the stale disk followed.
  - The desk's own storage preflight then re-attached the WORKING image (`/Volumes/QuantTermStorage`).
  - `/api/health` came back with `operational_ready: True`, `evidence_ready: True`, live lock `LOCKED` and verified. Nothing else on the host changed.
- **Relay:** retargeted at the owner's request, from the archived session `session_01JGESyQkM5SMQcAebqMciiJ` to `session_01JGESyQ8EfsS4H13HK9Yb8Q`.
  - Config backup: `config.json.bak-20261120-pre-retarget`. Ledger and acknowledgement history are preserved.
  - Verified: GPT instruction 7046791627 was delivered as `qr-9a22ec4b3666` and acked without being run twice (this cycle executes it). An owner round-trip TEST (7047033044) was delivered as `qr-5e240bd2c62c` and acked.
  - Both used slots freed under the 8-per-24-hour cap.

## Cycle 1: A-3, settlement → learning is exactly once across crashes

**Defect (confirmed on `f6081b26`).**

- The forward ledger was marked settled before learning ran.
- Inside learning, the observation was claimed before the policy was written.
- A crash at either point lost the observation for good, because every retry saw "settled" or "duplicate".

**Fix.**

1. A durable OUTBOX (`product/learning_observations.py`). Each event's id is the forward-ledger decision plus its eval spec. It is written before the ledger rewrite and replayed by `forward_soak.replay_learning_outbox()`.
2. Two-phase policy claims. A claim is CLAIMED and carries its apply payload. It becomes APPLIED after the policy write, which is stamped with `last_observation_key`.
3. `learning_policy_store.recover_claims()` replays a CLAIMED claim, or confirms it from the stamp. This runs under a cross-process writer lock.
4. The counterfactual unit claim is resumable.
5. Pre-existing claims are LEGACY: they still dedupe, but are never replayed.
6. Corrupt events are QUARANTINED, and failures retry the outcome job.

**Tests.** `tests/test_learning_exactly_once.py` has 9 tests. 8 fail on `f6081b26`; the concurrent-workers case was already safe there and stays as a control. They cover:

- a crash before the ledger write, after it, after the unit claim, and between the policy write and its mark;
- copies and repeated deliveries;
- concurrent workers;
- a corrupt event;
- a legacy claim and a legacy label.

Two older tests that pinned the at-most-once loss now assert exactly once.

**Production note.** Nothing was migrated. On first use, the claim table gains `state`/`payload` columns, with existing rows marked LEGACY, plus an `outbox` table. Both changes are additive.

## Cycle 2: B-6, a stock that stops trading cannot silently leave the evidence

**Defect (confirmed on `6d69603c`).** The walk-forward evaluated a signal only when a full 10-session window followed it. A delisted or suspended stock's final signals were dropped, even when the stop had already been hit in the bars that exist. That is an optimistic bias: dying stocks' losses disappeared.

**Fix (`scan/signal_backtest.py`).** For a symbol whose bars end before the last official session:

- a partial window counts when the trade RESOLVES inside it (stop, target or scratch), and must be exact official sessions up to the last bar;
- a trade still open when prices stop is CENSORED: never priced at the last close and never counted as a win or a loss, but reported (`stopped_symbol_trades_censored`, beside `_resolved`).

A still-trading symbol's recent signals stay pending. The target sweep uses full windows only.

**Offline before/after** (the same SSD copy and 200 symbols):

| Tier | Before | After |
|---|---|---|
| STRICT | 5,239 trades, −0.0922R [−0.186, +0.001] | 5,242 trades, −0.0924R [−0.186, +0.001] |
| COVERED_ALL | 6,031 trades, −0.0811R | 6,035 trades, −0.0812R |

The bias was real but small on this sample. The verdict is unchanged: NO_MATERIAL_EDGE.

**Tests.** `tests/test_stopped_symbol_censoring.py` has 3 tests (resolved loss counted; open trade censored; still-trading symbol pending). All 3 fail on `6d69603c`.

## Cycle 3: QT-12, CA coverage is bound to the exact events it vouches for

**Defect (confirmed on `b61ca527`).** The CA refresh persists each window's coverage as it walks, and writes the merged events only at the end. An interrupted refresh therefore left a newer coverage declaration over an older events file, and the point-in-time loader only checked that some events existed.

**Fix.**

- **Writer.** Coverage is marked unbound (`events_sha256: ""`) during the walk, and bound last to the SHA-256 of the events file actually written. `coverage_status()` reports `events_binding` (BOUND / UNBOUND / MISMATCH).
- **PIT loader (`research/edge_diagnosis.load_copy`).**
  - A MISMATCH is never trusted.
  - UNBOUND (legacy) is trusted only with the explicit research flag `--accept-unbound-coverage`. The report records `ca_coverage_binding`, and a labelled `ca_coverage_accepted_unbound`.
  - Production CA adjust-on-read is unchanged. The next nightly refresh binds the existing files.

**Tests.** 4 tests in `tests/test_corporate_actions_resilient.py`, all failing on `b61ca527`.

## Cycle 4: C-8 remainder, every consumer of a saved BUY re-checks it at read time

**Defect (confirmed on `746d0a4f`).**

- The decision board (`/api/decisions`, best trades) showed a BUY from the saved recommendations even after `load_scan()` had withdrawn it.
- The Telegram desk alert checked the scan generation but not each symbol's certificate.
- The board's API cache did not turn over when the scan or the evidence authority changed.

**Fix.**

- One certificate: `product/scan_store.current_buy_certificate(scan_scanned_at)` returns the symbols `load_scan()` certifies as BUY now, for that scan generation. It fails closed.
- **Paper entry** uses it.
- **Board:** an uncertified BUY shows as WATCH (`demoted_by: BUY_NOT_CERTIFIED_AT_READ`) and is never a best trade. The API cache key includes the scan file and `evidence_authority_key()`.
- **Desk alert:** it sends `withdraw_uncertified_cards(workspace, certificate)`.
- Publication-time consumers (`track_published_buys`, immediate alerts) were already re-certified in `f6081b26`.
- Historical publication records are untouched.

**Tests.** 2 new tests, both failing on `746d0a4f`. The board tests that are about ranking now use a labelled certify-all fixture, and the marker is registered.

## Cycle 5: C-2, the BUY evidence is bound to its statistical RESULT

**Defect (confirmed on `70e3f43a`).** The alpha look was bound to report metadata only, so rows edited afterwards still verified: a family flag flipped, or a bound changed. A look reserved but never completed also verified.

**Fix.**

1. `signal_backtest.result_digest()` is a canonical SHA-256. It covers every signal and combination row's admission fields (trades, means, lower bound, blocks, inference, Holm flag, p-values, se, df), the family summary and the provenance binding.
2. After inference, the reserved look is FINALISED to that digest (`alpha_ledger.finalize`: write-once; a different digest raises `LedgerConflict`), and only then is the report saved.
3. The evidence snapshot and the scanner's report fallback RECOMPUTE the digest from the rows they will use.
4. `evidence_provenance_note` requires the declared digest, the recomputed digest and the ledger's finalised digest to be equal.

Consequences:

- An interrupted report (reserved, not finalised), edited rows or a foreign digest certify nothing.
- No external trust anchor is claimed. An attacker with write access to the runtime is still out of scope.
- The column addition is additive. Old looks are unfinalised, so they cannot certify; that was already true after a deploy until the walk-forward reruns.

**Tests.** `tests/test_result_digest.py` has 11 tests (finalise-then-save on a real stub run; every admission field inside the digest; a JSON round trip is stable). The behavioural ones fail on `70e3f43a`.

## Cycle 6: GOOD_WAIT identity and age; legacy paths verified dormant

**GOOD_WAIT (P2, confirmed).** Every paper entry ran `note_later_entry(symbol)`, which stamped GOOD_WAIT on every pending counterfactual of that symbol, including hard rejections and waits weeks older than their horizon. It now credits only the symbol's pending WAIT decisions whose own 10-session horizon covers the entry day; other rows keep waiting for their rulebook outcome. 1 regression test.

**C-10 (legacy sniper and autopilot): dormant, verified and pinned by a test.**

- The legacy scanner worker, and its disk warm-up (`scan_store.json`, last written 30 Jul and ignored once older than 3 days), are started only by `start_background_scan()`, which only `legacy_app.py` and the archived `ui/` pages call.
- The Telegram `/trade` command reads that empty store, so it never reaches a legacy entry.
- The test asserts that no canonical module (API, market_ops, supervisor) calls `start_background_scan`.

## Cycle 7 (research): a pre-registered momentum portfolio alternative

**Pre-registration.** `PR-2026-11-21-MOM12-1` was frozen and committed in `ab4b1690` BEFORE any evaluation. It covers: 12-1 momentum, top 20% equal-weight, monthly, costs, an equal-weight benchmark, one primary test (m = 1, outside the BUY alpha budget), power (about 223 months for 0.5%/month), the missing-data rule, and a held-out year evaluated once.

**Code.** `research/momentum_portfolio.py`: SHA-pinned, refuses the live runtime, refuses the held-out year without an explicit single look. 6 hermetic tests (future-poisoning, coverage gating, missing bars, costs, traded-price floor, frozen spec).

**Discovery (37 months, Oct 2022 to Oct 2025).**

- Net excess +0.31%/month, t = 0.85, p = 0.20: **NOT SIGNIFICANT**.
- Regression alpha −0.04%/month, **beta 1.18**: the outperformance is market exposure, not selection.
- Maximum drawdown −28.7%; worst month −14.6%; turnover 31% one-way.
- 66 held names had no bar at exit. 50 of them resumed later (data gaps, not losses); 16 did not.
- The held-out year was NOT evaluated (the spec forbids it after a non-PASS).

Report: `docs/research/momentum_12_1_discovery_2026-11-21.md`.

**Scientific position.** It is unchanged: no strategy, old or new, shows positive risk-adjusted expectancy after costs. NO ELIGIBLE BUY remains correct.

## Cycle 8: independent GPT review of the overnight work, and its fixes

**Review.** Two independent, read-only Codex (GPT) reviews of `f6081b26..8281810c` ran on a frozen detached worktree, with no secrets and no network or broker access:

- Run E covered evidence, BUY admission, censoring and corporate-action coverage.
- Run L covered settlement → learning.

They reported 16 findings: E-01..E-07 and L-01..L-09, of which 14 were P0, 1 was P1 (E-07, L-08) and 1 was P2 (L-09). GPT ran no tests. Its claims were treated as hypotheses.

**Verification.** Every finding was first reproduced by a deterministic test on the base (`38429d98`), which failed there. It was then fixed, and the test passes on the fix. The tests are in `tests/test_overnight_review_20261121.py`, with 36 tests: 30 fail on the base, and the 6 that pass there are controls (a certified BUY still opens and still shows, a proven snapshot still certifies, a recent window stays pending, the HIST projection is still replaced, and a GOOD_WAIT inside the horizon still counts). All 36 pass on the fix. GPT's original sketches are used where they ran as written. E-06's original sketch patches a function the fix no longer calls, so it was run once on the base, where it fails, and the suite carries an equivalent hook on the new single read.

| Commit | Findings | What changed |
|---|---|---|
| `02010369` | E-01, E-02, E-03 | Every BUY is re-certified at its point of use. The paper entry re-reads the certificate inside the account lock. The cached `/api/decisions` is re-certified on every response, stale or fresh. The board and Why-This-Decision withdraw uncertified cards before enrichment, so the write-once feature observation never freezes an uncertified BUY. A committee BUY the current scan does not certify reads `BUY_WITHDRAWN_AT_READ`. |
| `3c7aae7a` | E-04, E-05, E-06, E-07 | A calibration snapshot carries the exact admission rows its ledger-finalised digest covers. Edges are derived from those rows (`verified_view`), never from its stored edges; snapshot schema 2 is a new id. The walk-forward simulates the verified prefix of official sessions: an exit before a suspension survives later bars, and a trade open at a gap is censored. CA commit protocol 2 writes events before the window record, so only committed records count. The PIT loader hashes and parses one read. |
| `9971fb06` | L-01, L-02, L-03, L-05, L-06, L-09 | Forward rows have one settlement owner (`settle_official_outcomes` delegates). Every learning apply is checked against the ledger as committed; a mismatch is quarantined, and replay waits for uncommitted rows. A resumed unit uses its own stored payload. Recovery requires that the scope equals the payload's policy. The feature label gets the decision id, and its errors are retried. The schema migration runs under `BEGIN IMMEDIATE`. |
| `ee6cf5e0` | L-04, L-07, L-08 | One writer protocol for the policy store (history publication, generation purge, replay priors, repair). Calibration is idempotent per taken outcome, and the ladder sink is not suppressed by the cell. GOOD_WAIT counts sessions through the fill session; an unverifiable ordinal stays pending. |

**Disagreements and limits.**

- **E-02.** GPT asked for scan publication to be coordinated with execution. The fix re-checks the certificate inside the account lock instead. That check is the entry's linearisation point: a BUY certified at that instant is a valid entry, and a scan published during the few milliseconds of the in-lock open does not undo it.
- **E-03.**
  - **Write-once freeze.** The board's freeze now records WATCH when the certificate is empty. That includes a transient unreadable scan, which fails closed. This is conservative.
  - **Held positions.** For a position already open under a BUY that was later withdrawn, the committee view shows `BUY_WITHDRAWN_AT_READ`, with its execution state kept for audit.
- **L-01.** Rows that the old second settler already settled without a learning event are NOT reconciled automatically. Applying them would change production learning records, so that is an owner decision (below).
- **L-06.** Legacy forward rows are labelled under their forward decision id. A call for an id with no feature observation is `not_found` and changes nothing.

**Deploy consequences (if the owner deploys).**

1. The next scan builds a schema-2 snapshot (a new id). Saved BUYs from the old snapshot stay WATCH until that scan re-judges them, which fails closed.
2. The first CA refresh re-walks all 11 half-year windows once under protocol 2. Until it finishes, `corporate_actions_incomplete` (and "new paper entries: limited") shows, which is honest.
3. GOOD_WAIT credits wait when the archive cannot verify the session ordinal.

**Production learning scorecard (read-only, unchanged tonight).**

- 0 decisions changed by learning; stage 0.
- Tracked BUYs: mean −0.127R, 95% CI [−0.31, +0.06].

## Open (next)

Owner decisions:

- `deploy <sha>`;
- `apply repair` (independent-unit policy rebuild);
- the stricter C-5 policy;
- starting the 52WH shadow test;
- a full-history whole-universe run;
- whether to reconcile legacy settled-without-event forward rows into learning (L-01).

Not done tonight:

- an edge-diagnosis rerun to report the E-07 censoring counts on the SSD copy;
- a GPT review of live-broker controls and account risk (outside both runs' scope).
