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

## Open (next)

C-2 result digest · GOOD_WAIT age · research cycle.
