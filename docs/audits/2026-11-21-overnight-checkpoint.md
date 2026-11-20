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

## Open (next)

B-6 delisting truncation · QT-12 CA digest · C-8 remainder · C-2 result digest · GOOD_WAIT age · research cycle.
