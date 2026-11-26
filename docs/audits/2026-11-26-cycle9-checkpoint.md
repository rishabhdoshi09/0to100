# Cycle 9 checkpoint: durability gaps on disposable runtimes (26 Nov 2026)

Branch `claude/cycle9-20261126`, from `161f7cbe` (`claude/integration-c8-codex-20261124`:
Claude cycle 8 + Codex's integrity work, 5/5 CI). Instruction: #333 comment 7121239222
(GPT review of `bf787f41`, items 1–4), plus the lifecycle audit's open F1 (GOOD_WAIT) and
F8 (freeze cost). Development and paper only; nothing deployed, no production data touched.

## Host (read-only, 26 Nov 13:40 IST)

- The desk has been paused since **24 Nov 20:01 IST** (`Errno 5`). The Mac slept in clamshell mode on battery
  25 Nov 14:26 → 26 Nov 13:26, reached 4% at one point, and is now on AC.
- `Expansion` (disk2s1) is mounted. The WORKING image is attached (disk3, APFS disk4s1
  "QuantTermStorage") but **not mounted**. Its diskimages-helper (pid 45609) dates from 24 Nov 09:44,
  before the outage. This is the stale-attach pattern of 19, 22 and 23 Nov.
- Recovery needs the owner's words ("detach and remount"): `hdiutil detach disk3`, then wait for the
  supervisor's own re-attach (no manual attach: it raced on 22 Nov). Not done.

## The cycle

| Item | Finding | Reproduced on `161f7cbe` | Fix | Tests | Mutants |
|---|---|---|---|---|---|
| **F1 GOOD_WAIT** (lifecycle audit) | A later-fill credit claimed its unit, updated two policies, then rewrote the ledger, with nothing on disk recording the fill. A crash lost the credit for good, and the next settlement priced the WAIT as MISSED_WINNER beside a policy that had counted GOOD_WAIT | 12/15 fail | The fill is marked durably first; every credit is made from the mark; a unit claimed before marks existed is finished with its own fill; a copy of a queued unit is never priced and takes the unit's label; a failed recovery prices no WAIT that pass | `test_good_wait_crash_safety.py` (15) | 10/10 |
| **Item 1** manifest writers | `_publish_plan`'s check-to-rename window: a concurrent apply's record (state applied + backup) was replaced by a fresh plan beside a repaired database, so the rollback evidence was gone | yes, two processes, a barrier at the publish write | One lock per manifest held by plan/apply/rollback from first read to last write: a never-replaced file keyed by the case-folded name in the manifest's physical directory; flock, non-blocking, fail closed; released by the kernel on death; manifest lock before SQLite's lock; no lock file before the place checks; writes refused without the lock or after it was removed; hard links refused | `test_cycle9_manifest_lock.py` (16; 11 fail on base) | 9/9 |
| **F8** freeze cost (lifecycle audit) | Every freeze read, parsed and hashed the whole ledger (plus a second full read for the torn tail): O(n) per freeze | measured | An index trusted only while the ledger's identity (dev, ino, size, mtime_ns, ctime_ns) is exactly what this process last read or wrote under the writer lock; same validation on every rebuild; the torn-tail check reads only the tail | `test_counterfactual_freeze_index.py` (11) | 11/11 |
| **Item 4** feature002 PIT | (a) An official-archive bar of the recording day was used by a row recorded before that session's close (the old provenance test encoded it); a bar dated after the recording day was used. (b) Under Kite authority a name missing from the snapshot was read from the store and ranked beside Kite frames (another adjustment, another as-of) | yes | (a) bars that did not exist at `recorded_at` are dropped and counted (`f002-input.v2`); (b) one ranked generation (Kite snapshot, else store, else one uncertified source, ranked but marked uncertified); mismatched or stale rows excluded with a reason; two uncertified sources rank nothing; two Kite snapshots still rank nothing | `test_cycle9_feature002_pit.py` (13) | 12/12 |
| **Item 4** F&O independence | None found: the F&O universe orders the walk and labels cards only; records are sorted by (−score, symbol) | n/a | none | pinned by a test (3 F&O sets, identical records) | n/a |
| **Item 2** cooperative stop | A SIGTERM during the tick's preparation (controls, the Telegram drain, scheduling) still LEASED and started a new job. Inside the 12 s grace, a long job is then SIGKILLed mid-run, and an interrupted automatic PAPER_CYCLE is retired as failed on restart | yes | the tick leases nothing once the stop is set | `test_cycle9_stop_boundaries.py` (5: pre-lease, mid-handler with a follow-up, the status write, start(), restart drains each job exactly once) | 1/1 |

### F8 measurements (disposable copy of a 32 MB / 6,430-row ledger, internal SSD, simulated)

| | base `161f7cbe` | cycle 9 |
|---|---|---|
| new freeze p50 / p95 / p99 | 845 / 909 / 965 ms | 85 / 102 / 121 ms |
| throughput | 1.18 freezes/s | 11.7 freezes/s |
| retried freeze p50 | 716 ms | 1.2 ms |
| another process appending between every two freezes | full read each time | full read each time (as designed) |

The remaining ~85 ms is the durable intent + append (4 syncs) and the canonical SQLite freeze. On the USB
HDD the syncs cost more (cycle 8 measured F_FULLFSYNC p50 58 ms, p99 ~0.6 s). These are simulated
numbers, not physical durability: physical durability rests on the barriers measured in cycle 8.

**Group commit was not built.** Freezes are interleaved with per-decision work that uses the returned
freeze id. Batching would acknowledge a freeze before its durable commit, or restructure every producer.

## GPT cross-review of `161f7cbe..a8cb2eef` (two read-only passes)

Part A covered learning, settlement and autonomy (A1–A6). Part B covered the maintenance tool and research inputs
(C9B-01–07). Each claim was made a test (`tests/test_cycle9_review.py`) and run on `a8cb2eef`: **13 of 13
tested claims failed there**. A1 and A5 were not built as tests; their verdicts follow.

| Claim | Verdict | Fix |
|---|---|---|
| **A2 (P0)** The forward settler priced a WAIT unit as MISSED_WINNER (with its reason-policy observation) while its verified fill was queued | **confirmed** | `good_wait_holds()` (queued marks plus claimed GOOD_WAIT units) is one source of truth. The forward settler settles a held WAIT only as GOOD_WAIT from that fill, else it waits. The recovery reads the winning claim, and a unit decided elsewhere first is never stamped GOOD_WAIT (`later_entry_superseded_by`) |
| **A3 (P1)** One unit could get two labels: a second copy with its own expired fill, or a copy frozen right after the recovery | **confirmed** | An expired or dropped copy takes its unit's label in the same pass. Official pricing reads the holds under its own lock, so claimed units are held too |
| **A4 (P1)** A failed GOOD_WAIT recovery still reported the EOD job SUCCEEDED and released learning | **confirmed** | Both failure forms go into `failed`, so the integrity wrapper retries |
| **A6 (P2)** A stop inside the lease left an unstarted paper cycle RUNNING, so it would be retired as failed after a kill | **confirmed** | `JobStore.release_unstarted` returns the reservation (PENDING, attempt restored) |
| **A1 (P0 per GPT)** A crash after the paper fill is saved and before its mark loses that credit | **known residual, not fixed this cycle** | It needs reconciliation against the paper book (decision id plus fill session). With A2's holds, the WAIT then settles on its price path, consistently on both settlers |
| **A5 (P2)** An unlocked external edit between the append and the post-append stat is trusted | **residual** | Only a writer outside the lock in a microsecond window can do this. The next interrupted append then refuses loudly (an intent mismatch). Hand edits need the desk stopped |
| **C9B-01 (P0)** A cleanup that deleted the lock file let a second command lock a new inode and erase an apply's record | **confirmed** | The lock is now the manifest's **directory**, pinned (`O_DIRECTORY\|O_NOFOLLOW`) and flocked. All manifest reads and writes go through that descriptor. A path that stops naming it refuses. No lock file exists |
| **C9B-07 (P2)** A parent swap created the lock file beside the database | **confirmed** | No file is created to lock. A swapped parent fails the no-follow open or the identity check against the target's directory |
| **C9B-02 (P0)** An overlay published after `recorded_at` (during the reads) reached a frozen row | **confirmed** | The hook stamps `recorded_at` AFTER every input is read |
| **C9B-03 (P0)** A split with an ex-date after the cutoff (applied on read by the store) rescaled the kept bars | **confirmed** (absolute prices only; ratios unchanged) | Factors with an ex-date after the cutoff are undone on the kept bars (`ca_undone_after_cutoff`) |
| **C9B-04 (P1)** A store rebuild between two symbol reads ranked two store generations together | **confirmed** | The store revision is read before and after. If it changed, a bhavcopy generation ranks nothing (`STORE_CHANGED_DURING_READ`) |
| **C9B-05 (P1)** Withheld sets still persisted trend ranks | **confirmed** | Withheld sets carry no trend rank either |
| **C9B-06 (P1)** One excluded row voided Spearman for the whole sample and inflated its n | **confirmed** | Only rows that carry a rank are paired |

Mutants: 14 of 14 killed. Two survivors on the first run exposed redundant guards: the A3 hold plus adoption on the next pass,
and the A4 holds-failure path. The tests now separate them (a one-pass assertion, and a credit-only failure).
One mutant is equivalent: dropping `O_NOFOLLOW` from the directory open is still refused, because the identity check
against the target's directory catches it.

## Residual risks

- **GOOD_WAIT (A1):** a crash after the paper fill is saved and before its mark loses that one credit. It needs a
  reconciliation against the paper book, which is not built. Both settlers then price the WAIT consistently.
- **Manifest lock:** a writer outside the tool (an edit, a restore, an older version) is caught only by
  the publication re-check. Commands on manifests in ONE directory are serialized (directory lock).
- **feature002:** rows frozen before `f002-input.v2` carry no record-time availability check and stay as
  they are, never re-derived. kite_historical / yfinance sets are uncertified.
- **Stop:** a job already running when SIGTERM arrives still runs to completion or to the host's SIGKILL
  (C8.7-03, one shutdown deadline, still open).

## Still owner-gated (unchanged)

Detach and remount; the Stage 2 re-run with the final SHA (the F1 manifest must be re-planned with this
tool); Stages 3–6; "repair freeze index"; deploy. Live trading stays HARD-LOCKED.
