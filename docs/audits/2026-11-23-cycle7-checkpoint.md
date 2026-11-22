# Cycle 7 checkpoint: correction binding, verified backup baseline, feature002 provenance (23 Nov 2026)

This journal records facts only and contains no secrets.

**Status:** development only.

- **Live trading:** HARD-LOCKED.
- **Production:** not deployed, not restarted, not repaired (`125880a6`).
- **Data:** no ledger mutated. The only production operation was the owner-approved Stage 1 backup (§4), which reads production and writes a new image.

## 1. Trigger

The relay delivered the owner's development-only review of `011dffb0` (#333 comment 7083413917, delivery `qr-18d3ea2874b1`, acked). It asked for:

- the correction manifest's binding to be made immutable;
- the backup final pass to verify unchanged files;
- feature002's input provenance to be recorded.

**Rule followed:** every hypothesis was reproduced by a failing test first.

## 2. Fixes (`2e082e44`), each reproduced first

**Learning-observation correction.** The plan binds the repair's IMMUTABLE identity: the target entry and the target and new sha256. Apply checks the live target against the binding's own recorded sha. Before any write it refuses:

- a swapped repair manifest;
- a redirected target with identical bytes;
- an edited plan binding;
- a missing scope, `changed_symbols` or binding field.

Two refusals that were missing before, both confirmed by test:

- copy mode can no longer authorize a production path;
- a live rollback needs the owner's words.

Legitimate repair-manifest transitions (`planned`, `swapping`, `applied`) still apply.

**Backup final pass.**

- `--baseline` re-verifies every unchanged backup file against the Stage 1 manifest. Confirmed: a corrupted same-size, same-mtime copy used to pass.
- `--verify-only` runs that check before the maintenance window.
- It refuses identical, nested or same-filesystem paths, which were not refused before.
- Stamp and baseline paths are made absolute. Found by the benchmark: a relative stamp broke `find -newer` after the `cd`.
- Deletions since the bulk pass are mirrored rather than reported as false failures (found by the benchmark).
- No BSD-awk `{n}` intervals.

**feature002.**

- New rows record their input source, snapshot id, per-symbol as-of and cutoff.
- Bars after the session cutoff are dropped.
- Legacy rows stay `unknown` (`research/feature002/provenance.py`).

**Benchmark** (SSD, 175 MB, 2,001 files):

| Mode | Time |
|---|---|
| Delta only | 0.25 s |
| Delta plus baseline | 3.8 s |
| Verify-only | 1.8 s |

On the USB backup the baseline is read-bound, so the runbook runs `--verify-only` before the stop.

**Mutation checks.** Every guard, reverted on its own, fails its test. One survivor (the canonical-target check) was killed by a new edited-plan test.

## 3. Independent GPT review (`011dffb0..2e082e44`, one pass)

**Result.** 8 P1 findings. **All 8 were confirmed**: each sketch fails on `2e082e44` at its claim. All are fixed, with the sketches adopted as tests. C7-08's sketch is adapted, as noted in its row.

| Finding | What it was | Fix |
|---|---|---|
| C7-01 | Newline-bearing names broke delta membership, so a corrupted unchanged file passed; empty or malformed baselines passed | The baseline and coverage verifier now works on exact path bytes (embedded Python). It refuses newline names, malformed lines and an empty baseline, and every regular file in the backup must be covered by the delta or the baseline |
| C7-02 | `--verify-only` skipped entries that had changed since the stamp | Verify-only checks every baseline entry |
| C7-03 | A directory named `source\n` resolved to its sibling `source`, and `--delete` emptied the backup | Newline arguments are refused before any `cd`; same-inode check (`-ef`) |
| C7-04 | A symlinked store target: the rebuild's `os.replace` replaces the link, and the correction plan then refused its own repair | The target is bound as the directory entry the rebuild writes (parent resolved, final name kept) |
| C7-05 | The runbook rollback lacked `--owner-approval` | Runbook and usage fixed; a test runs the documented command |
| C7-06 | Provenance was taken from the cache location (an offline official repair was labelled yfinance; Kite historical repairs were unknown) | `bulk_fetcher._tag_source` at acquisition: `kite_snapshot` (with id), `kite_historical`, `bhavcopy_store`, `yfinance`. No serving default by location |
| C7-07 | Trend features and provenance came from one read, the RS ranking from a second | RS is computed from the same captured, cut frames |
| C7-08 | The live overlay dated its bar with the host calendar, so on a non-IST host feature002's cutoff dropped the session's live bar | `nse_live` dates the bar with `core.market_clock.today_ist()`. The sketch built an already mis-dated frame; the adapted test drives the real producer with a host clock a day ahead of IST |

**Also fixed (the review's ordering limit).** A stale observation set now refuses before the manifest is written as `applying`. The in-transaction check stays as the backstop, pinned by a race test.

**Also fixed (the review's case limit).** The production-path checks compared `/Volumes/` case-sensitively. On this Mac, `/volumes/QuantTermStorage` exists, and `realpath` keeps the spelling given. So a different-case spelling passed:

- the correction tool's copy-mode guard;
- the rebuild tool's copy-mode desk check;
- the audit's live-path refusal.

All three now compare case-insensitively, and each is mutation-checked.

**Not fixed, with reasons:** the plan file's trust model. Apply trusts the caller-supplied plan file. Every authority check is against immutable content or live bytes, and an external signature is out of scope.

**Mutation checks.** Each reverted fix fails its test.

- Three backup-script guards first survived, because a deeper layer also failed the run.
- Specific tests now pin the newline refusal, the malformed-line refusal and verify-only's reported mismatch.

## 3b. GPT verification pass (`2e082e44..7f018681`, the one the protocol allows)

**Status of the C7 fixes:**

| Status | Findings |
|---|---|
| Closed for the reported cases | C7-01, C7-02, C7-05, C7-06 |
| Partly closed | C7-03, C7-04, C7-07, C7-08 |

**New findings.** 9 (1 P0, 8 P1), **all confirmed**: each sketch fails on `7f018681` at its claim.

- The C7V-05 audit sketch was adapted: its fake audit result gains `affected_sessions`, a key `main()` prints.
- The C7V-02 runbook test lands with the runbook correction commit.

There is no further GPT loop.

| Finding | What it was | Fix |
|---|---|---|
| C7V-01 (P0) | The scope digest hashed only id and vector: the same id and vector without a PIT-fill marker was judged differently, yet it would still have been quarantined | The digest covers every immutable input the planner judged: id, symbol, ts, kind, vector and `meta.pit_features`. Outcome columns are excluded |
| C7V-02 | Stage 3 ran the final pass without `--baseline` | The runbook wires the Stage 1 manifest into a pre-window verify-only and the in-window pass (runbook correction commit) |
| C7V-03 | A backup file replaced by a symlink to the source passed verify-only | An entry verifies only as a regular file, inside the root, with no symlinked component |
| C7V-04 | A symlink to `source\n` still had its newline stripped | Physical paths are taken with a sentinel; resolved paths containing a newline are refused |
| C7V-05 | The rebuild checked only the referent of a target entry; the audit's `--out` was unchecked | The rebuild checks the entry `os.replace` writes AND its referent; the audit refuses a live `--out` before running |
| C7V-06 | A public download overwrote a partial official repair in the same backfill | The public rung fetches only unresolved names; earlier (official) results win the merge |
| C7V-07 | A live bar's provenance was dropped | `live_overlay_date` is recorded when the overlaid bar survives the cutoff |
| C7V-08 | A snapshot switch mid-set mixed generations in the RS ranking | A set spanning more than one Kite snapshot is not ranked: RS and the ranks derived from it are withheld (`rank_inputs.withheld = MIXED_KITE_SNAPSHOTS`) |
| C7V-09 | An overlay call spanning IST midnight mis-dated the new session's quote | The IST session is re-checked under the publication lock, and a crossed fetch is discarded |

**Found while fixing.** `_prefetch_yf` parsed a single ticker as flat columns. yfinance can return `(ticker, field)` columns even for one ticker, so it now selects by ticker whenever the columns are grouped, and skips flat multi-ticker frames.

**Mutation checks.** Each reverted fix fails its test. Two first survived, each because a second layer also catches the case:

- the newline stripping;
- the merge order.

Each layer now has its own pin: a test of the resolved-path refusal, and an overlapping-public-result test.

## 4. Stage 1 backup (owner-approved; see the Stage 1 report)

Recorded separately in the Stage 1 report and the corrected runbook (§1).

## 5. Verification

See the #333 report for the regression and CI results of the final head.

## Open (owner decisions)

Stages 3–6 of the runbook, each with its own approval words. Also:

- `apply repair`;
- the 814 unlearned outcomes;
- `apply counterfactual migration`;
- any live unlock, which stays HARD-LOCKED.
