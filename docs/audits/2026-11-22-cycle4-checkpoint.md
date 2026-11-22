# Cycle 4 checkpoint: market-wide RS percentile, plus a store-integrity fix (22 Nov 2026)

This journal records facts only and contains no secrets.

- **Live trading:** HARD-LOCKED throughout.
- **Production:** not deployed and not restarted (`125880a6`).
- **Learning data:** no production learning record was modified.

## Start state

- **Candidate:** `8a9da347` on `claude/settlement-batch-20261118` (cycle 3).
  - All 5 workflows are green on that exact SHA, both on its development branch and on the candidate branch after the fast-forward.
  - It was reported as #333 comment 7074388885.
- **Cycle 3 publication.** #333 comment 7074388885 records that report; the relay verified the head and its CI before posting.

## 1. What was built (`claude/pit-rs-20261122`)

`rs_percentile` was `None` in all 20,493 stored DECISION observations. It is now filled into the LEARNING record only (producer `pit_v3_rs`), on the same published source session as the cycle-3 market context.

**Definition (declared before measurement).**

- **Score:** `research.sepa`'s `rs_cs_v1` composite, 0.40·r63 + 0.20·r126 + 0.20·r189 + 0.20·r252, with horizons counted in the name's own verified-archive bars.
- **Rank:** 100 × (names scoring strictly below) / N.
- **Universe:** the breadth universe, restricted to names with a verified bar on that session, and N must be at least 300.

**Declared differences from existing code.**

| Existing code | `rs_percentile` |
|---|---|
| `research.feature002` ranks within a scan's cards | Ranks market-wide |
| `build_rs_table` ranks a non-finite composite at 0 | Treats it as missing |
| — | Composites within 1e-12 are ties (RS-01) |

**Exactness.** The result equals `research.sepa.rs.build_rs_table` exactly when the universes coincide. This was tested on two sessions and on a market with an unverified archive mid-history.

## 2. Own review before GPT

**Finding.** `_cached_pass` re-filtered the ~3,500-symbol universe and re-sorted the archive on every call before checking its cache. That cost 39 ms per decision, or about 20 s for a 500-card board.

**Fix (`df6956e2`).** The cache is now checked on cheap source identities. The cost fell to 0.73 ms per decision.

## 3. Independent GPT cross-review (`8a9da347..9e597998`, one read-only pass)

GPT reported 5 findings: 4 P0 and 1 P2. **All 5 were confirmed**: GPT's own sketches fail on `9e597998` at their claim-bearing assertions, with the predicted values. They are now `tests/test_pit_rs_review_20261122.py`.

| Finding | Fix in `43cba9bb` |
|---|---|
| RS-01 (P0): a future corporate action split historical ties (0.0 became 99.666667) | Composites within 1e-12 are ties |
| RS-02 (P0): a later overlay made an earlier intraday quote look official (99.667774). **Store integrity:** `_incremental_append` updated only the symbols an archive lists, so an intraday overlay row of a symbol absent from that day's archive was kept and persisted to the store pickle as history, where every reader of history sees it. | An archive is the authority for its own day: appending it drops that date's row from every symbol it does not list, and logs `bhav_store_overlay_rows_dropped` |
| RS-03 (P0): a same-size store rebuild left the cache in place, so a corrected name ranked against stale peers (100.0 instead of 99.666667); a pass built across a change was still returned; a supplied frame could be ranked although absent from the pass | `bhavcopy_store.store_revision()` is a monotonic revision, bumped by every builder and never by the overlay. It joins the map identity in the generation. A pass or window built across a change is returned to nobody. A name's composite is read, by symbol id, from the window it is ranked in. |
| RS-04 (P0): a v3 refit could inherit v2's "exact-version" forward evidence | The model version hashes the producer, the ordered feature list and the training ids |
| RS-05 (P2): a breadth-only (v2) request built and retained the whole RS history (about 23 MiB retained, about 161 MiB transient) | The breadth pass computes no RS. RS is computed lazily per 64-session window, and at most 4 windows are kept. |

**Measured at production size** (3,469 names × 1,761 sessions, synthetic, nice'd):

| Measure | Result |
|---|---|
| Breadth pass | 5.2 s |
| First RS window | 6.4 s, with a traced peak of 2.5 MB |
| Retained memory | 0.04 MB for the live session; about 1.5 MB for a 64-session replay window |
| Per-decision RS | 0.15 ms |

**Residual.** A store pickle written before RS-02 can already contain such overlay rows, and they stay until the store is fully rebuilt from its archives. No production rebuild was run; that is a host operation.

## 4. Verification

- **The PIT and review suites pass:**
  - `test_pit_rs_review_20261122.py` (7), `test_pit_rs.py` (8), `test_pit_features.py` (42) and `test_pit_context_invariance.py` (6);
  - `test_pit_review_20261122.py` (9, plus 2 strict xfails);
  - `test_runtime_root_isolation.py` (21).
- **The affected-file sweeps (Mac guard, one interpreter per file).**
  - On `9e597998`: all 85 affected files pass.
  - On `43cba9bb`: **all 144 files pass**. The set widens because the store change reaches every reader of the bhavcopy store, the live overlay, the bulk fetcher and the learning-freeze path; it includes `test_money_paths.py` and the runtime-root isolation gate.
- **CI on `43cba9bb` (the exact SHA, on `claude/pit-rs-20261122`): all 5 workflows green.**
  - CI 44987647472;
  - Product Acceptance 44987647482;
  - Terminal UI 44987647453;
  - Host Migration Gate 44987647506;
  - Dependency Vulnerability Advisory 44987647551.
- **Integration.** `claude/settlement-batch-20261118` is fast-forwarded from `8a9da347` after an ancestor check, with no force. Any commit after `43cba9bb` is docs only.

## Open (next)

**Owner decisions:**

- `deploy <sha>`;
- `apply repair`;
- whether to replay the 814 unlearned outcomes;
- how to resolve any carried CA conflict;
- C-5;
- the 52WH prospective test;
- a full-history run;
- the regime vocabulary;
- an instrument master (QT-08).

New this cycle:

- **Whether to fully rebuild the production store pickle from its archives after a deploy.** This would clear any overlay rows persisted before RS-02. The rebuild is read-only on the archives and needs a restart window.

**Engineering, authorised and next:**

- a volatility percentile (a declared cross-sectional definition on the same windowed machinery);
- sector strength and percentiles from the NSE industry map (point-in-time membership caveat);
- setup geometry carried at the scan → card boundary;
- the `same_regime` display flag (display only).
