# Cycle 3 checkpoint: point-in-time market context in the learning record (22 Nov 2026)

This journal records facts only and contains no secrets.

- **Live trading:** HARD-LOCKED throughout.
- **Production:** not deployed and not restarted (`125880a6`).
- **Learning data:** no production learning record was modified. Measurements used the read-only SSD copy made in cycle 2 (`?mode=ro`).

## Start state

- **Candidate:** `38f1864e` on `claude/settlement-batch-20261118`. All 5 workflows are green on that exact SHA, and it was reported as #333 comment 7073537262. The head is `12292647` (docs only).
- **Host:** the desk was paused from Sat 21 Nov 09:46 to Sun 22 Nov ~08:49 IST (battery sleep, then a stale image attachment) and recovered; see the cycle-2 journal. The charger was connected at ~09:10 IST.

## 1. The measured gap

The SSD copy of `feature_store.db` holds 20,493 DECISION observations across 1,558 sessions, from 2019-12-20 to 2026-11-20.

| Feature | Stored value |
|---|---|
| `accum_score`, `base_days`, `base_depth_pct`, `breadth_pct_above_50dma`, `contraction_ratio`, `crude_usd`, `delivery_pct`, `pct_from_pivot`, `rel_strength`, `rs_percentile`, `sector_rank_percentile`, `sector_strength`, `usdinr`, `vix`, `volatility_percentile` | `None` in every row |
| `index_trend`, `correlation_regime` | `UNKNOWN` in every row |
| `regime` | `UNKNOWN` in 19,761 rows, `RISK_ON` in 732 |
| The 8 own-bar PIT features (`bded0395`) | filled in 1,310 rows (they exist only since 19 Nov) |

**Why `regime` is unusable.** The producers emit labels that `research.feature_schema` does not list:

- `core.regime_engine` emits CHOPPY, COMPRESSION, ...;
- the curriculum classifier emits BULL_TREND, RECOVERY, CHOPPY.

`_cat` maps every unlisted label to UNKNOWN.

## 2. What was built (`claude/pit-context-20261122`)

`product/pit_features.py` fills four features into the frozen LEARNING record (`evidence_intelligence.learning_features`). The design below is the one after the GPT review (§3); the first commit, `3391c665`, used a cutoff-relative 3-day window instead.

**One source session per decision.** Every context feature is read from the latest verified official session whose bhavcopy archive was published by the decision's own `generated_at`. Publication follows the NSE convention of 18:00 IST on the session day (`nse_calendar._PUBLISH_CUTOFF_HM`). The verified archives are `bhavcopy_store._dates_on_disk()`, not the settlement calendar. A decision at 15:31 therefore reads the previous session, however late it is frozen.

| Feature | Definition (on the source session S) | Missing when |
|---|---|---|
| `delivery_pct` | Official delivery % on the stock's bar dated exactly S | There is no bar on S (for example a suspension), the S row is the live overlay, or the value is outside 0–100 |
| `vix` | Official India VIX close dated exactly S | The index store has no S row (stale), or the value is not in (0, 200] |
| `index_trend` | Official Nifty 50 rows up to S, sorted and deduplicated, through `historical_curriculum.classify_session`: BULL_TREND is UP, BEAR is DOWN, anything else is FLAT | There are fewer than 200 distinct sessions, or no S row |
| `breadth_pct_above_50dma` | `scan.breadth`'s definition over the store's stocks. Inputs are bars of verified archives only (the live-overlay row is dropped). Each 50-bar window is averaged independently, and ties are decided within a relative band of 1e-12. A bad print counts as not above, as in `scan.breadth`. | Fewer than 300 names, or S is not a verified archive |

**Provenance.** The frozen record stores `pit_features = {filled, context_session, producer: pit_v2_context, publication_rule, …}`.

**Why `regime` is not filled.**

- `_analogs` weights stored rows by regime.
- `decision_service.decision_board` demotes a BUY on negative analog evidence (`historical_demotion`).
- Filling it would therefore change production decisions. It is recorded under Open (next).

## 3. Independent GPT cross-review (`12292647..3391c665`, one read-only pass)

There were 9 findings: 4 P0, 4 P1 and 1 P2. **All 9 were confirmed**:

- GPT's own test sketches all fail on `3391c665`, each at the assertion that carries its claim. The sketches are in `tests/test_pit_review_20261122.py`, and the adaptations to the fixed design are documented there.
- The one ambiguous failure (QT-05, line 129) was checked: it fails at the second assertion, so the veto exists before the fill and disappears after it.

| Finding | Severity | Fix in `b442c4f3` |
|---|---|---|
| QT-01: a later bhavcopy entered an earlier decision (a 15:31 decision frozen after 18:00 read that evening's delivery and breadth) | P0 | A single published source session for all context |
| QT-02: breadth accepted a live overlay as official. `official_dates()` adds calendar sessions with no archive, and an archive can be on disk before the store installs it | P0 | Verified-archive bars only; the overlay row is dropped before any mean |
| QT-03: a later split flipped a flat name (cumulative-sum drift: 33.666666666666664 against a mean of 33.66666666666424) | P0 | A per-window mean plus a relative tie band |
| QT-04: the same model version could serve a different predictor, since an active challenger with a VIX coefficient moved paper order by +3 | P0 | The model records `feature_producer`. It is scored with that producer, and older models are scored as `pit_v1_technical`. |
| QT-05: a stored fill could remove the live historical veto (the quality weight is 0.55 at 9 problems and 0.60 at 8) | P1 | `_live_projection`: live analogs read each stored row minus `meta.pit_features.filled`, with validation recomputed |
| QT-06: a cache built while the store was installing could freeze missing context | P1 | Breadth is keyed on the archive set plus the store generation, and the newest session is rechecked when it cannot be measured. Only a complete context is cached, keyed by session and generation. |
| QT-07: freshness was measured from the cutoff, not the decision; unsorted or duplicated rows; delivery depended on the 21-bar guard | P1 | Exact-session index rows; sorted and deduplicated frames; delivery read independently |
| QT-08: breadth universe and denominator | P1 | The denominator now matches `scan.breadth`. **Name-only fund classification is open:** the store keeps no instrument name or ISIN, so an EQ-series fund whose symbol has no ETF/BEES/LIQUID marker cannot be excluded point-in-time. Recorded as a strict xfail. |
| QT-09: a cold breadth pass inside a WHY request | P2 | **Accepted.** It is one vectorised pass per store generation, cached; a WHY request already re-enriches the whole workspace. Recorded as a strict xfail. |

**Disagreements, with reasons.**

- **VIX = 0 is rejected.** The schema treats it as an OUTLIER, but India VIX is strictly positive by construction, so a 0 is a data fault.
- **QT-09** is accepted as stated above.

## 4. Verification

- **Live decisions on real data are unchanged.**
  - The differential ran `evidence_read`, which feeds the BUY→WATCH historical veto, for 300 real stored decisions. They were posed as one board at 2026-11-21 19:00 IST against the SSD copy of the 20,493-row corpus.
  - The result is **byte-identical** (SHA-256 `28226d15…`) between the pre-slice code `12292647` and `b442c4f3`: 3,000 analogs, and 206 of 300 BUY queries vetoed.
  - The projection also removes the deployed `bded0395` technical fills from the live view. That restores the documented contract with no measurable change, because those rows sit at the quality floor.
- **Unit invariance (`tests/test_pit_context_invariance.py`).**
  - Evidence is identical with and without stored context, mutation-checked: a query that carries VIX does see it.
  - `enrich()` returns the same decision.
  - An un-promoted challenger cannot move paper order.
  - Only the learning-record builder, the challenger (which reads the producer version) and the offline coverage tool import the module.
- **Point-in-time (`tests/test_pit_features.py`, 42 tests).** The tests cover:
  - the publication-rule table (15:31, 17:59, 18:00, Monday, a missing Monday archive);
  - exact-session index rows (stale, unsliced, duplicated or unsorted input);
  - verified-only breadth bars (a mid-window missing archive with a spiked row);
  - flat ties after a 1:3 adjustment;
  - equality with `scan.breadth` on a random market with bad prints and flat names;
  - the store-generation cache;
  - the producer pins.
- **Local regression coverage (Mac guard, one interpreter per file).**
  - On `3391c665`, all 76 files that reach the learning-freeze path were clean.
  - On `b442c4f3`, all 84 files that touch any changed module or its consumers were clean: thesis, ranking, win probability, decision service, paper autopilot, replay and the feature store.
- **CI on `3391c665`:** 5/5 green.
- **CI on `b442c4f3`:** 4/5 green. The CI workflow's hermetic gate failed `tests/test_runtime_root_isolation.py`, because `research/pit_context_coverage.py` built `logs/...` paths itself. That is the same class as `38f1864e`.
  - Fixed: the tool now resolves the copy through `core.runtime_paths` and restores `QT_RUNTIME_ROOT` afterwards. It also passes stable source identities, since the breadth cache is keyed on them.
  - **Lesson.** A new module must also be swept with `test_runtime_root_isolation.py`, not only with the files matching its keywords.

## 5. Re-measure on the SSD copy (`research/pit_context_coverage.py`, read-only)

The producer was run over all 20,493 stored DECISION observations. The store copy holds 3,469 symbols over 1,761 sessions (2019-08-23 to 2026-11-19), with corporate-action events for 424 symbols. The run took 142 s with a peak RSS of 385 MB.

| Feature | Would fill | Values |
|---|---|---|
| `delivery_pct` | 20,475 (99.9%) | median 45.3%, p10 25.0, p90 62.4, range 2.45–100 |
| `breadth_pct_above_50dma` | 20,493 (100%) | median 43.0%, p10 35.6, p90 75.1, range 1.2–95.3 (the 2020 crash low is visible) |
| `vix` | 14,204 (69.3%) | median 17.4, range 9.2–26.7 |
| `index_trend` | 12,558 (61.3%) | 17 sessions: 9 DOWN, 8 FLAT |

- **Why VIX and trend coverage is lower.** The copy's index store holds only about a year of rows, and the trend needs 200 of them. Replay decisions older than that depth stay missing, which is honest.
- **Coherence check.** Breadth on DOWN-trend sessions has a median of 21.5%, against 40.6% on FLAT sessions.
- **Source-session lag** (cutoff minus source session, in days): 0 days for 7,663 decisions, 1 day for 9,752, 2–3 days for 2,865 and 4–5 days for 213, as weekends and the publication rule imply.
- **Limit.** The copy has no daily archives, so the verified-archive gate used the pickle's own session list.

## 6. Findings that change the next steps

- **Regime weighting in the live analogs is inert today.**
  - All 5,949 settled analog rows store regime UNKNOWN, with `meta.market_state` UNKNOWN.
  - Every analog therefore gets the same weight: 1.0 for an UNKNOWN query and 0.70 otherwise. A uniform weight cancels out of the weighted mean, the effective N and the veto.
  - The only visible effect is that `same_regime: True` is shown when both sides are UNKNOWN.
  - Fixing the vocabulary would ACTIVATE regime weighting and change BUY demotions. That is an owner decision.
- **The copy's 5,949 settled replay labels carry no `eval_spec`.**
  - They predate the one-rulebook stamping. `challenger_learning._comparable_label` correctly refuses them (GPT A-4), so the challenger has no comparable label on this copy, which is why it stays OBSERVING.
  - A pre-registered test of context against outcome on these labels would measure an unknown mix of rulebooks, so none was run.
  - A clean test needs stamped labels (written by the dev build after a deploy) or the production walk-forward. The cycle-2 loss attribution already found no positive breadth slice there.

## Open (next)

**Owner decisions:**

- `deploy <sha>`;
- `apply repair`;
- whether to replay the 814 settled outcomes that never reached learning;
- how to resolve any carried CA conflict;
- C-5;
- the 52WH prospective test;
- a full-history run.

New this cycle:

- the regime vocabulary, which would activate the inert regime weighting of the analog veto (§6);
- a point-in-time instrument master (name or ISIN) for fund exclusion from breadth (QT-08).

**Engineering, authorised and next:**

- `rs_percentile`, by reusing `rs_cs_v1` per verified session. Fidelity note: `build_rs_table` ranks names whose last bar precedes the as-of date.
- volatility and sector percentiles, once a definition is pre-declared.
- carrying setup geometry (pivot, base, contraction, accumulation) at the scan → card boundary.
- an index-store depth sufficient for replay-era VIX and trend.
- the displayed `same_regime` flag should not read True when both regimes are UNKNOWN. This is display only; the weights stay unchanged.
