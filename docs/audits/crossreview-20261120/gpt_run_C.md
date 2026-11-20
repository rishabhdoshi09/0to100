<!-- Raw output of an independent GPT (OpenAI Codex CLI, read-only sandbox) review of 9636b030, run C, 2026-11-20. Peer input, NOT verified and NOT an approval. Verification status per finding: ../2026-11-20-gpt-crossreview.md -->

## Findings

Audited `9636b03055961e5f53b5dfb7cfe5cfede03ba13f` against production commit `125880a6`, with priority on `dd4e9c34..9636b030`. The claimed **5% annual probability of any false BUY admission does not follow from the code**. The walk-forward can generate invalid p-values, select configurations using uncharged results, and admit setups whose exact hypothesis was never tested. Several downstream paper and display paths also bypass the scanner’s current evidence verdict.

### C-1 · P0 · CONFIRMED — One common regime shock can pass Holm as 40 independent observations

**Location** `research/overlap_inference.py:136–218` `overlap_robust_test`, `scan/signal_backtest.py:299–314,389–418` `_evidence_fields`, `apply_family_correction`.

**Counterexample** Take 40 filled trades on session ordinals `30*i`. Draw one fair sign `S` for the entire history and give each trade `S * (1 + 0.01*(-1)**i)` R. Each trade has true mean zero. The code counts 40 occupied bins and finds no lag-10 or lag-20 pairs. Its standard error is about `0.00160R`. When `S=+1`, the reported mean is `+1R`, the t statistic is about 624, and the result clears the lower-bound and Holm gates. This null is admitted **50%** of the time in a single look. Holm can handle dependence *between valid p-values*. The input p-value here is invalid.

Occupied bins also do not certify mutually non-overlapping windows. Decision ordinals `0, 19, 20, 39, 40, …, 179, 180, 199` occupy 20 bins, but at most 11 of their 10-session windows can be chosen without overlap.

**Deterministic test** Build `clusters = {f"d{i}": [s * (1 + .01 * (-1)**i), 1] for i in range(40)}` and `index = {f"d{i}": 30*i for i in range(40)}`. For `s` in `(-1, 1)`, run `_evidence_fields(clusters under the "clusters" key, index, 10)`, add `trades=40`, and call `apply_family_correction` on the single row. Assert that the exact null rejection rate over the two equiprobable histories is at most `REPORT_ALPHA`. It is `0.5`.

**Suggested fix** Specify the dependence assumptions required for admission and use a test calibrated at the actual Holm thresholds under those assumptions. Count genuinely separated windows for the minimum-data gate and withhold certification when residual dependence cannot be bounded.

### C-2 · P0 · CONFIRMED — A reservation authenticates metadata while the gate trusts mutable results

**Location** `research/alpha_ledger.py:71–84,167–173,264–279` `binding_sha256`, `reserve`, `_verify`, `scan/calibration_snapshot.py:147–180,286–290` `build_snapshot`, `load_current`, `scan/unified_scanner.py:212–228,358–368` `_report_rows`, `evidence_provenance_note`, `product/scan_store.py:419–434,491–528` `_evidence_authority`, `bind_evidence`.

**Counterexample** Keep a genuine reservation and its provenance. Change a failed combination’s `family_significant` to `True` and give it positive expectancy and lower-bound fields. The ledger check still succeeds. Neither verification nor snapshot construction recomputes the declared membership hash from the actual rows or authenticates their results. A new snapshot built from the changed report gives the edited combination a BUY. The same reservation also accepts substituted family rows if the declared metadata stays unchanged.

At the scan boundary, the authoritative branch reads the scan’s evidence ID for diagnostics but never compares it with the row ID or active snapshot ID. A legacy or mixed BUY with a missing or foreign ID survives if the current signal lookup passes. The authority cache also keys only on `current.json`, so replacing or deleting its target without changing the pointer can leave the cached snapshot in use.

**Deterministic test** Use `tests.evidence_fixtures.proven_provenance()`. Build a snapshot with `combos["MOMENTUM|VOL_SQUEEZE"]` containing `trades=90`, `independent_sessions=24`, `expectancy_r_precise=.3`, `lower_r=.1`, `inference="OVERLAP_ROBUST"`, and `family_significant=False`. Rebuild after changing only that flag to `True`. Pass a BUY with those signals and `edge_evidence_id="foreign"` to `bind_evidence(..., authority=edited_snapshot)`. Assert `WATCH`. It remains `BUY`. Run the result-edit and foreign-ID checks independently as well.

**Suggested fix** Finalize each reserved look once with a digest of the canonical family and the exact evidence rows used for admission. Verify that digest and the scan’s binding before trusting verdicts. Include the snapshot target in cache invalidation.

### C-3 · P0 · CONFIRMED — The annual budget is per SQLite file and per reservation year

**Location** `research/alpha_ledger.py:87–112,145–180,245–279` `ledger_path`, `_connect`, `reserve`, `_verify`, `core/runtime_paths.py:83–106` `runtime_root`, `scan/unified_scanner.py:334–365,389–398` `evidence_provenance_note`, `evidence_age_note`.

**Counterexample** Exhaust the default ledger, then start the report producer and verifier with `QT_ALPHA_LEDGER` pointing at a new file. The new report gets look 1 and verifies under the same program. Losing the database and running the producer has the same effect. A read of an existing attestation against a missing database fails closed, but the next reservation silently initializes a replacement budget. The override used by the test suite at `tests/conftest.py:245` has no test-only enforcement.

Separately, a report reserved at 23:59 IST on December 31 can first admit a setup on January 1. Its evidence remains valid after the new year’s 52 looks become available. A clock jump into a new year also opens a new bucket because reservation uses the supplied wall clock and verification imposes no year consistency check. Returning to a previously spent year in the same database preserves that year’s count.

**Deterministic test** Parameterize two cases. In the first, exhaust a temporary database, set the override to a second nonexistent file, reserve again, and assert the production reservation is refused. In the second, reserve with bound provenance at `2026-12-31T18:29:00+00:00`, then call `evidence_provenance_note` at `2026-12-31T18:31:00+00:00`. Assert it refuses to certify a BUY in the new admission year. The first case returns positive alpha and the second returns an empty error string.

**Suggested fix** Give production one durable ledger identity and distinguish initial provisioning from loss or rollback. Define the guarantee by reservation year, or prevent prior-year reports from authorizing new-year admissions. Use one trusted timestamp for provenance and reservation.

### C-4 · P0 · CONFIRMED — Discarded runs expose uncharged tests that can select the next charged run

**Location** `scan/signal_backtest.py:299–314,362–370,389–408,761–786,949–1005,1054–1069` `_evidence_fields`, `gate_program`, `apply_family_correction`, `run_backtest`, `_run_backtest_inner`. The discard change is in `ae55b791`.

**Counterexample** Run a 99-symbol preview at several `sample_step` and `lookback_sessions` settings. Every preview exposes its raw one-sided p-values and lower bounds before the code reaches the save decision. None reserves a look. Select the best preview and repeat it with a 100th symbol that has no usable frame. The save threshold counts requested symbols. The filled trades and winning p-value can stay identical, but only the selected report spends a look. `gate_program()` records `"even_stride"` for all the tested step sizes and lookbacks.

The family also shrinks after outcomes are observed. Signals enter the report only after a fill, and exact combinations with fewer than 30 fills are omitted. With a two-row family, `p=.0006` fails the first Holm threshold of about `.000481`. With the second row omitted, it passes `.000962`. The code does not establish that this filtering or preview-driven selection preserves the null distribution of the reported p-values.

**Deterministic test** In `tests/test_alpha_ledger.py`, use `_stub_walk_forward` and `_looks`. Run the default thin fixture with `sample_step=1, lookback_sessions=100`. Assert that the run either spent a look before `_evidence_fields` ran or returned no `p_one_sided_raw`, `p_one_sided` or `lower_r` for the signal. Today it returns statistics with `_looks() == []`. A second test can set `MIN_SYMBOLS_TO_SAVE=3`, make the last of three fixture symbols return `None`, and compare the same signal statistics at `max_symbols=2` and `3`.

**Suggested fix** Freeze the actual test configuration and family before evaluating outcomes. Keep unavailable members in the family at `p=1` where applicable. Use fresh holdout data after diagnostic selection, or charge the diagnostic looks that can influence admission.

### C-5 · P0 · CONFIRMED — A losing exact setup can inherit evidence from its component signals

**Location** `scan/signal_backtest.py:908–923,974–979` `_run_backtest_inner`, `scan/unified_scanner.py:245–269,429–442` `setup_evidence`, `_evidence_verdict`.

**Counterexample** `MOMENTUM` and `VOL_SQUEEZE` each pass their marginal tests because each does well with other signals. Their exact intersection has 29 fills and loses on all 29. The report drops the intersection. A score-80 row with those two signals then receives the average of their positive marginal bounds and becomes BUY. Neither marginal hypothesis establishes positive expectancy for this intersection. A signal with no evidence row can also disappear from the fallback calculation entirely.

**Deterministic test** Give both singles `{"mean": .30, "lower": .10, "n": 90, "blocks": 24, "inference": "OVERLAP_ROBUST", "family": True}`. Pass them to `_evidence_verdict(80, ["MOMENTUM", "VOL_SQUEEZE"], signal_edges=edges, combo_edges={}, gate=True)` and assert the verdict is `WATCH`. It is `BUY`.

**Suggested fix** Test the exact setup or a predeclared pooled policy matching the setup-selection rule. Withhold the qualified label when that hypothesis lacks sufficient evidence.

### C-6 · P0 · CONFIRMED — A report can certify trades measured over unchecked session windows

**Location** `scan/signal_backtest.py:602–649,805–825,837–845,944–945,1036–1049` `window_is_exact`, `realised_session_index`, `_run_backtest_inner`, `scan/unified_scanner.py:323–368` `evidence_provenance_note`.

**Counterexample** If `_dates_on_disk()` returns no dates or raises while symbol frames are available, the exact-window check becomes a no-op. A suspended symbol’s next 10 *rows* can span more than 10 official sessions. The later inference index comes from the union of symbol-frame dates, so those trades can still receive `OVERLAP_ROBUST`, enough blocks, and a positive Holm result. The report explicitly records `exact_horizon_check=False`. The admission verifier accepts that provenance. It also checks report creation age without testing the age of the underlying data span.

**Deterministic test** Use `tests.evidence_fixtures.proven_provenance(exact_horizon_check=False)` and assert `evidence_provenance_note(...)` rejects it. For a producer regression, use `_stub_walk_forward`, make `_dates_on_disk` return `[]`, and drop an interior bar from one symbol while another retains it. Assert no certifiable report is produced.

**Suggested fix** Require verified official session coverage before evaluating admission windows. Fail certification on absent coverage, gaps, or an unacceptable training-data cutoff. Bind those checks into the finalized report.

### C-7 · P0 · CONFIRMED — Historical replay uses future calibration even with `evidence_gate=False`

**Location** `product/historical_replay.py:302–322,428–436,900–912,1064–1080` `scan_session`, `decide_session`, `run_historical_replay`, `scan/unified_scanner.py:757–782,1144–1164` `UnifiedScanner.__init__`, `_analyze`.

**Counterexample** Replaying a past session truncates its price frame, but the default runner freezes **today’s** calibration and passes it to every past session. The constructor loads the snapshot’s multipliers with the evidence gate disabled. Those multipliers change the historical composite score before the score threshold and ranking run. A report generated using outcomes after the replay date can therefore change the reconstructed past decision.

**Deterministic test** Use the verified `_frame()` helper from `tests/test_buy_admission_end_to_end.py`. Call `scan_session` twice for the frame’s final day, with the same `ohlcv_fn` and two snapshots dated after that day. Give every `SIGNAL_META` key multiplier `1.0` in one and `.45` in the other. Assert the past row’s score is unchanged, or that both snapshots are refused for that date. The scanner uses both multipliers.

**Suggested fix** Resolve calibration as of each replay decision. Use a fixed, predeclared uncalibrated rule when no historical calibration is available.

### C-8 · P0 · CONFIRMED — A scan-store downgrade does not revoke derived BUYs or paper eligibility

**Location** `product/decision_discovery_store.py:89–127`, `product/decision_simulation_gate.py:126–160`, `product/recommendation_truth.py:162–199`, `product/home_os.py:215–252,346–396`, `product/recommendations_liveness.py:418–471`, `terminal_product_api.py:752–781`, `research/autonomy/jobs.py:1146–1158`, `product/paper_autopilot.py:533–578,1034–1051,1128–1151`, `research/autonomy/telegram_notifications.py:382–440,734–746,946–966`, `scan/market_scan_service.py:664–687`.

**Counterexample** Start with a current scan, a valid TCS BUY, matching persisted recommendation and discovery sidecars, and a same-scan committee BUY. Remove the alpha ledger or switch the calibration to one that rejects TCS. `load_scan()` now returns TCS as `WATCH` with the same `scanned_at`.

- **Home and discovery** reuse the old board and check committee lineage, not per-symbol current evidence. `build_buy_only` uses the downgraded scan index for annotations and still emits TCS as eligible.
- **React APIs** can serve the old Opportunities workspace and its admission state because its timestamps still match. `/api/decisions` and `/api/decisions/{symbol}/why` derive state from saved card tiers or committee fields. `decision_ranking.rank` preserves the incoming decision. React’s recommendation badge alone renders `CANDIDATE`, but that safeguard does not clear the stale API admission state or the decision-board BUY.
- **Paper** compares the scan and recommendation timestamps, then loads the saved card. If its other gates pass, the card still reaches `ENTER_NOW` and the book as a production entry. The evaluator requires an eligible tier and entry state, with no current scanner certificate check.
- **Telegram and learning** have two paths around revocation. The desk alert checks the recommendation timestamp and projects the old committee decision. Immediate scan alerts and `track_published_buys` consume the original in-memory payload. If the calibration changes after scanner construction and before publication, they can announce a qualified BUY and write `UNIFIED_BUY` for a row that `load_scan()` already withholds.

**Deterministic test** Extend `tests/test_no_buy_acceptance.py::test_a_stale_artifact_is_withheld_at_read_time_even_if_it_says_buy`. Preserve the scan timestamp and seed matching saved `ENTER_NOW` and committee BUY rows for the same symbol. After the real `load_scan` returns `WATCH`, assert `build_buy_only(..., verified=True)["rows"] == []`. It still includes the symbol. Use the same state at `build_fast_response` and `run_reco_paper_cycle`, with unrelated gates satisfied, and assert no eligible BUY and no production fill. For the publication branch, pass the original payload to `track_published_buys` with a spy on `log_signal` and assert it never receives `UNIFIED_BUY` after revocation.

**Suggested fix** Require each qualified display, notification, learning write and production fill to consume the same currently verified per-row certificate. Include the evidence revision and expiration in derived-artifact and cache validity.

### C-9 · P0 · CONFIRMED — A Telegram paper tap bypasses the account limits and can force a zero-sized trade

**Location** `alerts/telegram_actions.py:41–61,111–125` `build_setup_keyboard`, `_do_paper_trade`, `_handle_callback`, `risk/position_sizer.py:21–48` `size_position`, `execution/trade_executor.py:111–123,150–168` `_validate`, `place_trade`.

**Counterexample** At ₹100,000 capital, a button for entry ₹15,000, stop ₹13,500, target ₹17,500 sizes to zero. The handler forces one share, risking 1.5% and deploying 15%. The executor validates level geometry and immediately commits `PAPER_OPEN`. A second tap adds another open row for the same symbol. The path never checks whether the modern book already holds it, whether five positions are open, or whether the new trade breaches 5% total risk. Buttons carry rounded levels without expiry or refreshed prices. Direct legacy paper tickets reach the same permissive journal branch.

**Deterministic test** In an isolated runtime, patch `execution.trade_executor._DB` to `tmp_path / "trades.db"` and settings to ₹100,000, 1%, and 10%. Call `_do_paper_trade("TCS", 15000, 13500, 17500)` and assert `_open_trades()` is empty. Today it contains one row. Parameterize with a normal-sized trade and a pre-existing TCS in the temporary modern book to test cross-engine duplication.

**Suggested fix** Reject zero quantity. Resolve and validate the current setup on tap, and enforce sizing, duplication, open risk and position count under the shared account lock before journaling.

### C-10 · P0 · CONFIRMED — The legacy sniper buys WATCH setups and the legacy sizer exceeds hard caps

**Location** `legacy_app.py:141–162`, `scan/auto_scan.py:908–915`, `scan/breakout_sniper.py:145–194,200–239`, `execution/autopilot.py:75–98,617–680,760–840,941–952,978–1026,1060–1072`.

**Counterexample** With legacy daemons enabled, the paper autopilot armed and the ordinary tape gates passing, a `WATCH` / `edge_gated` pre-breakout enters the sniper watch map. The map discards evidence fields. A confirmed tick crossing calls `on_breakout`, which supplies score 70 and `edge=None`. `_passes_gates` rejects negative numeric edge and accepts missing edge. The engine can open the position and send a `BUY … (PAPER)` notification as an ordinary legacy entry, without the exploration limits.

This path has independent sizing and quote defects. A quote with yesterday’s `last_trade_time` and an in-range positive price passes `_anchor_live`. At a ₹100,000 pool, `entry=100`, `stop=90`, and conviction multiplier `1.5`, `_size` returns 150 shares with available cash. That is 1.5% trade risk and 15% name exposure. The final account gate checks aggregate risk and count, so it permits this order into an empty book.

**Deterministic test** Feed `build_watch_map` a gated WATCH pre-breakout with a stubbed instrument-token map, then run `process_ticks` through arm and confirmation and pass the hit to real `on_breakout`. Isolate the paper journal and satisfy the unrelated market gates. Assert no ordinary paper entry. Separately stub `get_live_quotes` to return yesterday’s `last_trade_time` and assert `_anchor_live` refuses it. With a stub pool of ₹100,000 and the default cap settings, assert `_size(100, 90, 1.5) <= 100`.

**Suggested fix** Route legacy entry through the same certificate and risk authority as the main engine. Make unproven sniper hits research alerts or send them through the bounded exploration lane. Validate exchange quote age before sizing.

### C-11 · P0 · CONFIRMED — The installed paper-cycle wrapper releases the account lock before persisting exposure

**Location** `research/autonomy/__init__.py:99–104,114–125` `install_runtime_handlers`, `run_supervisor`, `research/autonomy/paper_cycle_truth.py:141–168,220–222` `install_paper_cycle_truth`, `product/paper_autopilot.py:1326–1364` `run_reco_paper_cycle`, `product/paper_status.py:33–67` `modern_engine_open_positions`.

**Counterexample** The base `Deps.run_paper_cycle` passes `brain._save_intel_book` as the durability callback. The supervisor installs a replacement method that omits it. The modern engine opens TCS in memory and releases `account_mutation_lock`. A legacy paper entry can then acquire the same lock, read the old `intel_book.json`, and commit TCS to `trades.db`. The wrapper subsequently saves the modern book. Both books now hold TCS even though both automatic entry paths acquired the lock. The wrapper also suppresses a failure of its post-cycle save.

**Deterministic test** Adapt `tests/test_money_paths.py::TestAccountOpenRiskSpansEveryPaperBook.test_modern_engine_open_is_durable_before_account_lock_release` to invoke the installed `Deps.run_paper_cycle` with an isolated fake brain and the real recommendation cycle. Retain its nonblocking lock probe in `_save_intel_book`. Assert the first save after the position opens observes the lock held. It observes the lock free.

**Suggested fix** Pass the durability callback through the installed wrapper and require a successful persisted mutation before releasing the lock or reporting a fill.

### C-12 · P0 · CONFIRMED — Exploration’s one-per-day quota survives neither a failed write nor the commit gap

**Location** `product/paper_autopilot.py:87–109,123–169,1159–1174,1326–1364,1387–1395` `_exploration_taken`, `_record_exploration`, `_exploration_pick`, `run_reco_paper_cycle`.

**Counterexample** A cycle takes exploration trade A. The quota write fails and is silently ignored. The next cycle on the same day still sees zero used slots and can take B, which passes the duplicate check because it is a different symbol. A restart between book commit and quota recording creates the same state. The quota check also runs before the account lock, with no final quota recheck under that lock. The normal generated probe requests 0.5% risk and records `EXPLORATION` on the taken cycle row. The cap on the number of probes is unenforced in these cases.

**Deterministic test** Run two real `run_reco_paper_cycle` calls for the same day with two different otherwise eligible, edge-gated cards and an isolated `PaperBook`. Make `core.atomic_io.atomic_write_text` raise only for the temporary exploration quota file. Give the book a successful durability callback. Assert at most one position opens across the two calls. Today both can open. For the restart variant, persist the first book snapshot and restore it in the second cycle while leaving the quota file absent.

**Suggested fix** Persist exploration identity and its session slot atomically with the fill. If the quota cannot be read or committed, block the probe.

### C-13 · P1 · CONFIRMED — The new no-BUY diagnosis misclassifies an authority outage as a negative result

**Location** `product/scan_store.py:505–529` `bind_evidence`, `product/buy_admission.py:62–79` `from_scan_payload`, `product/home_os.py:312–325` `_scan_evidence_index`, `terminal_api.py:518–545` `_scan_payload_from`.

**Counterexample** A current scan was produced with valid evidence and an empty evidence note. Later the calibration pointer is unavailable. `bind_evidence` changes every BUY to WATCH and embeds the outage in each row’s reasons, but does not update the status note consumed by `buy_admission`. The classifier returns `NO_PROVEN_EDGE`, which says the strategy failed to prove edge, although the authority was unavailable. The React scan projection also drops `evidence_identity` and `evidence_binding`.

**Deterministic test** Pass `{"records": [{"verdict": "BUY"}], "evidence_identity": {"note": ""}}` through `bind_evidence(..., authority=None)`. Assert `from_scan_payload(result, scan_current=True)["code"] == EVIDENCE_UNAVAILABLE`. It returns `NO_PROVEN_EDGE`.

**Suggested fix** Carry a structured read-time authority status through the scan projection and give it precedence over row counts and the original scan note.

## Verified correct

- **One intact ledger enforces 52 positive reservations per IST reservation year across program IDs.** The count and insert run under `BEGIN IMMEDIATE`, the year/index uniqueness constraint is global, and a same-ID retry with a different program or declared binding is rejected. The legacy renumbering transaction retains rows and assigns zero alpha beyond 52. The verifier’s own cache checks main-file and journal/WAL identity, including ctime. `research/alpha_ledger.py:54–70,124–142,161–181,199–242`.
- **Holm’s step-down implementation is correct for the rows it receives.** The 53rd reservation gives zero alpha, which cannot reject any row. A thin report also cannot directly certify on its own zero-alpha attestation. `scan/signal_backtest.py:389–419,997–1005`, `research/alpha_ledger.py:267–272`.
- **The traced Kite broker mutations remain hard-locked.** The legacy live override can reach the wrapper, but the wrapper checks the canonical lock before SDK order placement. The raw compatibility proxy guards the traced order and GTT mutations. A Telegram callback explicitly selects `paper=True`. `product/live_execution_interlock.py:42–70`, `data/kite_client.py:111–122,294–324`, `alerts/telegram_actions.py:54–61,123–125`.
- **The normal scanner and conviction transformations have real gates.** `UnifiedScanner` defaults to the active evidence gate and checks provenance on construction. `scan.conviction` only upgrades an existing BUY to STRONG BUY and cannot promote WATCH. `product.conviction` ranks and labels conviction. The React scanner workspace reads through `load_scan` via `terminal_api._scan_payload`. The autonomous Telegram live-cross watcher gives a WATCH a research notice without a paper entry plan. `scan/unified_scanner.py:757–824,1156–1164`, `scan/conviction.py:191–212`, `product/conviction.py:119–132`, `product/observer_api.py:121–140,578–583`, `terminal_api.py:509–515`, `research/autonomy/telegram_notifications.py:654–671`.
- **F&O has a distinct paper candidate path.** The pipeline emits `PAPER_OPTION_CANDIDATE`, the runtime opens a simulated option, and Telegram explicitly labels the candidate as unvalidated research. React displays a paper options lab and paper candidates. F&O shares the account’s 5% open-risk calculation and has its own five position slots. `product/fo_options_pipeline.py:153–159`, `product/fo_paper_runtime.py:274–285,558–572,604–642`, `risk/portfolio_risk.py:160–164`, `research/autonomy/telegram_notifications.py:871–885`, `frontend/src/marketViews.tsx:184–201`.

## Not examined / limits

- The ledger only attempts to cover the walk-forward equity family. Outside that accounting are F&O paper, exploration, legacy sniper and manual paper tickets, historical replay and virtual paper, the separate US paper lane, and research programs such as the target and exit sweeps, feature confirmation, challenger experiments, edge diagnosis and the 52-week-high preregistration. The derived or stale BUY paths in C-8 also lack a fresh link to a charged, verified result. Historical `BUY`/`would_take` counts are explicitly replay output, although C-7 compromises their point-in-time validity.
- This was a static review of the requested admission and risk paths. It did not certify actual production configuration, market inputs, ledger contents, runtime trade history, the full settlement rulebook, or independent-unit deduplication across all learning consumers. Legacy-path reachability is conditional on the stated flags and arming state.
- No tests were executed. The pytest sketches are for an isolated test runtime. No files were changed, and no runtime data, credentials, broker, Telegram or network services were accessed.