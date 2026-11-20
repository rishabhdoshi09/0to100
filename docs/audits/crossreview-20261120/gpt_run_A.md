<!-- Raw output of an independent GPT (OpenAI Codex CLI, read-only sandbox) review of 9636b030, run A, 2026-11-20. Peer input, NOT verified and NOT an approval. Verification status per finding: ../2026-11-20-gpt-crossreview.md -->

## Findings

Reviewed `9636b03055961e5f53b5dfb7cfe5cfede03ba13f` against `125880a6`, with extra attention to `dd4e9c34..9636b030`. The canonical paper job can admit an entry after the scan reader has withdrawn its BUY. The settlement pipeline can also attach an old close to a new position and permanently lose learning updates after a retry.

### A-1 · P0 · CONFIRMED · Paper entry can use a withdrawn BUY

**Location** · `product/scan_store.py:515` `bind_evidence`, `research/autonomy/jobs.py:1146` `run_paper_cycle`, `research/autonomy/paper_cycle_truth.py:148` installed `Deps.run_paper_cycle`, `product/paper_autopilot.py:533` `evaluate_candidate` and `:1049` `run_reco_paper_cycle`.

**Counterexample** · Publish a qualified, ready recommendation. Before the paper job executes, its report ages out or the active calibration changes so `load_scan()` now returns that symbol as WATCH. The downgrade preserves `scanned_at`. The job compares only the scan and recommendation timestamps, so it still authorizes entries. The installed executor reloads the saved recommendation and evaluates its original tier and readiness. With the other gates satisfied, the symbol enters through the production lane. Exploration limits never apply.

**Deterministic test**

```python
def test_withdrawn_buy_blocks_paper_entry(monkeypatch):
    from types import SimpleNamespace as NS
    from product import recommendations_store as RS, scan_store as SS
    from research.autonomy import jobs as J

    stamp = "2026-11-20T05:00:00Z"
    card = dict(symbol="INFY", verdict="BUY", status="Ready to trade",
                reco_tier="good_setup", entry_state="ready", action_badge="Buy")
    scan = SS.bind_evidence(
        {"scanned_at": stamp, "records": [dict(card)]}, authority=None)
    assert scan["records"][0]["verdict"] == "WATCH"
    reco = {"scan_scanned_at": stamp, "categories": [{"cards": [card]}]}
    calls = []
    deps = NS(now_ist=lambda: None, holidays=set,
              run_paper_cycle=lambda allowed: calls.append(allowed) or {"status": "OK"})
    monkeypatch.setattr(J, "Deps", type(deps))
    monkeypatch.setattr(J, "_entry_reason", lambda *a: (True, "", "intraday"))
    monkeypatch.setattr(J, "_paper_market_data_source", lambda ctx: (True, "official"))
    monkeypatch.setattr(SS, "load_scan", lambda: scan)
    monkeypatch.setattr(RS, "load_recommendations", lambda: reco)
    J.run_paper_cycle(NS(deps=deps, active_failures=set()))
    assert calls == [False]  # Today [True]
```

**Suggested fix** · Bind each production entry candidate to the currently verified scan row and pass that validated workspace into execution. Recheck admission at the order boundary. Keep exploration explicitly tagged and capped.

### A-2 · P0 · CONFIRMED · An old close can settle a new position in the same symbol

**Location** · `product/forward_soak.py:307` `attach_closed_trades`, `product/forward_evidence.py:250` `_apply_settlement`, `product/_paper_learning_loop_core.py:118` `_lookup_taken` and `:214` `ingest_closed_trade`, `research/auto_research/paper_book.py:308` `_build_closed_trade`.

**Counterexample** · INFY trade A closes at +2R. A later INFY trade B opens and remains open. The next EOD pass iterates all historical closes, finds B as the newest unsettled INFY entry and stamps it with A’s +2R. The write-once settlement then prevents B’s own result from attaching. The learning path also picks the latest taken context by symbol, so delayed ingestion of A can assign it to B’s setup and execution geometry.

**Deterministic test**

```python
def test_old_close_cannot_settle_reentry(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from product import forward_evidence as FE, forward_soak as FS
    from product import paper_learning_loop as PLL

    path = tmp_path / "forward.jsonl"
    monkeypatch.setattr(PLL, "_lookup_taken", lambda symbol: {})
    for did, day in [("a", "2026-09-01"), ("b", "2026-09-03")]:
        FE.freeze_observation({"symbol": "INFY", "decision_id": did},
                              cycle_id=did, as_of=day, entered=True, path=path)
    close = dict(symbol="INFY", decision_id="a", entry_date="2026-09-01",
                 exit_date="2026-09-02", realized_R=2.0, exit_reason="TARGET")
    FS.attach_closed_trades(NS(closed=[close]), path=path)
    rows = {r["decision_id"]: r for r in FE.load_ledger(path)}
    assert rows["a"]["gross_R"] == 2.0 and rows["b"]["later_outcome"] is None
```

**Suggested fix** · Carry the paper intent or position ID into the taken journal and forward ledger and join on it. Leave ambiguous legacy matches unresolved.

### A-3 · P0 · CONFIRMED · Settlement retries permanently drop learning observations

**Location** · `product/forward_soak.py:236` `settle_pending_from_market` and `:242` `_flush`, `product/_paper_learning_loop_core.py:415` `ingest_counterfactual`, `product/learning_policy_store.py:265` `record_measured_outcome`, `research/autonomy/outcome_integrity.py:81` `strict_outcome_resolution`.

**Counterexample** · Five rejections settle in a batch. The ledger commits all five before policy ingest starts. Ingest raises on row three. The job can retry, but all five rows now have `later_outcome`, so the retry ingests zero. The last three observations are permanently missing from policy learning. There is a second loss window after a dedupe claim commits and before its policy update writes.

**Deterministic test** · In the existing `tests/test_forward_settlement_batch.py:116` `test_counterfactual_crash_mid_chunk_never_double_ingests`, keep the injected failure and retry. Replace the final assertion with `assert len(ingested) == len(set(ingested)) == 5`. The current code delivers two observations.

**Suggested fix** · Persist a replayable learning event keyed by the independent unit and make the policy update atomic with its dedupe record. Replay unapplied events during outcome retries.

### A-4 · P0 · CONFIRMED · Legacy labels survive in the learner and policy aggregates

**Location** · `product/evidence_intelligence.py:257` `settle_decision`, `product/challenger_learning.py:140` `_lane_weight`, `:151` `_rows` and `:512` `train`, `product/learning_policy_store.py:269` `record_measured_outcome`, `product/learning_policy_repair.py:125` `plan`, `product/evidence_policy_engine.py:220` `evaluate_policies`.

**Counterexample** · A pre-migration rejection already stored as `FORWARD_COUNTERFACTUAL` under the current thesis hash still enters challenger training at weight `0.65`. The feature outcome carries no evaluation spec, and the migration stages only the two JSONL ledgers. Separately, legacy missed winners and taken closes both update `SETUP::<setup>`. A taken close after the last legacy update changes the aggregate’s source to `paper_forward_taken_*` while retaining the old sample size and mean. Repair then skips the mixed policy based on that last source. Its stale aggregate can still support, penalize or block selection.

**Deterministic test**

```python
def test_legacy_counterfactual_excluded_from_training(monkeypatch):
    from product import challenger_learning as CL, trading_thesis as TT
    from research import feature_store

    legacy = {"meta": {"thesis_hash": "t"}, "outcome": 1.5,
              "outcome_meta": {"evidence_class": "FORWARD_COUNTERFACTUAL",
                               "not_pnl": True, "classification": "MISSED_WINNER"}}
    monkeypatch.setattr(TT, "manifest", lambda: {"thesis_hash": "t"})
    monkeypatch.setattr(feature_store, "load_observations", lambda **kw: [legacy])
    assert CL._rows() == []
```

**Suggested fix** · Version the feature outcomes, rebuild affected models and mixed policies from compatible per-observation records, and invalidate aggregates whose lineage cannot be reconstructed.

### A-5 · P0 · CONFIRMED · Migration can assign one freeze’s outcome to a different freeze

**Location** · `research/counterfactual_reconcile.py:91` `rejudge`, `:110` `_rates` and `:185` `stage`.

**Counterexample** · Two freezes share symbol, day, reason and old label, but have different stops. The dry run rejudges the two geometries separately. Staging indexes the results by `(symbol, as_of, reason_code)`, so the last result overwrites the first and gets written to both rows. The first freeze can turn from a correctly rejudged missed winner into an avoided loser. A second migration skips both because staging has already stamped them with the current spec. Retaining the old payload makes rollback possible from an intact backup. It does not make the staged outcome correct.

**Deterministic test**

```python
def test_stage_preserves_distinct_frozen_outcomes(tmp_path):
    from research import counterfactual_reconcile as RC
    from tests.evidence_fixtures import rulebook_outcome

    base = dict(symbol="INFY", as_of="2026-09-01", reason_code="LOW_QUALITY_SETUP",
                classification="MISSED_WINNER", hypothetical_entry=100,
                hypothetical_target=115, outcome={"forward_return_pct": 9})
    rows = [dict(base, counterfactual_id="a", hypothetical_stop=90),
            dict(base, counterfactual_id="b", hypothetical_stop=98)]
    def resolve(symbol, day, entry, stop, target):
        return rulebook_outcome(1, "WIN") if stop == 90 else rulebook_outcome(-1, "LOSS")
    report = RC.reconcile([], rows, resolve)
    paths = RC.stage(report, [], rows, tmp_path)
    staged = RC._read_jsonl(paths["counterfactuals"])
    assert [r["classification"] for r in staged] == ["MISSED_WINNER", "AVOIDED_LOSER"]
```

**Suggested fix** · Join staged outcomes to immutable freeze IDs, or to exact source-row identities for old rows. Use the independent observation key only when aggregating. Verify rollback across the policy, observation and feature stores as well as the staged ledgers.

### A-6 · P0 · CONFIRMED · Simulated ledger rows can update forward learning policies

**Location** · `product/forward_evidence.py:23` `PROVENANCES`, `:180` `freeze_observation`, `product/forward_soak.py:236` `settle_pending_from_market` and `:264` settlement payload construction, `product/learning_policy_repair.py:76` `independent_units`.

**Counterexample** · A forward-ledger rejection explicitly marked `BACKTEST` is accepted by the EOD settlement loop. The loop strips decision provenance while building the learning payload and updates `REJECT::<reason>` with source `counterfactual_not_pnl`. The first simulated copy can also claim the independent key ahead of a genuine observation. Repair accepts the same non-forward row. Filtering the dashboard ledger by provenance leaves those policy updates in place.

**Deterministic test**

```python
def test_backtest_rejection_cannot_feed_forward_policy(tmp_path, monkeypatch):
    from product import forward_evidence as FE, forward_soak as FS
    from product.learning_policy_store import load_policies
    from tests.evidence_fixtures import rulebook_outcome

    monkeypatch.setenv("QT_FORWARD_LEDGER", str(tmp_path / "forward.jsonl"))
    FE.freeze_observation({"symbol": "INFY", "reason_code": "LOW_QUALITY_SETUP"},
                          cycle_id="sim", as_of="2026-09-01", provenance=FE.BACKTEST)
    policy = tmp_path / "policy.json"
    FS.settle_pending_from_market(outcome_fn=lambda *a: rulebook_outcome(1, "WIN"),
                                  policy_path=policy)
    assert load_policies(policy)["policies"] == []
```

**Suggested fix** · Carry and validate decision provenance before any claim or policy write. Apply the same filter in repair.

### A-7 · P0 · CONFIRMED · A missing bar extends a ten-session trade past its deadline

**Location** · `core/outcome_resolver.py:90` `resolve_trade` and `:131` `counterfactual_outcome`, `core/signal_outcome_tracker.py:169` `_resolve_via_path`.

**Counterexample** · Omit a symbol’s bar for official session two. Give it a target hit on official session eleven. The resolver takes the first ten rows that happen to exist, includes session eleven and emits a WIN stamped with the ten-session rulebook. This affects both rejection outcomes and the surfaced-signal tracker.

**Deterministic test**

```python
def test_missing_symbol_bar_does_not_extend_horizon():
    import pandas as pd
    from core.outcome_resolver import counterfactual_outcome

    days = pd.bdate_range("2026-02-03", periods=11).delete(1)
    frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.},
                         index=days)
    frame.loc[days[-1], "high"] = 111.
    result = counterfactual_outcome("INFY", "2026-02-02", 100, 95, 110, frame=frame)
    assert result is None or result["outcome"] != "WIN"
```

**Suggested fix** · Use the calendar’s exact session window and validate coverage before issuing a final label. `product/challenger_shadow.py:513` already checks the expected dates against available symbol bars.

### A-8 · P0 · CONFIRMED · Evolution still grades equity counterfactuals with the old exit path

**Location** · `product/evolution/grading.py:26` `resolve_forward_outcome` and `:181` `grade_shadow_decision`, `core/outcome_resolver.py:230` `first_touch_detail`, `product/evolution/scorecard.py:57` `graded_rows`, `research/autonomy/jobs.py:1846` scheduled Evolution grading.

**Counterexample** · Entry 100, stop 95, target 110. The next session trades through entry, and the following session opens at 90. The canonical rulebook exits at 90 for −2R gross. Evolution exits at 95 and saves −1R from the legacy percentage calculation. Its scorecard and promotion inputs accept the persisted grade without checking the evaluation spec.

**Deterministic test**

```python
def test_evolution_uses_rulebook_gap_loss(tmp_path, monkeypatch):
    import pandas as pd
    from core import outcome_resolver as OR
    from product.evolution import grading as G

    frame = pd.DataFrame(
        [(100, 101, 99, 100), (90, 93, 89, 92)],
        columns=["open", "high", "low", "close"],
        index=pd.to_datetime(["2026-09-02", "2026-09-03"]))
    monkeypatch.setattr(OR, "_ohlcv", lambda symbol: frame)
    row = dict(shadow_id="s", domain="EQUITY", decision="WAIT", symbol="INFY",
               as_of="2026-09-01", entry=100, stop=95, target=110)
    grade = G.grade_shadow_decision(row, path=tmp_path / "shadow.jsonl")
    expected = OR.counterfactual_outcome("INFY", row["as_of"], 100, 95, 110)
    assert grade["counterfactual_R"] == expected["net_r"]
```

**Suggested fix** · Route equity shadow grades through the canonical resolver and version their outcomes. Regrade or quarantine existing promotion inputs measured by the old rules.

### A-9 · P1 · CONFIRMED · A signal-log write failure still unblocks learning

**Location** · `research/autonomy/jobs.py:1332` `settle_signal_log`, `research/autonomy/outcome_liveness.py:394` `_run_outcome_resolution_light`, `research/autonomy/outcome_integrity.py:78` `strict_outcome_resolution`.

**Counterexample** · `update_outcomes()` returns `written=0` and a `write_error`. The lightweight handler includes the error in metadata, returns `SUCCEEDED` and unblocks the session’s outcome dependency. The installed integrity wrapper checks paper ingest, forward soak and official counterfactual failures. It lets the failed signal write pass.

**Deterministic test**

```python
def test_signal_write_error_blocks_outcome_success(monkeypatch):
    from data import bhavcopy_runtime, bhavcopy_store
    from core import signal_outcome_tracker as S
    from product import autonomous_loop
    from research.autonomy import job_store as JS, outcome_integrity as OI
    from research.autonomy.outcome_liveness import _run_outcome_resolution_light
    from tests.test_outcome_liveness import _eod_ctx

    monkeypatch.setattr(autonomous_loop, "settle_official_outcomes", lambda day: {})
    monkeypatch.setattr(bhavcopy_runtime, "ensure_loaded", lambda **kw: None)
    monkeypatch.setattr(bhavcopy_store, "is_ready", lambda: True)
    monkeypatch.setattr(S, "update_outcomes",
                        lambda **kw: {"written": 0, "write_error": "disk full"})
    monkeypatch.setattr(OI, "_BASE_HANDLER", _run_outcome_resolution_light)
    monkeypatch.setattr(OI, "_unreported_forward_failures", lambda: [])
    assert OI.strict_outcome_resolution(_eod_ctx()).status == JS.RETRYABLE_FAILED
```

**Suggested fix** · Make the installed wrapper retry on a signal settlement write or fatal error before issuing the learning dependency.

### A-10 · P1 · CONFIRMED · The scorecard misses policy effects the ledger records

**Location** · `product/paper_autopilot.py:238` `AutopilotDecision.as_dict`, `product/forward_evidence.py:175` `freeze_observation` and `:193` persisted policy field, `product/learning_scorecard.py:648` `learning_influence`.

**Counterexample** · An active policy blocks a candidate. The autopilot passes `policy_effect="BLOCK"` and the forward ledger saves `policy_result="BLOCK"`. The learning scorecard reads `policy_effect` and defaults the row to NEUTRAL. It reports zero changed decisions even with the qualifying policy present.

**Deterministic test**

```python
def test_learning_scorecard_reads_recorded_policy_effect(tmp_path):
    from product.forward_evidence import freeze_observation
    from product.learning_scorecard import learning_influence

    row = freeze_observation(
        {"symbol": "INFY", "decision": "BLOCK", "policy_effect": "BLOCK"},
        cycle_id="one", as_of="2026-09-01", path=tmp_path / "forward.jsonl")
    policy = {"policy_id": "SETUP::VCP", "production_status": "ACTIVE"}
    assert learning_influence([policy], [row])["decisions_changed_by_learning"] == 1
```

**Suggested fix** · Read the persisted field with a compatibility fallback. Record whether the policy changed the decision or rank against the same candidate evaluated without the overlay.

## Verified correct

- **Canonical wiring.** `run_whole_market_scan` delegates to the unified scanner, projects recommendations and discovery, and publishes the scan. Discovery builds decisions and calls `rank`; paper entry separately evaluates and ranks the saved recommendation cards. The supervisor dispatches through `HANDLERS`, and startup installs the lightweight outcome lane followed by its integrity wrapper. The surfaced-signal settlement call is on that installed path. `scan/market_scan_service.py:393`, `:645`, `:671`; `product/desk_scan_overlays.py:63`, `:82`; `product/decision_service.py:324`; `research/autonomy/supervisor.py:2110`; `research/autonomy/__init__.py:90`; `research/autonomy/outcome_liveness.py:396`.
- **What the migration fixed.** Both forward-ledger rejection loops skip rows marked `entered`. For a complete, ordered bar window they use the common buy-stop, gap-aware stop and target, and next-bar breakeven trail. Fresh ingestion rejects legacy forward-outcome classes without a current spec before taking a dedupe claim. `product/forward_soak.py:234`; `product/autonomous_loop.py:629`; `core/outcome_resolver.py:131`; `core/evaluation_spec.py:119`; `product/_paper_learning_loop_core.py:393`.
- **Feedback that changes decisions.** Linked paper closes feed conditional evidence, which can demote discovery rank after its sample floor. Closes also feed setup policies. A rulebook missed winner following `EVIDENCE_POLICY_BLOCK` adds `+0.40` to that setup policy. Status and session floors gate subsequent rank adjustments and vetoes. The ML challenger’s paper-rank adjustment requires `PAPER_ACTIVE`, with promotion checking exact-version forward performance. Surfaced-signal and edge-gated shadow outcomes also feed the scanner’s demote-only forward multipliers. The scanner still applies the separate walk-forward proof and ledger-attestation checks before issuing a BUY. `product/_paper_learning_loop_core.py:140`, `:265`, `:482`; `product/conditional_evidence.py:318`; `product/learning_policy_store.py:141`; `product/evidence_policy_engine.py:220`; `product/challenger_learning.py:647`, `:750`; `scan/live_edge.py:30`; `scan/calibration_snapshot.py:125`; `scan/unified_scanner.py:278`, `:323`.
- **Reporting and write-only artifacts.** The nightly `challenger_shadow` result and learning scorecard are reporting outputs. I found writers for `learning_memory.json` and `learning_observations.jsonl`, with no readers of either file in the traced selection path. This is separate from the SQLite observation claims that drive policy sample accounting. `research/autonomy/jobs.py:1644`, `:1653`, `:1661`; `product/autonomous_loop.py:728`; `product/learning_observations.py:55`.

## Not examined / limits

- Static audit only. The pytest sketches were not executed. I did not inspect runtime stores, production state, environment files, or external services. The counterexamples establish behavior at this commit, not how often it occurred in production.
- The main rejection settlement defaults to ten, but there is no repo-wide ten-session invariant. `product/decision_simulator.py:177` and `product/historical_replay.py:281` retain five-session close-to-close grading. `research/autonomy/outcome_integrity.py:51` also uses five for its error probe without writing labels. The core decision journal uses close-to-close and a separate WAIT experiment. Case memory defaults to fifteen and uses `first_touch_path`. Paper holds and outcome settlement also read separately overridable settings. `core/decision_journal.py:218`; `product/case_memory.py:30`, `:463`; `core/evaluation_spec.py:48`; `product/paper_autopilot.py:67`.
- I did not independently rederive the Holm or overlap statistics, audit all alpha-ledger crash paths, or exhaustively review risk, broker interlocks, Telegram, F&O, or the legacy scanner and UI.