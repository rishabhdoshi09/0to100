<!-- Raw output of the independent GPT verification pass over 9636b030..01dec27c (Codex CLI, read-only), 2026-11-20. Peer input, NOT verified and NOT an approval. Resolutions: ../2026-11-20-gpt-crossreview.md -->

## Findings

Reviewed `9636b030..01dec27c` at the requested detached HEAD. **The fixes still leave confirmed paths to an uncertified production paper fill, a reset annual alpha budget, and an oversized Telegram paper trade.** The pytest sketches below are for the coordinator. I traced them against the source and did not run them.

### QT-01 · P0 · CONFIRMED · A-1 / C-8 · The paper executor accepts a missing certificate

`research/autonomy/jobs.py:1151` `run_paper_cycle` checks that the scan and recommendations agree, then calls the installed cycle at `:1177`. The cycle reloads recommendations at `product/paper_autopilot.py:1117` and reads the scan again at `:1197`. If that second scan read fails or sees a different generation, `_current_scan_certificate` returns `None` at `:210–217`. `_withdraw_uncertified` then returns the original production-tier card at `:224–225`. The card can pass selection and reach `_execute` at `:1426`.

The first job check can succeed just before a scan replacement or read failure. The money-path test at `tests/test_money_paths.py:5979` explicitly expects the fail-open behavior on a generation mismatch. The exploration assertion at `:5975` also passes with zero fills.

**Fix.** Validate the same recommendation generation the executor consumes. Block production entry on a missing or mismatched certificate, and recheck authority at the final mutation boundary.

```python
def test_paper_cycle_requires_its_own_scan_certificate(monkeypatch):
    from product import paper_autopilot as PA, scan_store as SS
    from tests.test_paper_autopilot_money_path import (
        PaperBook, _eligible_card, _now, _workspace,
    )

    ws = _workspace([_eligible_card()])
    monkeypatch.setenv("QT_PAPER_EXPLORATION", "0")
    monkeypatch.setattr(SS, "load_scan", lambda: None)
    book = PaperBook(capital=100_000)
    out = PA.run_reco_paper_cycle(
        book=book, workspace=ws, as_of="2026-09-01", now=_now(),
        entries_allowed=True, enforce_history=False,
    )
    assert not book.open and not out["taken"]
```

### QT-02 · P0 · CONFIRMED · C-3 · The first run after ledger loss erases the evidence of prior spending

`scan/signal_backtest.py:442` `_alpha_ledger_lost` checks only the last report’s `look_index`. A saveable run after ledger loss detects the old look and emits a zero-index attestation at `:1046–1055`, then overwrites that same report at `:1117–1120`. The next run finds index zero and calls `reserve`. The empty database starts again at look 1 at `research/alpha_ledger.py:175–191`. Losing the ledger after 52 looks can therefore reopen the budget in the same reservation year and production configuration.

`tests/test_crossreview_20261120.py:327` tests the predicate on two hand-written reports. It never exercises the write that changes the predicate’s answer.

**Fix.** Preserve the last spent report and latch ledger loss in durable state. Keep certification blocked until the ledger is restored or reconciled. The guard must also cover a reservation committed before any report was written.

```python
def test_ledger_loss_stays_blocked_across_two_runs(tmp_path, monkeypatch):
    from tests.test_alpha_ledger import _stub_walk_forward

    sb = _stub_walk_forward(monkeypatch, tmp_path)
    ledger = tmp_path / "alpha.db"
    monkeypatch.setenv("QT_ALPHA_LEDGER", str(ledger))
    monkeypatch.setattr(sb, "MIN_SYMBOLS_TO_SAVE", 1)

    def accounting():
        return sb.run_backtest()["evidence_provenance"]["alpha_accounting"]

    assert accounting()["look_index"] == 1
    ledger.unlink()
    assert accounting()["alpha"] == 0
    assert accounting()["alpha"] == 0
    assert not ledger.exists()
```

### QT-03 · P0 · CONFIRMED · C-9 · Telegram can exceed both per-trade hard caps

`alerts/telegram_actions.py:68–85` `_do_paper_trade` sizes from `settings.risk_per_trade_pct` and `settings.max_position_size_pct` via `risk/position_sizer.py:21–43`. The shared gate at `risk/portfolio_risk.py:247–277` checks duplicate, slots, sector and aggregate risk. It does not enforce 1% per trade or 10% per name. With ₹100,000 capital, configured limits of 2% and 20%, and an empty book, a ₹100 entry with a ₹90 stop journals 200 shares. The 2% open risk passes the 5% check.

**Fix.** Clamp configurable sizing to the hard ceilings and enforce both ceilings again in the shared mutation gate with the final quantity.

```python
def test_tap_respects_hard_caps_with_looser_settings(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import config
    from alerts.telegram_actions import _do_paper_trade
    from execution import trade_executor as TE
    from product import paper_status
    from risk import portfolio_risk as PR
    from risk.position_manager import _open_trades

    monkeypatch.setattr(config, "settings", SimpleNamespace(
        trading_capital=100_000, risk_per_trade_pct=.02,
        max_position_size_pct=.20, max_open_positions=5,
    ))
    monkeypatch.setattr(TE, "_DB", tmp_path / "trades.db")
    monkeypatch.setattr(paper_status, "modern_engine_open_positions", lambda **kw: [])
    monkeypatch.setattr(PR, "_fno_open_rows", lambda: [])
    _do_paper_trade("TCS", 100, 90, 120)
    assert sum(row["qty"] for row in _open_trades(strict=True)) <= 100
```

### QT-04 · P0 · CONFIRMED · C-11 · A failed in-lock save releases invisible exposure

The installed wrapper now passes the save callback at `research/autonomy/paper_cycle_truth.py:148–159`. In `product/paper_autopilot.py:1425–1445` `run_reco_paper_cycle`, the book is mutated before the callback runs. If the save returns `False`, the account lock releases with the new position still in memory. The second save at `:1516–1522` happens outside the lock. In that interval another paper writer can see the old disk book via `risk/portfolio_risk.py:100–105` and open the same symbol or spend the same capacity. The wrapper test at `tests/test_money_paths.py:5993` only checks callback injection.

**Fix.** Roll back a failed open under the lock, or make its exposure visible to every writer through a durable reservation before releasing the lock.

```python
def test_failed_persist_keeps_exposure_consistent_at_unlock(monkeypatch):
    from contextlib import contextmanager
    from risk import portfolio_risk as PR
    from tests.test_paper_autopilot_money_path import PaperBook, _cycle, _eligible_card

    book = PaperBook(capital=100_000)
    saves, releases = [], []

    def save():
        saves.append(True)
        return len(saves) > 1

    @contextmanager
    def lock():
        yield
        releases.append((bool(book.open), len(saves) > 1))

    monkeypatch.setattr(PR, "account_mutation_lock", lock)
    _cycle(book, [_eligible_card()], durable_book_save=save)
    assert releases and all(not opened or durable for opened, durable in releases)
```

### QT-05 · P0 · CONFIRMED · C-8 · Home can retain a BUY after its authority disappears

`product/scan_store.py:419–443` `_evidence_authority` now notices a missing or replaced snapshot target. Home can skip that reader entirely. `_scan_evidence_index` returns its cached verdict at `product/home_os.py:276–285`. Its `evidence_authority_key` at `product/scan_store.py:446–465` omits the target file and any expiry boundary. After one successful Home read, deleting just the snapshot target makes a fresh `load_scan` withhold the BUY while `build_buy_only` at `product/home_os.py:348–363` still accepts the cached BUY. The same cache can retain a BUY across evidence expiry.

**Fix.** Cache the parsed scan and reverify its BUYs when serving Home, or key the verified result on the full authority identity and expire it by the earliest validity deadline.

```python
def test_home_revokes_buy_when_authority_target_disappears(tmp_path, monkeypatch):
    import json
    from product import home_os as HO, scan_store as SS
    from scan import calibration_snapshot as CS
    from tests.test_crossreview_20261120 import _authority

    target = tmp_path / "cal.json"
    target.write_text(json.dumps(_authority("cal")))
    (tmp_path / "current.json").write_text('{"snapshot_id":"cal"}')
    scan = tmp_path / "scan.json"
    scan.write_text(json.dumps({
        "schema_version": 2, "provenance": {}, "evidence_identity": {"id": "cal"},
        "records": [{"symbol": "TCS", "verdict": "BUY", "signals": ["VCP"],
                     "edge_evidence_id": "cal"}],
    }))
    monkeypatch.setattr(CS, "DEFAULT_DIR", tmp_path)
    monkeypatch.setattr(SS, "DEFAULT_SCAN_PATH", scan)
    monkeypatch.setattr(SS, "scan_source_status", lambda _: (True, ""))
    monkeypatch.setattr(HO, "_SCAN_EVIDENCE_CACHE", {"key": None, "value": None})
    load_scan = SS.load_scan
    monkeypatch.setattr(SS, "load_scan", lambda: load_scan(scan))
    rows = [{"symbol": "TCS", "decision": "BUY", "entry": 100, "stop": 95, "target": 110}]
    assert HO.build_buy_only(rows, verified=True)["count"] == 1
    target.unlink()
    assert load_scan(scan)["records"][0]["verdict"] == "WATCH"
    assert HO.build_buy_only(rows, verified=True)["count"] == 0
```

### QT-06 · P0 · CONFIRMED · A-7 / B-4 / B-5 · Missing calendar authority restores positional settlement

`core/session_window.py:65–87` `official_dates` returns `[]` when the archive listing is empty or fails. `core/outcome_resolver.py:111–135` `resolve_trade` treats that as permission to simulate the next ten symbol rows. The bar source is independent of the archive listing: `data/bhavcopy_store.py:329–332` can restore the in-memory store from its cache, and `get_ohlcv` at `:636–656` reads that store. A cached frame with a missing second session and a target touch on official session 11 can thus produce a WIN. `core/signal_outcome_tracker.py:171–188` consumes it as a settled result. The A-7 test supplies a nonempty `sessions` list, so it misses the production fallback.

**Fix.** Return unresolved when official-session authority is unavailable. An empty verified index must authorize zero bars.

```python
def test_cached_bars_cannot_settle_without_session_authority(monkeypatch):
    import pandas as pd
    from core import outcome_resolver as OR, session_window as SW

    days = pd.bdate_range("2026-02-03", periods=11).delete(1)
    frame = pd.DataFrame(
        {"open": 100., "high": 101., "low": 99., "close": 100.}, index=days,
    )
    frame.loc[days[-1], "high"] = 111.
    monkeypatch.setattr(OR, "_ohlcv", lambda _: frame)
    monkeypatch.setattr(SW, "official_dates", lambda: [])
    assert OR.resolve_trade("TCS", "2026-02-02", 100, 95, 110) is None
```

### QT-07 · P0 · CONFIRMED · B-8 · A missing open can still create a fill below the day’s low

`data/bhavcopy_store.py:257–268` coerces a malformed open to `NaN` and retains the row when its close is valid. `core/evaluation_spec.py:127–138` `simulate_trade` turns that open into `None` and fills at the trigger if the high reaches it. A bar with low ₹105, high ₹111 and close ₹108 produces a ₹100 fill and +0.8 gross R, even though its entire traded range was above ₹100. `scan/signal_backtest.py:882–899,949–964` counts the invented return in BUY evidence. The B-8 regression exercises a missing high and a valid gap open.

**Fix.** Require finite, coherent OHLC for a gap-aware fill or exit when an open is expected. Keep an unusable session unresolved for certification.

```python
def test_missing_open_cannot_fill_below_traded_range():
    from core.evaluation_spec import simulate_trade

    trade = simulate_trade(100, 90, 120, [111], [105], [108], opens=[float("nan")])
    assert not trade.filled
```

### QT-08 · P0 · CONFIRMED · A-2 · A close can still borrow another entry’s context

`product/_paper_learning_loop_core.py:120–138` `_lookup_taken` tries the paper ID, then falls back to the latest same-day row regardless of that row’s ID. If the close for `pos-a` lacks its taken-journal record and `pos-b` for the same symbol and date is present, `ingest_closed_trade` uses `pos-b`’s setup and execution context at `:228–237` and updates its setup policy at `:280–289`. `forward_soak.attach_closed_trades` uses the same lookup at `product/forward_soak.py:364–367`. The added A-2 test replaces `_lookup_taken` with an empty stub.

The dedupe also misses closes attached before the upgrade. `attach_closed_trades` builds `used` only from `later_outcome.closed_trade_key` at `:330–331`. A historical close already recorded without that key can settle a second legacy row from the same session when it is the only remaining pending match at `:346–358`.

**Fix.** Exclude conflicting IDs from date fallback and require an unambiguous legacy match across settled and pending rows. Reconcile unique legacy close keys before attaching more closes.

```python
def test_taken_context_never_falls_back_to_conflicting_position_id(monkeypatch):
    from product.paper_learning_loop import _core

    monkeypatch.setattr(_core, "_read_jsonl", lambda _: [{
        "symbol": "TCS", "paper_decision_id": "pos-b",
        "entry_date": "2026-09-01", "setup_label": "VCP",
    }])
    assert _core._lookup_taken(
        "TCS", decision_id="pos-a", entry_date="2026-09-01",
    ) == {}
```

### QT-09 · P0 · CONFIRMED · A-8 · Evolution promotion still consumes legacy equity grades

`product/evolution/grading.py:159–160,238–245` leaves previously graded equity rows untouched. `product/evolution/scorecard.py:57–64` `graded_rows` accepts every stored outcome without checking its rulebook. `paired_comparison` at `:182–221` puts legacy `counterfactual_R` into `incremental_diffs`, and `product/evolution/promotion.py:329–344` uses those pairs and that expectancy for promotion. A legacy first-touch win survives beside a new gap-aware loss and can supply the required pair count. The grading test covers a fresh row only.

**Fix.** Filter equity promotion inputs by the current rulebook, or regrade old outcomes from frozen inputs before admitting them to the comparison.

```python
def test_equity_promotion_ignores_legacy_grade(tmp_path):
    import json
    from product.evolution.scorecard import paired_comparison

    ledger = tmp_path / "shadow.jsonl"
    base = {"domain": "EQUITY", "market_snapshot_id": "s", "outcome": {"settled": True}}
    rows = [
        dict(base, policy_id="champ", decision="WAIT", counterfactual_R=2),
        dict(base, policy_id="chal", decision="ENTER_NOW", counterfactual_R=2),
    ]
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert paired_comparison("champ", "chal", domain="EQUITY", path=ledger)["paired_snapshots"] == 0
```

### QT-10 · P1 · CONFIRMED · B-7 · Replay scorecards recreate a return for orders that never filled

`product/historical_replay.py:729–743` `evaluate_outcomes` clears the rulebook R for `NO_FILL` or unresolved rows, but keeps the close-path `forward_return_pct` written by `product/decision_outcomes.py:141–147`. `product/scorecards.py:26–34` falls back to that return when `r_multiple` is `None`. `run_historical_replay` calls the scorecards at `product/historical_replay.py:1280–1284`. A ₹100 buy-stop with a high of ₹95 and close of ₹94 throughout can contribute −0.6R to the reason scorecard even though it never triggered.

**Fix.** Keep close-path measures under `path`. Make scorecards and attribution honor the rulebook’s explicit missing R.

```python
def test_replay_no_fill_does_not_contribute_to_reason_expectancy(monkeypatch):
    import pandas as pd
    from core import session_window as SW
    from product import historical_replay as HR
    from product.scorecards import reason_scorecards

    days = pd.bdate_range("2026-02-03", periods=10)
    frame = pd.DataFrame(
        {"open": 94., "high": 95., "low": 93., "close": 94.}, index=days,
    )
    monkeypatch.setattr(HR, "ohlcv_as_of", lambda *a, **kw: frame)
    monkeypatch.setattr(SW, "official_dates", lambda: list(days.strftime("%Y-%m-%d")))
    rows = HR.evaluate_outcomes([{
        "symbol": "TCS", "as_of": "2026-02-02", "decision": "AVOID",
        "entry": 100, "stop": 90, "target": 110,
    }])
    assert rows[0]["rulebook_outcome"] == "NO_FILL"
    assert reason_scorecards(rows)["rows"][0]["r_expectancy"] is None
```

### QT-11 · P1 · CONFIRMED · C-4 / C-6 · Frame length stands in for usable and current evidence

`scan/signal_backtest.py:852–873` `_run_backtest_inner` credits a symbol and adds its dates before checking any exact window or attempting analysis. A 100-row frame with no OHLC columns and zero eligible windows advances the save threshold. Its endpoint can also become `data_last_session` at `:1035–1039,1105–1109` while every measured trade is from older frames. `evidence_provenance_note` at `scan/unified_scanner.py:366–380` checks that single endpoint. The C-4 test exercises program hashes and Holm arithmetic without running the save gate.

**Fix.** Count a symbol after validating its input and evaluating an eligible window. Bind evidence freshness to the bars actually used by each certifying hypothesis.

```python
def test_unusable_frame_cannot_turn_thin_run_into_saved_report(tmp_path, monkeypatch):
    import pandas as pd
    from data import bhavcopy_store as BS
    from tests.test_alpha_ledger import _stub_walk_forward

    sb = _stub_walk_forward(monkeypatch, tmp_path)
    read = BS.get_ohlcv
    junk = pd.DataFrame(index=pd.bdate_range("2026-06-01", periods=100))
    monkeypatch.setattr(BS, "store_symbols", lambda: ["S0", "S1", "S2", "JUNK"])
    monkeypatch.setattr(BS, "get_ohlcv", lambda symbol: junk if symbol == "JUNK" else read(symbol))
    monkeypatch.setattr(sb, "MIN_SYMBOLS_TO_SAVE", 4)
    report = sb.run_backtest()
    assert report["evidence_provenance"]["symbols_evaluated"] == 3
    assert not sb._OUT.exists()
```

### QT-12 · P1 · CONFIRMED · B-9 · An older nonempty events file validates newer coverage

`data/corporate_actions_resilient.py:654–671` writes each successful coverage window before it writes the merged events. If interrupted between those writes, a nonempty older events file remains. `research/edge_diagnosis.py:578–589` `load_copy` only checks whether *any* events exist. `analyse` at `:483–520` can then include an unadjusted split in the newest covered window as `STRICT`.

**Fix.** Publish coverage with a digest of the exact event payload, and reject coverage when the loaded payload differs. Advance coverage only after that payload is durable.

```python
def test_coverage_without_matching_event_generation_is_unverified(tmp_path, monkeypatch):
    import json
    import pickle
    from data import corporate_actions as CA
    from research import edge_diagnosis as ED

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    logs = tmp_path / "logs"
    (logs / "bhav").mkdir(parents=True)
    (logs / "bhav" / "store_cache.pkl").write_bytes(pickle.dumps({"store": {}}))
    (logs / "ca_coverage.json").write_text(json.dumps({"windows": {"latest": {
        "success": True, "covered_from": "2025-01-01", "covered_to": "2026-11-19",
    }}}))
    monkeypatch.setattr(CA, "load_events", lambda _: {
        "OLD": [{"ex_date": "2024-01-01", "factor": 2}],
    })
    assert ED.coverage_span(ED.load_copy(tmp_path)["coverage"]) == (None, None)
```

### QT-13 · P1 · CONFIRMED · B-10 · Parseable invalid membership can still certify completeness

`data/nse_universe.py:564–579` `point_in_time_universe` increments `valid` before checking membership. `pd.Timestamp(None)` returns `NaT`; a list containing only `{"symbol": "TCS", "listed": null}` therefore returns an empty universe marked complete. `product/historical_replay.py:206–225,251–257` can fall back to store symbols and carry that complete flag into replay. Separately, `research/edge_diagnosis.py:489–520` still includes unknown membership in `STRICT` statistics.

**Fix.** Validate nonempty symbols and finite, ordered membership dates before claiming completeness. Require verified membership for statistics labelled strict.

```python
def test_null_listing_date_does_not_certify_membership(tmp_path, monkeypatch):
    from data import nse_universe as U

    table = tmp_path / "membership.json"
    table.write_text('[{"symbol":"TCS","listed":null}]')
    monkeypatch.setattr(U, "get_nse_universe", lambda: ["TCS"])
    assert U.point_in_time_universe("2026-09-01", table)["survivorship_complete"] is False
```

### QT-14 · P2 · CONFIRMED · B-2 / C-7 · The async replay cache uses the wrong default calibration identity

`product/historical_replay.py:937–948` uses the neutral snapshot ID for a default run. `start_replay_async` passes `None` to `replay_is_current` at `:1352–1359`, and `replay_identity` at `:861–866` hashes that as an empty ID. A completed neutral replay misses the async cache at `:910–918` even when all inputs agree. The scheduled path through `product/autonomous_learning.py:497–501` uses this entry point.

**Fix.** Resolve the effective calibration once and apply the same default to cache lookup and execution.

```python
def test_default_replay_lookup_recognizes_neutral_report(tmp_path):
    import json
    from product import historical_replay as HR

    dates = lambda: ["2026-09-01", "2026-09-02"]
    saved = HR.replay_identity(
        dates_fn=dates, calibration_snapshot=HR.NEUTRAL_REPLAY_CALIBRATION,
    )
    HR.report_path(tmp_path).write_text(json.dumps(dict(saved, status="SUCCEEDED")))
    assert HR.replay_is_current(directory=tmp_path, dates_fn=dates)
```

## Verified correct

These checks cover the original counterexample where applicable. The findings above give the remaining failures.

| Audit claim | Verified behavior |
|---|---|
| A-1 / C-8 | A matching scan with a read-time WATCH demotes an eligible paper card. A fresh Home index also excludes it. `product/paper_autopilot.py:215–235`, `product/home_os.py:354–362`. |
| A-2 | An exact paper-ID match attaches to its own pending row. Newly attached closes carry a dedupe key. `product/forward_soak.py:340–352,379–399`, `product/forward_evidence.py:271–276`. |
| A-4 | The challenger’s training-row loader rejects untagged simulated labels. The feature outcome writer records the supplied spec. `product/challenger_learning.py:151–182`, `product/evidence_intelligence.py:247–274`. |
| A-5 / A-6 | Staging uses each freeze’s ID or legacy geometry. The soak settler, official forward-ledger loop and repair exclude `BACKTEST` and `WALK_FORWARD` rejection rows. `research/counterfactual_reconcile.py:158–164,195–211`, `product/forward_soak.py:234–239`, `product/autonomous_loop.py:629–635`, `product/learning_policy_repair.py:76–88`. |
| A-7 / B-4 / B-5 | With a verified index, a missing covered weekday cuts the window and uncovered future years are removed. The resolver drops the marked overlay before simulation. `core/session_window.py:50–62,90–103`, `core/outcome_resolver.py:105–124`, `scan/signal_backtest.py:863–867`. |
| A-8 / B-7 | Newly graded equity shadows and replayed rejections use the gap-aware rulebook net R. An unfilled replay order does not write a feature-training label. `product/evolution/grading.py:192–206`, `product/historical_replay.py:718–763`. |
| A-9 / A-10 | The installed outcome wrapper converts the signal-log failure marker into `RETRYABLE_FAILED`. The scorecard reads persisted `policy_result`. `research/autonomy/__init__.py:90–97`, `research/autonomy/jobs.py:1327–1345`, `research/autonomy/outcome_integrity.py:107–118`, `product/learning_scorecard.py:648–656`. |
| B-1 | The ₹20 floor uses the historical traded price recovered from post-bar corporate-action factors. Order geometry still rounds in adjusted price units, as disclosed. `data/corporate_actions.py:109–134`, `scan/unified_scanner.py:943–953,1164–1165,1265`. |
| B-2 / C-7 | The default replay analyzer gets neutral weights, with empty regime calibration on direct `_analyze` calls. `product/historical_replay.py:285–315,937–948`, `scan/unified_scanner.py:781–806,1172–1177`. |
| B-3 / C-2 / C-13 | A fresh scan read rejects a foreign evidence ID. The authority reader includes the target file in its cache identity. Authority loss with a blank published note reaches the admission status as `EVIDENCE_UNAVAILABLE`. `product/scan_store.py:419–443,514–553,608`, `product/buy_admission.py:49–52,75–79`. |
| B-8 / B-9 / B-10 | A valid gap beyond the target scratches at its fill. Coverage ends at the recorded end date and an absent events file clears coverage. Empty and wrong top-level membership schemas are rejected. `core/evaluation_spec.py:143–146`, `research/edge_diagnosis.py:311–317,470–472,583–589`, `data/nse_universe.py:554–561`. |
| C-3 / C-4 / C-6 | Ordinary production reads reject an environment-redirected ledger. The public backtest binds its run parameters, pads Holm with untested catalogue members, and the verifier checks the exact-window flag and reported data age. `research/alpha_ledger.py:256–273`, `scan/signal_backtest.py:427–438,815–820,1027–1063`, `scan/unified_scanner.py:363–383`. |
| C-9 / C-11 / C-12 | Telegram refuses zero quantity, preserves paise in callback data and journals with `paper=True` under the account lock. A successful modern book save runs under that lock. The normal exploration path reserves before execution and blocks on a failed write. `alerts/telegram_actions.py:41–85`, `execution/trade_executor.py:154–168`, `product/paper_autopilot.py:182–187,1398–1445`. |

**C-1, common shock.** The sorted greedy algorithm correctly finds the maximum number of disjoint equal-length windows (`research/overlap_inference.py:146–160`). I agree a single realised history cannot identify the stipulated common-sign shock. The reported inference is conditional on the dependence assumption, and the disjoint-window count alone provides no protection against that null.

**C-5, intersection.** The minimum tightens the former average for the marginal rows present, but `setup_evidence` omits missing components and can certify from one surviving marginal row (`scan/unified_scanner.py:245–274`). Positive marginals give no lower bound for the exact intersection. Under the stated BUY invariant, an untested intersection remains WATCH.

**Open items.**

- **A-3:** Accurate. The batch commits settlement before policy ingest, and retries exclude already-settled rows. `product/forward_soak.py:237–256`.
- **A-4, mixed aggregates:** Accurate. Repair skips policies whose latest source does not begin with `counterfactual`, even when earlier observations contributed to the same aggregate. `product/learning_policy_repair.py:135–136`.
- **B-6:** Accurate. The backtest requires room for a complete positional horizon before it simulates, so an observable early stop near the last print is omitted. `scan/signal_backtest.py:852–886`.
- **B-8, target sweep:** Accurate. The sweep call supplies highs, lows and closes without opens. `scan/signal_backtest.py:935–942`.
- **C-2, result digest:** Accurate under the stated scope. The ledger binds report metadata and family membership, and the stored result rows lack a digest in the attestation. `scan/signal_backtest.py:1032–1058`, `research/alpha_ledger.py:293–301`.
- **C-8, board, desk alert and tracker:** Accurate. The board reads saved recommendations, the desk alert uses the scan to check generation only, and the tracker receives the publication payload. `product/decision_service.py:302–337`, `research/autonomy/telegram_notifications.py:950–966`, `scan/market_scan_service.py:672–676`.
- **C-10:** Accurate for the canonical startup call graph examined. The legacy scan loop starts the sniper, and `legacy_app.py` starts that loop behind its daemon opt-in. `legacy_app.py:142–162`, `scan/auto_scan.py:908–915`.
- **Developer’s open `note_later_entry` item:** Accurate. The update is keyed by symbol and the pending-row loop imposes no age bound. `product/_paper_learning_loop_core.py:525–531,549–562`.

## Not examined / limits

This was a single static pass over the requested change set and relevant callers. I did not run pytest, inspect production or runtime artifacts, contact external services, or verify deployment settings and historical data quality. The findings establish reachable behavior from the supplied code and fixtures. They do not establish that these counterexamples occurred on the host.