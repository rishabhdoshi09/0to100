<!-- Raw output of an independent GPT (OpenAI Codex CLI, read-only sandbox) review of 9636b030, run B, 2026-11-20. Peer input, NOT verified and NOT an approval. Verification status per finding: ../2026-11-20-gpt-crossreview.md -->

## Findings

### B-1 · P0 · CONFIRMED · Future splits change historical setup eligibility

**Location** · `data/bhavcopy_store.py:636` `get_ohlcv`, `data/corporate_actions.py:78` `adjust_frame`, `scan/signal_backtest.py:812` `_run_backtest_inner`, `scan/unified_scanner.py:928` `_analyze`, `product/historical_replay.py:103` `ohlcv_as_of`.

**Counterexample** · A ₹30 stock qualifies for analysis on September 18. Add a 2-for-1 split with a September 21 ex-date, leaving every historical bar unchanged. `get_ohlcv()` now returns the September 18 close as ₹15. The backtest and replay slice the frame after adjustment, so the scanner drops the same historical setup at its ₹20 price floor. Price and volume ratios can survive uniform scaling, but this absolute filter and rounded order levels cannot. `research/edge_diagnosis.py:562` and `research/full_history_probe.py:38` also adjust before the historical slice.

**Deterministic test**

```python
def test_future_split_cannot_remove_historical_setup(monkeypatch):
    import pandas as pd
    from data import bhavcopy_store as BS
    from product.historical_replay import ohlcv_as_of
    from scan.unified_scanner import UnifiedScanner

    day = "2026-09-18"
    bars = pd.DataFrame(
        {"open": 30., "close": 30., "high": 31., "low": 29., "volume": 1e6},
        index=pd.bdate_range(end=day, periods=75),
    )
    monkeypatch.setattr(BS, "_store", {"TCS": bars})
    monkeypatch.setattr(BS, "_CA_EVENTS", {})
    scanner = UnifiedScanner(calibration_snapshot={"snapshot_id": "neutral"},
                             evidence_gate=False)

    def setup():
        return scanner._analyze("TCS", ohlcv_as_of("TCS", day, ohlcv_fn=BS.get_ohlcv))

    assert setup() is not None
    monkeypatch.setattr(BS, "_CA_EVENTS", {"TCS": [
        {"ex_date": pd.Timestamp("2026-09-21"), "factor": 2., "type": "split"}
    ]})
    assert setup() is not None
```

**Suggested fix** · Build decision frames from raw bars and events available and effective by the decision cutoff. Translate frozen order levels and outcome bars onto a common share basis when an action occurs during the holding window.

### B-2 · P0 · CONFIRMED · Replay scores historical decisions with future calibration

**Location** · `product/historical_replay.py:302` `scan_session`, `product/historical_replay.py:900` and `:1066` `run_replay`, `scan/calibration_snapshot.py:55` `_load_inputs`, `scan/unified_scanner.py:764` `__init__` and `:1149` `_analyze`.

**Counterexample** · Replay a September flat-base setup with identical September OHLCV. Change the November report’s `FLAT_BASE` multiplier from `0.45` to `1.25`. The September score changes, and scores feed the replay ranking and verdict. `evidence_gate=False` disables the proof gate and leaves the score multipliers active. With multiple signals, the same change can cross the scanner’s 55-point BUY threshold. The replay also freezes decision features into the learning store at `product/historical_replay.py:618`.

**Deterministic test**

```python
def test_replay_score_is_invariant_to_future_calibration(monkeypatch):
    import pandas as pd
    from product import historical_replay as HR, scan_store as SS
    from scan import calibration_snapshot as CS

    day = "2026-09-18"
    bars = pd.DataFrame(
        {"close": 100., "high": 101., "low": 99., "volume": 1e6},
        index=pd.bdate_range(end=day, periods=75),
    )
    monkeypatch.setattr(SS, "scan_provenance", lambda **kwargs: {})

    def score(weight):
        monkeypatch.setattr(CS, "get_or_create_snapshot", lambda: {
            "snapshot_id": f"november-{weight}",
            "created_at": "2026-11-20T12:00:00Z",
            "backtest_generated_at": "2026-11-20T11:00:00Z",
            "multipliers": {"FLAT_BASE": weight},
        })
        return HR.scan_session(day, ["TCS"], ohlcv_fn=lambda _: bars)["records"][0]["score"]

    assert score(0.45) == score(1.25)
```

**Suggested fix** · Resolve calibration per decision timestamp using the availability of both backtest and forward inputs. Use explicit neutral weights and mark comparability as limited when no eligible snapshot exists.

### B-3 · P0 · CONFIRMED · `load_scan()` can certify a BUY with a report generated after the scan

**Location** · `product/scan_store.py:470` `bind_evidence`, `product/scan_store.py:532` `load_scan`, `scan/unified_scanner.py:371` `evidence_age_note`.

**Counterexample** · A scan stamped 10:30 IST carries BUY and names snapshot `old`. A report generated at 15:30 IST becomes current under snapshot `new` and proves that signal. At 16:30, `load_scan()` retains the morning BUY. The authoritative branch checks report age against read time, evaluates the new edges, and never requires the scan or row evidence ID to equal the authority ID. The IDs are merely reported in `evidence_binding`.

**Deterministic test**

```python
def test_scan_cannot_be_certified_by_a_later_report(tmp_path, monkeypatch):
    from datetime import datetime
    from product import scan_store as SS
    from scan import unified_scanner as US
    from tests.evidence_fixtures import proven_provenance

    monkeypatch.setenv("QT_ALPHA_LEDGER", str(tmp_path / "alpha.db"))
    check = US.evidence_provenance_note
    read_at = datetime.fromisoformat("2026-11-20T11:00:00+00:00")
    monkeypatch.setattr(US, "evidence_provenance_note",
                        lambda p, *, now=None: check(p, now=now or read_at))
    snap = {
        "snapshot_id": "new",
        "backtest_provenance": proven_provenance("2026-11-20T10:00:00+00:00"),
        "backtest_edges": {"VCP": dict(mean=.5, lower=.2, n=90, blocks=30,
                                       inference="OVERLAP_ROBUST", family=True)},
    }
    monkeypatch.setattr(SS, "_evidence_authority", lambda: snap)
    monkeypatch.setattr(SS, "scan_source_status", lambda _: (True, ""))
    path = SS.save_scan({
        "schema_version": 2, "scanned_at": "2026-11-20T05:00:00+00:00",
        "evidence_identity": {"id": "old"},
        "records": [{"symbol": "TCS", "signals": ["VCP"], "verdict": "BUY",
                     "edge_evidence_id": "old"}],
    }, tmp_path / "scan.json")
    assert SS.load_scan(path)["records"][0]["verdict"] == "WATCH"
```

**Suggested fix** · Require matching row, scan, and active evidence IDs. Check that the evidence existed by scan start, and check freshness again at read time.

### B-4 · P0 · CONFIRMED · The backtest can count the eleventh official session as the tenth

**Location** · `scan/signal_backtest.py:602` `window_is_exact`, `:620` `realised_session_index`, `:805` and `:823` `_run_backtest_inner`, `:1036` report provenance. Compare `research/intelligence/data/nse_calendar.py:314` `forward_sessions`.

**Counterexample** · For a November 18 decision, the bundled calendar puts session ten on December 3. Remove the November 20 archive market-wide. The realised index compresses the missing date, approves December 4 as bar ten, and sets `exact_horizon_check=True`. A target first reached on December 4 can enter the weekly evidence as a win. An empty archive index bypasses the check entirely. The same realised-index path accepts weekdays in uncovered 2027.

**Deterministic test**

```python
def test_missing_archive_and_uncovered_year_are_not_exact_windows(monkeypatch):
    import json
    from datetime import date
    from pathlib import Path
    import pandas as pd
    from research.intelligence.data import nse_calendar as CAL
    from scan import signal_backtest as SB

    holidays = set(json.loads(Path("data/nse_holidays.json").read_text()))
    monkeypatch.setattr(CAL, "load_holidays", lambda: holidays)
    start = date(2026, 11, 18)
    official = CAL.forward_sessions(start, 11, holidays)
    missing = [str(start)] + [str(d) for d in official if str(d) != "2026-11-20"]
    uncovered = pd.bdate_range("2026-12-31", periods=11).strftime("%Y-%m-%d").tolist()
    accepted = [
        SB.window_is_exact(days, 1, 10, SB.realised_session_index(days))
        for days in (missing, uncovered)
    ]
    assert accepted == [False, False]
```

**Suggested fix** · Derive the expected dates from a verified exchange calendar and check observed bars against them. Treat missing sessions and uncovered years as unverified evidence.

### B-5 · P0 · CONFIRMED · Forward settlement can use an intraday overlay or a bar past the deadline

**Location** · `core/outcome_resolver.py:63` `resolve_trade` and `:109` `counterfactual_outcome`, `data/nse_live.py:215` `apply_live_to_store`, `data/bhavcopy_store.py:636` `get_ohlcv`, `product/autonomous_loop.py:572` and `:633` `settle_official_outcomes`.

**Counterexample** · For a November 18 rejection, a marked November 19 intraday overlay with open ₹100 and high ₹111 resolves a ₹100/₹90/₹110 plan immediately as a WIN. The resolver ignores the overlay marker and returns on a terminal touch, so the rejection can be saved permanently as a rulebook `MISSED_WINNER`. Separately, if the first ten *stored* rows end on December 4 because November 20 is absent, a target first reached on December 4 also resolves as a WIN.

**Deterministic test**

```python
def test_counterfactual_rejects_overlay_and_late_target():
    import pandas as pd
    from core.outcome_resolver import counterfactual_outcome

    overlay = pd.DataFrame({"open": [100.], "high": [111.], "low": [99.],
                            "close": [105.]}, index=pd.to_datetime(["2026-11-19"]))
    overlay.attrs["quantterm_live_overlay_date"] = "2026-11-19"

    dates = pd.bdate_range("2026-11-19", "2026-12-04").difference(
        pd.to_datetime(["2026-11-20", "2026-11-24"]))
    late = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.},
                        index=dates)
    late.loc["2026-12-04", "high"] = 111.

    got = [counterfactual_outcome("TCS", "2026-11-18", 100, 90, 110,
                                  horizon=10, frame=f) for f in (overlay, late)]
    assert got == [None, None]
```

**Suggested fix** · Settle from published official bars checked against expected session dates. Allow early terminal settlement only when the official sequence through the exit is complete.

### B-6 · P0 · CONFIRMED · A delisting can remove an already observable loss from the evidence

**Location** · `data/bhavcopy_store.py:343` `_full_build` and `:672` `store_symbols`, `scan/signal_backtest.py:745` `_spread_sample`, `:778` `run_backtest`, `:820` `_run_backtest_inner`, `research/edge_diagnosis.py:241` `collect` and `:280` `symbol_end_truncations`.

**Counterexample** · A stock with 90 bars signals on bar 87, fills at ₹100 and hits its ₹90 stop on bar 88, then ceases trading after bar 90. The loop ends before it analyzes this signal. Append otherwise irrelevant bars after the stop and the same loss enters the report. The store can retain delisted names, but even-stride sampling does nothing to prevent this outcome-dependent exclusion. The diagnosis reports an approximate truncation count after collection without recovering the loss.

**Deterministic test**

```python
def test_observed_stop_counts_when_symbol_ceases_trading(monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import pandas as pd
    from data import bhavcopy_store as BS
    from research.intelligence.data import nse_calendar as CAL
    from scan import calibration_snapshot as CS, signal_backtest as SB
    from scan.unified_scanner import UnifiedScanner

    holidays = set(json.loads(Path("data/nse_holidays.json").read_text()))
    dates = pd.bdate_range("2026-01-05", periods=125).difference(pd.to_datetime(list(holidays)))
    bars = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.},
                        index=dates[:90])
    bars.loc[dates[87], "low"] = 89.
    plan = SimpleNamespace(signals=["VCP"], entry=100., stop=90., target=120.)
    monkeypatch.setattr(CAL, "load_holidays", lambda: holidays)
    monkeypatch.setattr(BS, "store_symbols", lambda: ["TCS"])
    monkeypatch.setattr(BS, "_dates_on_disk", lambda: list(dates.date))
    monkeypatch.setattr(BS, "get_ohlcv", lambda _: bars)
    monkeypatch.setattr(CS, "get_or_create_snapshot", lambda: {"snapshot_id": "neutral"})
    monkeypatch.setattr(SB, "_nifty_regime_series", lambda: None)
    monkeypatch.setattr(SB, "_calibration_version", lambda: "test")
    monkeypatch.setattr(UnifiedScanner, "_analyze",
                        lambda self, sym, hist: plan if hist.index[-1] == dates[86] else None)
    trades = []
    SB.run_backtest(sample_step=1, max_symbols=1, on_trade=trades.append)
    assert [r["exit_reason"] for r in trades] == ["LOSS"]
```

**Suggested fix** · Schedule decisions using the historical exchange session and membership, independent of the symbol’s eventual last bar. Count verified early terminal outcomes and explicitly track unresolved delistings and suspensions before certifying evidence.

### B-7 · P0 · CONFIRMED · Historical replay writes a positive training label for a stopped-out trade

**Location** · `product/historical_replay.py:281` `_forward_return` and `:669` `evaluate_outcomes`, `product/decision_outcomes.py:66` `path_metrics` and `:130` `settle_frozen`, `product/evidence_intelligence.py:270` `record_resolved_prediction`.

**Counterexample** · A rejected ₹100/₹94/₹115 plan fills on session one, stops out on session two, and recovers to ₹108 by session five. Replay classifies the rejection as `MISSED_WINNER` using the five-session close return. It then derives `+1.33R` from the later close, although its own path metrics saw the stop, and writes that positive R to the feature store. The new rulebook settlement in the forward lane leaves this replay path in place.

**Deterministic test**

```python
def test_replay_learns_the_stop_even_if_price_recovers(monkeypatch):
    from core import outcome_resolver as OR
    from product import evidence_intelligence as EI, historical_replay as HR
    from tests.test_counterfactual_horizon import D, _bars

    frame = _bars([(99.5, 100.5, 99, 100), (99, 99.5, 93, 94.5)]
                  + [(108, 109, 107, 108)] * 8)
    monkeypatch.setattr(OR, "_ohlcv", lambda _: frame)
    monkeypatch.setattr(HR, "ohlcv_as_of", lambda *a, **kw: frame)
    labels = []
    monkeypatch.setattr(EI, "record_resolved_prediction",
                        lambda decision_id, r, **kw: labels.append(r))
    row, = HR.evaluate_outcomes([dict(
        symbol="TCS", as_of=D, decision="AVOID", canonical_decision_id="test",
        entry=100., stop=94., target=115.,
    )])
    assert row["classification"] == "AVOIDED_LOSER"
    assert len(labels) == 1 and labels[0] < 0
```

**Suggested fix** · Use the versioned rulebook outcome for replay classification and training R. Keep close returns and excursion metrics as diagnostics.

### B-8 · P0 · CONFIRMED · The simulator reports impossible executions on the fill bar

**Location** · `core/evaluation_spec.py:119` `simulate_trade`, `data/bhavcopy_store.py:254` `_read_day`, `scan/signal_backtest.py:206` `sweep_targets` and `:894` `_run_backtest_inner`.

**Counterexample** · With entry ₹100, stop ₹90, target ₹120, and a bar of O/H/L/C ₹130/₹135/₹125/₹132, the engine fills at ₹130 and reports `WIN` at ₹120, below every trade on that bar and worth **−0.25R**. It disables gap exits on the fill bar but still permits a target print below the bar’s low. A missing high also passes `if h < entry` because `NaN < entry` is false. The bhavcopy parser keeps rows with a valid close and missing high. The target sweep has a related optimistic error because it omits opens entirely and assumes the trigger fill even when the whole bar is above the trigger.

**Deterministic test**

```python
def test_fill_bar_cannot_invent_execution_prices():
    from core.evaluation_spec import simulate_trade

    gap = simulate_trade(100, 90, 120, [135], [125], [132], opens=[130])
    missing = simulate_trade(100, 90, 120, [float("nan")], [79], [81], opens=[80])
    assert (not gap.filled or 125 <= gap.exit_price <= 135) and not missing.filled
```

**Suggested fix** · Validate finite, consistent OHLC. Define an explicit policy for opening beyond the target, and use the same open-aware simulation for target sweeps.

### B-9 · P1 · CONFIRMED · `STRICT` corporate-action coverage has no end date or artifact binding

**Location** · `research/edge_diagnosis.py:128` `coverage_start`, `:296` `tier_map`, `:469` `analyse`, `:562` `load_copy`, `data/corporate_actions_resilient.py:654` coverage persistence.

**Counterexample** · A coverage file describes one successful interval from August 28, 2021 through August 31, 2022. A decision on October 3, 2022 is still classified `STRICT` once 252 observed sessions have elapsed. `coverage_start()` discards `covered_to`. If `ca_events.json` is missing, `load_copy()` also returns raw frames alongside the stale coverage file, and the same trade still enters `STRICT`. Coverage and event files are written separately, with coverage written first.

**Deterministic test**

```python
def test_strict_tier_ends_with_verified_coverage():
    import pandas as pd
    from research import edge_diagnosis as ED

    days = pd.bdate_range("2021-08-30", "2022-10-03").strftime("%Y-%m-%d")
    coverage = {"windows": {"one": {
        "covered_from": "2021-08-28", "covered_to": "2022-08-31", "success": True,
    }}}
    tiers = ED.tier_map(dict(zip(days, range(len(days)))), ED.coverage_start(coverage))
    assert tiers["2022-10-03"] == "UNCOVERED"
```

**Suggested fix** · Carry both coverage bounds, bind them to the exact events artifact, and require coverage through the measured exit as well as across the feature lookback.

### B-10 · P1 · CONFIRMED · Incomplete membership can be presented as complete or included in diagnosis

**Location** · `data/nse_universe.py:523` `point_in_time_universe`, `data/bhavcopy_store.py:677` `membership_rows`, `research/edge_diagnosis.py:476` `analyse`, `product/historical_replay.py:1064` and `:1136` `run_replay`.

**Counterexample** · A syntactically valid history file containing `{"symbols":[]}` returns an empty universe with `survivorship_complete=True`. A valid list also carries no coverage bounds, and generated “delisted” dates are inferred from the last bar. In edge diagnosis, an absent membership file makes every symbol’s membership `None`, but only `False` is excluded. Default replay does surface explicitly incomplete membership as `DEGRADED`, after it has already built and evaluated the session decisions.

**Deterministic test**

```python
def test_parseable_wrong_schema_is_incomplete(tmp_path, monkeypatch):
    from data import nse_universe as U

    path = tmp_path / "universe_history.json"
    path.write_text('{"symbols":[]}')
    monkeypatch.setattr(U, "get_nse_universe", lambda: ["TCS"])
    assert U.point_in_time_universe("2026-09-18", path=path)["survivorship_complete"] is False
```

**Suggested fix** · Validate the membership schema and date coverage, distinguish missing prints from confirmed delistings, and exclude unknown membership from certified statistics and replay learning.

## Verified correct

- Once the input frame is fixed, the main backtest computes per-symbol indicators on `df.iloc[:t]` and passes a separate forward slice to the simulator. The 52-week high excludes the decision bar. Historical regime buckets query the causal rolling index series at `t-1`. `scan/signal_backtest.py:151`, `:827`, `:837`, `:864` · `scan/unified_scanner.py:942`, `:982`.
- For valid bars, an already-open position honors a gap at the open before checking intraday touches. If both the active stop and target are touched after the open, the stop wins. The breakeven stop becomes active on the next bar. The main backtest deducts the modeled round-trip charge once per fill, using entry-price notional as the approximation for both legs. `core/evaluation_spec.py:135` · `scan/signal_backtest.py:850` · `core/costs.py:9`, `:34`, `:43`.
- Successful corporate-action adjustment divides OHLC and multiplies volume by the same factor, preserving price-times-volume. The standalone integrity verifier requires a nonempty event table before reporting success. `data/corporate_actions.py:95` · `core/data_integrity.py:90`.
- The holiday-bhavcopy repair rejects a mismatched embedded `DATE1` on download and parse, and `load_scan()` rejects a known holiday session. The strict calendar helper skips listed holidays and rejects a window that crosses into uncovered 2027. The bundled manifest covers 2026 only. `data/bhavcopy_store.py:192`, `:208`, `:242` · `product/scan_store.py:562` · `research/intelligence/data/nse_calendar.py:303` · `data/nse_calendar_manifest.json:4`.

## Not examined / limits

- Static review of `9636b03055961e5f53b5dfb7cfe5cfede03ba13f`. The pytest sketches were traced against the checkout and were not executed.
- Runtime data was outside scope. Actual delisted-name retention, the claimed August 28, 2021 coverage start, Kite’s upstream adjustment basis, historical publication timestamps, and the incidence of each defect in saved reports remain unverified.
- I traced the live regime and breadth inputs to their current-feed paths and the historical index helper to its date cutoff. I did not establish a common publication-time cutoff across live equity, index, VIX, and sector feeds. I did not audit broker locking, position sizing, Telegram execution, or alpha-ledger concurrency.