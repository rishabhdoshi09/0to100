"""The full F&O lifecycle, proved end to end with real production functions:

    scan -> rank -> option select -> paper execute -> durable position ->
    supervise -> exit -> settle -> evidence ingestion -> future ranking

and the safety properties layered on top of it:
  - a real settled PAPER_FORWARD trade can affect a LATER ranking call once
    genuine evidence threshold is met
  - historical (COUNTERFACTUAL) evidence, however large the sample, can
    never override that forward-evidence hierarchy
  - a tiny sample cannot materially alter ranking
  - rejected historical candidates are graded, not silently dropped
  - ranking_context_key stays populated end to end (candidate -> position ->
    trade row -> evidence cell). This is a SEPARATE field from the legacy
    context_key fo_snapshot_engine.py always sets for the unrelated "Forward
    evidence" UI panel -- test_real_candidate_shape_keeps_the_two_context_keys_separate
    below proves the two are never conflated, closing a real gap where an
    earlier version of this fix read the wrong field and silently never
    matured any evidence cell for a real production candidate.
  - evidence durability survives what a process restart would see (a fresh
    read of the same on-disk store, no in-memory state)
  - live money stays locked throughout

Individual halves of this are already proved in isolation elsewhere
(tests/test_fno_evidence_wiring.py proves the settlement path with
product.decision_ranking.rank() against hand-built Decision objects;
tests/test_fno_historical_loop.py proves historical classification and
checkpoint durability). This file proves the same lifecycle through the
actual functions wired into production for F&O specifically:
product.fno_ranking.rank_fno_candidates (what fo_runtime.py and
fo_paper_runtime.py._candidate_rows both call), product.fo_paper_runtime
.run_fo_paper_cycle (the real open/settle entrypoint), and
product.fno_historical_loop.run_next_batch (the real historical driver).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import product.conditional_evidence as CE
from product.evidence_class import COUNTERFACTUAL, PAPER_FORWARD
from product.fno_evidence import fno_context_key, record_fno_settlement
from product.fno_ranking import rank_fno_candidates
from product.fo_paper_runtime import run_fo_paper_cycle
from product.fo_paper_store import FoPaperStore
from product.live_execution_interlock import get_live_execution_state
from tests.test_fo_paper_runtime import _IntradayQuoteClient, _QuoteClient

IST = timezone(timedelta(hours=5, minutes=30))


def _setup(*, direction="LONG", score=82.0, nifty_alignment=3.0, sector_strength=2.5,
           atr_pct=2.0, breakout_distance_pct=1.5):
    return {
        "score": score,
        "direction": direction,
        "atr_pct": atr_pct,
        "breakout_distance_pct": breakout_distance_pct,
        "components": {"nifty_alignment": nifty_alignment, "sector_strength": sector_strength},
        "expected_move": {"holding_days": 2},
        "underlying_trade_plan": {"entry": 50.0, "stop": 40.0, "target": 70.0},
    }


def _candidate(symbol: str, *, setup: dict, option_score=86.0, instrument_token=9001) -> dict:
    """The shape product.fo_options_pipeline.evaluate_fo_opportunity returns.
    Deliberately omits selected_contract["context_key"] the way
    fo_options_pipeline.py itself never populates one -- but real candidates
    from product.fo_runtime.run_fo_directional_scan go one layer further
    through product.fo_snapshot_engine.evaluate_fo_snapshot, which DOES
    unconditionally set contract["context_key"] to a legacy, unrelated
    FOCTX_V1-format key (see test_real_candidate_shape_keeps_the_two_context_keys_separate
    below, which exercises that real layer). This fixture is only valid for
    proving the ranking_context_key path in isolation; it is not proof that
    the legacy context_key field is ever actually empty in production."""
    return {
        "symbol": symbol,
        "direction": setup["direction"],
        "decision": "PAPER_OPTION_CANDIDATE",
        "setup": setup,
        "selected_contract": {
            "symbol": f"{symbol}26OCTCE",
            "option_type": "CE",
            "eligible": True,
            "lot_size": 25,
            "ask": 50.0,
            "score": option_score,
            "instrument_token": instrument_token,
            "trade_plan": {"entry": 50.0, "stop": 40.0, "target": 70.0},
        },
        "paper_only": True,
        "live_execution_allowed": False,
    }


def _directional(candidates: list[dict]) -> dict:
    return {
        "available": True,
        "status": "READY",
        "decision": "PAPER_CANDIDATES",
        "candidates": candidates,
        "paper_only": True,
        "live_execution_allowed": False,
    }


def _settled_row(i: int, *, net_pnl: float, trade_id_prefix="LIFECYCLE") -> dict:
    return {
        "trade_id": f"{trade_id_prefix}-{i}",
        "underlying": "LOSERCO",
        "option_symbol": "LOSERCO26OCTCE",
        "entry_price": 50.0,
        "stop_price": 40.0,
        "exit_price": 50.0 + net_pnl / 10.0,
        "quantity": 10,
        "net_pnl": net_pnl,
        "exit_reason": "STOP",
        "opened_at": "2026-09-01T04:00:00+00:00",
        "settled_at": f"2026-09-{2 + (i % 25):02d}T10:00:00+00:00",
        "mfe_pct": 0.3,
        "mae_pct": -1.2,
        "production_evidence_eligible": True,
    }


def test_full_fno_lifecycle_scan_to_future_ranking(tmp_path, monkeypatch):
    evidence_path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(evidence_path))
    monkeypatch.setenv("QT_LEARNING_POLICIES", str(tmp_path / "policies.json"))

    # ---- Live money stays locked throughout every step of this lifecycle. ----
    state = get_live_execution_state()
    assert state.locked is True and state.authorized is False

    loser_setup = _setup()
    control_setup = _setup(sector_strength=0.0)  # a different regime bucket
    loser_key = fno_context_key(loser_setup)
    control_key = fno_context_key(control_setup)
    assert loser_key and control_key and loser_key != control_key

    # ---- SCAN -> RANK (before any evidence exists): raw score decides order. ----
    early_scan = [
        _candidate("LOSERCO", setup=loser_setup, instrument_token=1),
        _candidate("CONTROLCO", setup=control_setup, option_score=80.0, instrument_token=2),
    ]
    ranked_before = rank_fno_candidates(early_scan)
    assert [r["symbol"] for r in ranked_before] == ["LOSERCO", "CONTROLCO"], (
        "with no evidence yet, ranking must follow the raw setup score"
    )
    # context_key must be populated end to end even before any trade exists.
    for row in ranked_before:
        assert row["ranking_evidence"]["context_key"], "context_key must never be empty on a real candidate"

    # ---- OPTION SELECT -> PAPER EXECUTE -> DURABLE POSITION -> SUPERVISE ->
    # EXIT -> SETTLE -> EVIDENCE INGESTION, through the real entrypoint, with
    # NO context_key supplied by the candidate (the real production shape). ----
    fo_db = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 1, 10, 15, 0, tzinfo=IST)
    with FoPaperStore(fo_db) as store:
        opened = run_fo_paper_cycle(
            _directional([_candidate("LOSERCO", setup=loser_setup, instrument_token=1)]),
            client=_QuoteClient(last=50, high=50, low=50, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
    assert opened["opened_count"] == 1
    with FoPaperStore(fo_db) as store:
        positions = store.load_positions()
    assert len(positions) == 1
    position = positions[0]
    assert position["ranking_context_key"] == loser_key, (
        "the durable position must carry the real ranking-evidence context key -- "
        "context_key (no prefix) is a SEPARATE, legacy FOCTX_V1 key that "
        "fo_snapshot_engine.evaluate_fo_snapshot always sets for the unrelated "
        "'Forward evidence' UI panel and product.fno_ranking never reads"
    )
    # SUPERVISE: an open position is tracked with bars-held / mark bookkeeping.
    assert "bars_held" in position and "max_mark" in position and "min_mark" in position

    # EXIT -> SETTLE: a genuine loss (price craters through the stop).
    client = _IntradayQuoteClient(
        last=30, high=32, low=28, bid=29,
        entry_minute_rows=[{
            "date": "2026-09-01T10:15:00+05:30",
            "open": 49.0, "high": 55.0, "low": 45.0, "close": 51.0,
        }],
        intraday_rows=[{
            "date": "2026-09-01T10:16:00+05:30",
            "open": 45.0, "high": 46.0, "low": 30.0, "close": 32.0,
        }],
    )
    with FoPaperStore(fo_db) as store:
        settled_cycle = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=opened_at.replace(hour=10, minute=20),
            allow_new_entries=False,
            store=store,
            capital=200_000,
            cost_model=lambda entry, exit, qty: 25.0,
            cost_model_name="TEST_COSTS",
        )
    assert settled_cycle["settled_count"] == 1
    real_trade = settled_cycle["settled"][0]
    assert real_trade["ranking_context_key"] == loser_key, (
        "settlement must update the SAME cell the candidate was ranked against"
    )
    assert real_trade["production_evidence_eligible"] is True
    assert float(real_trade["net_pnl"]) < 0.0, "the fixture is engineered to be a real loss"

    cell_after_one = CE.read(loser_key, path=evidence_path)
    assert cell_after_one["count"] == 1, "the real production entrypoint must have recorded exactly one evidence outcome"

    # Top up to the sample floor with the SAME function run_fo_paper_cycle's
    # settlement loop calls internally (product.fno_evidence
    # .record_fno_settlement) -- already proven equivalent to a real cycle
    # by the assertion above; this only proves the sample-gate boundary
    # without twenty-nine slow simulated intraday cycles.
    from product.conditional_evidence import MIN_SAMPLE
    for i in range(MIN_SAMPLE - 1):
        update = record_fno_settlement(_settled_row(i, net_pnl=-900.0), context_key=loser_key, path=evidence_path)
        assert update is not None

    cell = CE.read(loser_key, path=evidence_path)
    assert cell["count"] == MIN_SAMPLE
    assert cell["wins"] == 0
    assert cell["expectancy_R"] < 0

    # ---- FUTURE RANKING: a LATER scan, with a brand-new instrument for the
    # same underlying/setup/regime/sector/vol/ext/confidence bucket, must now
    # rank behind the unmeasured control -- this is the actual production
    # ranking function, not a parallel equity-shaped proof. ----
    later_scan = [
        _candidate("LOSERCO", setup=loser_setup, instrument_token=99),  # new instrument, same bucket
        _candidate("CONTROLCO", setup=control_setup, option_score=80.0, instrument_token=2),
    ]
    ranked_after = rank_fno_candidates(later_scan)
    assert [r["symbol"] for r in ranked_after] == ["CONTROLCO", "LOSERCO"], (
        "a real, fully-costed, sample-sufficient run of settled F&O losses in "
        "this exact context must demote it behind an unmeasured control in a "
        "LATER scan's ranking"
    )
    demoted = next(r for r in ranked_after if r["symbol"] == "LOSERCO")
    assert demoted["base_score"] == loser_setup["score"], "the raw scorer itself is untouched"
    assert demoted["ranking_adjustment"] < 0
    assert demoted["ranking_evidence"]["reason"] == "MEASURED_NEGATIVE_EXPECTANCY"
    assert demoted["ranking_evidence"]["context_key"] == loser_key

    # ---- SAFETY: historical (COUNTERFACTUAL) evidence, however large,
    # cannot override the forward-evidence hierarchy for this ranking
    # function. Seed a large COUNTERFACTUAL sample for the CONTROL context
    # and prove rank_fno_candidates still ignores it entirely. ----
    from product.decision_chain import Outcome

    for i in range(200):
        outcome = Outcome(
            position_id=f"CF-{i}", paper_order_id=f"CF-{i}", paper_intent_id=f"CF-{i}",
            decision_id=f"CF-{i}", symbol="CONTROLCO", realized_R=-5.0,
            exit_reason="STOP", entry_session="2026-09-01",
            exit_session=f"2026-09-{2 + (i % 25):02d}",
            evidence_class=COUNTERFACTUAL,
            resolved_at=f"2026-09-{2 + (i % 25):02d}T10:00:00+00:00",
        )
        CE.record_outcome(outcome, context_key=control_key, evidence_class=COUNTERFACTUAL, path=evidence_path)

    ranked_with_historical_noise = rank_fno_candidates(later_scan)
    assert [r["symbol"] for r in ranked_with_historical_noise] == ["CONTROLCO", "LOSERCO"], (
        "200 COUNTERFACTUAL (historical replay) outcomes for CONTROLCO must "
        "never demote it -- only PAPER_FORWARD evidence may move ranking"
    )
    control_row = next(r for r in ranked_with_historical_noise if r["symbol"] == "CONTROLCO")
    assert control_row["ranking_adjustment"] == 0.0
    assert control_row["ranking_evidence"]["reason"] == "INSUFFICIENT_EVIDENCE"

    # ---- SAFETY: a tiny forward sample cannot materially alter ranking. ----
    tiny_setup = _setup(sector_strength=-4.0)  # yet another, fresh bucket
    tiny_key = fno_context_key(tiny_setup)
    for i in range(5):
        record_fno_settlement(_settled_row(i, net_pnl=-900.0, trade_id_prefix="TINY"), context_key=tiny_key, path=evidence_path)
    tiny_candidates = [
        _candidate("TINYCO", setup=tiny_setup, instrument_token=55),
        _candidate("CONTROLCO", setup=control_setup, option_score=80.0, instrument_token=2),
    ]
    tiny_ranked = rank_fno_candidates(tiny_candidates)
    tiny_row = next(r for r in tiny_ranked if r["symbol"] == "TINYCO")
    assert tiny_row["ranking_adjustment"] == 0.0
    assert tiny_row["ranking_evidence"]["reason"] == "INSUFFICIENT_EVIDENCE"

    # ---- RESTART DURABILITY: a fresh, independent read of the on-disk store
    # (exactly what a new process after a restart would see -- this module
    # keeps no in-memory cache) reproduces the same matured cell and the
    # same demoted ranking. ----
    reloaded_cell = CE.load(evidence_path)
    reloaded_key = CE.cell_key(PAPER_FORWARD, loser_key)
    assert reloaded_cell["cells"][reloaded_key]["count"] == MIN_SAMPLE
    ranked_after_restart = rank_fno_candidates(later_scan)
    assert [r["symbol"] for r in ranked_after_restart] == ["CONTROLCO", "LOSERCO"], (
        "restarting the process must not lose learned F&O evidence"
    )

    # ---- Live money is still locked at the very end of the lifecycle. ----
    assert get_live_execution_state().locked is True


def test_rejected_historical_candidates_are_graded_not_dropped(tmp_path, monkeypatch):
    """The historical half of the lifecycle: a rejected (below-conviction)
    breakout candidate is still graded (CORRECT_REJECTION/MISSED_WINNER/...),
    and that grading can never leak into the forward-evidence-only ranking
    function proved above."""
    from product.fno_historical_loop import run_next_batch
    from tests.test_fno_walk_forward_learning import _repeated_losing_breakouts

    hist_evidence_path = tmp_path / "hist_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(hist_evidence_path))
    checkpoint_path = tmp_path / "checkpoint.json"

    frame, _ = _repeated_losing_breakouts(8, start=date(2026, 1, 1))
    result = run_next_batch(
        universe=["LOSER"], history_provider=lambda s: frame if s == "LOSER" else None,
        max_sessions_per_run=999, path=checkpoint_path,
    )
    assert result["status"] == "OK"
    assert sum(result["classification_counts"].values()) == 8, (
        "every graded historical candidate -- taken or rejected -- must show "
        "up in the classification breakdown, not be silently dropped"
    )

    # This historical evidence lives in a COUNTERFACTUAL cell and must be
    # invisible to the production forward-only ranking function.
    control_setup = _setup(sector_strength=0.0)
    candidates = [_candidate("CONTROLCO", setup=control_setup, instrument_token=2)]
    ranked = rank_fno_candidates(candidates)
    assert ranked[0]["ranking_adjustment"] == 0.0


def test_real_candidate_shape_keeps_the_two_context_keys_separate(tmp_path, monkeypatch):
    """A code-review finding on an earlier version of this fix: every real
    candidate reaches product.fo_paper_runtime through
    product.fo_snapshot_engine.evaluate_fo_snapshot, which unconditionally
    sets selected_contract["context_key"] to product.fo_evidence.fo_context_key
    -- an older, unrelated, contract-level (option_type/rvol/adx/delta/dte/iv)
    bucketing that feeds ONLY the pre-existing "Forward evidence" UI panel
    (_fo_forward_evidence_overlay). The hand-built candidates used elsewhere in
    this file never set that field, so a version of the fix that read the
    wrong field back at settlement time still passed every test while being
    completely disconnected from ranking in real production. This test builds
    a candidate the same way test_fo_snapshot_engine.py does -- through the
    real evaluate_fo_snapshot_auto, with real bars/instruments/quotes, never a
    hand-built dict -- and proves the durable position and settled trade carry
    BOTH keys, correctly distinct, with evidence recorded under the ranking
    one only.
    """
    from datetime import date as _date

    from options.directional_selector import black_scholes
    from product.fo_snapshot_engine import evaluate_fo_snapshot_auto
    from tests.test_fo_snapshot_engine import _bars, _nfo, _option_quotes

    evidence_path = tmp_path / "conditional_evidence.json"
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(evidence_path))

    as_of = _date(2026, 9, 26)
    bars = _bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _nfo(spot)
    result = evaluate_fo_snapshot_auto(
        # Must match _nfo()'s hardcoded instrument "name": "TEST", or the
        # universe/future lookup fails closed with NOT_FO_UNIVERSE.
        symbol="TEST",
        daily_bars=bars,
        nfo_instruments=instruments,
        underlying_quote={
            "last_price": spot,
            "average_price": spot - 3.0,
            "depth": {"buy": [{"price": spot - 0.1}], "sell": [{"price": spot + 0.1}]},
        },
        futures_quote={"last_price": spot + 2.0, "oi": 106_000},
        previous_futures_price=spot - 10.0,
        previous_futures_oi=100_000,
        option_quotes=_option_quotes(spot, instruments),
        benchmark_20d_return_pct=1.0,
        nifty_change_pct=0.6,
        sector_relative_strength_pct=1.2,
        iv_percentile=45.0,
        as_of=as_of,
    )
    assert result["decision"] == "PAPER_OPTION_CANDIDATE"
    real_candidate = result["selected"]
    contract = real_candidate["selected_contract"]

    # The legacy field is genuinely populated -- never empty -- for a real
    # candidate, and it is NOT the ranking-evidence key format.
    legacy_key = contract["context_key"]
    assert legacy_key, "fo_snapshot_engine must always set the legacy context_key"
    assert legacy_key.startswith("FOCTX_V1|"), "legacy key format must not drift"

    ranked = rank_fno_candidates([real_candidate], path=evidence_path)
    ranking_key = ranked[0]["ranking_evidence"]["context_key"]
    assert ranking_key, "ranking must resolve a real context key for a real candidate"
    assert ranking_key != legacy_key, (
        "the two evidence systems must never accidentally share one key"
    )

    fo_db = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, 0, tzinfo=IST)
    with FoPaperStore(fo_db) as store:
        opened = run_fo_paper_cycle(
            _directional([real_candidate]),
            client=_QuoteClient(last=spot, high=spot, low=spot, bid=spot - 1.0),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
    assert opened["opened_count"] == 1
    with FoPaperStore(fo_db) as store:
        position = store.load_positions()[0]
    assert position["context_key"] == legacy_key
    assert position["ranking_context_key"] == ranking_key

    with FoPaperStore(fo_db) as store:
        settled_cycle = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=_IntradayQuoteClient(
                last=spot * 0.5, high=spot * 0.55, low=spot * 0.45, bid=spot * 0.49,
                entry_minute_rows=[{
                    "date": "2026-09-26T10:15:00+05:30",
                    "open": contract["premium"] - 1.0, "high": contract["premium"] + 5.0,
                    "low": contract["premium"] - 5.0, "close": contract["premium"],
                }],
                intraday_rows=[{
                    "date": "2026-09-26T10:16:00+05:30",
                    "open": contract["premium"] - 1.0, "high": contract["premium"],
                    "low": max(0.05, contract["premium"] * 0.3), "close": contract["premium"] * 0.5,
                }],
            ),
            now_ist=opened_at.replace(hour=10, minute=20),
            allow_new_entries=False,
            store=store,
            capital=200_000,
            cost_model=lambda entry, exit, qty: 5.0,
            cost_model_name="TEST_COSTS",
        )
    assert settled_cycle["settled_count"] == 1
    trade = settled_cycle["settled"][0]
    assert trade["context_key"] == legacy_key, "the legacy field/format must survive settlement unchanged"
    assert trade["ranking_context_key"] == ranking_key

    # Evidence must be recorded under the ranking key -- never the legacy one.
    assert CE.read(ranking_key, path=evidence_path)["count"] == 1
    assert CE.read(legacy_key, path=evidence_path)["count"] == 0
