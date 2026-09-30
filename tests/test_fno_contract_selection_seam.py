"""Deterministic acceptance tests for the CONTRACT-SELECTION SEAM.

product.fo_options_pipeline.evaluate_fo_opportunity is the ONE place a real
F&O candidate's option contract gets chosen. Before this file's fix,
contract evidence was computed only AFTER a contract had already been picked
by raw score (product.fno_ranking annotated the already-selected contract),
so mature, validated evidence could never actually change which contract got
traded -- only how its already-fixed score was displayed. These tests prove
the real failure mode this fix closes: two genuinely eligible contracts
(never a hand-built final "selected_contract" dict) where the higher-raw-
score contract is chosen before learning, and the other is chosen after
mature, validated forward evidence demotes the first one.

Every test drives the real production path:

    evaluate_fo_snapshot_auto -> evaluate_fo_opportunity -> option candidates
    -> contract_ranking_adjustment (learned contract ranking) -> selected_contract

never a hand-built final candidate/contract dict.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import product.conditional_evidence as CE
from options.directional_selector import black_scholes
from product.decision_chain import Outcome
from product.evidence_class import COUNTERFACTUAL, PAPER_FORWARD
from product.fno_contract_evidence import (
    MIN_SAMPLE_DEMOTE,
    MIN_SAMPLE_PROMOTE,
    NEGATIVE_CAP,
    POSITIVE_CAP,
    contract_context_key,
)
from product.fo_snapshot_engine import evaluate_fo_snapshot_auto
from product.live_execution_interlock import get_live_execution_state

AS_OF = date(2026, 9, 26)

# Empirically chosen (see the module-level comment on each) so that, with
# this fixture's bars/quotes, strike 1120 CE (CONTRACT_A) and strike 1180 CE
# (CONTRACT_B) are BOTH genuinely eligible -- cleared every hard liquidity/
# risk/expiry/theta/delta-band/score gate in
# options.directional_selector.score_option_contract on raw score alone --
# with CONTRACT_A's raw score (~88) comfortably ahead of CONTRACT_B's (~70),
# and strike 1210 CE (INELIGIBLE_CONTRACT) fails DELTA_OUTSIDE_DIRECTIONAL_BAND
# / OPTION_SCORE_BELOW_THRESHOLD regardless of evidence.
CONTRACT_A_STRIKE = None  # resolved at runtime: round(spot/10)*10 - 30
CONTRACT_B_STRIKE = None  # resolved at runtime: round(spot/10)*10 + 60
INELIGIBLE_STRIKE_OFFSET = 60.0


def _bars():
    n = 90
    x = np.arange(n, dtype=float)
    close = 1000.0 + x * 1.3 + np.sin(x / 2.7) * 20.0
    high = close + 8.0
    low = close - 8.0
    volume = np.full(n, 100_000.0)
    prior_high = float(high[-21:-1].max())
    close[-1] = prior_high + 8.0
    high[-1] = close[-1] + 2.0
    low[-1] = close[-1] - 14.0
    volume[-1] = 250_000.0
    return pd.DataFrame({
        "open": close - 2.0, "high": high, "low": low, "close": close, "volume": volume,
    })


def _nfo(spot: float, strikes: list[float]) -> list[dict]:
    out = [{
        "instrument_token": 1, "tradingsymbol": "TEST26OCTFUT", "name": "TEST",
        "expiry": "2026-10-29", "strike": 0, "tick_size": 0.05, "lot_size": 50,
        "instrument_type": "FUT", "segment": "NFO-FUT", "exchange": "NFO",
    }]
    token = 2
    for strike in strikes:
        for kind in ("CE", "PE"):
            out.append({
                "instrument_token": token,
                "tradingsymbol": f"TEST26OCT{int(strike)}{kind}",
                "name": "TEST", "expiry": "2026-10-29", "strike": strike,
                "tick_size": 0.05, "lot_size": 50, "instrument_type": kind,
                "segment": "NFO-OPT", "exchange": "NFO",
            })
            token += 1
    return out


def _option_quotes(spot: float, instruments: list[dict]) -> dict:
    out = {}
    for row in instruments:
        if row["instrument_type"] not in {"CE", "PE"}:
            continue
        price = black_scholes(
            spot=spot, strike=float(row["strike"]), dte=33, iv=0.24,
            option_type=row["instrument_type"],
        )["price"]
        out[row["tradingsymbol"]] = {
            "timestamp": f"{AS_OF.isoformat()}T10:00:00+05:30",
            "last_price": price, "volume": 8000, "oi": 40000,
            "depth": {
                "buy": [{"price": max(0.05, price - 0.15)}],
                "sell": [{"price": price + 0.15}],
            },
        }
    return out


def _strikes(spot: float) -> dict[str, float]:
    atm = round(spot / 10.0) * 10.0
    return {
        "a": atm - 30.0,
        "b": atm + 30.0,
        "ineligible": atm + INELIGIBLE_STRIKE_OFFSET,
        "atm": atm,
    }


def _scan(*, path: str | None = None):
    bars = _bars()
    spot = float(bars["close"].iloc[-1])
    strikes = _strikes(spot)
    instruments = _nfo(spot, [strikes["a"], strikes["b"], strikes["ineligible"]])
    result = evaluate_fo_snapshot_auto(
        symbol="TEST",
        daily_bars=bars,
        nfo_instruments=instruments,
        underlying_quote={
            "last_price": spot, "average_price": spot - 3.0,
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
        as_of=AS_OF,
        path=path,
    )
    return result, spot, strikes


def _eligible_ce_by_strike(result: dict, strike: float) -> dict | None:
    """A learned-and-ranked eligible contract by strike -- read from
    best_contracts (which carries raw_contract_score/contract_evidence/
    learned_contract_score), never from the raw, pre-learning all_candidates.
    """
    long_row = next(row for row in result["directions"] if row["direction"] == "LONG")
    for row in long_row["options"]["best_contracts"]:
        if row["option_type"] == "CE" and float(row["strike"]) == strike:
            return row
    return None


def _raw_ce_by_strike(result: dict, strike: float) -> dict | None:
    """The raw, pre-learning scored contract (eligible or not) by strike."""
    long_row = next(row for row in result["directions"] if row["direction"] == "LONG")
    for row in long_row["options"]["all_candidates"]:
        if row["option_type"] == "CE" and float(row["strike"]) == strike:
            return row
    return None


def _seed(*, context_key: str, evidence_class: str, n: int,
          realized_R: float | list[float], path, prefix: str) -> None:
    values = realized_R if isinstance(realized_R, list) else [realized_R] * n
    assert len(values) == n
    for i, r in enumerate(values):
        outcome = Outcome(
            position_id=f"{prefix}-{i}", paper_order_id=f"{prefix}-{i}",
            paper_intent_id=f"{prefix}-{i}", decision_id=f"{prefix}-{i}",
            symbol="TEST", realized_R=r, exit_reason="TEST",
            entry_session="2026-01-01", exit_session=f"2026-01-{2 + (i % 25):02d}",
            evidence_class=evidence_class,
            resolved_at=f"2026-01-{2 + (i % 25):02d}T10:00:00+00:00",
        )
        CE.record_outcome(outcome, context_key=context_key, evidence_class=evidence_class, path=path)


def _assert_fixture_sane(result: dict, strikes: dict) -> None:
    assert result["decision"] == "PAPER_OPTION_CANDIDATE"
    contract_a = _eligible_ce_by_strike(result, strikes["a"])
    contract_b = _eligible_ce_by_strike(result, strikes["b"])
    assert contract_a is not None and contract_a["eligible"] is True
    assert contract_b is not None and contract_b["eligible"] is True
    assert contract_a["score"] > contract_b["score"] + NEGATIVE_CAP, (
        "fixture sanity: A's raw-score lead over B must be recoverable by a "
        "single full demotion, or the 'before/after' flip this test proves "
        "cannot happen with these caps"
    )


def test_before_learning_the_higher_raw_score_contract_is_selected(tmp_path, monkeypatch):
    path = str(tmp_path / "evidence.json")
    result, spot, strikes = _scan(path=path)
    _assert_fixture_sane(result, strikes)
    assert result["selected"]["selected_contract"]["strike"] == strikes["a"]
    assert result["selected"]["selected_contract"]["learned_contract_score"] == (
        result["selected"]["selected_contract"]["raw_contract_score"]
    ), "no evidence exists yet -- learned score must equal raw score"


def test_after_mature_negative_evidence_a_different_eligible_contract_is_selected(tmp_path, monkeypatch):
    """The real failure mode this fix closes: mature, validated forward
    evidence demoting the raw-score leader must change which REAL contract
    evaluate_fo_snapshot_auto selects, not just annotate the same one.
    """
    path = str(tmp_path / "evidence.json")
    before, spot, strikes = _scan(path=path)
    contract_a = _eligible_ce_by_strike(before, strikes["a"])
    key_a = contract_context_key(contract_a)

    _seed(context_key=key_a, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE,
          realized_R=-3.0, path=path, prefix="A-DEMOTE")

    after, _, _ = _scan(path=path)
    selected = after["selected"]["selected_contract"]
    assert selected["strike"] == strikes["b"], (
        "mature negative forward evidence on the raw-score leader must "
        "flip the REAL selection to the other genuinely eligible contract"
    )
    assert selected["contract_evidence"]["context_key"] != key_a
    a_after = _eligible_ce_by_strike(after, strikes["a"])
    assert a_after["contract_evidence"]["direction"] == "DEMOTE"
    assert a_after["contract_evidence"]["adjustment"] == NEGATIVE_CAP
    assert a_after["learned_contract_score"] < a_after["raw_contract_score"]
    assert a_after["learned_contract_score"] < selected["learned_contract_score"]


def test_tiny_contract_sample_cannot_change_selection(tmp_path, monkeypatch):
    path = str(tmp_path / "evidence.json")
    before, spot, strikes = _scan(path=path)
    contract_a = _eligible_ce_by_strike(before, strikes["a"])
    key_a = contract_context_key(contract_a)

    _seed(context_key=key_a, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE - 1,
          realized_R=-3.0, path=path, prefix="A-TINY")

    after, _, _ = _scan(path=path)
    assert after["selected"]["selected_contract"]["strike"] == strikes["a"], (
        "a sample below the demotion floor must never move the real selection"
    )
    a_after = _eligible_ce_by_strike(after, strikes["a"])
    assert a_after["contract_evidence"]["reason"] == "INSUFFICIENT_CONTRACT_SAMPLE"


def test_positive_evidence_can_cautiously_promote_a_contract(tmp_path, monkeypatch):
    path = str(tmp_path / "evidence.json")
    before, spot, strikes = _scan(path=path)
    contract_b = _eligible_ce_by_strike(before, strikes["b"])
    key_b = contract_context_key(contract_b)

    _seed(context_key=key_b, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_PROMOTE,
          realized_R=1.5, path=path, prefix="B-PROMOTE")

    after, _, _ = _scan(path=path)
    b_after = _eligible_ce_by_strike(after, strikes["b"])
    assert b_after["contract_evidence"]["direction"] == "PROMOTE"
    assert 0.0 < b_after["contract_evidence"]["adjustment"] <= POSITIVE_CAP
    assert b_after["learned_contract_score"] > b_after["raw_contract_score"]
    # Still not enough to beat A's much larger raw-score lead on its own
    # (fixture sanity: promotion is capped well below the A/B raw-score gap).
    assert after["selected"]["selected_contract"]["strike"] == strikes["a"]


def test_adjustment_caps_hold_under_extreme_inputs(tmp_path, monkeypatch):
    path = str(tmp_path / "evidence.json")
    before, spot, strikes = _scan(path=path)
    contract_a = _eligible_ce_by_strike(before, strikes["a"])
    contract_b = _eligible_ce_by_strike(before, strikes["b"])
    key_a = contract_context_key(contract_a)
    key_b = contract_context_key(contract_b)

    _seed(context_key=key_a, evidence_class=PAPER_FORWARD, n=200, realized_R=-50.0, path=path, prefix="A-EXTREME")
    _seed(context_key=key_b, evidence_class=PAPER_FORWARD, n=200, realized_R=50.0, path=path, prefix="B-EXTREME")

    after, _, _ = _scan(path=path)
    a_after = _eligible_ce_by_strike(after, strikes["a"])
    b_after = _eligible_ce_by_strike(after, strikes["b"])
    assert a_after["contract_evidence"]["adjustment"] == NEGATIVE_CAP
    assert b_after["contract_evidence"]["adjustment"] == POSITIVE_CAP
    assert after["selected"]["selected_contract"]["strike"] == strikes["b"]


def test_historical_counterfactual_evidence_never_changes_real_selection(tmp_path, monkeypatch):
    """No historical NSE option-chain data source exists in this repository.
    Even a huge, robust COUNTERFACTUAL cell parked under contract A's exact
    context key must be completely invisible to the real selection seam.
    """
    path = str(tmp_path / "evidence.json")
    before, spot, strikes = _scan(path=path)
    contract_a = _eligible_ce_by_strike(before, strikes["a"])
    key_a = contract_context_key(contract_a)

    _seed(context_key=key_a, evidence_class=COUNTERFACTUAL, n=500, realized_R=-50.0, path=path, prefix="A-FAKE-HISTORY")

    after, _, _ = _scan(path=path)
    assert after["selected"]["selected_contract"]["strike"] == strikes["a"], (
        "fabricated/historical option-chain evidence must never move the real selection"
    )
    a_after = _eligible_ce_by_strike(after, strikes["a"])
    assert a_after["contract_evidence"]["count"] == 0
    assert a_after["learned_contract_score"] == a_after["raw_contract_score"]


def test_ineligible_contract_never_selected_regardless_of_evidence(tmp_path, monkeypatch):
    """The learned adjustment must never make an otherwise ineligible
    contract eligible -- it can only re-order contracts that already
    cleared every hard gate on raw score alone.
    """
    path = str(tmp_path / "evidence.json")
    before, spot, strikes = _scan(path=path)
    long_row = next(row for row in before["directions"] if row["direction"] == "LONG")
    ineligible = next(
        row for row in long_row["options"]["all_candidates"]
        if row["option_type"] == "CE" and float(row["strike"]) == strikes["ineligible"]
    )
    assert ineligible["eligible"] is False, "fixture sanity: this strike must genuinely fail a hard gate"
    ineligible_key = contract_context_key(ineligible)
    assert ineligible_key

    _seed(context_key=ineligible_key, evidence_class=PAPER_FORWARD, n=200,
          realized_R=50.0, path=path, prefix="IMPOSSIBLE-PROMOTE")

    after, _, _ = _scan(path=path)
    after_long = next(row for row in after["directions"] if row["direction"] == "LONG")
    selected_strikes = {row["strike"] for row in after_long["options"]["best_contracts"]}
    assert strikes["ineligible"] not in selected_strikes, (
        "even maximal, validated positive evidence must never rescue a "
        "contract that failed a hard liquidity/risk/expiry/theta/delta/score gate"
    )
    assert after["selected"]["selected_contract"]["strike"] != strikes["ineligible"]


def test_restart_preserves_contract_evidence_and_selection(tmp_path, monkeypatch):
    path = str(tmp_path / "evidence.json")
    before, spot, strikes = _scan(path=path)
    contract_a = _eligible_ce_by_strike(before, strikes["a"])
    key_a = contract_context_key(contract_a)

    _seed(context_key=key_a, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE,
          realized_R=-3.0, path=path, prefix="A-RESTART")

    # A fresh, independent read of the on-disk store -- exactly what a new
    # process after a restart would see (no in-memory cache anywhere in this
    # chain) -- reproduces the same matured cell and the same selection.
    reloaded = CE.load(path)
    reloaded_key = CE.cell_key(PAPER_FORWARD, key_a)
    assert reloaded["cells"][reloaded_key]["count"] == MIN_SAMPLE_DEMOTE

    after_restart, _, _ = _scan(path=path)
    assert after_restart["selected"]["selected_contract"]["strike"] == strikes["b"], (
        "restarting the process must not lose learned contract evidence"
    )


def test_live_money_stays_locked_throughout(tmp_path, monkeypatch):
    path = str(tmp_path / "evidence.json")
    assert get_live_execution_state().locked is True
    before, spot, strikes = _scan(path=path)
    contract_a = _eligible_ce_by_strike(before, strikes["a"])
    key_a = contract_context_key(contract_a)
    _seed(context_key=key_a, evidence_class=PAPER_FORWARD, n=MIN_SAMPLE_DEMOTE,
          realized_R=-3.0, path=path, prefix="A-LOCK")
    after, _, _ = _scan(path=path)
    assert after["selected"]["selected_contract"]["strike"] == strikes["b"]
    assert get_live_execution_state().locked is True
    assert get_live_execution_state().authorized is False
