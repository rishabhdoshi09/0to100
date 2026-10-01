"""First-class F&O Evolution tests.

No historical option-chain evidence is fabricated. Underlying shadows are
graded from official underlying bars; contract shadows are PAPER_FORWARD-only.
"""
from __future__ import annotations

import pandas as pd
import pytest

from product.evidence_class import COUNTERFACTUAL, PAPER_FORWARD
from product.evolution import grading
from product.evolution import policy_registry as PR
from product.evolution import shadow_decisions as SD
from product.evolution.fno_adapter import (
    contract_policy_adjustment,
    run_contract_tournament,
    run_underlying_tournament,
)
from product.fo_options_pipeline import _rerank_by_learned_score


def _contract(symbol: str, score: float, *, liquidity: float, delta_fit: float = 20.0, theta: float = 8.0):
    return {
        "symbol": symbol,
        "option_type": "CE",
        "strike": 100.0,
        "expiry": "2026-10-08",
        "dte": 7,
        "instrument_token": 101 if symbol == "A" else 202,
        "moneyness": "ATM",
        "delta": 0.60,
        "iv_percentile": 40.0,
        "spread_pct": 0.8,
        "oi": 5000,
        "volume": 5000,
        "score": score,
        "components": {
            "delta_fit": delta_fit,
            "liquidity": liquidity,
            "theta": theta,
            "iv": 8.0,
            "expiry_fit": 8.0,
            "expected_payoff": 12.0,
        },
        "projected_return_at_expected_move_pct": 15.0,
        "gamma_delta_change_for_1pct_move": 0.01,
        "vega_pct_of_premium_per_vol_point": 1.0,
        "trade_plan": {"entry": 10.0, "stop": 8.0, "target": 14.0},
        "eligible": True,
        "blockers": [],
        "paper_only": True,
        "live_execution_allowed": False,
    }


def _setup(direction="LONG"):
    return {
        "direction": direction,
        "score": 80.0,
        "tradable": True,
        "components": {
            "futures_oi": 8.0,
            "sector_strength": 5.0,
            "nifty_alignment": 4.0,
        },
        "futures_oi_state": "LONG_BUILDUP" if direction == "LONG" else "SHORT_BUILDUP",
        "breakout_distance_pct": 0.8,
        "underlying_trade_plan": (
            {"entry": 100.0, "stop": 95.0, "target": 110.0}
            if direction == "LONG"
            else {"entry": 100.0, "stop": 105.0, "target": 90.0}
        ),
        "expected_move": {
            "horizon": "1_TO_2D",
            "holding_days": 1,
            "exit_policy": "SESSION_HOLD",
        },
    }


def test_contract_champion_reorders_only_eligible_contracts(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="CONTRACT_CHAMP", domain=PR.FNO_CONTRACT,
        hypothesis="liquidity-heavy", weights={"liquidity_mult": 3.0},
        status=PR.CHAMPION,
    )
    a = _contract("A", 80.0, liquidity=1.0)
    b = _contract("B", 75.0, liquidity=25.0)
    ranked = _rerank_by_learned_score([a, b], setup=_setup(), path=str(tmp_path / "evidence.json"))
    assert ranked[0]["symbol"] == "B"
    assert ranked[0]["final_contract_score"] > ranked[1]["final_contract_score"]

    ineligible = dict(b, eligible=False, blockers=["BID_ASK_TOO_WIDE"])
    verdict = contract_policy_adjustment(ineligible, PR.current_champion(PR.FNO_CONTRACT))
    assert verdict["eligible"] is False
    assert verdict["adjustment"] == 0.0


def test_contract_challenger_can_choose_different_already_eligible_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="C_BASE", domain=PR.FNO_CONTRACT, hypothesis="neutral",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="C_LIQ", domain=PR.FNO_CONTRACT, hypothesis="prefer liquidity",
        weights={"liquidity_mult": 3.0}, status=PR.CHALLENGER,
        parent_policy_id="C_BASE",
    )
    a = dict(_contract("A", 80.0, liquidity=1.0), learned_contract_score=80.0)
    b = dict(_contract("B", 75.0, liquidity=25.0), learned_contract_score=75.0)

    result = run_contract_tournament(
        underlying_symbol="RELIANCE",
        direction="LONG",
        setup=_setup(),
        eligible_contracts=[a, b],
        selected_contract=a,
        as_of="2026-09-30",
    )
    assert result["results"]["C_BASE"]["contract_symbol"] == "A"
    assert result["results"]["C_LIQ"]["contract_symbol"] == "B"
    assert result["results"]["C_LIQ"]["grading_mode"] == "PAPER_FORWARD_CONTRACT_ONLY"


def test_contract_shadow_refuses_historical_option_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="C_BASE", domain=PR.FNO_CONTRACT, hypothesis="neutral",
        weights={}, status=PR.CHAMPION,
    )
    a = dict(_contract("A", 80.0, liquidity=10.0), learned_contract_score=80.0)
    result = run_contract_tournament(
        underlying_symbol="RELIANCE", direction="LONG", setup=_setup(),
        eligible_contracts=[a], selected_contract=a, as_of="2026-09-30",
    )
    row = result["results"]["C_BASE"]
    assert grading.grade_shadow_decision(row) is None

    with pytest.raises(ValueError, match="PAPER_FORWARD"):
        grading.record_observed_contract_shadow_outcome(
            row["shadow_id"], realized_R=1.2, evidence_class=COUNTERFACTUAL,
            observed_source="synthetic-history",
        )

    graded = grading.record_observed_contract_shadow_outcome(
        row["shadow_id"], realized_R=1.2, evidence_class=PAPER_FORWARD,
        observed_source="observed-paper-option-mark",
    )
    assert graded["counterfactual_R"] == 1.2
    assert graded["evidence_class"] == PAPER_FORWARD


def test_short_fno_underlying_shadow_grades_directionally_from_official_bars(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="U_BASE", domain=PR.FNO_UNDERLYING, hypothesis="neutral",
        weights={}, status=PR.CHAMPION,
    )
    candidate = {
        "symbol": "RELIANCE",
        "direction": "SHORT",
        "setup": _setup("SHORT"),
        "pre_evolution_ranking_score": 80.0,
        "ranking_score": 80.0,
    }
    result = run_underlying_tournament(
        [candidate], selected=candidate, as_of="2026-09-30",
    )
    row = result["results"]["U_BASE"]["SHORT"]

    idx = pd.to_datetime(["2026-09-30", "2026-10-01"])
    bars = pd.DataFrame(
        {
            "high": [101.0, 102.0],
            "low": [99.0, 89.0],
            "close": [100.0, 90.0],
        },
        index=idx,
    )
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda _symbol: bars)
    graded = grading.grade_shadow_decision(row)
    assert graded is not None
    assert graded["counterfactual_R"] > 0
    assert graded["classification"] == "WINNER_TAKEN"



def test_contract_challenger_is_graded_from_its_own_forward_option_bars(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from product.evolution import scorecard

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    PR.register_policy(
        policy_id="OBS_BASE", domain=PR.FNO_CONTRACT, hypothesis="neutral",
        weights={}, status=PR.CHAMPION,
    )
    PR.register_policy(
        policy_id="OBS_LIQ", domain=PR.FNO_CONTRACT, hypothesis="prefer liquidity",
        weights={"liquidity_mult": 3.0}, status=PR.CHALLENGER,
        parent_policy_id="OBS_BASE",
    )
    a = dict(_contract("A", 80.0, liquidity=1.0), learned_contract_score=80.0)
    b = dict(_contract("B", 75.0, liquidity=25.0), learned_contract_score=75.0)
    result = run_contract_tournament(
        underlying_symbol="RELIANCE",
        direction="LONG",
        setup=_setup(),
        eligible_contracts=[a, b],
        selected_contract=a,
        as_of="2026-09-30",
    )
    assert result["results"]["OBS_BASE"]["contract_symbol"] == "A"
    assert result["results"]["OBS_LIQ"]["contract_symbol"] == "B"

    def observed_bars(token, *, from_dt, to_dt, client=None, interval="minute"):
        # Contract A loses; independently-observed Contract B hits target.
        if int(token) == 101:
            high, low, close = 10.5, 7.5, 8.0
        else:
            high, low, close = 14.5, 9.5, 14.0
        return [{
            "timestamp": (from_dt + timedelta(minutes=1)).isoformat(),
            "open": 10.0,
            "high": high,
            "low": low,
            "close": close,
            "last_price": close,
        }]

    monkeypatch.setattr("data.nfo_market.read_option_intraday_bars", observed_bars)
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata")) + timedelta(days=2)
    graded = grading.grade_pending_contract_decisions(
        client=object(), now_ist=now_ist,
    )
    by_policy = {row["policy_id"]: row for row in graded}
    assert by_policy["OBS_BASE"]["counterfactual_R"] < 0
    assert by_policy["OBS_LIQ"]["counterfactual_R"] > 0
    assert by_policy["OBS_BASE"]["observed_source"] == "NFO_INTRADAY_FORWARD_SHADOW"
    assert by_policy["OBS_LIQ"]["evidence_class"] == PAPER_FORWARD

    paired = scorecard.paired_comparison(
        "OBS_BASE", "OBS_LIQ", domain=PR.FNO_CONTRACT,
    )
    assert paired["paired_snapshots"] == 1
    assert paired["incremental_expectancy_R"] > 0
