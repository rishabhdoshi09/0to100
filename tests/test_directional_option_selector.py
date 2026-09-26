from options.directional_selector import (
    black_scholes,
    score_option_contract,
    select_option_contracts,
)


def _contract(symbol: str, strike: float, delta=None, *, option_type="CE"):
    row = {
        "symbol": symbol,
        "option_type": option_type,
        "strike": strike,
        "dte": 9,
        "ltp": 82.0,
        "bid": 81.5,
        "ask": 82.5,
        "volume": 5500,
        "oi": 32000,
        "iv": 24.0,
    }
    if delta is not None:
        row["delta"] = delta
    return row


def test_black_scholes_has_sane_call_and_put_greeks():
    call = black_scholes(spot=1000, strike=1000, dte=10, iv=0.25, option_type="CE")
    put = black_scholes(spot=1000, strike=1000, dte=10, iv=0.25, option_type="PE")
    assert call["price"] > 0
    assert put["price"] > 0
    assert 0 < call["delta"] < 1
    assert -1 < put["delta"] < 0
    assert call["gamma"] > 0
    assert call["theta_per_day"] < 0


def test_directional_contract_is_scored_and_scenario_repriced():
    result = score_option_contract(
        _contract("RELIANCE27SEP3050CE", 3050.0, 0.62),
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        iv_percentile=45.0,
    )
    assert result["eligible"] is True
    assert result["score"] >= 60
    assert result["score_is_probability"] is False
    assert result["scenario_model"] == "BLACK_SCHOLES_CONSTANT_IV_ESTIMATE"
    moves = {row["underlying_move_pct"]: row for row in result["scenarios"]}
    assert moves[2.0]["projected_option_return_pct"] > moves[1.0]["projected_option_return_pct"]


def test_missing_bid_ask_fails_closed_even_with_attractive_delta():
    contract = _contract("RELIANCE27SEP3050CE", 3050.0, 0.62)
    contract["bid"] = 0
    contract["ask"] = 0
    result = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
    )
    assert result["eligible"] is False
    assert "BID_ASK_UNAVAILABLE" in result["blockers"]


def test_wrong_side_and_far_otm_delta_are_rejected():
    wrong = score_option_contract(
        _contract("RELIANCE27SEP3000PE", 3000.0, -0.6, option_type="PE"),
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
    )
    assert wrong["eligible"] is False
    assert "WRONG_OPTION_SIDE" in wrong["blockers"]

    far = score_option_contract(
        _contract("RELIANCE27SEP3300CE", 3300.0, 0.18),
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
    )
    assert far["eligible"] is False
    assert "DELTA_OUTSIDE_DIRECTIONAL_BAND" in far["blockers"]


def test_selector_ranks_eligible_contracts_only():
    rows = [
        _contract("GOOD", 3050.0, 0.62),
        _contract("DEEP", 2900.0, 0.82),
        _contract("WRONG", 3050.0, -0.60, option_type="PE"),
    ]
    result = select_option_contracts(
        rows,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        iv_percentile=40.0,
    )
    assert result["eligible_count"] >= 1
    assert all(row["eligible"] for row in result["best_contracts"])
    assert all(row["option_type"] == "CE" for row in result["best_contracts"])
