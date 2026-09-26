from product.fo_evidence import (
    FORWARD_PAPER,
    HISTORICAL_REPLAY,
    fo_context_key,
    summarize_fo_outcomes,
)


def _rows(lane=FORWARD_PAPER, n=40):
    rows = []
    for i in range(n):
        win = i % 3 != 0
        rows.append({
            "settled": True,
            "evidence_lane": lane,
            "context_key": "CTX",
            "net_option_return_pct": 8.0 if win else -4.0,
            "production_evidence_eligible": lane == FORWARD_PAPER,
            "mfe_pct": 12.0 if win else 2.0,
            "mae_pct": -2.0 if win else -6.0,
            "false_breakout": not win,
            "settled_at": f"2026-09-{(i % 28) + 1:02d}T10:00:00",
        })
    return rows


def test_context_key_buckets_continuous_inputs():
    key = fo_context_key(
        direction="long",
        futures_oi_state="long_buildup",
        rvol=2.2,
        adx=31,
        delta=0.62,
        dte=10,
        iv_percentile=45,
    )
    assert key == "LONG|LONG_BUILDUP|RVOL_2_2.5|ADX_GE30|D_55_65|DTE_8_14|IV_NORMAL"


def test_forward_paper_can_create_probability_claim_after_minimum_n():
    result = summarize_fo_outcomes(_rows(), context_key="CTX", min_n=30)
    assert result["probability_claim_available"] is True
    assert result["win_probability_pct"] is not None
    assert result["win_probability_wilson_lb_pct"] < result["win_probability_pct"]
    assert result["expectancy_pct"] is not None
    assert result["max_drawdown_pct"] is not None
    assert result["production_influence_allowed"] is True


def test_historical_replay_never_masquerades_as_forward_probability():
    result = summarize_fo_outcomes(
        _rows(HISTORICAL_REPLAY, 100),
        context_key="CTX",
        evidence_lane=HISTORICAL_REPLAY,
        min_n=30,
    )
    assert result["n"] == 100
    assert result["probability_claim_available"] is False
    assert result["win_probability_pct"] is None
    assert result["production_influence_allowed"] is False


def test_small_forward_sample_stays_uncalibrated():
    result = summarize_fo_outcomes(_rows(n=12), context_key="CTX", min_n=30)
    assert result["n"] == 12
    assert result["probability_claim_available"] is False
    assert result["expectancy_pct"] is None


def test_gross_only_forward_rows_never_create_probability_claim():
    rows = _rows(n=40)
    for row in rows:
        row["production_evidence_eligible"] = False
        row["cost_model_status"] = "UNCONFIGURED_GROSS_ONLY"

    result = summarize_fo_outcomes(rows, context_key="CTX", min_n=30)

    assert result["observed_n"] == 40
    assert result["n"] == 0
    assert result["excluded_unpriced_costs"] == 40
    assert result["probability_claim_available"] is False
    assert result["win_probability_pct"] is None
    assert result["expectancy_pct"] is None
    assert result["production_influence_allowed"] is False
