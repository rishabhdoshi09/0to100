from product.fo_evidence import (
    FO_CONTEXT_SCHEMA_VERSION,
    FORWARD_PAPER,
    HISTORICAL_REPLAY,
    fo_context_key,
    fo_evidence_coverage,
    summarize_fo_outcomes,
)


def _canonical_context() -> str:
    return (
        f"{FO_CONTEXT_SCHEMA_VERSION}|LONG|LONG_BUILDUP|RVOL_2_2.5|"
        "ADX_GE30|D_55_65|DTE_8_14|IV_NORMAL"
    )


def _rows(lane=FORWARD_PAPER, n=40, *, context_key: str | None = None):
    rows = []
    for i in range(n):
        win = i % 3 != 0
        rows.append({
            "settled": True,
            "evidence_lane": lane,
            "context_key": context_key or _canonical_context(),
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
    assert key == (
        f"{FO_CONTEXT_SCHEMA_VERSION}|LONG|LONG_BUILDUP|RVOL_2_2.5|"
        "ADX_GE30|D_55_65|DTE_8_14|IV_NORMAL"
    )


def test_forward_paper_can_create_probability_claim_after_minimum_n():
    result = summarize_fo_outcomes(_rows(), context_key=_canonical_context(), min_n=30)
    assert result["probability_claim_available"] is True
    assert result["win_probability_pct"] is not None
    assert result["win_probability_wilson_lb_pct"] < result["win_probability_pct"]
    assert result["expectancy_pct"] is not None
    assert result["max_drawdown_pct"] is not None
    assert result["production_influence_allowed"] is True


def test_historical_replay_never_masquerades_as_forward_probability():
    result = summarize_fo_outcomes(
        _rows(HISTORICAL_REPLAY, 100),
        context_key=_canonical_context(),
        evidence_lane=HISTORICAL_REPLAY,
        min_n=30,
    )
    assert result["n"] == 100
    assert result["probability_claim_available"] is False
    assert result["win_probability_pct"] is None
    assert result["production_influence_allowed"] is False


def test_small_forward_sample_stays_uncalibrated():
    result = summarize_fo_outcomes(_rows(n=12), context_key=_canonical_context(), min_n=30)
    assert result["n"] == 12
    assert result["probability_claim_available"] is False
    assert result["expectancy_pct"] is None


def test_gross_only_forward_rows_never_create_probability_claim():
    rows = _rows(n=40)
    for row in rows:
        row["production_evidence_eligible"] = False
        row["cost_model_status"] = "UNCONFIGURED_GROSS_ONLY"

    result = summarize_fo_outcomes(rows, context_key=_canonical_context(), min_n=30)

    assert result["observed_n"] == 40
    assert result["n"] == 0
    assert result["excluded_unpriced_costs"] == 40
    assert result["probability_claim_available"] is False
    assert result["win_probability_pct"] is None
    assert result["expectancy_pct"] is None
    assert result["production_influence_allowed"] is False



def test_broader_evidence_coverage_is_counts_only_and_fully_costed_forward_only():
    exact = _canonical_context()
    same_thesis_other_contract = (
        f"{FO_CONTEXT_SCHEMA_VERSION}|LONG|LONG_BUILDUP|RVOL_2_2.5|ADX_GE30|D_65_75|DTE_15_30|IV_HIGH"
    )
    same_direction_oi = (
        f"{FO_CONTEXT_SCHEMA_VERSION}|LONG|LONG_BUILDUP|RVOL_1.5_2|ADX_20_25|D_55_65|DTE_8_14|IV_NORMAL"
    )
    same_direction = (
        f"{FO_CONTEXT_SCHEMA_VERSION}|LONG|SHORT_COVERING|RVOL_2_2.5|ADX_GE30|D_55_65|DTE_8_14|IV_NORMAL"
    )
    other_direction = (
        f"{FO_CONTEXT_SCHEMA_VERSION}|SHORT|SHORT_BUILDUP|RVOL_2_2.5|ADX_GE30|D_55_65|DTE_8_14|IV_NORMAL"
    )

    def row(context_key, *, eligible=True, lane=FORWARD_PAPER):
        return {
            "settled": True,
            "evidence_lane": lane,
            "context_key": context_key,
            "net_option_return_pct": 3.0,
            "production_evidence_eligible": eligible,
        }

    outcomes = (
        [row(exact) for _ in range(3)]
        + [row(same_thesis_other_contract) for _ in range(2)]
        + [row(same_direction_oi) for _ in range(2)]
        + [row(same_direction)]
        + [row(other_direction) for _ in range(4)]
        + [row(exact, eligible=False) for _ in range(5)]
        + [row(exact, lane=HISTORICAL_REPLAY) for _ in range(6)]
    )

    coverage = fo_evidence_coverage(outcomes, context_key=exact, min_n=30)

    assert coverage["valid_context"] is True
    assert coverage["exact_n"] == 3
    assert coverage["thesis_n"] == 5
    assert coverage["direction_oi_n"] == 7
    assert coverage["direction_n"] == 8
    assert coverage["remaining_to_exact_min_n"] == 27
    assert coverage["research_only"] is True
    assert coverage["counts_only"] is True
    assert coverage["probability_claim_available"] is False
    assert coverage["production_influence_allowed"] is False


def test_evidence_coverage_rejects_noncanonical_context_shape():
    coverage = fo_evidence_coverage(_rows(), context_key="CTX", min_n=30)

    assert coverage["valid_context"] is False
    assert coverage["exact_n"] == 0
    assert coverage["thesis_n"] == 0
    assert coverage["direction_oi_n"] == 0
    assert coverage["direction_n"] == 0
    assert coverage["production_influence_allowed"] is False



def test_legacy_unversioned_context_cannot_create_production_probability():
    legacy = "LONG|LONG_BUILDUP|RVOL_2_2.5|ADX_GE30|D_55_65|DTE_8_14|IV_NORMAL"
    result = summarize_fo_outcomes(
        _rows(n=100, context_key=legacy),
        context_key=legacy,
        min_n=30,
    )

    assert result["n"] == 100
    assert result["valid_context"] is False
    assert result["context_schema_version"] == FO_CONTEXT_SCHEMA_VERSION
    assert result["probability_claim_available"] is False
    assert result["production_influence_allowed"] is False


def test_broader_coverage_never_crosses_context_schema_versions():
    current = _canonical_context()
    legacy_like = current.replace(FO_CONTEXT_SCHEMA_VERSION, "FOCTX_V0", 1)
    outcomes = (
        _rows(n=5, context_key=current)
        + _rows(n=40, context_key=legacy_like)
    )

    coverage = fo_evidence_coverage(outcomes, context_key=current, min_n=30)

    assert coverage["valid_context"] is True
    assert coverage["context_schema_version"] == FO_CONTEXT_SCHEMA_VERSION
    assert coverage["exact_n"] == 5
    assert coverage["direction_n"] == 5
    assert coverage["production_influence_allowed"] is False



def test_path_uncertain_forward_rows_never_create_probability_claim():
    rows = _rows(n=40)
    for row in rows:
        row["production_evidence_eligible"] = False
        row["cost_model_status"] = "CONFIGURED:TEST_COSTS"
        row["entry_minute_status"] = "AMBIGUOUS_BOUNDARY_TOUCH"
        row["path_observation_complete"] = False
        row["evidence_exclusion_reason"] = "ENTRY_MINUTE_AMBIGUOUS_BOUNDARY_TOUCH"

    result = summarize_fo_outcomes(rows, context_key=_canonical_context(), min_n=30)

    assert result["observed_n"] == 40
    assert result["n"] == 0
    assert result["excluded_unpriced_costs"] == 0
    assert result["excluded_path_observation"] == 40
    assert result["excluded_other_ineligible"] == 0
    assert result["probability_claim_available"] is False
    assert result["win_probability_pct"] is None
    assert result["expectancy_pct"] is None
    assert result["production_influence_allowed"] is False
