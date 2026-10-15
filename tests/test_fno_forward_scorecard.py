"""Only settled, fully costed and observed option paper trades count as forward edge."""
from product.fno_forward_scorecard import forward_scorecard


def trade(i, *, net=100.0, cost=10.0, kind="CE", eligible=True):
    return {
        "trade_id": str(i), "evidence_lane": "FORWARD_PAPER",
        "production_evidence_eligible": eligible,
        "cost_model_status": "CONFIGURED:TEST",
        "entry_price": 100.0, "stop_price": 90.0, "quantity": 10,
        "gross_pnl": net + cost, "costs": cost, "net_pnl": net,
        "option_type": kind, "entry_delta": 0.61, "entry_dte": 14,
    }


def test_valid_paper_forward_report_segments_and_costs():
    rows = [trade(i, net=(100 if i % 3 else -50)) for i in range(45)]
    report = forward_scorecard(rows)
    assert report["eligible_trades"] == 45
    assert report["overall"]["total_costs"] == 450
    assert report["overall"]["net_expectancy_R"] > 0
    assert report["overall"]["verdict"] == "PAPER_EDGE_CANDIDATE_REQUIRES_INDEPENDENT_REVIEW"
    assert report["by_setup_type"]["20D_BREAKOUT_LONG"]["n"] == 45
    assert report["by_contract_shape"]["CE|DELTA_050_TO_070|DTE_8_TO_21"]["n"] == 45
    assert report["live_execution_allowed"] is False


def test_ambiguous_uncosted_duplicate_and_counterfactual_never_count():
    rows = [trade(1), trade(1), trade(2, eligible=False),
            {**trade(3), "cost_model_status": "UNCONFIGURED_GROSS_ONLY"},
            {**trade(4), "evidence_lane": "COUNTERFACTUAL"},
            {**trade(5), "net_pnl": 100_000}]
    report = forward_scorecard(rows)
    assert report["eligible_trades"] == 1
    assert report["overall"]["verdict"] == "INSUFFICIENT_FORWARD_EVIDENCE"
    assert report["excluded"]["NOT_COSTED_FULLY_OBSERVED_FORWARD"] == 3
    assert report["excluded"]["MISSING_OR_DUPLICATE_TRADE_ID"] == 1
    assert report["excluded"]["COST_RECONCILIATION_FAILED"] == 1


def test_loss_making_sample_can_never_be_promoted_as_positive():
    report = forward_scorecard([trade(i, net=-20) for i in range(50)])
    assert report["overall"]["net_expectancy_R"] < 0
    assert report["overall"]["verdict"] == "NO_DEMONSTRATED_POSITIVE_FORWARD_EDGE"
