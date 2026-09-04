from product.operator_language import explain_opportunity


def test_scanner_buy_verdict_stays_research_candidate_without_committee_decision():
    card = explain_opportunity({
        "symbol": "ABC",
        "status": "Ready to trade",
        "verdict": "BUY",
        "setup_label": "Momentum breakout",
    })
    assert card["label"] == "Research candidate"
    assert "did not make an investment BUY decision" in card["meaning"]


def test_committee_buy_is_the_home_truth_even_when_broker_is_blocked():
    card = explain_opportunity({
        "symbol": "ABC",
        "decision": "BUY",
        "entry_state": "ENTER_NOW",
        "execution_state": "BLOCKED_BROKER_AUTH",
        "reason_code": "COMMITTEE_BUY",
    })
    assert card["label"] == "BUY — execution blocked"
    assert "broker execution waits" in card["meaning"]


def test_wait_and_avoid_are_committee_language():
    wait = explain_opportunity({"symbol": "ABC", "decision": "WAIT", "reason_code": "ENTRY_TOO_EXTENDED"})
    avoid = explain_opportunity({"symbol": "XYZ", "decision": "AVOID", "reason_code": "DD_GATE_FAILED"})
    assert wait["label"] == "WAIT"
    assert "Waiting" in wait["meaning"]
    assert avoid["label"] == "AVOID"
    assert "company check did not pass" in avoid["meaning"]


def test_no_judgment_is_not_rendered_as_rejection():
    card = explain_opportunity({"symbol": "BAD", "decision": "NO_JUDGMENT", "reason_code": "INVALID_SYMBOL"})
    assert card["label"] == "NO JUDGMENT"
    assert "not a real stock" in card["meaning"]
