import pytest

from research.autonomy.stopping_law import evaluate_realized_gain


def row(i, samples, origin="HISTORICAL_REPLAY", request="req-1"):
    return {
        "event": "REALIZED",
        "evidence_origin": origin,
        "request_id": request,
        "record_fingerprint": f"real-{i}",
        "eligible_samples": samples,
    }


def test_insufficient_evidence_continues():
    decision = evaluate_realized_gain([row(i, 0) for i in range(3)], request_id="req-1")
    assert decision.stop is False
    assert decision.reason == "INSUFFICIENT_REALIZED_EVIDENCE"


def test_sustained_zero_gain_stops_after_minimum_evidence():
    rows = [row(i, 3) for i in range(4)] + [row(i + 4, 0) for i in range(4)]
    decision = evaluate_realized_gain(rows, request_id="req-1")
    assert decision.stop is True
    assert decision.reason == "SUSTAINED_ZERO_INFORMATION_GAIN"
    assert decision.zero_yield_streak == 4


def test_improving_gain_continues():
    rows = [row(i, x) for i, x in enumerate([1, 1, 1, 1, 2, 2, 2, 2])]
    decision = evaluate_realized_gain(rows, request_id="req-1")
    assert decision.stop is False
    assert decision.reason == "CONTINUE_INFORMATION_ACQUISITION"


def test_flat_gain_is_plateau():
    rows = [row(i, 2) for i in range(8)]
    decision = evaluate_realized_gain(rows, request_id="req-1")
    assert decision.stop is True
    assert decision.reason == "REALIZED_INFORMATION_GAIN_PLATEAU"


def test_other_requests_are_not_mixed():
    rows = [row(i, 0, request="other") for i in range(20)] + [row(30, 2)]
    decision = evaluate_realized_gain(rows, request_id="req-1")
    assert decision.observations == 1
    assert decision.stop is False


def test_forward_evidence_fails_closed():
    with pytest.raises(ValueError, match="non-historical"):
        evaluate_realized_gain([row(1, 3, origin="FORWARD_PAPER")], request_id="req-1")


def test_missing_fingerprint_fails_closed():
    bad = row(1, 3)
    bad["record_fingerprint"] = ""
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_realized_gain([bad], request_id="req-1")
