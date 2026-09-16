from product.autonomous_loop import _should_download


def test_completed_due_diligence_is_consumed_without_recursive_download() -> None:
    prev = {"scan_run_id": "scan-1"}

    assert _should_download("MARKET_SCAN", prev, "scan-1") is False
    assert _should_download("DUE_DILIGENCE_ACQUIRE", prev, "scan-1") is False


def test_explicit_research_triggers_still_allow_download() -> None:
    prev = {"scan_run_id": "scan-1"}

    assert _should_download("research_cycle", prev, "scan-1") is True
    assert _should_download("pipeline", prev, "scan-1") is True
    assert _should_download("manual", prev, "scan-1") is True


def test_repeated_outcome_resolution_does_not_download() -> None:
    prev = {"scan_run_id": "scan-1"}

    assert _should_download("outcome_resolution", prev, "scan-1") is False
