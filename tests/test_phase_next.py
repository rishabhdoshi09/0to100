"""Focused tests for phase-next protocol freezes (network-free)."""
from research.phase_next import protocol as P
from research.phase_next.eval_utils import final_after_confirm, map_discovery_verdict


def test_snapshot_and_partitions_frozen():
    assert P.SNAPSHOT_ID == "a7a9828ec37e09e4"
    assert P.DISCOVERY_START < P.DISCOVERY_END < P.CONFIRM_START
    assert P.REVERSAL_FORMATIONS == (1, 3, 5)
    assert 60 not in P.REVERSAL_FORMATIONS  # not rejected momentum lookback


def test_final_verdict_mapping():
    assert final_after_confirm("FAIL", None) == "FAIL"
    assert final_after_confirm("PASS", "PASS") == "CONFIRMED"
    assert final_after_confirm("PASS", "FAIL") == "FAILED_CONFIRMATION"
    assert final_after_confirm("PASS", None) == "DISCOVERY_PASS_NEEDS_FUTURE_CONFIRMATION"


def test_map_discovery_negative_net_is_fail():
    assert map_discovery_verdict("PROMOTE", mean_net=-0.01, fdr_ok=True) == "FAIL"
