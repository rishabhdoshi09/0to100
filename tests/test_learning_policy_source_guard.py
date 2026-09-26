from __future__ import annotations

import pytest

from product.learning_policy_store import (
    ACTIVE,
    ELIGIBLE,
    EXPERIMENTAL,
    _status_for,
)


@pytest.mark.parametrize(
    "source",
    [
        "backtest",
        "backtest_walkforward",
        "historical_replay",
        "decision_simulation",
        "pit_replay",
        "historical_virtual_paper",
    ],
)
def test_replay_only_sources_never_become_selection_active(source):
    assert _status_for(100, 1.0, source=source) == EXPERIMENTAL


def test_forward_paper_source_can_promote_after_sample_and_edge_floors():
    status = _status_for(100, 1.0, source="paper_forward")
    assert status in {ELIGIBLE, ACTIVE}
