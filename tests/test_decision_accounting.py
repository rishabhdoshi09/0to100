from __future__ import annotations

from research.intelligence.runtime.cycle_result import IntelligenceCycleResult
from research.intelligence.runtime.decision_accounting import (
    BLOCKED,
    NOT_SELECTED,
    TAKEN,
    finalize_cycle_decisions,
)


class _Ctx:
    as_of_date = "2026-08-28"
    data_snapshot_id = "snap-1"
    market_regime = "RISK_ON"

    @staticmethod
    def cycle_id():
        return "cycle-1"


class _Store(list):
    pass


def test_every_generated_signal_gets_one_terminal_decision():
    result = IntelligenceCycleResult(cycle_id="cycle-1", as_of_date="2026-08-28", mode="PAPER_AUTO")
    result.signals_generated = [
        ("s1", "AAA"),
        ("s1", "BBB"),
        ("s2", "CCC"),
    ]
    result.positions_opened = [("s1", "AAA")]
    result.blocked_target_positions = [("s1", "BBB", "TOTAL_OPEN_RISK_CAP")]
    result.allocation_decisions = [("s1", "DEPLOY"), ("s2", "PAUSE")]

    store = _Store()
    finalize_cycle_decisions(_Ctx(), result, store=store)

    by_symbol = {row["symbol"]: row for row in result.decision_outcomes}
    assert by_symbol["AAA"]["decision"] == TAKEN
    assert by_symbol["BBB"]["decision"] == BLOCKED
    assert by_symbol["BBB"]["reason"] == "TOTAL_OPEN_RISK_CAP"
    assert by_symbol["CCC"]["decision"] == NOT_SELECTED
    assert by_symbol["CCC"]["reason"] == "ALLOCATION_PAUSE"

    rejected = {(row[0], row[1], row[2]) for row in result.signals_rejected}
    assert ("BBB", "s1", "TOTAL_OPEN_RISK_CAP") in rejected
    assert ("CCC", "s2", "ALLOCATION_PAUSE") in rejected
    assert not any(row[0] == "AAA" for row in rejected)


def test_decision_accounting_is_idempotent_for_rejection_projection():
    result = IntelligenceCycleResult(cycle_id="cycle-1", as_of_date="2026-08-28", mode="PAPER_AUTO")
    result.signals_generated = [("s1", "AAA")]
    result.allocation_decisions = [("s1", "MAINTAIN")]

    finalize_cycle_decisions(_Ctx(), result, store=_Store())
    finalize_cycle_decisions(_Ctx(), result, store=_Store())

    assert result.signals_rejected == [("AAA", "s1", "ALLOCATION_MAINTAIN")]
    assert len(result.decision_outcomes) == 1
