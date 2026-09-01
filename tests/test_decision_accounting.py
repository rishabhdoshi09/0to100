from __future__ import annotations

from research.intelligence import schemas as SC
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


class _EventStore(list):
    def all(self):
        return list(self)


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


def test_decision_ledger_preserves_exact_signal_levels_and_provenance():
    result = IntelligenceCycleResult(cycle_id="cycle-1", as_of_date="2026-08-28", mode="PAPER_AUTO")
    result.signals_generated = [("s1", "AAA")]
    result.allocation_decisions = [("s1", "PAUSE")]
    store = _EventStore([
        SC.CanonicalSignal(
            strategy_id="s1",
            strategy_version=3,
            rules_hash="rules-3",
            data_snapshot_id="snap-1",
            event_ts="2026-08-28",
            source="signal",
            symbol="AAA",
            entry=101.0,
            stop=96.0,
            target=114.0,
            max_hold=10,
            rationale="breakout with volume",
        )
    ])

    finalize_cycle_decisions(_Ctx(), result, store=store)

    row = result.decision_outcomes[0]
    assert row["decision"] == NOT_SELECTED
    assert row["entry"] == 101.0
    assert row["stop"] == 96.0
    assert row["target"] == 114.0
    assert row["max_hold"] == 10
    assert row["rules_hash"] == "rules-3"
    assert row["strategy_version"] == 3
    assert row["signal_record_id"]
