from datetime import datetime
from zoneinfo import ZoneInfo

from operations.zerodha_observer import BROKER_MUTATIONS_ENABLED, observation_slot

IST = ZoneInfo("Asia/Kolkata")


def _at(hour: int, minute: int, *, day: int = 3):
    # 2026-08-03 is Monday; 2026-08-08 is Saturday.
    return datetime(2026, 8, day, hour, minute, tzinfo=IST)


def test_premarket_slot_is_durable_and_deduplicated():
    due = observation_slot(_at(8, 50), set())
    assert due == ("2026-08-03:premarket", "PREMARKET")
    assert observation_slot(_at(9, 0), {due[0]}) is None


def test_intraday_slots_are_bucketed_every_fifteen_minutes():
    assert observation_slot(_at(9, 15), set()) == (
        "2026-08-03:intraday:00",
        "INTRADAY",
    )
    assert observation_slot(_at(10, 7), set()) == (
        "2026-08-03:intraday:03",
        "INTRADAY",
    )
    assert observation_slot(
        _at(10, 7), {"2026-08-03:intraday:03"}
    ) is None


def test_eod_slot_runs_once_after_market_close():
    due = observation_slot(_at(15, 40), set())
    assert due == ("2026-08-03:eod", "EOD")
    assert observation_slot(_at(18, 0), {due[0]}) is None


def test_weekends_do_not_schedule_observation_slots():
    assert observation_slot(_at(10, 0, day=8), set()) is None


def test_scheduler_explicitly_has_no_broker_mutation_capability():
    assert BROKER_MUTATIONS_ENABLED is False
