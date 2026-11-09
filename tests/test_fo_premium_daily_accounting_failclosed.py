"""F&O premium daily spending cap must not reset on a corrupt persisted row.

These are offline regression tests. No broker requests or live orders.
"""
import sqlite3

import pytest

from product.fo_paper_store import FoPaperStore


def _closed_row(trade_id="T1", opened_at="2026-11-09T10:00:00+05:30"):
    return {
        "trade_id": trade_id, "option_symbol": "RELIANCE26NOVCE",
        "underlying": "RELIANCE", "entry_price": 50.0,
        "quantity": 25, "opened_at": opened_at,
        "settled_at": "2026-11-09T10:30:00+05:30",
    }


def test_closed_trade_corrupt_json_blocks_new_daily_budget(tmp_path):
    with FoPaperStore(tmp_path / "fo.sqlite3") as store:
        store.append_trades([_closed_row()])
        store.conn.execute(
            "UPDATE fo_closed_trades SET payload_json = ? WHERE trade_id = ?",
            ("{invalid", "T1"),
        )
        store.conn.commit()
        with pytest.raises(ValueError, match="corrupt premium-accounting"):
            store.premium_deployed_on_session("2026-11-09")
        assert store.status()["closed_trades"] == 1


@pytest.mark.parametrize("bad", [
    {"entry_price": "not-a-number"},
    {"entry_price": 0},
    {"quantity": 0},
    {"quantity": "1.5"},
    {"quantity": None},
])
def test_invalid_same_day_amount_blocks_budget(tmp_path, bad):
    with FoPaperStore(tmp_path / "fo.sqlite3") as store:
        row = {**_closed_row(), **bad}
        store.append_trades([row])
        with pytest.raises(ValueError, match="invalid current-session"):
            store.premium_deployed_on_session("2026-11-09")


def test_missing_entry_date_blocks_unknown_daily_exposure(tmp_path):
    with FoPaperStore(tmp_path / "fo.sqlite3") as store:
        row = _closed_row()
        del row["opened_at"]
        store.append_trades([row])
        with pytest.raises(ValueError, match="missing entry date"):
            store.premium_deployed_on_session("2026-11-09")


def test_mismatched_trade_id_blocks_repeated_premium_accounting(tmp_path):
    with FoPaperStore(tmp_path / "fo.sqlite3") as store:
        store.append_trades([_closed_row()])
        store.conn.execute(
            "UPDATE fo_closed_trades SET payload_json = ? WHERE trade_id = ?",
            ('{"trade_id":"OTHER","opened_at":"2026-11-09","entry_price":50,"quantity":25}', "T1"),
        )
        store.conn.commit()
        with pytest.raises(ValueError, match="mismatched premium-accounting"):
            store.premium_deployed_on_session("2026-11-09")


def test_valid_prior_session_not_counted_and_current_session_once(tmp_path):
    with FoPaperStore(tmp_path / "fo.sqlite3") as store:
        store.append_trades([
            _closed_row("OLDER", "2026-11-06T10:00:00+05:30"),
            {**_closed_row("TODAY"), "option_symbol": "TCS26NOVCE"},
        ])
        # A split legacy book may represent the same trade both open/closed;
        # daily premium is counted exactly once by durable trade ID.
        store.replace_positions([_closed_row("TODAY")])
        assert store.premium_deployed_on_session("2026-11-09") == 1250.0
