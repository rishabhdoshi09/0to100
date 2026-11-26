"""Cycle 11: the market report's index tiles say WHICH session they describe, and when it is stale.

23 Nov (owner's Mac): during the session the report's "Latest session" tile showed NIFTY
+1.122% (the 20 Nov close against 19 Nov, from the index store) beside the F&O context's
live -1.406% (23 Nov intraday; the session closed -1.683%). Both were right, but the tile
carried no date, so a previous session's move read as today's. With Kite logged out the
index store stopped at 23 Nov: on 26 Nov the same tile showed -1.683% as the "latest session",
two sessions behind. Each tile now carries its session (as_of), source and basis, and is
stale when that session is older than the expected latest completed session.
"""
from __future__ import annotations

import pytest

from reports import street_pulse as SP


@pytest.fixture
def prints(monkeypatch):
    def install(as_of, expected):
        monkeypatch.setattr("data.index_store.latest_index_print",
                            lambda ticker: {"price": 25366.4, "chg_pct": -1.68, "as_of": as_of,
                                            "source": "kite_index_history"} if ticker == "^NSEI" else {})
        if isinstance(expected, Exception):
            monkeypatch.setattr("data.bhavcopy_runtime.expected_latest_completed_session",
                                lambda **k: (_ for _ in ()).throw(expected))
        else:
            from datetime import date
            monkeypatch.setattr("data.bhavcopy_runtime.expected_latest_completed_session",
                                lambda **k: date.fromisoformat(expected))
        return SP._market_snapshot()["indices"][0]
    return install


def test_a_tile_two_sessions_behind_is_stale_and_dated(prints):
    tile = prints("2026-11-23", "2026-11-26")
    assert tile["as_of"] == "2026-11-23" and tile["stale"] is True
    assert tile["source"] == "kite_index_history" and tile["basis"] == "completed_session_close"


def test_a_current_tile_is_not_stale(prints):
    tile = prints("2026-11-26", "2026-11-26")
    assert tile["as_of"] == "2026-11-26" and tile["stale"] is False


def test_a_tile_without_a_session_is_stale(prints):
    assert prints("", "2026-11-26")["stale"] is True


def test_an_unknown_expected_session_is_unknown_not_fresh(prints):
    tile = prints("2026-11-26", RuntimeError("no calendar"))
    assert tile["stale"] is None and tile["as_of"] == "2026-11-26"
