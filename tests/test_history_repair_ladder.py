"""A missing Zerodha session is a fact about our login, not about the market.

History repair used to try exactly one source. Outside market hours, with no
daily Kite session, every gap stayed open and the scan walked a universe it did
not have data for — while reporting KITE_HISTORY_UNAVAILABLE and carrying on.

The repair is a gap to close, so the ladder fills the remainder rather than
picking a winner: the broker supplies what it can, the official store covers
the rest, and a public source is the last resort. Whatever no source can
supply stays missing, visibly.
"""
from __future__ import annotations

import pandas as pd
import pytest

import scan.bulk_fetcher as BF
from data.acquisition import ACQUIRED, LAST_KNOWN_GOOD_STATE


def _frame(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame({
        "open": [10.0] * rows, "high": [11.0] * rows,
        "low": [9.0] * rows, "close": [10.5] * rows, "volume": [1000] * rows,
    })


@pytest.fixture(autouse=True)
def _clean_caches(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(BF, "_kite_cache", {})
    monkeypatch.setattr(BF, "_yf_cache", {})
    monkeypatch.setattr(BF, "_bhav_symbols", lambda: set())


def test_nothing_missing_is_a_cheap_no_op(monkeypatch):
    monkeypatch.setattr(BF, "_bhav_symbols", lambda: {"AAA", "BBB"})
    report = BF.backfill_missing(["AAA", "BBB"])
    assert report["missing"] == 0
    assert report["loaded"] == 0


def test_the_broker_supplies_the_gap_when_it_can(monkeypatch):
    monkeypatch.setattr(BF, "_repair_via_kite",
                        lambda missing, client=None, now=None: ({s: _frame() for s in missing},
                                                                {"attempted": len(missing)}))
    report = BF.backfill_missing(["AAA", "BBB"])
    assert report["loaded"] == 2
    assert report["unresolved"] == 0
    assert report["sources"] == ["zerodha_kite_data_only"]
    assert report["state"] == ACQUIRED


def test_no_kite_session_no_longer_leaves_every_gap_open(monkeypatch):
    """The exact defect: the login is missing, so nothing was repaired."""
    def no_session(missing, client=None, now=None):
        raise RuntimeError("empty Kite profile")

    monkeypatch.setattr(BF, "_repair_via_kite", no_session)
    monkeypatch.setattr(BF, "_repair_via_bhavcopy", lambda missing: {s: _frame() for s in missing})

    report = BF.backfill_missing(["AAA", "BBB"])
    assert report["loaded"] == 2, "the official store should have covered the gap"
    assert report["attempted"] == 0, "the broker tried nothing; the store did the work"
    assert report["attempted_total"] == 2
    assert report["unresolved"] == 0
    assert report["sources"] == ["nse_bhavcopy"]
    assert report["attempts"][0]["source"] == "zerodha_kite_data_only"
    assert report["attempts"][0]["outcome"] != "OK"


def test_the_ladder_fills_the_remainder_rather_than_picking_a_winner(monkeypatch):
    """A source that supplies three of ten has not failed."""
    monkeypatch.setattr(BF, "_repair_via_kite",
                        lambda missing, client=None, now=None: ({"AAA": _frame()}, {"attempted": 1}))
    monkeypatch.setattr(BF, "_repair_via_bhavcopy", lambda missing: {"BBB": _frame()})
    monkeypatch.setattr(BF, "_repair_via_yfinance", lambda missing: {"CCC": _frame()})

    report = BF.backfill_missing(["AAA", "BBB", "CCC"])
    assert report["loaded"] == 3
    assert report["unresolved"] == 0
    assert report["sources"] == ["zerodha_kite_data_only", "nse_bhavcopy", "yfinance_daily"]


def test_the_walk_stops_as_soon_as_the_gap_is_closed(monkeypatch):
    reached = []

    def yf(missing):
        reached.append(1)
        return {s: _frame() for s in missing}

    monkeypatch.setattr(BF, "_repair_via_kite",
                        lambda missing, client=None, now=None: ({s: _frame() for s in missing},
                                                                {"attempted": len(missing)}))
    monkeypatch.setattr(BF, "_repair_via_yfinance", yf)
    BF.backfill_missing(["AAA"])
    assert reached == [], "no reason to call a public source once the gap is closed"


def test_what_no_source_can_supply_stays_missing(monkeypatch):
    monkeypatch.setattr(BF, "_repair_via_kite",
                        lambda missing, client=None, now=None: ({"AAA": _frame()}, {"attempted": 1}))
    monkeypatch.setattr(BF, "_repair_via_bhavcopy", lambda missing: {})
    monkeypatch.setattr(BF, "_repair_via_yfinance", lambda missing: {})

    report = BF.backfill_missing(["AAA", "BBB"])
    assert report["loaded"] == 1
    assert report["unresolved"] == 1, "the gap must stay visible, not be papered over"


def test_total_failure_is_reported_not_swallowed(monkeypatch):
    monkeypatch.setattr(BF, "_repair_via_kite",
                        lambda missing, client=None, now=None: (_ for _ in ()).throw(
                            RuntimeError("no session")))
    monkeypatch.setattr(BF, "_repair_via_bhavcopy",
                        lambda missing: (_ for _ in ()).throw(LookupError("store is current")))
    monkeypatch.setattr(BF, "_repair_via_yfinance", lambda missing: {})

    report = BF.backfill_missing(["AAA"])
    assert report["loaded"] == 0
    assert report["unresolved"] == 1
    assert report["state"] == "DATA_UNAVAILABLE"
    assert len(report["attempts"]) == 3


def test_a_public_source_contribution_is_named_in_the_report(monkeypatch):
    """Half of this came from Yahoo is exactly what must not be invisible."""
    monkeypatch.setattr(BF, "_repair_via_kite",
                        lambda missing, client=None, now=None: ({}, {"attempted": 0}))
    monkeypatch.setattr(BF, "_repair_via_bhavcopy", lambda missing: {})
    monkeypatch.setattr(BF, "_repair_via_yfinance", lambda missing: {s: _frame() for s in missing})

    report = BF.backfill_missing(["AAA"])
    assert report["sources"] == ["yfinance_daily"]
    assert report["loaded"] == 1


def test_repaired_frames_reach_the_cache_the_scanner_reads(monkeypatch):
    monkeypatch.setattr(BF, "_repair_via_kite",
                        lambda missing, client=None, now=None: ({s: _frame() for s in missing},
                                                                {"attempted": len(missing)}))
    BF.backfill_missing(["AAA"])
    assert "AAA" in BF._kite_cache


def test_an_up_to_date_store_says_so_instead_of_re_downloading(monkeypatch):
    """Re-fetching the same sessions cannot conjure a symbol that is not in them."""
    from data import bhavcopy_store as BS
    from research.intelligence.data import nse_calendar as CAL

    built: list[int] = []
    monkeypatch.setattr(BS, "is_ready", lambda: True)
    monkeypatch.setattr(BS, "store_symbols", lambda: ["ZZZ"])
    monkeypatch.setattr(BS, "latest_two_eq_sessions", lambda: ["2099-01-01", "2099-01-02"])
    monkeypatch.setattr(BS, "build_store", lambda *a, **k: built.append(1))
    monkeypatch.setattr(CAL, "load_holidays", lambda *a, **k: set())
    monkeypatch.setattr(CAL, "latest_required_session",
                        lambda *a, **k: __import__("datetime").date(2098, 1, 1))

    with pytest.raises(LookupError):
        BF._repair_via_bhavcopy(["AAA"])
    assert built == [], "an up-to-date store must not be rebuilt to prove a negative"


def test_a_behind_store_is_brought_up_to_date_before_giving_up(monkeypatch):
    from data import bhavcopy_store as BS
    from research.intelligence.data import nse_calendar as CAL

    built: list[int] = []
    monkeypatch.setattr(BS, "is_ready", lambda: True)
    monkeypatch.setattr(BS, "store_symbols", lambda: ["ZZZ"])
    monkeypatch.setattr(BS, "latest_two_eq_sessions", lambda: ["2020-01-01"])
    monkeypatch.setattr(BS, "build_store", lambda *a, **k: built.append(1))
    monkeypatch.setattr(BS, "get_ohlcv", lambda symbol: _frame())
    monkeypatch.setattr(CAL, "load_holidays", lambda *a, **k: set())
    monkeypatch.setattr(CAL, "latest_required_session",
                        lambda *a, **k: __import__("datetime").date(2026, 9, 11))

    frames = BF._repair_via_bhavcopy(["AAA"])
    assert built == [1], "a store behind the exchange should be caught up"
    assert set(frames) == {"AAA"}


def test_symbols_already_in_the_store_are_not_re_fetched(monkeypatch):
    from data import bhavcopy_store as BS

    monkeypatch.setattr(BS, "store_symbols", lambda: ["AAA"])
    assert BF._repair_via_bhavcopy(["AAA"]) == {}


def test_an_unbuilt_store_is_a_bootstrap_problem_not_a_scan_problem(monkeypatch):
    """A scan asking for three symbols must not trigger a full history build."""
    from data import bhavcopy_store as BS

    built: list[int] = []
    monkeypatch.setattr(BS, "is_ready", lambda: False)
    monkeypatch.setattr(BS, "store_symbols", lambda: [])
    monkeypatch.setattr(BS, "build_store", lambda *a, **k: built.append(1))

    with pytest.raises(LookupError, match="bootstrap"):
        BF._repair_via_bhavcopy(["AAA"])
    assert built == []
