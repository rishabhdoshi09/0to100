"""Street Pulse is a local projection — never a full-universe bhav walk on page open."""
from __future__ import annotations

import pandas as pd

from reports.street_pulse import (
    _is_confirmed_breakout,
    _losing_from_scan,
    _movers_from_scan,
    _movers_from_session,
    build_pulse,
)
from reports import street_pulse as SP


def test_movers_and_weak_come_from_scan_rows():
    rows = [
        {"symbol": "UP", "price": 100, "change_pct": 8.2, "volume_ratio": 3},
        {"symbol": "DOWN", "price": 50, "change_pct": -6.4, "volume_ratio": 1.1,
         "above_sma50": False, "momentum_5d": -7, "pct_below_20d_high": 12},
        {"symbol": "FLAT", "price": 80, "change_pct": 0.2, "volume_ratio": 1},
    ]
    gainers, losers = _movers_from_scan(rows, top_n=2)
    assert gainers[0]["symbol"] == "UP"
    assert losers[0]["symbol"] == "DOWN"
    weak = _losing_from_scan(rows)
    assert weak and weak["symbol"] == "DOWN"


def test_confirmed_breakout_accepts_scan_codes():
    assert _is_confirmed_breakout({"signals": ["BREAKOUT_52W"]})
    assert _is_confirmed_breakout({"signals": ["52-week high breakout"]})
    assert _is_confirmed_breakout({"breakout_grade": "A", "signals": []})
    assert not _is_confirmed_breakout({"signals": ["MOMENTUM"]})


def test_session_movers_use_two_csvs_not_get_ohlcv(monkeypatch):
    called = {"ohlcv": 0, "symbols": 0}

    def boom_ohlcv(*_a, **_k):
        called["ohlcv"] += 1
        raise AssertionError("get_ohlcv must not run on Pulse open")

    def boom_symbols():
        called["symbols"] += 1
        raise AssertionError("store_symbols must not run on Pulse open")

    today = pd.DataFrame({
        "symbol": ["LIQ", "THIN", "DROP", "CA"],
        "close": [100.0, 10.0, 50.0, 5.0],
        "volume": [1_000_000.0, 100.0, 2_000_000.0, 5_000_000.0],
    })
    prev = pd.DataFrame({
        "symbol": ["LIQ", "THIN", "DROP", "CA"],
        "close": [90.0, 9.0, 60.0, 50.0],
        "volume": [800_000.0, 100.0, 2_000_000.0, 5_000_000.0],
    })
    monkeypatch.setattr(
        "data.bhavcopy_store.latest_two_eq_sessions",
        lambda: (today, prev, None),
    )
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", boom_ohlcv)
    monkeypatch.setattr("data.bhavcopy_store.store_symbols", boom_symbols)
    gainers, losers = _movers_from_session(top_n=2)
    assert called["ohlcv"] == 0
    assert called["symbols"] == 0
    assert gainers[0]["symbol"] == "LIQ"
    assert gainers[0]["chg_pct"] == 11.11
    assert losers[0]["symbol"] == "DROP"
    assert losers[0]["chg_pct"] == round((50 / 60 - 1) * 100, 2)
    assert all(row["symbol"] != "CA" for row in gainers + losers)


def test_build_pulse_uses_scan_file_not_store(monkeypatch):
    monkeypatch.setattr(
        SP, "_scan_rows_latest",
        lambda: ([{"symbol": "AAA", "price": 10, "change_pct": 5, "volume_ratio": 2.5,
                   "signals": ["BREAKOUT_52W"], "categories": ["PreBreakout"],
                   "pivot_distance_pct": 1.0, "reasons": ["tight"]}], 2000),
    )
    monkeypatch.setattr(SP, "_market_snapshot", lambda: {
        "indices": [{"name": "NIFTY 50", "price": 25000, "chg_pct": 0.4}],
        "commentary": "Choppy market — chhote positions, quick profits.",
    })
    monkeypatch.setattr(SP, "_headlines", lambda: ["Headline one"])
    monkeypatch.setattr(SP, "_movers_from_bhav", lambda: ([], []))
    SP._PULSE_CACHE["pulse"] = None
    SP._PULSE_CACHE["ts"] = 0
    pulse = build_pulse(force=True)
    assert pulse["gainers"][0]["symbol"] == "AAA"
    assert pulse["scanned"] == 2000
    assert "NIFTY" in pulse["takeaways"][0]
    assert pulse["headlines"] == ["Headline one"]
    assert pulse["as_of_ist"]

    called = {"n": 0}

    def boom():
        called["n"] += 1
        raise AssertionError("cache miss")

    monkeypatch.setattr(SP, "_scan_rows_latest", boom)
    again = build_pulse()
    assert called["n"] == 0
    assert again["scanned"] == 2000


def test_snapshot_skips_option_chain_and_google(monkeypatch):
    monkeypatch.setattr(
        "data.index_store.latest_index_print",
        lambda ticker: {"price": 25000.0, "chg_pct": 0.4} if ticker == "^NSEI" else {},
    )
    monkeypatch.setattr(
        "data.live_quotes.get_index_quotes",
        lambda names: (_ for _ in ()).throw(AssertionError("google/kite quotes")),
    )
    snap = SP._market_snapshot()
    assert snap["indices"][0]["name"] == "NIFTY 50"
    assert "options" not in snap
