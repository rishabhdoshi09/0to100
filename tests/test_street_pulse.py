"""Street Pulse is a scan projection — never a full-universe bhav walk on page open."""
from __future__ import annotations

from reports.street_pulse import (
    _is_confirmed_breakout,
    _losing_from_scan,
    _movers_from_scan,
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


def test_build_pulse_uses_scan_file_not_store(monkeypatch):
    monkeypatch.setattr(
        SP, "_scan_rows",
        lambda: ([{"symbol": "AAA", "price": 10, "change_pct": 5, "volume_ratio": 2.5,
                   "signals": ["BREAKOUT_52W"], "categories": ["PreBreakout"],
                   "pivot_distance_pct": 1.0, "reasons": ["tight"]}], 2000),
    )
    monkeypatch.setattr(SP, "_market_snapshot", lambda: {
        "indices": [{"name": "NIFTY 50", "price": 25000, "chg_pct": 0.4}],
        "commentary": "Choppy market — chhote positions, quick profits.",
    })
    monkeypatch.setattr(SP, "_headlines", lambda: ["Headline one"])
    SP._PULSE_CACHE["pulse"] = None
    SP._PULSE_CACHE["ts"] = 0
    pulse = build_pulse(force=True)
    assert pulse["gainers"][0]["symbol"] == "AAA"
    assert pulse["scanned"] == 2000
    assert "NIFTY" in pulse["takeaways"][0]
    assert pulse["headlines"] == ["Headline one"]

    called = {"n": 0}

    def boom():
        called["n"] += 1
        raise AssertionError("cache miss")

    monkeypatch.setattr(SP, "_scan_rows", boom)
    again = build_pulse()
    assert called["n"] == 0
    assert again["scanned"] == 2000
