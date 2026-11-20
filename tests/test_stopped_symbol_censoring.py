"""A stock that stops trading must not silently leave the walk-forward evidence (cross-review B-6).

The walk-forward needed a full 10-session window after every signal, so the
final signals of a symbol that was delisted or suspended were never
evaluated, even when the trade was already stopped out in the bars that do
exist. That drops observable losses of dying stocks: an optimistic bias.
Now a stopped symbol's partial window counts when the trade resolved inside
it, and is CENSORED (counted, never invented) when the trade was still open
when prices stopped. A still-trading symbol's recent signals stay pending.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _run(monkeypatch, bars_by_symbol, signal_day_index, dates):
    from data import bhavcopy_store as BS
    from research.intelligence.data import nse_calendar as CAL
    from scan import calibration_snapshot as CS, signal_backtest as SB
    from scan.unified_scanner import UnifiedScanner

    holidays = set(json.loads(Path("data/nse_holidays.json").read_text()))
    plan = SimpleNamespace(signals=["VCP"], entry=100., stop=90., target=120., confidence=0.0)
    monkeypatch.setattr(CAL, "load_holidays", lambda: holidays)
    monkeypatch.setattr(BS, "store_symbols", lambda: sorted(bars_by_symbol))
    monkeypatch.setattr(BS, "_dates_on_disk", lambda: list(dates.date))
    monkeypatch.setattr(BS, "get_ohlcv", lambda s: bars_by_symbol.get(s))
    monkeypatch.setattr(CS, "get_or_create_snapshot", lambda: {"snapshot_id": "neutral"})
    monkeypatch.setattr(SB, "_nifty_regime_series", lambda: None)
    monkeypatch.setattr(SB, "_calibration_version", lambda: "test")
    monkeypatch.setattr(UnifiedScanner, "_analyze",
                        lambda self, sym, hist: plan if hist.index[-1] == dates[signal_day_index] else None)
    trades = []
    rep = SB.run_backtest(sample_step=1, max_symbols=len(bars_by_symbol), on_trade=trades.append)
    return trades, rep["evidence_provenance"]


def _dates():
    holidays = set(json.loads(Path("data/nse_holidays.json").read_text()))
    return pd.bdate_range("2026-01-05", periods=125).difference(pd.to_datetime(sorted(holidays)))


def _flat(dates):
    return pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 1e6}, index=dates)


def test_an_observed_stop_counts_when_the_symbol_then_stops_trading(monkeypatch):
    dates = _dates()
    dying = _flat(dates[:90])
    dying.loc[dates[87], "low"] = 89.0                     # the buy-stop fills, then the stop is hit
    trades, prov = _run(monkeypatch, {"DYING": dying}, 86, dates)
    assert [t["exit_reason"] for t in trades] == ["LOSS"]
    assert prov["stopped_symbol_trades_resolved"] == 1 and prov["stopped_symbol_trades_censored"] == 0


def test_a_trade_still_open_when_prices_stop_is_censored_not_invented(monkeypatch):
    dates = _dates()
    dying = _flat(dates[:90])                              # fills at 100, never hits 90 or 120
    trades, prov = _run(monkeypatch, {"DYING": dying}, 86, dates)
    assert trades == []
    assert prov["stopped_symbol_trades_censored"] == 1 and prov["stopped_symbol_trades_resolved"] == 0


def test_a_still_trading_symbols_recent_signal_stays_pending(monkeypatch):
    dates = _dates()
    alive = _flat(dates)                                   # bars through the last official session
    alive.loc[dates[-2], "low"] = 89.0
    trades, prov = _run(monkeypatch, {"ALIVE": alive}, len(dates) - 4, dates)
    assert trades == []                                    # its full window has not happened yet
    assert prov["stopped_symbol_trades_resolved"] == 0 and prov["stopped_symbol_trades_censored"] == 0
