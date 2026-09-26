from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

from options.directional_selector import black_scholes
from product import fo_runtime


AS_OF = date(2026, 9, 26)


def _breakout_bars():
    n = 90
    x = np.arange(n, dtype=float)
    close = 1000.0 + x * 1.0 + np.sin(x / 2.4) * 10.0
    high = close + 7.0
    low = close - 7.0
    volume = np.full(n, 100_000.0)
    prior_high = float(high[-21:-1].max())
    close[-1] = prior_high + 7.0
    high[-1] = close[-1] + 2.0
    low[-1] = close[-1] - 13.0
    volume[-1] = 250_000.0
    return pd.DataFrame({
        "open": close - 2.0,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _quiet_bars():
    frame = _breakout_bars()
    frame.loc[frame.index[-1], "close"] = float(frame["close"].iloc[-2])
    frame.loc[frame.index[-1], "high"] = float(frame["high"].iloc[-2])
    frame.loc[frame.index[-1], "low"] = float(frame["low"].iloc[-2])
    frame.loc[frame.index[-1], "volume"] = 100_000.0
    return frame


def _report(spot):
    return SimpleNamespace(underlyings=(
        SimpleNamespace(
            symbol="TEST",
            future_symbol="TEST26OCTFUT",
            instrument_token=1,
            lot_size=50,
            expiry="2026-10-29",
        ),
    ))


def _instruments(spot):
    strike = round(spot / 10.0) * 10.0
    return [
        {
            "instrument_token": 1, "tradingsymbol": "TEST26OCTFUT",
            "name": "TEST", "expiry": "2026-10-29", "strike": 0,
            "tick_size": 0.05, "lot_size": 50, "instrument_type": "FUT",
            "segment": "NFO-FUT", "exchange": "NFO",
        },
        {
            "instrument_token": 2, "tradingsymbol": "TEST26OCTCE",
            "name": "TEST", "expiry": "2026-10-29", "strike": strike,
            "tick_size": 0.05, "lot_size": 50, "instrument_type": "CE",
            "segment": "NFO-OPT", "exchange": "NFO",
        },
        {
            "instrument_token": 3, "tradingsymbol": "TEST26OCTPE",
            "name": "TEST", "expiry": "2026-10-29", "strike": strike,
            "tick_size": 0.05, "lot_size": 50, "instrument_type": "PE",
            "segment": "NFO-OPT", "exchange": "NFO",
        },
    ]


class _Client:
    def __init__(self, spot, instruments):
        self.spot = spot
        self.instruments_rows = instruments
        self.quote_calls = []

    def quote(self, keys):
        self.quote_calls.append(list(keys))
        out = {}
        for key in keys:
            if key == "NSE:NIFTY 50":
                out[key] = {"last_price": 22000.0, "ohlc": {"close": 21900.0}}
            elif key == "NSE:TEST":
                out[key] = {
                    "last_price": self.spot,
                    "average_price": self.spot - 3.0,
                    "depth": {
                        "buy": [{"price": self.spot - 0.1}],
                        "sell": [{"price": self.spot + 0.1}],
                    },
                }
            elif key == "NFO:TEST26OCTFUT":
                out[key] = {"last_price": self.spot + 2.0, "oi": 106_000}
            elif key.startswith("NFO:TEST26OCT"):
                kind = "CE" if key.endswith("CE") else "PE"
                strike = next(
                    float(row["strike"]) for row in self.instruments_rows
                    if row["tradingsymbol"] == key.split(":", 1)[1]
                )
                fair = black_scholes(
                    spot=self.spot, strike=strike, dte=33, iv=0.24, option_type=kind,
                )["price"]
                out[key] = {
                    "last_price": fair,
                    "volume": 8000,
                    "oi": 40000,
                    "depth": {
                        "buy": [{"price": max(0.05, fair - 0.15)}],
                        "sell": [{"price": fair + 0.15}],
                    },
                }
        return out

    def historical_with_oi(self, token, from_date, to_date, interval="day"):
        assert token == 1
        return [{"date": "2026-09-25", "close": self.spot - 10.0, "oi": 100_000}]


class _NoQuoteClient:
    def quote(self, keys):
        raise AssertionError("no deep quote should be requested for a valid empty prefilter")


def test_runtime_scans_canonical_universe_and_returns_ranked_paper_candidate(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _instruments(spot)
    client = _Client(spot, instruments)

    monkeypatch.setattr(
        fo_runtime,
        "_index_context",
        lambda: {"return_20d_pct": 1.0, "return_5d_pct": 0.5, "last_close": 21900.0},
    )
    monkeypatch.setattr(fo_runtime, "_sector_strength_map", lambda nifty: {"Tech": 1.2})
    monkeypatch.setattr(fo_runtime, "_sector_for", lambda symbol: "Tech")

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=instruments,
        client=client,
        as_of=AS_OF,
        history_getter=lambda symbol: bars,
    )
    assert result["available"] is True
    assert result["status"] == "READY"
    assert result["universe_size"] == 1
    assert result["prefilter_passed"] == 1
    assert result["deep_evaluated"] == 1
    assert result["decision"] == "PAPER_CANDIDATES"
    assert result["candidate_count"] == 1
    selected = result["candidates"][0]
    assert selected["direction"] == "LONG"
    assert selected["selected_contract"]["option_type"] == "CE"
    assert selected["selected_contract"]["trade_plan"]["entry"] > selected["selected_contract"]["trade_plan"]["stop"]
    assert selected["paper_only"] is True
    assert selected["live_execution_allowed"] is False


def test_empty_breakout_set_is_valid_no_trade_and_does_not_request_deep_quotes():
    bars = _quiet_bars()
    spot = float(bars["close"].iloc[-1])
    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=_instruments(spot),
        client=_NoQuoteClient(),
        as_of=AS_OF,
        history_getter=lambda symbol: bars,
    )
    assert result["available"] is True
    assert result["status"] == "READY"
    assert result["decision"] == "NO_ELIGIBLE_TRADE"
    assert result["candidate_count"] == 0
    assert result["quote_scope"]["option_contracts_requested"] == 0
