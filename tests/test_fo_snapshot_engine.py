from datetime import date

import numpy as np
import pandas as pd

from options.directional_selector import black_scholes
from product.fo_snapshot_engine import evaluate_fo_snapshot_auto


AS_OF = date(2026, 9, 26)


def _bars():
    n = 90
    x = np.arange(n, dtype=float)
    # Choppy uptrend: strong enough for a breakout, but deliberately below
    # the production RSI>=82 blow-off rejection and above the liquidity floor.
    close = 1000.0 + x * 1.3 + np.sin(x / 2.7) * 20.0
    high = close + 8.0
    low = close - 8.0
    volume = np.full(n, 100_000.0)
    prior_high = float(high[-21:-1].max())
    close[-1] = prior_high + 8.0
    high[-1] = close[-1] + 2.0
    low[-1] = close[-1] - 14.0
    volume[-1] = 25_000.0
    return pd.DataFrame({
        "open": close - 2.0,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _nfo(spot):
    strike = round(spot / 10.0) * 10.0
    return [
        {
            "instrument_token": 1,
            "tradingsymbol": "TEST26OCTFUT",
            "name": "TEST",
            "expiry": "2026-10-29",
            "strike": 0,
            "tick_size": 0.05,
            "lot_size": 50,
            "instrument_type": "FUT",
            "segment": "NFO-FUT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 2,
            "tradingsymbol": "TEST26OCTCE",
            "name": "TEST",
            "expiry": "2026-10-29",
            "strike": strike,
            "tick_size": 0.05,
            "lot_size": 50,
            "instrument_type": "CE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 3,
            "tradingsymbol": "TEST26OCTPE",
            "name": "TEST",
            "expiry": "2026-10-29",
            "strike": strike,
            "tick_size": 0.05,
            "lot_size": 50,
            "instrument_type": "PE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
    ]


def _option_quotes(spot, instruments):
    out = {}
    for row in instruments:
        if row["instrument_type"] not in {"CE", "PE"}:
            continue
        price = black_scholes(
            spot=spot,
            strike=float(row["strike"]),
            dte=33,
            iv=0.24,
            option_type=row["instrument_type"],
        )["price"]
        out[row["tradingsymbol"]] = {
            "last_price": price,
            "volume": 8000,
            "oi": 40000,
            "depth": {
                "buy": [{"price": max(0.05, price - 0.15)}],
                "sell": [{"price": price + 0.15}],
            },
        }
    return out


def test_auto_snapshot_selects_long_ce_only_when_full_evidence_is_present():
    bars = _bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _nfo(spot)
    result = evaluate_fo_snapshot_auto(
        symbol="TEST",
        daily_bars=bars,
        nfo_instruments=instruments,
        underlying_quote={
            "last_price": spot,
            "average_price": spot - 3.0,
            "depth": {
                "buy": [{"price": spot - 0.1}],
                "sell": [{"price": spot + 0.1}],
            },
        },
        futures_quote={"last_price": spot + 2.0, "oi": 106_000},
        previous_futures_price=spot - 10.0,
        previous_futures_oi=100_000,
        option_quotes=_option_quotes(spot, instruments),
        benchmark_20d_return_pct=1.0,
        nifty_change_pct=0.6,
        sector_relative_strength_pct=1.2,
        iv_percentile=45.0,
        as_of=AS_OF,
    )
    assert result["decision"] == "PAPER_OPTION_CANDIDATE"
    selected = result["selected"]
    assert selected["direction"] == "LONG"
    assert selected["selected_contract"]["option_type"] == "CE"
    assert selected["selected_contract"]["lot_size"] == 50
    assert selected["selected_contract"]["context_key"]
    assert selected["live_execution_allowed"] is False


def test_missing_previous_futures_oi_blocks_instead_of_assuming_neutral():
    bars = _bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _nfo(spot)
    result = evaluate_fo_snapshot_auto(
        symbol="TEST",
        daily_bars=bars,
        nfo_instruments=instruments,
        underlying_quote={"last_price": spot, "average_price": spot - 3.0},
        futures_quote={"last_price": spot + 2.0, "oi": 106_000},
        previous_futures_price=spot - 10.0,
        previous_futures_oi=None,
        option_quotes=_option_quotes(spot, instruments),
        benchmark_20d_return_pct=1.0,
        nifty_change_pct=0.6,
        sector_relative_strength_pct=1.2,
        as_of=AS_OF,
    )
    assert result["decision"] == "NO_TRADE"
    long_row = next(row for row in result["directions"] if row["direction"] == "LONG")
    assert "MISSING_FUTURES_OI_CHANGE_PCT" in long_row["setup"]["blockers"]
