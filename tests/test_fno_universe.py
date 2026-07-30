"""Complete-current-universe tests for retail F&O momentum."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from data.fno_universe import build_fno_universe, evaluate_all_underlyings


def _eq(symbol: str, name: str | None = None):
    return {
        "exchange": "NSE",
        "segment": "NSE",
        "instrument_type": "EQ",
        "tradingsymbol": symbol,
        "name": name or symbol,
        "instrument_token": 1,
        "lot_size": 1,
    }


def _fut(symbol: str, expiry: str, token: int, *, name: str | None = None):
    underlying = name or symbol
    return {
        "exchange": "NFO",
        "segment": "NFO-FUT",
        "instrument_type": "FUT",
        "tradingsymbol": f"{symbol}{expiry.replace('-', '')}FUT",
        "name": underlying,
        "expiry": expiry,
        "instrument_token": token,
        "lot_size": 500,
    }


def test_every_current_stock_future_underlying_enters_the_funnel():
    rows = [
        _eq("AAA"),
        _eq("BBB"),
        _eq("CCC"),
        _fut("AAA", "2026-08-27", 11),
        _fut("BBB", "2026-08-27", 12),
        _fut("CCC", "2026-08-27", 13),
        _fut("NIFTY", "2026-08-27", 99),
    ]
    report = build_fno_universe(rows, as_of=date(2026, 7, 31))

    assert report.unique_stock_underlyings == 3
    assert report.symbols == ["AAA", "BBB", "CCC"]
    assert report.index_future_contracts == 1


def test_duplicate_expiries_collapse_to_one_underlying_with_nearest_contract():
    rows = [
        _eq("AAA"),
        _fut("AAA", "2026-09-24", 22),
        _fut("AAA", "2026-08-27", 11),
        _fut("AAA", "2026-10-29", 33),
    ]
    report = build_fno_universe(rows, as_of=date(2026, 7, 31))

    assert report.unique_stock_underlyings == 1
    assert report.mapped_underlyings == 1
    assert report.underlyings[0].contract_count == 3
    assert report.underlyings[0].instrument_token == 11


def test_unmapped_underlying_is_never_silently_dropped():
    rows = [_fut("MISSING", "2026-08-27", 11)]
    report = build_fno_universe(rows, as_of=date(2026, 7, 31))

    assert report.unique_stock_underlyings == 1
    assert report.mapped_underlyings == 0
    assert len(report.exclusions) == 1
    assert report.exclusions[0].stage == "canonical_mapping"
    assert "could not be mapped" in report.exclusions[0].reason


@dataclass
class _Signal:
    signals: list[str]
    reasons: list[str]
    score: float = 72.0
    verdict: str = "BUY"
    price: float = 100.0
    momentum_5d: float = 8.0
    rsi: float = 65.0
    volume_ratio: float = 1.8


def test_filters_do_not_change_which_underlyings_were_evaluated():
    universe = build_fno_universe(
        [
            _eq("AAA"),
            _eq("BBB"),
            _eq("CCC"),
            _fut("AAA", "2026-08-27", 11),
            _fut("BBB", "2026-08-27", 12),
            _fut("CCC", "2026-08-27", 13),
        ],
        as_of=date(2026, 7, 31),
    )
    histories = {
        "AAA": pd.DataFrame({"close": range(100)}, index=pd.date_range("2026-01-01", periods=100)),
        "BBB": pd.DataFrame({"close": range(100)}, index=pd.date_range("2026-01-01", periods=100)),
        "CCC": pd.DataFrame({"close": range(20)}, index=pd.date_range("2026-01-01", periods=20)),
    }
    calls: list[str] = []

    def analyzer(symbol, history):
        calls.append(symbol)
        if symbol == "AAA":
            return _Signal(signals=["MOMENTUM"], reasons=["Up 8% in 5 days"])
        return _Signal(signals=["PRE_BREAKOUT"], reasons=["Near breakout"])

    funnel = evaluate_all_underlyings(
        universe,
        history_getter=histories.get,
        analyzer=analyzer,
        minimum_sessions=60,
    )

    assert funnel.total_underlyings == 3
    assert funnel.data_ready == 2
    assert funnel.evaluated == 2
    assert calls == ["AAA", "BBB"]
    assert [row.symbol for row in funnel.qualified] == ["AAA"]
    assert {row.symbol: row.reason for row in funnel.excluded}["CCC"].startswith("Insufficient history")
    # A UI filter may choose only qualified rows, but the funnel still records all 3.
    assert len(funnel.rows) == 3
