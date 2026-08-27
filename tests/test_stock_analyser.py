"""Click-through Minervini live analyser — official history only."""
from __future__ import annotations

import pandas as pd
import pytest

from product.stock_analyser import analyse_stock


@pytest.fixture(autouse=True)
def _no_index_network(monkeypatch):
    monkeypatch.setattr("product.stock_analyser.analyser_benchmark_frame", lambda: (None, "Nifty 50"))
    monkeypatch.setattr("product.monitor_context.nifty_frame", lambda: None)


def _frame(*, periods=280, start=70.0, step=0.55, volume=3_000_000, last_open_delta=-0.4,
           last_high_delta=1.2, last_low_delta=-0.4) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    close = pd.Series([start + i * step for i in range(periods)], index=index)
    data = pd.DataFrame(
        {
            "open": close - 0.4,
            "high": close + 1.1,
            "low": close - 0.9,
            "close": close,
            "volume": [volume] * periods,
        },
        index=index,
    )
    data.iloc[-1, data.columns.get_loc("open")] = float(close.iloc[-1]) + last_open_delta
    data.iloc[-1, data.columns.get_loc("high")] = float(close.iloc[-1]) + last_high_delta
    data.iloc[-1, data.columns.get_loc("low")] = float(close.iloc[-1]) + last_low_delta
    return data


def test_uptrend_watchlist_or_strong_with_quote_strip():
    sepa = analyse_stock(_frame())
    assert sepa["available"] is True
    assert sepa["total"] == 7
    assert sepa["score"] >= 55
    assert sepa["verdict"] in {"STRONG", "WATCHLIST"}
    by_id = {c["id"]: c for c in sepa["criteria"]}
    assert by_id["near_52w_high"]["passed"] is True
    assert by_id["off_52w_low"]["passed"] is True
    assert by_id["stage"]["passed"] is True
    assert by_id["liquidity"]["passed"] is True
    quote = sepa["quote"]
    assert quote["open"] and quote["high"] and quote["low"] and quote["prev_close"]
    assert quote["high_52w"] >= quote["close"]


def test_bearish_wide_thin_name_is_not_a_setup():
    frame = _frame(start=220.0, step=-0.45, volume=90_000,
                   last_open_delta=2.0, last_high_delta=3.0, last_low_delta=-8.0)
    sepa = analyse_stock(frame)
    assert sepa["available"] is True
    assert sepa["score"] < 55
    by_id = {c["id"]: c for c in sepa["criteria"]}
    assert by_id["intraday"]["passed"] is False
    assert by_id["liquidity"]["passed"] is False
    assert "WAIT" in sepa["headline"] or "NOT IDEAL" in sepa["headline"] or "MIXED" in sepa["headline"]


def test_screenshot_style_watchlist_headline_when_three_checks_pass():
    """Aeroenter-like: near highs, off lows, stage 2, but weak tape/volume/VCP."""
    frame = _frame(volume=96_000, last_open_delta=2.8, last_high_delta=4.0, last_low_delta=-2.0)
    sepa = analyse_stock(frame)
    by_id = {c["id"]: c for c in sepa["criteria"]}
    assert by_id["near_52w_high"]["passed"] is True
    assert by_id["off_52w_low"]["passed"] is True
    assert by_id["liquidity"]["passed"] is False
    assert sepa["passed"] >= 3
    assert sepa["headline"] in {
        "WATCHLIST — WAIT FOR SETUP",
        "READY — SETUP QUALIFIED",
        "MIXED — WAIT FOR STRUCTURE",
        "WEAK — NOT IDEAL FOR SWING",
    }


def _aeroenter_like_frame() -> pd.DataFrame:
    """Last bar matches the public STOCK ANALYSER screenshot numbers."""
    index = pd.date_range("2024-08-01", periods=252, freq="B")
    n = len(index)
    prev_close, last_close = 143.64, 142.65
    closes = [62.2 + (prev_close - 62.2) * i / (n - 2) for i in range(n - 1)] + [last_close]
    data = pd.DataFrame(
        {
            "open": [c - 0.4 for c in closes],
            "high": [c + 1.1 for c in closes],
            "low": [c - 0.9 for c in closes],
            "close": closes,
            "volume": [960_000] * n,
        },
        index=index,
    )
    data["low"] = data["low"].clip(lower=62.2)
    data.iloc[0, data.columns.get_loc("low")] = 62.2
    peak = n // 2
    data.iloc[peak, data.columns.get_loc("high")] = 149.55
    data.iloc[-2, data.columns.get_loc("close")] = prev_close
    data.iloc[-1, data.columns.get_loc("open")] = 145.45
    data.iloc[-1, data.columns.get_loc("high")] = 147.0
    data.iloc[-1, data.columns.get_loc("low")] = 141.01
    data.iloc[-1, data.columns.get_loc("close")] = last_close
    return data


def test_aeroenter_style_card_is_watchlist_55_of_100():
    bench_index = pd.date_range("2024-08-01", periods=252, freq="B")
    flat = pd.Series([100.0] * len(bench_index), index=bench_index)
    bench = pd.DataFrame(
        {"close": flat, "open": flat, "high": flat + 0.1, "low": flat - 0.1, "volume": [1_000_000] * len(bench_index)},
        index=bench_index,
    )
    sepa = analyse_stock(_aeroenter_like_frame(), bench_frame=bench, bench_label="Nifty 500")
    by_id = {c["id"]: c for c in sepa["criteria"]}
    assert sepa["score"] == 55
    assert sepa["passed"] == 3
    assert sepa["headline"] == "WATCHLIST — WAIT FOR SETUP"
    assert by_id["near_52w_high"]["passed"] is True
    assert "4.6% below" in by_id["near_52w_high"]["detail"]
    assert by_id["off_52w_low"]["passed"] is True
    assert "129.3% above" in by_id["off_52w_low"]["detail"]
    assert "doubled from lows" in by_id["off_52w_low"]["note"]
    assert by_id["relative_strength"]["passed"] is False
    assert "Nifty 500" in by_id["relative_strength"]["title"]
    assert by_id["intraday"]["passed"] is False
    assert "Bearish" in by_id["intraday"]["detail"]
    assert by_id["vcp"]["passed"] is False
    assert "Moderate range" in by_id["vcp"]["note"]
    assert by_id["liquidity"]["passed"] is False
    assert "9.6L" in by_id["liquidity"]["detail"]
    assert by_id["stage"]["passed"] is True
    assert by_id["stage"]["detail"] == "Stage 2 — Advancing"
    quote = sepa["quote"]
    assert quote["open"] == 145.45
    assert quote["high"] == 147.0
    assert quote["low"] == 141.01
    assert quote["prev_close"] == 143.64
    assert quote["high_52w"] == 149.55


def test_thin_9p6_lakh_book_fails_liquidity_even_when_notional_exceeds_10cr():
    """9.6 lakh shares at ~₹143 is ~₹13.7 Cr notional, still a thin book."""
    sepa = analyse_stock(_frame(volume=960_000))
    liq = next(c for c in sepa["criteria"] if c["id"] == "liquidity")
    assert liq["passed"] is False
    assert "9.6L" in liq["detail"]


def test_short_history_is_incomplete_not_a_fail():
    sepa = analyse_stock(_frame(periods=12))
    assert sepa["available"] is False
    assert sepa["verdict"] == "INCOMPLETE"
    assert sepa["score"] == 0


def test_relative_strength_uses_last_session_vs_benchmark():
    stock = _frame()
    bench_index = stock.index
    bench_close = pd.Series([100.0] * (len(bench_index) - 1) + [100.0], index=bench_index)
    bench = pd.DataFrame({"close": bench_close, "open": bench_close, "high": bench_close + 0.1,
                          "low": bench_close - 0.1, "volume": [1_000_000] * len(bench_index)},
                         index=bench_index)
    sepa = analyse_stock(stock, bench_frame=bench, bench_label="Nifty 500")
    rs = next(c for c in sepa["criteria"] if c["id"] == "relative_strength")
    assert rs["passed"] is True
    assert "Nifty 500" in rs["detail"]
    assert sepa["benchmark"] == "Nifty 500"
