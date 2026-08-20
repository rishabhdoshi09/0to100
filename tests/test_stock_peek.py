"""Snapshot popup numbers — on-file + timed tape, never a scrape gate."""
from __future__ import annotations

import time

import pandas as pd

from product.stock_peek import build_stock_peek, _upside


def _frame(periods: int = 280) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="B")
    close = pd.Series([100 + i * 0.4 for i in range(periods)], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [120000] * periods,
        },
        index=index,
    )


def test_upside_from_buy_and_target():
    assert _upside(743.90, 842.70) == 13.3
    assert _upside(0, 100) is None
    assert _upside(100, None) is None


def test_peek_fills_change_fundamentals_and_upside_without_live_tape():
    payload = build_stock_peek(
        "MSTCLTD",
        scan_payload={
            "records": [
                {
                    "symbol": "MSTCLTD",
                    "company": "MSTC",
                    "price": 743.9,
                    "entry": 743.9,
                    "stop": 694.5,
                    "target": 842.7,
                    "rsi": 69.2,
                    "volume_ratio": 4.94,
                    "sector": "Trading",
                }
            ]
        },
        long_term_payload={
            "records": [
                {
                    "symbol": "MSTCLTD",
                    "company": "MSTC",
                    "sector": "Trading",
                    "classification": "GARP_CANDIDATE",
                    "fundamental_coverage": 0.7,
                    "fundamentals": {"pe": 18.4, "roe": 22.1, "roce": 19.0},
                }
            ]
        },
        raw_fundamentals={"data": {}, "cache_status": "TODAY"},
        frame=_frame(),
        quote={"price": 748.5, "chg_pct": 1.25, "source": "nse"},
        load_history=False,
        load_live=False,
    )
    assert payload["cmp"] == 748.5
    assert payload["change_pct"] == 1.25
    assert payload["upside_from_buy_pct"] == 13.3
    assert payload["technical"]["change_pct"] == 1.25
    metric_keys = {m["key"] for m in payload["technical"]["metrics"]}
    assert "change_pct" in metric_keys
    assert "close" in metric_keys
    keys = {m["key"] for m in payload["fundamentals"]["metrics"]}
    assert "pe" in keys and "roe" in keys
    assert all(m.get("value") is not None for m in payload["fundamentals"]["metrics"])


def test_peek_does_not_wait_on_hung_history_or_scrape(monkeypatch):
    import product.stock_peek as peek

    def hang():
        time.sleep(30)
        raise AssertionError("history must be timed out")

    def scrape(*_a, **_k):
        raise AssertionError("snapshot must not scrape fundamentals")

    monkeypatch.setattr(peek, "_load_frame", hang)
    monkeypatch.setattr("fundamentals.lazy.ensure_deep_fundamentals", scrape)
    t0 = time.monotonic()
    payload = build_stock_peek(
        "MSTCLTD",
        scan_payload={
            "records": [
                {
                    "symbol": "MSTCLTD",
                    "company": "MSTC",
                    "price": 743.9,
                    "entry": 743.9,
                    "stop": 694.5,
                    "target": 842.7,
                    "rsi": 69.2,
                    "volume_ratio": 4.94,
                    "change_pct": 0.8,
                }
            ]
        },
        long_term_payload={"records": []},
        raw_fundamentals={"data": {}},
        load_history=True,
        load_live=False,
    )
    assert time.monotonic() - t0 < 4.0
    assert payload["upside_from_buy_pct"] == 13.3
    assert payload["rsi"] == 69.2
    assert payload["technical"]["change_pct"] == 0.8
    assert "history" in (payload.get("history_note") or "").lower()
