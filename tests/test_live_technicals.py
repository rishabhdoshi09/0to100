from __future__ import annotations

import numpy as np
import pandas as pd

from product import live_technicals as LT


def _frame():
    close = np.linspace(100.0, 125.0, 40)
    return pd.DataFrame(
        {
            "close": close,
            "volume": [1_000_000.0] * 39 + [2_000_000.0],
        },
        index=pd.date_range("2026-07-01", periods=40, freq="B"),
    )


def test_apply_levels_with_no_quote_is_network_free_and_never_invents_plan(monkeypatch):
    import data.live_quotes as live_quotes

    monkeypatch.setattr(
        live_quotes,
        "get_live_quotes",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )
    row = {"symbol": "AAA", "price": 100.0, "freshness": "2026-09-21"}

    out = LT.apply_current_trade_levels(row, None)

    assert out["price"] == 100.0
    assert out["cmp"] == 100.0
    assert out["price_tag"].startswith("EOD")
    assert "entry" not in out
    assert "target" not in out
    assert "stop" not in out


def test_worker_refresh_uses_one_bulk_quote_call_and_official_cache(monkeypatch):
    calls = []
    monkeypatch.setattr("scan.bulk_fetcher.get_cached", lambda _s: _frame())

    def quotes(symbols):
        calls.append(list(symbols))
        return {
            "AAA": {"price": 130.0, "chg_pct": 1.2, "source": "kite"},
            "BBB": {"price": 131.0, "chg_pct": 0.8, "source": "kite"},
        }

    monkeypatch.setattr("data.live_quotes.get_live_quotes", quotes)
    rows = [
        {"symbol": "AAA", "entry": 120.0, "stop": 114.0, "target": 138.0},
        {"symbol": "BBB", "entry": 121.0, "stop": 115.0, "target": 139.0},
    ]

    out = LT.refresh_rows_technicals(rows, bulk_overlay=True)

    assert calls == [["AAA", "BBB"]]
    assert [row["cmp"] for row in out] == [130.0, 131.0]
    assert all(row["rsi"] is not None for row in out)
    assert all(row["volume_ratio"] > 0 for row in out)
    assert all("official_ohlcv_cache+kite" == row["tech_source"] for row in out)
    assert out[0]["upside_to_target_pct"] > 0
