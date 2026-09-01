"""PIT correlation must fail closed when no historical time anchor is available."""
from __future__ import annotations

import pandas as pd


def test_unanchored_local_correlation_is_diagnostic_only(monkeypatch):
    from data import bhavcopy_store as store
    from product.pit_correlation import correlations_for_candidates

    dates = pd.bdate_range("2026-01-01", periods=80)
    a = pd.DataFrame({"close": [100.0 + i for i in range(80)]}, index=dates)
    b = pd.DataFrame({"close": [200.0 + 2 * i for i in range(80)]}, index=dates)
    monkeypatch.setattr(store, "is_ready", lambda: True)
    monkeypatch.setattr(store, "get_ohlcv", lambda symbol: a.copy() if symbol == "AAA" else b.copy() if symbol == "BBB" else None)

    out = correlations_for_candidates([{"symbol": "AAA"}, {"symbol": "BBB"}])
    assert out["point_in_time"] is False
    assert out["production_usable"] is False
    assert out["warning"] == "NO_TIME_ANCHOR_DIAGNOSTIC_ONLY"
    assert out["correlations"] == {}
    assert out["diagnostic_correlations"]["AAA|BBB"] > 0.99
    assert out["network_used"] is False
