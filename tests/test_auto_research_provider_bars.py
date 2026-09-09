from __future__ import annotations

import pandas as pd

from research.auto_research import providers as P


def test_daily_bars_reads_exact_normalized_bhavcopy_session(monkeypatch):
    from data import bhavcopy_store as BS

    seen = []

    def _read(day):
        seen.append(day.isoformat())
        return pd.DataFrame([
            {"symbol": "AAA", "open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0, "volume": 1000},
            {"symbol": "BBB", "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.5, "volume": 2000},
        ])

    monkeypatch.setattr(BS, "_read_day", _read)
    bars = P.daily_bars("2026-09-04")

    assert seen == ["2026-09-04"]
    assert bars["AAA"] == (100.0, 105.0, 98.0, 103.0)
    assert bars["BBB"] == (50.0, 51.0, 49.0, 50.5)


def test_daily_bars_does_not_substitute_another_session(monkeypatch):
    from data import bhavcopy_store as BS

    seen = []

    def _read(day):
        seen.append(day.isoformat())
        return None

    monkeypatch.setattr(BS, "_read_day", _read)
    assert P.daily_bars("2026-09-04") == {}
    assert seen == ["2026-09-04"]


def test_daily_bars_ignores_nan_rows(monkeypatch):
    from data import bhavcopy_store as BS

    monkeypatch.setattr(
        BS,
        "_read_day",
        lambda _day: pd.DataFrame([
            {"symbol": "BAD", "open": 100.0, "high": float("nan"), "low": 98.0, "close": 99.0},
            {"symbol": "GOOD", "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.5},
        ]),
    )
    assert P.daily_bars("2026-09-04") == {"GOOD": (10.0, 11.0, 9.5, 10.5)}
