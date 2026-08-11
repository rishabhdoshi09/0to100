"""Long-term scan progress must be reportable (no silent STARTING freeze)."""
from __future__ import annotations


def test_scan_long_term_reports_progress(monkeypatch):
    import pandas as pd

    from scan import long_term as LT

    idx = pd.date_range("2020-01-01", periods=260, freq="B")
    close = pd.Series(range(100, 100 + len(idx)), dtype=float)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1_000_000.0] * len(idx),
        },
        index=idx,
    )
    monkeypatch.setattr(LT, "get_ohlcv", lambda _s: frame, raising=False)
    monkeypatch.setattr(
        "data.bhavcopy_store.get_ohlcv",
        lambda _s: frame,
    )
    monkeypatch.setattr(
        "data.bhavcopy_store.store_symbols",
        lambda: ["AAA", "BBB", "CCC"],
    )

    events: list[tuple] = []

    def progress(current, total, message):
        events.append((current, total, message))

    picks = LT.scan_long_term(symbols=["AAA", "BBB", "CCC"], top=10, progress=progress)
    assert isinstance(picks, list)
    assert events
    assert events[-1][0] == 3
    assert events[-1][1] == 3


def test_run_long_term_scan_progress_phases():
    from scan.long_term_service import run_long_term_scan

    events: list[str] = []

    def progress(current, total, message):
        events.append(str(message))

    technical = [
        {
            "symbol": "AAA",
            "score": 70,
            "factors": ["uptrend"],
            "extension_pct": 5,
            "verdict": "LONG_TERM_BUY",
        }
    ]
    report = run_long_term_scan(
        technical_scanner=lambda **_kw: technical,
        fundamental_provider=lambda _s, _r: {},
        sector_lookup=lambda _s: "IT",
        save=False,
        progress=progress,
    )
    assert report.ok
    blob = " | ".join(events).lower()
    assert "fundamental" in blob
    assert "sav" in blob or "top" in blob
