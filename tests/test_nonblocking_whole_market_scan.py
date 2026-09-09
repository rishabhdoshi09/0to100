from __future__ import annotations

from types import SimpleNamespace


def test_whole_market_scan_emits_progress_before_cached_context(monkeypatch):
    """The scan lane must never wait on network-capable regime/index builders."""
    import core.regime_engine as regime_engine
    import data.index_store as index_store
    import scan.bulk_fetcher as bulk_fetcher
    from scan.unified_scanner import UnifiedScanner

    monkeypatch.setattr(bulk_fetcher, "cached_symbols", lambda: ["AAA"])
    monkeypatch.setattr(bulk_fetcher, "get_cached", lambda _symbol: object())
    monkeypatch.setattr(bulk_fetcher, "prefetch", lambda _symbols: 1)
    monkeypatch.setattr(index_store, "recent_index_closes", lambda _ticker, n=31: [100.0] * int(n))

    # If the old network-capable path is ever called, fail loudly.
    monkeypatch.setattr(
        regime_engine,
        "compute_regime",
        lambda: (_ for _ in ()).throw(AssertionError("compute_regime must not run in MARKET_SCAN")),
    )
    monkeypatch.setattr(
        regime_engine,
        "peek_cached_regime",
        lambda: SimpleNamespace(market_regime="BULLISH"),
    )

    scanner = UnifiedScanner(max_workers=1)
    monkeypatch.setattr(scanner, "_analyze", lambda _symbol, _frame: None)

    events: list[tuple[int, int]] = []
    scanner.scan(["AAA"], progress=lambda current, total: events.append((current, total)), prefetch=False)

    assert events[0] == (0, 1)
    assert events[-1] == (1, 1)
    assert scanner._nifty_ret30 == 0.0
    assert scanner._regime == "BULLISH"


def test_whole_market_scan_tolerates_missing_cached_regime(monkeypatch):
    import core.regime_engine as regime_engine
    import data.index_store as index_store
    import scan.bulk_fetcher as bulk_fetcher
    from scan.unified_scanner import UnifiedScanner

    monkeypatch.setattr(bulk_fetcher, "cached_symbols", lambda: ["AAA"])
    monkeypatch.setattr(bulk_fetcher, "get_cached", lambda _symbol: object())
    monkeypatch.setattr(bulk_fetcher, "prefetch", lambda _symbols: 1)
    monkeypatch.setattr(index_store, "recent_index_closes", lambda _ticker, n=31: [])
    monkeypatch.setattr(regime_engine, "peek_cached_regime", lambda: None)

    scanner = UnifiedScanner(max_workers=1)
    monkeypatch.setattr(scanner, "_analyze", lambda _symbol, _frame: None)

    events: list[tuple[int, int]] = []
    scanner.scan(["AAA"], progress=lambda current, total: events.append((current, total)), prefetch=False)

    assert events == [(0, 1), (1, 1)]
    assert scanner._regime == ""
    assert scanner._regime_calib == {}
