from types import SimpleNamespace

from product.scan_progress import eta_label, eta_seconds, finish_progress, read_progress, write_progress


def test_eta_needs_a_real_pace():
    assert eta_seconds(0, 2000, 100.0, now=110.0) is None
    assert eta_seconds(400, 2000, 100.0, now=140.0) == 160.0
    assert eta_label(160) == "about 3 min"
    assert eta_label(12) == "under 15s"
    assert eta_label(None) == ""


def test_progress_file_carries_eta(tmp_path, monkeypatch):
    path = tmp_path / "scan_progress.json"
    write_progress(current=0, total=2000, stage="STARTING", path=path, now=1000.0)
    monkeypatch.setattr("product.scan_progress._started_at", 1000.0)
    payload = write_progress(current=500, total=2000, stage="SCANNING", path=path, now=1040.0)
    assert payload["active"] is True
    assert payload["pct"] == 25.0
    assert payload["eta_label"] == "about 2 min"
    saved = read_progress(path)
    assert saved["current"] == 500
    assert saved["total"] == 2000
    done = finish_progress(records=40, setups=12, path=path)
    assert done["active"] is False
    assert done["setups"] == 12


def test_unified_scanner_reports_analyze_progress(monkeypatch):
    from scan.unified_scanner import UnifiedScanner

    seen: list[tuple[int, int]] = []

    class Fast(UnifiedScanner):
        def _analyze(self, symbol, df):
            return SimpleNamespace(symbol=symbol, signals=["MOMENTUM"], score=50)

    monkeypatch.setattr("scan.bulk_fetcher.cached_symbols", lambda: ["AAA", "BBB"])
    monkeypatch.setattr("scan.bulk_fetcher.get_cached", lambda symbol: object())
    out = Fast(max_workers=2).scan(
        ["AAA", "BBB"],
        progress=lambda current, total: seen.append((current, total)),
        prefetch=False,
    )
    assert [row.symbol for row in out] == ["AAA", "BBB"] or {row.symbol for row in out} == {"AAA", "BBB"}
    assert seen[0] == (0, 2)
    assert seen[-1] == (2, 2)


def test_unified_scanner_warms_cache_when_outer_prefetch_was_skipped(monkeypatch):
    from scan.unified_scanner import UnifiedScanner

    state = {"warm": False}

    class Fast(UnifiedScanner):
        def _analyze(self, symbol, df):
            return SimpleNamespace(symbol=symbol, signals=["MOMENTUM"], score=50)

    def cached():
        return ["AAA", "BBB"] if state["warm"] else []

    def prefetch(symbols, progress=None):
        state["warm"] = True
        return 2

    monkeypatch.setattr("scan.bulk_fetcher.cached_symbols", cached)
    monkeypatch.setattr("scan.bulk_fetcher.prefetch", prefetch)
    monkeypatch.setattr("scan.bulk_fetcher.get_cached", lambda symbol: object())
    out = Fast(max_workers=2).scan(["AAA", "BBB"], prefetch=False)
    assert {row.symbol for row in out} == {"AAA", "BBB"}
    assert state["warm"] is True


def test_market_scan_passes_progress_into_analyze():
    from scan.market_scan_service import run_whole_market_scan

    seen: list[tuple[int, int]] = []

    class Scanner:
        def scan(self, symbols, progress=None, prefetch=True):
            assert prefetch is False
            if progress:
                progress(1, 2)
                progress(2, 2)
            return [SimpleNamespace(symbol="AAA", signals=["MOMENTUM"], score=80,
                                    verdict="BUY", chase_risk=False, price=100,
                                    momentum_5d=2, rsi=55, volume_ratio=1.5,
                                    entry=101, stop=95, target=120, reasons=["ok"])]

    report = run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha", "BBB": "Beta"},
        prefetch_fn=lambda symbols, progress=None: 2,
        scanner=Scanner(),
        fno_provider=lambda: set(),
        progress_callback=lambda current, total=0, **k: seen.append((int(current), int(total))),
        save=False,
    )
    assert report.ok
    assert seen == [(1, 2), (2, 2)]


def test_empty_ohlcv_cache_keeps_last_scan(monkeypatch):
    from scan.market_scan_service import DATA_UNAVAILABLE, run_whole_market_scan

    saved: list[dict] = []

    class EmptyScanner:
        def scan(self, symbols, progress=None, prefetch=True):
            return []

    monkeypatch.setattr("scan.bulk_fetcher.cached_symbols", lambda: [])
    monkeypatch.setattr("product.scan_store.save_scan", lambda payload, path=None: saved.append(payload))
    report = run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha", "BBB": "Beta"},
        prefetch_fn=lambda symbols, progress=None: len(symbols),
        scanner=EmptyScanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    assert report.status == DATA_UNAVAILABLE
    assert report.error_code == "OHLCV_CACHE_EMPTY"
    assert saved == []


def test_market_ops_prefetch_warms_ohlcv():
    import inspect
    from operations.market_ops import MarketOperationsWorker
    source = inspect.getsource(MarketOperationsWorker._run_market_scan)
    assert "warm_ohlcv" in source
    assert "return len(symbols)" not in source
    assert "warm_ohlcv(symbols, progress=progress)" not in source


def test_stack_scripts_restart_children_instead_of_stopping_the_desk():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    inner = (root / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")
    complete = (root / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    assert "Ctrl-C is the stop signal" in inner
    assert "restarting" in inner.lower()
    assert 'exit 1' not in inner.split("QuantTerm is running")[-1]
    assert "exited unexpectedly" not in complete
    assert "restarting it" in complete.lower()
