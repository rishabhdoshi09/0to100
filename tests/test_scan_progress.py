from types import SimpleNamespace

from product.scan_progress import eta_label, eta_seconds, finish_progress, read_progress, write_progress


def test_eta_needs_a_real_pace():
    assert eta_seconds(0, 2000, 100.0, now=110.0) is None
    assert eta_seconds(400, 2000, 100.0, now=140.0) == 160.0
    assert eta_label(160) == "about 3 min"
    assert eta_label(12) == "under 15s"
    assert eta_label(None) == ""


def test_stale_active_progress_does_not_keep_a_fake_eta(tmp_path):
    path = tmp_path / "scan_progress.json"
    path.write_text(
        '{"active": true, "stage": "SCANNING", "current": 11, "total": 47, '
        '"eta_label": "about 20 min", "updated_at": 1}',
        encoding="utf-8",
    )
    saved = read_progress(path)
    assert saved["active"] is False
    assert saved["eta_label"] == ""
    assert saved.get("stale") is True


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


def test_default_market_scanner_uses_parallel_workers():
    from scan.market_scan_service import _default_scanner

    scanner = _default_scanner()
    assert scanner._max_workers >= 8
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
    history = inspect.getsource(MarketOperationsWorker._ensure_history)
    assert "not scanning stocks yet" in history
    assert "Downloading official NSE bhavcopy" in history
    assert "current_count,\n                    total," not in history


def test_market_scan_does_not_forward_prefetch_days_as_stocks():
    from scan.market_scan_service import run_whole_market_scan

    seen: list[tuple[int, int]] = []
    prefetch_progress: list = []

    def prefetch(symbols, progress=None):
        if progress:
            progress(11, 47)
            prefetch_progress.append((11, 47))
        return len(symbols)

    class Scanner:
        def scan(self, symbols, progress=None, prefetch=True):
            if progress:
                progress(1, 2)
                progress(2, 2)
            return [SimpleNamespace(symbol="AAA", signals=["MOMENTUM"], score=80,
                                    verdict="BUY", chase_risk=False, price=100,
                                    momentum_5d=2, rsi=55, volume_ratio=1.5,
                                    entry=101, stop=95, target=120, reasons=["ok"])]

    report = run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha", "BBB": "Beta"},
        prefetch_fn=prefetch,
        scanner=Scanner(),
        fno_provider=lambda: set(),
        progress_callback=lambda current, total=0, **k: seen.append((int(current), int(total))),
        save=False,
    )
    assert report.ok
    assert (11, 47) not in seen
    assert prefetch_progress == []
    assert seen == [(1, 2), (2, 2)]


def test_market_scan_service_overlays_long_term_after_save():
    import inspect
    from scan.market_scan_service import run_whole_market_scan
    source = inspect.getsource(run_whole_market_scan)
    assert "overlay_long_term_from_market_scan" in source
    assert "persist_public_best_setups" in source
    assert "persist_desks_from_market_scan" in source


def test_unified_scanner_does_not_report_prefetch_days(monkeypatch):
    from scan.unified_scanner import UnifiedScanner

    seen: list[tuple[int, int]] = []
    state = {"warm": False}

    class Fast(UnifiedScanner):
        def _analyze(self, symbol, df):
            return SimpleNamespace(symbol=symbol, signals=["MOMENTUM"], score=50)

    def cached():
        return ["AAA", "BBB"] if state["warm"] else []

    def prefetch(symbols, progress=None):
        if progress:
            progress(11, 47)
        state["warm"] = True
        return 2

    monkeypatch.setattr("scan.bulk_fetcher.cached_symbols", cached)
    monkeypatch.setattr("scan.bulk_fetcher.prefetch", prefetch)
    monkeypatch.setattr("scan.bulk_fetcher.get_cached", lambda symbol: object())
    Fast(max_workers=2).scan(
        ["AAA", "BBB"],
        progress=lambda current, total: seen.append((current, total)),
        prefetch=False,
    )
    assert (11, 47) not in seen
    assert seen[0] == (0, 2)
    assert seen[-1] == (2, 2)


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
    assert "scripts/local_stack.py" in complete
    assert "python scripts/local_stack.py stop --ports 5173,8765,8766" in complete
    assert "scripts/local_stack.py scan" in inner
    assert "Queueing market scan, news and long-term funds in this terminal" in inner
    assert "Do not start a second terminal" in complete
    assert "curl" not in inner
    assert "curl" not in complete
    assert "run_quantterm_complete.sh --restart" not in inner
    assert "Use --restart" not in complete


def test_scan_report_separates_approved_universe_from_scanned():
    from scan.market_scan_service import run_whole_market_scan

    class Scanner:
        def scan(self, symbols, progress=None, prefetch=True):
            assert set(symbols) == {"AAA", "BBB"}
            if progress:
                progress(1, 1)
            return [SimpleNamespace(symbol="AAA", signals=["MOMENTUM"], score=80,
                                    verdict="BUY", chase_risk=False, price=100,
                                    momentum_5d=2, rsi=55, volume_ratio=1.5,
                                    entry=101, stop=95, target=120, reasons=["ok"])]

    report = run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha", "aaa": "dup", "BBB": "Beta"},
        prefetch_fn=lambda symbols, progress=None: len(symbols),
        scanner=Scanner(),
        fno_provider=lambda: set(),
        save=False,
    )
    assert report.ok
    assert report.approved_universe == 3
    assert report.scanned == 1
    assert report.universe_size == 1
    assert report.payload["approved_universe"] == 3
    assert report.payload["scanned"] == 1
    assert report.payload["universe_size"] == 1
    assert report.payload["qualified_rows"] == 1
