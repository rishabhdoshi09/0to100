"""One whole-market scan fills every setup, including the long-term overlay."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scan.long_term_service import (
    overlay_long_term_from_market_scan,
    run_long_term_scan,
    technical_rows_from_market_scan,
)
from scan.market_scan_service import run_whole_market_scan


def _scan_payload():
    return {
        "schema_version": 1,
        "scanned_at": "2026-08-26T10:00:00+00:00",
        "records": [
            {
                "symbol": "AAA",
                "score": 82,
                "price": 100,
                "reasons": ["Clean base breakout"],
                "signals": ["BREAKOUT_52W"],
                "verdict": "BUY",
                "chase_risk": False,
            },
            {
                "symbol": "BBB",
                "score": 70,
                "price": 50,
                "reasons": ["Extended after breakout"],
                "signals": ["MOMENTUM"],
                "verdict": "WATCH",
                "chase_risk": True,
            },
            {
                "symbol": "CCC",
                "score": 20,
                "reasons": ["Too weak"],
                "chase_risk": False,
            },
        ],
    }


def test_technical_rows_map_reasons_and_do_not_invent_extension():
    rows = technical_rows_from_market_scan(_scan_payload(), min_score=45, top=40, enrich=False)
    assert [r["symbol"] for r in rows] == ["AAA", "BBB"]
    assert "CCC" not in [r["symbol"] for r in rows]
    assert rows[0]["factors"] == ["Clean base breakout"]
    assert "extension_pct" not in rows[0]
    assert rows[0]["from_saved_market_scan"] is True
    assert rows[1]["chase_risk"] is True


def test_chase_risk_without_extension_waits_instead_of_inventing_zero(monkeypatch):
    monkeypatch.setattr(
        "scan.long_term_service._enrich_with_long_term_score",
        lambda rows, **_k: rows,
    )
    report = overlay_long_term_from_market_scan(
        _scan_payload(),
        save=False,
        refresh_fundamentals=False,
        fundamental_provider=lambda _s, _r: None,
    )
    by_symbol = {r["symbol"]: r for r in report.payload["records"]}
    assert by_symbol["AAA"]["timing"] == "TECHNICALLY_FAVORABLE"
    assert by_symbol["BBB"]["timing"] == "WAIT_FOR_BASE"
    assert "extension_pct" not in by_symbol["AAA"]
    assert report.payload["technical_from_saved_scan"] is True
    assert report.payload["scope"] == "saved_market_scan"


def test_run_long_term_scan_uses_saved_scan_and_skips_ohlcv_walk(monkeypatch):
    called = {"walk": 0}

    def boom(*_a, **_k):
        called["walk"] += 1
        raise AssertionError("scan_long_term must not run when a market scan is on file")

    import scan.long_term_service as lts

    monkeypatch.setattr("product.scan_store.load_scan", lambda: _scan_payload())
    monkeypatch.setattr("scan.long_term.scan_long_term", boom)
    monkeypatch.setattr("scan.long_term_service._enrich_with_long_term_score", lambda rows, **_k: rows)
    monkeypatch.setattr(
        "scan.long_term_service._prepare_official_history",
        lambda: (_ for _ in ()).throw(AssertionError("history walk must not run")),
    )
    report = lts.run_long_term_scan(
        save=False,
        scope="saved_market_scan",
        fundamental_provider=lambda _s, _r: None,
        sector_lookup=lambda _s: "Technology",
    )
    assert called["walk"] == 0
    assert report.ok
    assert {r["symbol"] for r in report.payload["records"]} == {"AAA", "BBB"}
    assert report.payload["technical_from_saved_scan"] is True
    assert report.payload["scope"] == "saved_market_scan"


def test_nifty500_scope_projects_saved_scan_without_ohlcv_walk(monkeypatch):
    """Default nifty500 scope still projects the saved scan, then filters.

    The consistency gate must not convert projection into a second market walk.
    """
    called = {"walk": 0}

    def boom(*_a, **_k):
        called["walk"] += 1
        raise AssertionError("nifty500 scope must not OHLCV-walk when a saved scan exists")

    payload = {
        "schema_version": 1,
        "scanned_at": "2026-08-26T10:00:00+00:00",
        "records": [
            {"symbol": "TCS", "score": 82, "price": 100, "reasons": ["Clean"], "verdict": "BUY", "chase_risk": False},
            {"symbol": "INFY", "score": 70, "price": 50, "reasons": ["Leader"], "verdict": "WATCH", "chase_risk": False},
            {"symbol": "ZZZNOTINDEX", "score": 90, "price": 10, "reasons": ["Off universe"], "verdict": "BUY", "chase_risk": False},
        ],
    }
    import scan.long_term_service as lts

    monkeypatch.setattr("product.scan_store.load_scan", lambda: payload)
    monkeypatch.setattr("scan.long_term.scan_long_term", boom)
    monkeypatch.setattr("scan.long_term_service._enrich_with_long_term_score", lambda rows, **_k: rows)
    monkeypatch.setattr(
        "scan.long_term_service._prepare_official_history",
        lambda: (_ for _ in ()).throw(AssertionError("history walk must not run")),
    )
    report = lts.run_long_term_scan(
        save=False,
        fundamental_provider=lambda _s, _r: None,
        sector_lookup=lambda _s: "Technology",
    )
    assert called["walk"] == 0
    assert report.ok
    symbols = {r["symbol"] for r in report.payload["records"]}
    assert symbols == {"TCS", "INFY"}
    assert "ZZZNOTINDEX" not in symbols
    assert report.payload["technical_from_saved_scan"] is True


def test_market_scan_save_writes_long_term_overlay(monkeypatch, tmp_path):
    saved_lt: list[dict] = []
    monkeypatch.setattr("product.scan_store.save_scan", lambda payload, path=None: tmp_path / "scan.json")
    monkeypatch.setattr("product.sepa_setup.persist_public_best_setups", lambda payload: ([], ""))
    monkeypatch.setattr(
        "product.desk_scan_overlays.persist_desks_from_market_scan",
        lambda payload: {"recommendations": "skipped", "market_reports": "skipped"},
    )
    monkeypatch.setattr(
        "product.long_term_store.save_long_term_scan",
        lambda payload, path=None: saved_lt.append(dict(payload)) or tmp_path / "lt.json",
    )
    monkeypatch.setattr("scan.long_term_service._enrich_with_long_term_score", lambda rows, **_k: rows)
    monkeypatch.setattr("scan.long_term_service._default_fundamental_provider", lambda _s, _r: None)

    class Scanner:
        def scan(self, symbols, progress=None, prefetch=True):
            return [
                SimpleNamespace(
                    symbol="AAA", signals=["MOMENTUM"], score=80, verdict="BUY",
                    chase_risk=False, price=100, momentum_5d=2, rsi=55, volume_ratio=1.5,
                    entry=101, stop=95, target=120, reasons=["ok"],
                )
            ]

    report = run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha"},
        prefetch_fn=lambda symbols, progress=None: 1,
        scanner=Scanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    assert report.ok
    overlay = report.payload.get("long_term_overlay") or {}
    assert overlay.get("status") in {"SUCCEEDED", "NO_CANDIDATES"}
    assert saved_lt, "market scan must persist the long-term overlay"
    assert saved_lt[0]["technical_from_saved_scan"] is True


def test_overlay_failure_does_not_fail_the_market_scan(monkeypatch, tmp_path):
    monkeypatch.setattr("product.scan_store.save_scan", lambda payload, path=None: tmp_path / "scan.json")
    monkeypatch.setattr("product.sepa_setup.persist_public_best_setups", lambda payload: ([], ""))
    monkeypatch.setattr(
        "product.desk_scan_overlays.persist_desks_from_market_scan",
        lambda payload: {"recommendations": "skipped", "market_reports": "skipped"},
    )

    def boom(*_a, **_k):
        raise RuntimeError("overlay exploded")

    monkeypatch.setattr("scan.long_term_service.overlay_long_term_from_market_scan", boom)

    class Scanner:
        def scan(self, symbols, progress=None, prefetch=True):
            return [
                SimpleNamespace(
                    symbol="AAA", signals=["MOMENTUM"], score=80, verdict="BUY",
                    chase_risk=False, price=100, momentum_5d=2, rsi=55, volume_ratio=1.5,
                    entry=101, stop=95, target=120, reasons=["ok"],
                )
            ]

    report = run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha"},
        prefetch_fn=lambda symbols, progress=None: 1,
        scanner=Scanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    assert report.ok
    assert report.payload["long_term_overlay"]["status"] == "FAILED"
    assert report.payload["long_term_overlay"]["error_code"] == "RuntimeError"


def test_market_scan_save_persists_recommendation_and_report_desks(monkeypatch, tmp_path):
    called = {"n": 0}

    def persist(payload):
        called["n"] += 1
        assert payload.get("records")
        return {"recommendations": "saved", "recommendation_cards": 1, "market_reports": "saved"}

    monkeypatch.setattr("product.scan_store.save_scan", lambda payload, path=None: tmp_path / "scan.json")
    monkeypatch.setattr("product.sepa_setup.persist_public_best_setups", lambda payload: ([], ""))
    monkeypatch.setattr(
        "scan.long_term_service.overlay_long_term_from_market_scan",
        lambda payload, **_k: SimpleNamespace(status="SUCCEEDED", payload={"records": []}, error_code=""),
    )
    monkeypatch.setattr("product.desk_scan_overlays.persist_desks_from_market_scan", persist)

    class Scanner:
        def scan(self, symbols, progress=None, prefetch=True):
            return [
                SimpleNamespace(
                    symbol="AAA", signals=["MOMENTUM"], score=80, verdict="BUY",
                    chase_risk=False, price=100, momentum_5d=2, rsi=55, volume_ratio=1.5,
                    entry=101, stop=95, target=120, reasons=["ok"],
                )
            ]

    report = run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha"},
        prefetch_fn=lambda symbols, progress=None: 1,
        scanner=Scanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    assert report.ok
    assert called["n"] == 1
    assert report.payload["desk_overlays"]["recommendations"] == "saved"
    assert report.payload["desk_overlays"]["market_reports"] == "saved"


def test_desk_overlay_failure_does_not_fail_the_market_scan(monkeypatch, tmp_path):
    monkeypatch.setattr("product.scan_store.save_scan", lambda payload, path=None: tmp_path / "scan.json")
    monkeypatch.setattr("product.sepa_setup.persist_public_best_setups", lambda payload: ([], ""))
    monkeypatch.setattr(
        "scan.long_term_service.overlay_long_term_from_market_scan",
        lambda payload, **_k: SimpleNamespace(status="SUCCEEDED", payload={"records": []}, error_code=""),
    )
    monkeypatch.setattr(
        "product.desk_scan_overlays.persist_desks_from_market_scan",
        lambda payload: (_ for _ in ()).throw(RuntimeError("desk overlay exploded")),
    )

    class Scanner:
        def scan(self, symbols, progress=None, prefetch=True):
            return [
                SimpleNamespace(
                    symbol="AAA", signals=["MOMENTUM"], score=80, verdict="BUY",
                    chase_risk=False, price=100, momentum_5d=2, rsi=55, volume_ratio=1.5,
                    entry=101, stop=95, target=120, reasons=["ok"],
                )
            ]

    report = run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha"},
        prefetch_fn=lambda symbols, progress=None: 1,
        scanner=Scanner(),
        fno_provider=lambda: set(),
        save=True,
    )
    assert report.ok
    assert report.payload["desk_overlays"]["error"] == "RuntimeError"


def test_priority_ordered_symbols_put_category_names_first():
    from scan.market_scan_service import priority_ordered_symbols

    ordered = priority_ordered_symbols(
        ["ZZZ", "AAA", "SEP", "FNO", "REC", "WAT"],
        scan_payload={"records": [{"symbol": "AAA", "signals": ["BREAKOUT_52W"]}]},
        reco_payload={"records": [{"symbol": "REC"}]},
        long_term_payload={"records": [{"symbol": "SEP"}]},
        fno_symbols={"FNO"},
        watchlist=["WAT"],
    )
    assert ordered[:5] == ["AAA", "WAT", "REC", "SEP", "FNO"]
    assert ordered[-1] == "ZZZ"
    assert set(ordered) == {"ZZZ", "AAA", "SEP", "FNO", "REC", "WAT"}


def test_desk_ui_uses_one_scan_now_for_every_setup():
    root = Path(__file__).resolve().parents[1]
    scanner = (root / "frontend" / "src" / "marketRadarViews.tsx").read_text(encoding="utf-8")
    reco = (root / "frontend" / "src" / "recommendationsViews.tsx").read_text(encoding="utf-8")
    experience = (root / "frontend" / "src" / "experience.tsx").read_text(encoding="utf-8")
    views = (root / "frontend" / "src" / "views.tsx").read_text(encoding="utf-8")
    product = (root / "frontend" / "src" / "productViews.tsx").read_text(encoding="utf-8")
    assert "Run long-term scan" not in scanner
    assert "one scan fills every tab" in scanner
    assert "tab === 'Long-Term' ? longTermScan : marketScan" not in scanner
    assert "Refresh funds" in reco
    assert "keepRicherMemory" in reco
    assert "Refresh long-term" not in reco
    # Contract the actual mechanism, not brittle button copy: Recommendations
    # must invoke the same canonical market-scan runner used by the rest of desk.
    assert "marketScan.start()" in reco
    assert "mode === 'Long-Term' ? longTermScan : marketScan" not in experience
    assert "Run long-term scan" not in experience
    assert "RUN_SCAN_NOW" in views
    assert "Find long-term candidates" not in product
    assert "Refresh funds" in product
