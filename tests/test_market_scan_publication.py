from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from scan import market_scan_service as S


class _Scanner:
    def scan(self, symbols, *, progress=None, prefetch=False):
        if progress is not None:
            progress(1, 1)
        return [{
            "symbol": "AAA",
            "score": 80.0,
            "verdict": "BUY",
            "signals": ["MOMENTUM"],
            "reasons": ["qualified"],
            "price": 100.0,
            "entry": 101.0,
            "stop": 95.0,
            "target": 113.0,
        }]


class _Probe:
    def finalize(self, results, *, cached, walked_total):
        return {
            "summary": {
                "scanner_instrumented": True,
                "checked": 1,
                "failed": 0,
                "state": "COMPLETE",
                "reason_counts": {"QUALIFIED": 1},
            },
            "ledger": [],
        }


@contextmanager
def _observe_scanner(_scanner, _symbols):
    yield _Probe()


def test_scan_is_published_only_after_identity_and_overlays(monkeypatch):
    events = []
    saved = {}

    monkeypatch.setattr(S, "_saved_priority_inputs", lambda: (None, None, None, []))
    monkeypatch.setattr(S, "_feature002_hook", None)
    monkeypatch.setattr("scan.scan_coverage.observe_scanner", _observe_scanner)
    monkeypatch.setattr("scan.scan_coverage.save_audit", lambda _audit: None)
    monkeypatch.setattr("scan.bulk_fetcher.cached_symbols", lambda: ["AAA"])
    monkeypatch.setattr(
        "product.scan_store._history_freshness",
        lambda: ({
            "available_session": "2026-09-18",
            "expected_latest_completed_session": "2026-09-18",
            "reason_code": "HISTORY_CURRENT",
            "stale_sessions": 0,
            "current": True,
        }, ""),
    )

    def save_scan(payload):
        events.append("save")
        saved.update(dict(payload))

    monkeypatch.setattr("product.scan_store.save_scan", save_scan)
    monkeypatch.setattr(
        "product.sepa_setup.persist_public_best_setups",
        lambda _payload: events.append("best"),
    )

    def overlay(_payload, *, refresh_fundamentals=False, save=True):
        events.append("long")
        return SimpleNamespace(status="SUCCEEDED", payload={"records": []}, error_code="")

    monkeypatch.setattr(
        "scan.long_term_service.overlay_long_term_from_market_scan",
        overlay,
    )

    def desks(_payload):
        events.append("desks")
        return {"recommendations": "saved", "market_reports": "saved"}

    monkeypatch.setattr(
        "product.desk_scan_overlays.persist_desks_from_market_scan",
        desks,
    )

    report = S.run_whole_market_scan(
        universe_provider=lambda: {"AAA": "Alpha"},
        prefetch_fn=lambda *_args, **_kwargs: None,
        scanner=_Scanner(),
        fno_provider=lambda: set(),
        save=True,
    )

    assert report.ok is True
    assert events == ["best", "long", "desks", "save"]
    assert saved["as_of_session"] == "2026-09-18"
    assert saved["history_latest_date"] == "2026-09-18"
    assert saved["scan_status"] == "SUCCEEDED"
    assert saved["long_term_overlay"]["status"] == "SUCCEEDED"
    assert saved["desk_overlays"]["recommendations"] == "saved"
    assert saved["records"][0]["status_scope"] == "SCANNER_SETUP"
    assert saved["records"][0]["selection_required"] is True
    assert saved["summary"]["setup_ready"] == 1
    assert saved["summary"]["ready_to_trade_scope"] == "SCANNER_SETUP_ONLY"


def test_scan_publication_filters_non_stock_funds(monkeypatch):
    monkeypatch.setattr(
        "product.scan_store._history_freshness",
        lambda: ({
            "available_session": "2026-09-24",
            "expected_latest_completed_session": "2026-09-24",
            "reason_code": "HISTORY_CURRENT",
            "stale_sessions": 0,
            "current": True,
        }, ""),
    )

    from product.scan_store import build_scan_payload

    rows = [
        {
            "symbol": "AAA",
            "score": 80.0,
            "verdict": "BUY",
            "signals": ["MOMENTUM"],
            "reasons": ["qualified"],
            "price": 100.0,
            "entry": 101.0,
            "stop": 95.0,
            "target": 113.0,
        },
        {
            "symbol": "HDFCLIQUID",
            "score": 99.0,
            "verdict": "BUY",
            "signals": ["PRE_BREAKOUT"],
            "reasons": ["synthetic liquid-fund setup"],
            "price": 1079.6,
            "entry": 1079.6,
            "stop": 1079.2,
            "target": 1080.5,
        },
    ]
    payload = build_scan_payload(
        {"AAA": "Alpha", "HDFCLIQUID": "HDFC NIFTY 1D RATE LIQUID - GROWTH ETF"},
        rows,
        scanned=2,
        approved_universe=2,
    )

    assert [r["symbol"] for r in payload["records"]] == ["AAA"]
    assert payload["qualified_rows"] == 1
    assert payload["summary"]["setup_ready"] == 1
