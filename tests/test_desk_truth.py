"""Desk truth: real engines, provenance, PIT safety, and health — not theatre."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from product.counterfactual_learning import CORRECT_REJECTION, MISSED_WINNER
from product.forward_evidence import BACKTEST
from product.historical_replay import (
    ENGINE,
    decide_session,
    evaluate_outcomes,
    ohlcv_as_of,
    run_historical_replay,
    scan_session,
)
from product.paper_autopilot import evaluate_candidate
from product.runtime_lifecycle import LIFECYCLES, inspect_runtime
from reporting.evidence_intake import PARSER_VERSION, evidence_requirements, save_upload


def _bars(end: str, n: int = 80, *, future_days: int = 0, start: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(end) + pd.Timedelta(days=future_days), periods=n + future_days)
    close = [start + i * 0.4 for i in range(len(idx))]
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [200_000] * len(idx),
        },
        index=idx,
    )


def test_a_acquire_survives_failing_external_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "fundamentals.fetcher.get_deep_fundamentals",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("screener down")),
    )
    monkeypatch.setattr(
        "product.due_diligence.acquire._nse_session",
        lambda: (_ for _ in ()).throw(RuntimeError("nse down")),
    )
    from product.due_diligence.acquire import acquire_symbol

    payload = acquire_symbol("QTTRUTHA", force=True)
    assert payload["symbol"] == "QTTRUTHA"
    assert payload["places_orders"] is False
    errors = [step.get("error") for step in payload.get("steps") or [] if step.get("error")]
    assert errors, payload.get("steps")


def test_b_auto_acquisition_stores_provenance(tmp_path, monkeypatch):
    from product.due_diligence import acquire as acquire_mod

    facts_dir = tmp_path / "INFY" / "autonomy"
    facts_dir.mkdir(parents=True)
    monkeypatch.setattr(acquire_mod, "EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr("reporting.evidence_intake._autonomy_pack", lambda symbol: {
        "acquired_at": "2026-09-01T10:00:00+00:00",
        "downloads": [{
            "url": "https://nsearchives.nseindia.com/annual/INFY.pdf",
            "ok": True,
            "path": "nse_ar_INFY.pdf",
        }],
        "steps": [{"id": "nse_annual_reports", "ok": True}],
        "_files": [{
            "kind": "annual_report",
            "filename": "nse_ar_INFY.pdf",
            "sha256": "abc123",
        }],
        "commentary": [{
            "source": "NSE filing",
            "source_url": "https://www.nseindia.com/get-quotes/equity?symbol=INFY",
            "event_date": "2026-07-15",
            "commentary": "Management said demand stayed healthy.",
        }],
    })
    monkeypatch.setattr(
        "reporting.evidence_intake.load_raw_fundamentals",
        lambda symbol: {
            "available": True,
            "data": {
                "about": "Infosys is an IT services company.",
                "url": "https://www.screener.in/company/INFY/",
                "quarterly_results": [{"period": "Jun 2026", "sales": 1}, {"period": "Mar 2026", "sales": 1}, {"period": "Dec 2025", "sales": 1}, {"period": "Sep 2025", "sales": 1}],
                "profit_loss": [{"period": "Mar 2026", "sales": 1}],
            },
            "fetched_at": "2026-09-02T00:00:00+00:00",
            "age_days": 0,
            "freshness": "FRESH",
            "section_as_of": {},
        },
    )
    payload = evidence_requirements("INFY")
    profile = next(row for row in payload["requirements"] if row["key"] == "business_profile")
    assert profile["acquisition"] == "AUTO_SOURCED"
    assert profile["source_url"]
    assert profile["acquired_at"]
    assert profile["parser"] == PARSER_VERSION
    assert "IT services" in profile["evidence"]
    assert profile["status"] == "UNKNOWN_DATE"
    annual = next(row for row in payload["requirements"] if row["key"] == "annual_report")
    assert annual["acquisition"] == "AUTO_SOURCED"
    assert annual["sha256"] == "abc123"


def test_c_failed_auto_offers_manual_fallback(monkeypatch):
    monkeypatch.setattr("reporting.evidence_intake._autonomy_pack", lambda symbol: {
        "acquired_at": "2026-09-01T10:00:00+00:00",
        "downloads": [{"url": "https://www.nseindia.com/companies-listing/corporate-filings-annual-reports", "ok": False, "error": "403"}],
        "steps": [{"id": "nse_annual_reports", "ok": False, "error": "NSE annual report download failed"}],
        "_files": [],
    })
    monkeypatch.setattr(
        "reporting.evidence_intake.load_raw_fundamentals",
        lambda symbol: {"available": False, "data": {}, "fetched_at": "", "age_days": None, "freshness": "MISSING", "section_as_of": {}},
    )
    payload = evidence_requirements("QTFAIL")
    annual = next(row for row in payload["requirements"] if row["key"] == "annual_report")
    assert annual["acquisition"] == "AUTOMATION_FAILED"
    assert annual["available"] is False
    assert "failed" in annual["failure_reason"].lower()
    assert annual["sources_attempted"]
    assert "Manual" in annual["instructions"] or "fallback" in annual["instructions"].lower()


def test_d_duplicate_upload_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("reporting.evidence_intake.ROOT", tmp_path)
    monkeypatch.setattr("reporting.evidence_intake.EVIDENCE_ROOT", tmp_path / "research_evidence")
    content = b"as_of_date,business_summary,customers,demand_drivers,source_url\n2026-03-31,Software services,Enterprises,IT spend,https://www.nseindia.com/get-quotes/equity?symbol=INFY\n"
    first = save_upload(
        "INFY", "business_profile", content,
        filename="profile.csv", as_of="2026-03-31",
        source_url="https://www.nseindia.com/get-quotes/equity?symbol=INFY",
    )
    second = save_upload(
        "INFY", "business_profile", content,
        filename="profile.csv", as_of="2026-03-31",
        source_url="https://www.nseindia.com/get-quotes/equity?symbol=INFY",
    )
    assert first["sha256"] == second["sha256"]
    from reporting.evidence_intake import list_uploads
    rows = [row for row in list_uploads("INFY") if row.get("kind") == "business_profile"]
    assert len(rows) == 1


def test_e_replay_calls_evaluate_candidate(tmp_path):
    called: list[str] = []

    def decide(card, **kwargs):
        called.append(str(card.get("symbol") or ""))
        return evaluate_candidate(card, **kwargs)

    scan = {
        "records": [{
            "symbol": "INFY",
            "status": "Ready to trade",
            "verdict": "BUY",
            "price": 1500,
            "score": 80,
            "rsi": 58,
            "volume_ratio": 1.4,
            "entry": 1490,
            "stop": 1400,
            "target": 1650,
            "chase_risk": False,
            "signals": ["MOMENTUM"],
            "reasons": ["relative strength"],
            "reco_tier": "HIGH",
            "entry_state": "ready",
        }],
        "scanned_at": "2026-06-12T15:30:00+05:30",
        "as_of_session": "2026-06-12",
    }
    rows = decide_session("2026-06-12", scan, decide_fn=decide)
    assert called == ["INFY"]
    assert rows
    assert rows[0]["engine"] == ENGINE
    assert rows[0]["pit"]["future_evidence_used"] is False
    assert rows[0]["pit"]["degraded"]


def test_f_historical_decision_cannot_see_future_bars():
    frame = _bars("2026-06-12", n=80, future_days=10)
    sliced = ohlcv_as_of("INFY", "2026-06-12", ohlcv_fn=lambda _s: frame)
    assert sliced is not None
    last = str(sliced.index[-1].date())
    assert last <= "2026-06-12"
    assert len(sliced) < len(frame)

    leaked: list[str] = []

    def analyzer(symbol: str, bars):
        last_bar = str(bars.index[-1].date())
        if last_bar > "2026-06-12":
            leaked.append(last_bar)
        return None

    scan_session("2026-06-12", ["INFY"], ohlcv_fn=lambda _s: frame, analyzer=analyzer)
    assert leaked == []


def test_g_replay_persists_decision_records(tmp_path):
    frame = _bars("2026-06-20", n=90)

    def analyzer(symbol: str, bars):
        return SimpleNamespace(
            symbol=symbol, signals=["MOMENTUM"], reasons=["tape"],
            verdict="WATCH", price=float(bars["close"].iloc[-1]),
            score=70, rsi=60, volume_ratio=1.2, momentum_5d=3.5,
            entry=100, stop=90, target=120, chase_risk=False,
        )

    def decide(card, **_k):
        return evaluate_candidate(
            {**card, "reco_tier": "WATCH", "entry_state": "watch", "chase_risk": False},
            book=None, entries_allowed=True, paper_enabled=True, workspace=None,
            now=datetime(2026, 6, 12, tzinfo=timezone.utc),
        )

    first = run_historical_replay(
        sessions=2,
        universe_limit=1,
        symbols=["INFY"],
        force=True,
        directory=tmp_path,
        ohlcv_fn=lambda _s: frame,
        dates_fn=lambda: ["2026-06-11", "2026-06-12", "2026-06-13"],
        analyzer=analyzer,
        decide_fn=decide,
    )
    assert first["status"] in {"SUCCEEDED", "DEGRADED"}
    assert first["run_id"]
    assert first["engine"] == ENGINE
    assert first["provenance"] == BACKTEST
    assert Path(tmp_path / "latest.json").exists()
    assert first["decisions_tested"] >= 1
    ledger = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert ledger
    row = json.loads(ledger[0])
    assert row["symbol"] == "INFY"
    assert row["as_of"] in {"2026-06-11", "2026-06-12"}
    assert row["decision"]


def test_h_outcome_links_to_historical_decision():
    rows = evaluate_outcomes([
        {"symbol": "AAA", "as_of": "2026-06-01", "decision": "REJECT", "entry": 100, "stop": 95, "target": 110, "forward_return_pct": -6},
        {"symbol": "BBB", "as_of": "2026-06-01", "decision": "REJECT", "entry": 100, "stop": 95, "target": 110, "forward_return_pct": 9},
        {"symbol": "CCC", "as_of": "2026-06-01", "decision": "WAIT", "forward_return_pct": None},
    ])
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["AAA"]["classification"] == CORRECT_REJECTION
    assert by_symbol["BBB"]["classification"] == MISSED_WINNER
    assert by_symbol["CCC"]["outcome_status"] == "UNRESOLVED"


def test_i_scanner_request_executes_real_analyzer():
    from scan.unified_scanner import UnifiedScanner

    frame = _bars("2026-06-12", n=90, start=200)
    frame["volume"] = 2_000_000
    analyzed: list[str] = []
    scanner = UnifiedScanner(max_workers=1)
    original = scanner._analyze

    def spy(symbol, bars):
        analyzed.append(symbol)
        return original(symbol, bars)

    payload = scan_session("2026-06-12", ["INFY"], ohlcv_fn=lambda _s: frame, analyzer=spy)
    assert analyzed == ["INFY"]
    assert payload["engine"] == "scan.unified_scanner.UnifiedScanner._analyze"
    assert payload["pit"] is True
    assert "rejected_candidates" in payload


def test_j_recommendation_request_uses_production_workspace():
    from product.recommendations_workspace import build_recommendations_workspace

    scan = {
        "records": [{
            "symbol": "TCS",
            "status": "Ready to trade",
            "verdict": "BUY",
            "price": 3500,
            "score": 88,
            "rsi": 55,
            "volume_ratio": 1.6,
            "entry": 3480,
            "stop": 3300,
            "target": 3800,
            "chase_risk": False,
            "signals": ["MOMENTUM", "PRE_BREAKOUT"],
            "reasons": ["breakout quality"],
        }],
        "scanned_at": "2026-06-12T15:30:00+05:30",
    }
    live = build_recommendations_workspace(scan_payload=scan, persist_ledger=False)
    pit = build_recommendations_workspace(
        scan_payload=scan, persist_ledger=False, point_in_time=True, as_of="2026-06-12",
    )
    assert pit["point_in_time"] is True
    assert pit["pit_degraded"]
    assert live.get("point_in_time") is False
    decision = evaluate_candidate(
        {
            "symbol": "TCS",
            "reco_tier": "HIGH",
            "entry_state": "ready",
            "entry": 3480,
            "stop": 3300,
            "target": 3800,
            "volume_ratio": 1.6,
        },
        book=None,
        entries_allowed=True,
        paper_enabled=True,
        workspace=None,
    )
    assert decision.decision
    assert decision.reason_code


def test_k_health_endpoint_reports_real_lifecycle():
    from fastapi.testclient import TestClient
    import terminal_api

    client = TestClient(terminal_api.app)
    payload = client.get("/api/health").json()
    assert payload["lifecycle"] in LIFECYCLES
    assert "reason" in payload
    runtime = inspect_runtime(api_serving=True)
    assert runtime["lifecycle"] in LIFECYCLES
    assert runtime["live_locked"] is True


def test_l_restart_does_not_duplicate_replay(tmp_path):
    frame = _bars("2026-06-20", n=90)

    def analyzer(symbol: str, bars):
        return SimpleNamespace(
            symbol=symbol, signals=["MOMENTUM"], reasons=["tape"],
            verdict="WATCH", price=100, score=70, rsi=50, volume_ratio=1.0, momentum_5d=3.0,
            entry=100, stop=90, target=120, chase_risk=False,
        )

    kwargs = dict(
        sessions=2,
        universe_limit=1,
        symbols=["INFY"],
        directory=tmp_path,
        ohlcv_fn=lambda _s: frame,
        dates_fn=lambda: ["2026-06-11", "2026-06-12", "2026-06-13"],
        analyzer=analyzer,
        decide_fn=lambda card, **_k: evaluate_candidate(
            {**card, "reco_tier": "WATCH"}, book=None, entries_allowed=True, paper_enabled=True, workspace=None,
        ),
    )
    first = run_historical_replay(force=True, **kwargs)
    ledger_one = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    second = run_historical_replay(force=False, **kwargs)
    ledger_two = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert second.get("cache_hit") is True
    assert second["run_id"] == first["run_id"]
    assert ledger_two == ledger_one


def test_report_auto_acquire_route_exists():
    import report_api
    paths = {route.path for route in report_api.app.routes}
    assert "/evidence/{symbol}/actions/auto-acquire" in paths
