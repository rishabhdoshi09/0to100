"""Tests for Reco-style recommendations / market-reports projections."""
from __future__ import annotations

from product.recommendations_workspace import (
    build_market_reports_workspace,
    build_recommendations_workspace,
    card_from_row,
    primary_scan_category,
    risk_tier,
    upside_metrics,
)
from product.radar_workspace import enrich_scan_row


def test_upside_and_risk_from_real_fields_only():
    ups = upside_metrics({"price": 115, "entry": 100, "target": 130})
    assert ups["upside_from_entry_pct"] == 15.0
    assert ups["upside_to_target_pct"] == 13.0
    assert upside_metrics({"price": 0, "entry": 0})["upside_from_entry_pct"] is None
    assert risk_tier({"chase_risk": True}) == "High"
    assert risk_tier({"rsi": 50, "chase_risk": False}) == "Low"


def test_categories_project_from_scan_and_long_term_without_invention():
    scan = {
        "scanned_at": "2026-08-15T10:00:00+00:00",
        "records_status": "CURRENT_DAY",
        "same_ist_day": True,
        "records": [
            {
                "symbol": "TRENDY", "company": "Trendy Co", "score": 70,
                "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                "chase_risk": False, "price": 110, "entry": 100, "target": 125,
                "stop": 95, "rsi": 58, "volume_ratio": 1.5, "avg_vol20": 1e6,
                "momentum_5d": 5, "above_sma50": True, "reasons": ["RS leadership"],
            },
            {
                "symbol": "SNIPE", "company": "Snipe Ltd", "score": 80,
                "signals": ["PRE_BREAKOUT", "BREAKOUT_52W"], "verdict": "BUY",
                "status": "Watch for breakout", "categories": ["PreBreakout"],
                "pivot_distance_pct": 1.0, "chase_risk": False,
                "price": 200, "entry": 201, "target": 230, "stop": 190,
                "rsi": 55, "volume_ratio": 1.8, "avg_vol20": 2e6,
                "breakout_grade": "A", "breakout_conviction": 85,
                "reasons": ["Clean pivot"],
            },
            {
                "symbol": "RECOV", "company": "Recovery Inc", "score": 60,
                "signals": ["DOUBLE_BOTTOM", "ACCUMULATION"], "verdict": "WATCH",
                "status": "Watch", "chase_risk": False,
                "price": 50, "entry": 48, "target": 60, "stop": 45,
                "rsi": 45, "volume_ratio": 1.2, "avg_vol20": 5e5,
                "reasons": ["Base forming"],
            },
            {
                "symbol": "CHASE", "company": "Chase Co", "score": 90,
                "signals": ["MOMENTUM"], "verdict": "WATCH", "status": "Wait for pullback",
                "chase_risk": True, "price": 300, "entry": 280, "target": 340,
                "rsi": 80, "volume_ratio": 2.0, "avg_vol20": 1e6,
            },
            {
                # Coil ≠ recovery — must not land in Recovery Setups.
                "symbol": "COIL", "company": "Coil Ltd", "score": 65,
                "signals": ["CUP_HANDLE", "POCKET_PIVOT", "NR7_COIL"], "verdict": "WATCH",
                "status": "Watch", "chase_risk": False,
                "price": 90, "entry": 88, "target": 100, "rsi": 50,
                "volume_ratio": 1.1, "avg_vol20": 8e5,
            },
        ],
    }
    long_term = {
        "scanned_at": "2026-08-15T09:00:00+00:00",
        "records": [
            {
                "symbol": "QUAL", "company": "Quality Ltd",
                "classification": "QUALITY_COMPOUNDER",
                "fundamental_coverage": 0.8, "fundamental_score": 75,
                "combined_score": 82, "price": 500, "entry": 480, "target": 600,
                "quality_factors": ["ROCE", "Low debt"], "risk_flags": [],
            },
            {
                "symbol": "THIN", "classification": "QUALITY_COMPOUNDER",
                "fundamental_coverage": 0.2, "combined_score": 70,
            },
        ],
    }
    payload = build_recommendations_workspace(
        scan_payload=scan, long_term_payload=long_term, refresh_technicals=False,
        compute_sepa=False,
    )
    by_id = {c["id"]: c for c in payload["categories"]}
    assert payload["schema_version"] == 3
    assert by_id["best_setups"]["count"] == 0
    assert by_id["wealth_builders"]["count"] == 1
    assert by_id["wealth_builders"]["cards"][0]["symbol"] == "QUAL"
    assert "coverage" in (by_id["wealth_builders"]["cards"][0].get("qualify_reason") or "").lower()
    assert by_id["super_trends"]["count"] >= 1
    assert any(c["symbol"] == "TRENDY" for c in by_id["super_trends"]["cards"])
    assert not any(c["symbol"] == "CHASE" for c in by_id["super_trends"]["cards"])
    assert any(c["symbol"] == "SNIPE" for c in by_id["momentum_breakouts"]["cards"])
    assert any(c["symbol"] == "RECOV" for c in by_id["recovery_setups"]["cards"])
    assert not any(c["symbol"] == "COIL" for c in by_id["recovery_setups"]["cards"])
    card = card_from_row(scan["records"][0], category_id="super_trends", category_label="Super Trends")
    assert card["action_badge"] == "Buy"
    assert card["upside_from_entry_pct"] == 10.0
    assert "lifecycle" in payload
    assert payload["disclaimer"]


def test_scan_categories_are_exclusive_primary():
    """A symbol must not appear in two scan buckets."""
    scan = {
        "scanned_at": "2026-08-15T10:00:00+00:00",
        "records": [
            {
                # Would match breakout AND could look trendy — breakout wins.
                "symbol": "BOTH", "score": 85,
                "signals": ["PRE_BREAKOUT", "MOMENTUM", "BREAKOUT_52W"],
                "verdict": "BUY", "status": "Watch for breakout",
                "categories": ["PreBreakout"], "pivot_distance_pct": 0.8,
                "chase_risk": False, "rsi": 55, "volume_ratio": 1.5, "avg_vol20": 1e6,
                "breakout_grade": "B", "above_sma50": True, "momentum_5d": 4,
            },
            {
                # Recovery + momentum signal — recovery before super-trend.
                "symbol": "TURN", "score": 62,
                "signals": ["DOUBLE_BOTTOM", "MOMENTUM"], "verdict": "WATCH",
                "status": "Watch", "chase_risk": False, "rsi": 48,
                "volume_ratio": 1.2, "avg_vol20": 5e5, "above_sma50": True,
                "momentum_5d": 3,
            },
        ],
    }
    payload = build_recommendations_workspace(
        scan_payload=scan, long_term_payload={"records": []}, refresh_technicals=False,
        compute_sepa=False,
    )
    by_id = {c["id"]: c for c in payload["categories"]}
    scan_syms = []
    for cat_id in ("super_trends", "momentum_breakouts", "recovery_setups"):
        scan_syms.extend(c["symbol"] for c in by_id[cat_id]["cards"])
    assert len(scan_syms) == len(set(scan_syms))
    assert any(c["symbol"] == "BOTH" for c in by_id["momentum_breakouts"]["cards"])
    assert not any(c["symbol"] == "BOTH" for c in by_id["super_trends"]["cards"])
    assert any(c["symbol"] == "TURN" for c in by_id["recovery_setups"]["cards"])
    assert not any(c["symbol"] == "TURN" for c in by_id["super_trends"]["cards"])


def test_cup_handle_alone_is_not_recovery():
    row = enrich_scan_row({
        "symbol": "CUP", "signals": ["CUP_HANDLE"], "score": 70,
        "chase_risk": False, "rsi": 50, "volume_ratio": 1.2, "avg_vol20": 1e6,
        "verdict": "WATCH", "status": "Watch",
    })
    assert primary_scan_category(row) is None or primary_scan_category(row)[0] != "recovery_setups"


def test_negative_momentum_not_super_trend():
    row = enrich_scan_row({
        "symbol": "FADE", "signals": ["MOMENTUM"], "score": 90,
        "verdict": "BUY", "status": "Ready to trade", "chase_risk": False,
        "rsi": 58, "volume_ratio": 1.4, "avg_vol20": 1e6,
        "momentum_5d": -1.5, "above_sma50": True,
    })
    assigned = primary_scan_category(row)
    assert assigned is None or assigned[0] != "super_trends"


def test_refresh_stamps_shortlist_quotes_not_scan_head(monkeypatch):
    import product.recommendations_workspace as rw

    called = {}

    def fake_quotes(symbols, ttl=8.0, **kwargs):
        called["symbols"] = list(symbols)
        called["ttl"] = ttl
        return {symbols[0]: {"price": 210.0, "source": "nse"}}

    monkeypatch.setattr("data.live_quotes.get_live_quotes", fake_quotes)
    payload = build_recommendations_workspace(
        scan_payload={
            "scanned_at": "2026-08-18T10:00:00+00:00",
            "records": [
                {
                    "symbol": "SNIPE", "company": "Snipe Ltd", "score": 80,
                    "signals": ["PRE_BREAKOUT", "BREAKOUT_52W"], "verdict": "BUY",
                    "status": "Watch for breakout", "categories": ["PreBreakout"],
                    "pivot_distance_pct": 1.0, "chase_risk": False,
                    "price": 200, "entry": 201, "target": 230, "stop": 190,
                    "rsi": 55, "volume_ratio": 1.8, "avg_vol20": 2e6,
                    "breakout_grade": "A", "breakout_conviction": 85,
                },
            ],
        },
        long_term_payload={"records": []},
        refresh_technicals=True,
        compute_sepa=False,
    )
    assert payload["typical_seconds"] == 8
    assert "load_note" in payload
    assert called["symbols"] == ["SNIPE"]
    card = next(
        c for cat in payload["categories"] if cat["id"] == "momentum_breakouts"
        for c in cat["cards"]
    )
    assert card["cmp"] == 210.0
    assert card["price_tag"] == "LIVE"


def test_wealth_empty_explains_needs_fundamentals():
    payload = build_recommendations_workspace(
        scan_payload={"records": [], "scanned_at": ""},
        long_term_payload={
            "records": [
                {
                    "symbol": "X", "classification": "NEEDS_FUNDAMENTALS",
                    "fundamental_coverage": 0.0, "combined_score": 45,
                },
            ],
        },
        refresh_technicals=False,
        compute_sepa=False,
    )
    wealth = next(c for c in payload["categories"] if c["id"] == "wealth_builders")
    assert wealth["count"] == 0
    assert "NEEDS_FUNDAMENTALS" in wealth["empty_detail"]


def test_empty_recovery_is_honest_empty():
    payload = build_recommendations_workspace(
        scan_payload={"records": [], "scanned_at": ""},
        long_term_payload={"records": []},
        refresh_technicals=False,
        compute_sepa=False,
    )
    recovery = next(c for c in payload["categories"] if c["id"] == "recovery_setups")
    assert recovery["count"] == 0
    assert recovery["cards"] == []
    assert "double-bottom" in recovery["empty_detail"].lower() or "accumulation" in recovery["empty_detail"].lower()


def test_market_reports_lists_today_pulse(tmp_path, monkeypatch):
    import product.recommendations_workspace as rw

    monkeypatch.setattr(rw, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(
        "reports.street_pulse.build_pulse",
        lambda: {
            "date": "15 August 2026",
            "takeaways": ["Nifty steady", "Breadth mixed"],
            "gainers": [], "losers": [], "breakouts_today": [],
        },
    )
    payload = build_market_reports_workspace(persist_today=True)
    assert payload["reports"]
    assert payload["reports"][0]["is_new"] is True
    assert payload["reports"][0]["title"] == "Market Pulse"
    assert (tmp_path / list(tmp_path.glob("market_pulse_*.json"))[0]).exists()
    assert "Nifty" in payload["reports"][0]["summary"]


def test_market_reports_reuse_fresh_file(tmp_path, monkeypatch):
    import json
    import product.recommendations_workspace as rw
    from core.market_clock import today_ist

    monkeypatch.setattr(rw, "REPORTS_DIR", tmp_path)
    day = today_ist().isoformat()
    pulse = {"date": day, "takeaways": ["Saved pulse"], "gainers": []}
    path = tmp_path / f"market_pulse_{day}.json"
    path.write_text(json.dumps({
        "id": f"market_pulse_{day}", "title": "Market Pulse", "kind": "market_pulse",
        "date": day, "created_at": "2026-08-18T10:00:00+00:00", "pulse": pulse,
    }))
    monkeypatch.setattr(
        "reports.street_pulse.build_pulse",
        lambda: (_ for _ in ()).throw(AssertionError("should reuse file")),
    )
    payload = build_market_reports_workspace(persist_today=True)
    assert payload["today_pulse"]["takeaways"] == ["Saved pulse"]
    assert payload["typical_seconds"] == 8


def test_best_setups_ranks_sepa_and_does_not_invent_weak_names():
    import pandas as pd

    index = pd.date_range("2024-01-01", periods=280, freq="B")
    close = pd.Series([80 + i * 0.6 for i in range(280)], index=index)
    strong = pd.DataFrame(
        {
            "open": close - 0.4,
            "high": close + 1.2,
            "low": close - 1.0,
            "close": close,
            "volume": [200000] * 280,
        },
        index=index,
    )
    down = pd.Series([200 - i * 0.4 for i in range(280)], index=index)
    weak = pd.DataFrame(
        {
            "open": down + 0.3,
            "high": down + 1,
            "low": down - 1,
            "close": down,
            "volume": [150000] * 280,
        },
        index=index,
    )
    frames = {"LEADER": strong, "LAGGER": weak}

    payload = build_recommendations_workspace(
        scan_payload={
            "scanned_at": "2026-08-18T10:00:00+00:00",
            "records": [
                {
                    "symbol": "LEADER", "company": "Leader Ltd", "score": 88,
                    "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                    "chase_risk": False, "price": float(close.iloc[-1]),
                    "entry": float(close.iloc[-1]), "target": float(close.iloc[-1]) * 1.1,
                    "stop": float(close.iloc[-1]) * 0.95, "rsi": 58, "volume_ratio": 1.4,
                    "avg_vol20": 1e6,
                },
                {
                    "symbol": "LAGGER", "company": "Lagger Ltd", "score": 70,
                    "signals": ["MOMENTUM"], "verdict": "WATCH", "status": "Watch",
                    "chase_risk": False, "price": float(down.iloc[-1]),
                    "rsi": 40, "volume_ratio": 1.1, "avg_vol20": 1e6,
                },
            ],
        },
        long_term_payload={"records": []},
        refresh_technicals=False,
        compute_sepa=True,
        sepa_load_frame=lambda symbol: frames.get(symbol),
    )
    best = next(c for c in payload["categories"] if c["id"] == "best_setups")
    assert payload["categories"][0]["id"] == "best_setups"
    assert best["count"] == 1
    card = best["cards"][0]
    assert card["symbol"] == "LEADER"
    assert card["sepa_score"] >= 80
    assert card["sepa_passed"] >= 6
    assert card["sepa_verdict"] == "STRONG"
    assert not any(c["symbol"] == "LAGGER" for c in best["cards"])


def test_serve_recommendations_reuses_disk_cache(tmp_path, monkeypatch):
    import product.recommendations_workspace as rw
    monkeypatch.setattr(rw, "WORKSPACE_CACHE", tmp_path / "recommendations_workspace.json")
    scan = {
        "scanned_at": "2026-08-19T10:00:00+00:00",
        "records": [
            {
                "symbol": "TRENDY", "company": "Trendy Co", "score": 70,
                "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                "chase_risk": False, "price": 110, "entry": 100, "target": 125,
                "stop": 95, "rsi": 58, "volume_ratio": 1.5, "avg_vol20": 1e6,
            },
        ],
    }
    first = rw.serve_recommendations_workspace(
        scan_payload=scan,
        long_term_payload={"scanned_at": "2026-08-19T09:00:00+00:00", "records": []},
        refresh_technicals=False,
        compute_sepa=False,
    )
    assert first["served_from_cache"] is False
    assert (tmp_path / "recommendations_workspace.json").exists()
    second = rw.serve_recommendations_workspace(
        scan_payload=scan,
        long_term_payload={"scanned_at": "2026-08-19T09:00:00+00:00", "records": []},
        refresh_technicals=False,
        compute_sepa=False,
    )
    assert second["served_from_cache"] is True
    assert second["categories"][0]["id"] == first["categories"][0]["id"]
    stale = rw.serve_recommendations_workspace(
        scan_payload={**scan, "scanned_at": "2026-08-20T10:00:00+00:00"},
        long_term_payload={"scanned_at": "2026-08-19T09:00:00+00:00", "records": []},
        refresh_technicals=False,
        compute_sepa=False,
        background_rebuild=False,
    )
    assert stale["served_from_cache"] is True
    assert stale.get("stale_ranking") is True
    assert rw._card_count(stale) == rw._card_count(first)


def test_serve_keeps_last_ranking_when_scan_file_goes_empty(tmp_path, monkeypatch):
    import product.recommendations_workspace as rw
    monkeypatch.setattr(rw, "WORKSPACE_CACHE", tmp_path / "recommendations_workspace.json")
    scan = {
        "scanned_at": "2026-08-19T10:00:00+00:00",
        "records": [
            {
                "symbol": "TRENDY", "company": "Trendy Co", "score": 70,
                "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                "chase_risk": False, "price": 110, "entry": 100, "target": 125,
                "stop": 95, "rsi": 58, "volume_ratio": 1.5, "avg_vol20": 1e6,
            },
        ],
    }
    first = rw.serve_recommendations_workspace(
        scan_payload=scan,
        long_term_payload={"scanned_at": "2026-08-19T09:00:00+00:00", "records": []},
        refresh_technicals=False,
        compute_sepa=False,
    )
    assert rw._card_count(first) > 0
    empty = rw.serve_recommendations_workspace(
        scan_payload={"scanned_at": "", "records": []},
        long_term_payload={"scanned_at": "", "records": []},
        refresh_technicals=False,
        compute_sepa=False,
    )
    assert empty["served_from_cache"] is True
    assert empty.get("stale_ranking") is True
    assert rw._card_count(empty) == rw._card_count(first)


def test_serve_does_not_block_the_request_on_sepa(tmp_path, monkeypatch):
    import product.recommendations_workspace as rw
    monkeypatch.setattr(rw, "WORKSPACE_CACHE", tmp_path / "recommendations_workspace.json")
    calls: list[str] = []

    def hang(symbol: str):
        calls.append(symbol)
        raise AssertionError("SEPA must not run on the Ideas request thread")

    scan = {
        "scanned_at": "2026-08-20T10:00:00+00:00",
        "records": [
            {
                "symbol": "TRENDY", "company": "Trendy Co", "score": 70,
                "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                "chase_risk": False, "price": 110, "entry": 100, "target": 125,
                "stop": 95, "rsi": 58, "volume_ratio": 1.5, "avg_vol20": 1e6,
            },
        ],
    }
    import time
    t0 = time.monotonic()
    first = rw.serve_recommendations_workspace(
        scan_payload=scan,
        long_term_payload={"scanned_at": "2026-08-20T09:00:00+00:00", "records": []},
        refresh_technicals=False,
        compute_sepa=True,
        sepa_load_frame=hang,
        background_rebuild=False,
    )
    assert time.monotonic() - t0 < 5.0
    assert calls == []
    assert first["served_from_cache"] is False
    assert first.get("sepa_pending") is True
    assert rw._card_count(first) > 0

    later = rw.serve_recommendations_workspace(
        scan_payload={**scan, "scanned_at": "2026-08-21T10:00:00+00:00"},
        long_term_payload={"scanned_at": "2026-08-20T09:00:00+00:00", "records": []},
        refresh_technicals=False,
        compute_sepa=True,
        sepa_load_frame=hang,
        background_rebuild=False,
    )
    assert calls == []
    assert later["served_from_cache"] is True
    assert later.get("stale_ranking") is True
    assert rw._card_count(later) == rw._card_count(first)


def test_serve_kicks_background_sepa_when_scan_stamp_moves(tmp_path, monkeypatch):
    import product.recommendations_workspace as rw
    monkeypatch.setattr(rw, "WORKSPACE_CACHE", tmp_path / "recommendations_workspace.json")
    kicked: list[str] = []

    def capture(scan, lt, sepa_load_frame=None):
        kicked.append(str(scan.get("scanned_at") or ""))
        return True

    monkeypatch.setattr(rw, "kick_sepa_rebuild", capture)
    scan = {
        "scanned_at": "2026-08-19T10:00:00+00:00",
        "records": [
            {
                "symbol": "TRENDY", "company": "Trendy Co", "score": 70,
                "signals": ["MOMENTUM"], "verdict": "BUY", "status": "Ready to trade",
                "chase_risk": False, "price": 110, "entry": 100, "target": 125,
                "stop": 95, "rsi": 58, "volume_ratio": 1.5, "avg_vol20": 1e6,
            },
        ],
    }
    rw.serve_recommendations_workspace(
        scan_payload=scan,
        long_term_payload={"scanned_at": "2026-08-19T09:00:00+00:00", "records": []},
        refresh_technicals=False,
        compute_sepa=True,
        background_rebuild=True,
    )
    assert kicked == ["2026-08-19T10:00:00+00:00"]
    kicked.clear()
    rw.serve_recommendations_workspace(
        scan_payload={**scan, "scanned_at": "2026-08-20T10:00:00+00:00"},
        long_term_payload={"scanned_at": "2026-08-19T09:00:00+00:00", "records": []},
        refresh_technicals=False,
        compute_sepa=True,
        background_rebuild=True,
    )
    assert kicked == ["2026-08-20T10:00:00+00:00"]


def test_cached_workspace_mismatch_is_ignored(tmp_path, monkeypatch):
    import product.recommendations_workspace as rw
    monkeypatch.setattr(rw, "WORKSPACE_CACHE", tmp_path / "recommendations_workspace.json")
    assert rw.load_cached_recommendations_workspace("a", "b") is None
    rw.persist_recommendations_workspace("a", "b", {"categories": []})
    assert rw.load_cached_recommendations_workspace("a", "b")["categories"] == []
    assert rw.load_cached_recommendations_workspace("other", "b") is None

