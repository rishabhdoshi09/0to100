"""One-origin evidence, session honesty, options EOD watch, gated EV, journal, rotation."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from product.recommendations_workspace import card_from_row
from product.session_honesty import format_session_label, session_payload
from product.system_loops import decision_journal_workspace, portfolio_intel_workspace


def test_session_weekend_banner_uses_last_official_session():
    now = datetime(2026, 8, 16, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    payload = session_payload(now=now, last_session="2026-08-14", market_open=False)
    assert payload["is_weekend"] is True
    assert payload["market_open"] is False
    assert payload["state"] == "weekend"
    assert payload["last_session"] == "2026-08-14"
    assert "Friday 14 Aug 2026" in payload["banner"]
    assert "5 minutes" in payload["retry_note"]
    assert format_session_label("2026-08-14") == "Friday 14 Aug 2026"


def test_session_open_banner_does_not_look_closed():
    now = datetime(2026, 8, 14, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    payload = session_payload(now=now, last_session="2026-08-14", market_open=True)
    assert payload["state"] == "open"
    assert payload["banner"].startswith("NSE open")


def test_eod_watch_merges_defaults_and_caps(tmp_path):
    from options.eod_watch import MAX_WATCH, add_watch, capture_list, watched_symbols

    path = tmp_path / "eod_watch.json"
    first = add_watch("reliance", path=path)
    assert first["accepted"] is True
    assert "RELIANCE" in first["watched"]
    again = add_watch("RELIANCE", path=path)
    assert again["already"] is True
    assert watched_symbols(path) == ["RELIANCE"]
    capture = capture_list(path=path)
    assert capture[:3] == ["NIFTY", "BANKNIFTY", "FINNIFTY"]
    assert "RELIANCE" in capture
    for i in range(MAX_WATCH + 5):
        add_watch(f"SYM{i}", path=path)
    assert len(watched_symbols(path)) == MAX_WATCH


def test_capture_universe_uses_watch_when_symbols_omitted(monkeypatch, tmp_path):
    from options import eod_snapshot as snap
    from options import eod_watch as watch

    path = tmp_path / "eod_watch.json"
    watch.add_watch("RELIANCE", path=path)
    monkeypatch.setattr(watch, "WATCH_PATH", path)
    monkeypatch.setattr(snap, "capture_symbol", lambda symbol, as_of=None: {
        "symbol": symbol, "available": False, "saved": False, "message": "test",
    })
    monkeypatch.setattr(snap, "store_status", lambda: {"available": False, "latest_as_of": ""})
    out = snap.capture_universe()
    requested = [row["symbol"] for row in out["results"]]
    assert requested[:3] == ["NIFTY", "BANKNIFTY", "FINNIFTY"]
    assert "RELIANCE" in requested


def test_card_from_row_gates_ev_below_thirty():
    thin = card_from_row(
        {"symbol": "THIN", "ev_pct": 1.4, "ev_n": 12, "entry": 100, "target": 110},
        category_id="super_trends",
        category_label="Super Trends",
    )
    assert "ev_pct" not in thin
    fat = card_from_row(
        {
            "symbol": "FAT",
            "ev_pct": 1.4,
            "ev_lb_pct": 0.6,
            "ev_n": 40,
            "ev_conf": "HIGH",
            "p_win": 58,
            "entry": 100,
            "target": 110,
        },
        category_id="super_trends",
        category_label="Super Trends",
    )
    assert fat["ev_pct"] == 1.4
    assert fat["ev_n"] == 40
    assert fat["ev_conf"] == "HIGH"


def test_decision_journal_workspace_is_honest_when_empty():
    payload = decision_journal_workspace()
    assert payload["available"] is True
    assert payload["advice_only"] is True
    assert payload["thin"] is True
    assert "No claim" in payload["message"]


def test_portfolio_intel_workspace_is_advice_only():
    payload = portfolio_intel_workspace()
    assert payload.get("advice_only") is True
    assert payload.get("swap") in (None, payload.get("swap"))
    assert "message" in payload


def test_terminal_app_exposes_evidence_and_loop_routes():
    import terminal_product_api as tpa

    paths = {route.path for route in tpa.app.routes}
    assert "/evidence/{symbol}" in paths
    assert "/evidence/templates/{kind}.csv" in paths
    assert "/api/decision-journal" in paths
    assert "/api/portfolio-intel" in paths
    assert "/api/session-honesty" in paths
    assert "/api/market/options/{symbol}/watch-eod" in paths


def test_evidence_template_still_served_from_report_api():
    import report_api

    response = report_api.evidence_template("financial_history")
    assert response.media_type == "text/csv"
    assert b"period_end" in response.body


def test_chain_backoff_payload_explains_retry(monkeypatch):
    from options import chain_fetch as CF

    CF._WS_FAIL_S["NIFTY"] = 1_000_000.0
    monkeypatch.setattr(CF.time, "time", lambda: 1_000_010.0)
    monkeypatch.setattr(CF, "chain_workspace", lambda symbol, spot=None: {
        "available": False, "symbol": symbol, "message": "forced miss",
    })
    payload = CF.chain_workspace_cached("NIFTY")
    assert payload["available"] is False
    assert payload["backoff"] is True
    assert payload["force_bypasses_backoff"] is True
    assert payload["retry_after_s"] > 0
    assert "dead button" in payload["message"]
    forced = CF.chain_workspace_cached("NIFTY", force=True)
    assert forced["message"] == "forced miss"
    CF._WS_FAIL_S.pop("NIFTY", None)
