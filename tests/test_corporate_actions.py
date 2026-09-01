from __future__ import annotations

import product.corporate_actions as ca


def test_official_empty_is_valid_and_does_not_scrape_secondary(monkeypatch, tmp_path):
    monkeypatch.setattr(ca, "CACHE_ROOT", tmp_path)
    called = {"secondary": 0}
    monkeypatch.setattr(ca, "_fetch_nse", lambda symbol: [])

    def secondary(symbol):
        called["secondary"] += 1
        return [{"symbol": symbol}]

    monkeypatch.setattr(ca, "_fetch_screener", secondary)
    result = ca.get_corporate_actions("TCS", force_refresh=True)
    assert result["available"] is True
    assert result["delivery_state"] == "FRESH_OFFICIAL"
    assert result["actions"] == []
    assert called["secondary"] == 0


def test_nse_failure_uses_reputable_secondary(monkeypatch, tmp_path):
    monkeypatch.setattr(ca, "CACHE_ROOT", tmp_path)

    def fail(symbol):
        raise RuntimeError("NSE blocked")

    monkeypatch.setattr(ca, "_fetch_nse", fail)
    monkeypatch.setattr(ca, "_fetch_screener", lambda symbol: [{
        "symbol": symbol,
        "action_type": "DIVIDEND",
        "subject": "Final dividend announced",
        "source": "Screener.in",
        "source_tier": "reputable_secondary",
    }])
    result = ca.get_corporate_actions("TCS", force_refresh=True)
    assert result["delivery_state"] == "FALLBACK_SECONDARY"
    assert result["source_tier"] == "reputable_secondary"
    assert result["actions"][0]["action_type"] == "DIVIDEND"
    assert result["attempts"][0]["ok"] is False


def test_live_failures_serve_stale_last_good(monkeypatch, tmp_path):
    monkeypatch.setattr(ca, "CACHE_ROOT", tmp_path)
    ca._save_cache("TCS", {
        "schema_version": 1,
        "symbol": "TCS",
        "available": True,
        "delivery_state": "FRESH_OFFICIAL",
        "source": "NSE India",
        "source_tier": "official_exchange",
        "retrieved_at": "2026-08-01T00:00:00+00:00",
        "actions": [{"symbol": "TCS", "action_type": "DIVIDEND", "subject": "Dividend"}],
        "count": 1,
    })

    def fail(symbol):
        raise RuntimeError("internet down")

    monkeypatch.setattr(ca, "_fetch_nse", fail)
    monkeypatch.setattr(ca, "_fetch_screener", fail)
    result = ca.get_corporate_actions("TCS", force_refresh=True)
    assert result["available"] is True
    assert result["delivery_state"] == "STALE_LAST_GOOD"
    assert result["actions"][0]["action_type"] == "DIVIDEND"
    assert len(result["attempts"]) == 2


def test_no_source_and_no_snapshot_is_explicit_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(ca, "CACHE_ROOT", tmp_path)

    def fail(symbol):
        raise RuntimeError("down")

    monkeypatch.setattr(ca, "_fetch_nse", fail)
    monkeypatch.setattr(ca, "_fetch_screener", fail)
    result = ca.get_corporate_actions("TCS", force_refresh=True)
    assert result["available"] is False
    assert result["delivery_state"] == "UNAVAILABLE"
    assert result["actions"] == []


def test_action_classifier_does_not_guess_dates():
    assert ca._action_type("Dividend - Rs 30 per share") == "DIVIDEND"
    assert ca._action_type("Bonus 1:1") == "BONUS"
    assert ca._action_type("Sub-division of equity shares") == "SPLIT"
