from __future__ import annotations



def test_radar_home_workspace_is_persisted_read_only(monkeypatch):
    import product.live_technicals as live_technicals
    import product.observer_api as observer
    import product.sepa_setup as sepa

    scan = {
        "scanned_at": "2026-09-22T10:00:00+00:00",
        "universe_size": 1,
        "records": [{
            "symbol": "AAA",
            "score": 80,
            "verdict": "BUY",
            "status": "Ready to trade",
            "signals": ["BREAKOUT_52W", "MOMENTUM"],
            "chase_risk": False,
            "volume_ratio": 1.5,
            "rsi": 58,
            "breakout_grade": "A",
            "breakout_conviction": 80,
            "avg_vol20": 1_000_000,
        }],
    }

    monkeypatch.setattr(observer.core, "_scan_payload", lambda: scan)
    monkeypatch.setattr(observer.core, "_long_term_payload", lambda: {"scanned_at": "", "records": []})
    monkeypatch.setattr(
        observer.core,
        "_market_payload",
        lambda: {"health": "Healthy", "breadth": "60%", "trade_stance": "Open", "leaders": [], "laggards": []},
    )
    monkeypatch.setattr(observer.core, "_autonomy_payload", lambda: {})
    monkeypatch.setattr(sepa, "load_persisted_best_setups", lambda _scan_at: None)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("radar GET must not perform heavy enrichment")

    monkeypatch.setattr(sepa, "public_best_setups", forbidden)
    monkeypatch.setattr(live_technicals, "refresh_rows_technicals", forbidden)

    payload = observer.radar_home_workspace()

    assert payload["scan_scanned_at"] == scan["scanned_at"]
    assert payload["counts"]["breakouts"] == 1
    assert payload["best_setups"] == []
    assert "does not recompute" in payload["best_setups_note"]
