"""Telegram /trade must gate on live/EOD RSI — not frozen scan oscillators."""
from __future__ import annotations


def test_trade_now_refreshes_technicals_before_rsi_gate(monkeypatch):
    import alerts.telegram_commands as tc

    monkeypatch.setattr(
        "execution.autopilot.get_status",
        lambda: {"armed": True, "mode": "PAPER"},
    )

    store_rows = [
        {
            "symbol": "HOT",
            "verdict": "BUY",
            "rsi": 55.0,  # stale scan — looks fine
            "volume_ratio": 2.0,
            "price": 100.0,
            "entry": 100.0,
            "stop": 95.0,
            "score": 80,
            "breakout_grade": "A",
            "breakout_conviction": 70,
            "chase_risk": False,
            "signals": ["BREAKOUT_52W"],
        }
    ]
    monkeypatch.setattr(
        "scan.auto_scan.get_results",
        lambda: (store_rows, 1, "2026-08-13T04:00:00+00:00", "ok"),
    )

    refreshed = [
        {
            **store_rows[0],
            "rsi": 81.0,  # live blow-off — must skip
            "volume_ratio": 2.1,
            "tech_source": "live",
        }
    ]
    monkeypatch.setattr(
        "product.live_technicals.refresh_rows_technicals",
        lambda rows, **kwargs: refreshed,
    )
    monkeypatch.setattr(
        "product.radar_workspace.is_sniper_breakout_candidate",
        lambda row: True,
    )
    monkeypatch.setattr(
        "product.radar_workspace.breakout_quality_score",
        lambda row: 80.0,
    )
    monkeypatch.setattr("scan.ev_engine.ev_rank_key", lambda row: 1.0)

    placed = []

    def _consider(**kwargs):
        placed.append(kwargs)
        return True

    monkeypatch.setattr("execution.autopilot.consider", _consider)

    msg = tc._trade_now()
    assert placed == []
    assert "gates" in msg.lower() or "atke" in msg.lower()


def test_trade_now_skips_thin_volume_after_refresh(monkeypatch):
    import alerts.telegram_commands as tc

    monkeypatch.setattr(
        "execution.autopilot.get_status",
        lambda: {"armed": True, "mode": "PAPER"},
    )
    store_rows = [
        {
            "symbol": "THIN",
            "verdict": "BUY",
            "rsi": 50.0,
            "volume_ratio": 1.8,
            "price": 100.0,
            "entry": 100.0,
            "stop": 95.0,
            "score": 75,
            "breakout_grade": "B",
            "breakout_conviction": 60,
            "chase_risk": False,
            "signals": ["BREAKOUT_52W"],
        }
    ]
    monkeypatch.setattr(
        "scan.auto_scan.get_results",
        lambda: (store_rows, 1, "t", "ok"),
    )
    monkeypatch.setattr(
        "product.live_technicals.refresh_rows_technicals",
        lambda rows, **kwargs: [{**store_rows[0], "volume_ratio": 0.2, "rsi": 48.0}],
    )
    monkeypatch.setattr(
        "product.radar_workspace.is_sniper_breakout_candidate",
        lambda row: False,
    )
    monkeypatch.setattr(
        "product.radar_workspace.breakout_quality_score",
        lambda row: 40.0,
    )
    monkeypatch.setattr("scan.ev_engine.ev_rank_key", lambda row: 0.0)

    placed = []
    monkeypatch.setattr(
        "execution.autopilot.consider",
        lambda **kwargs: placed.append(kwargs) or True,
    )

    msg = tc._trade_now()
    assert placed == []
    assert "THIN" not in msg or "gates" in msg.lower() or "atke" in msg.lower()
