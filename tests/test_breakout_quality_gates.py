"""Hard volume/tech gates for breakouts; fundamentals only for best-among."""
from __future__ import annotations


def test_gate_rejects_thin_volume_technical_path():
    from product.breakout_quality import gate_breakout_quality

    ok, reasons, status = gate_breakout_quality({
        "volume_ratio": 0.1, "rsi": 55, "chase_risk": False,
    })
    assert ok is False
    assert status["volume"] == "fail"
    assert any("volume" in r for r in reasons)


def test_avoid_review_does_not_kill_technical_sniper():
    from product.breakout_quality import gate_breakout_quality

    ok, reasons, status = gate_breakout_quality({
        "volume_ratio": 1.5, "rsi": 55, "chase_risk": False,
        "classification": "AVOID_REVIEW", "fundamental_coverage": 0.8,
        "fundamental_score": 15,
    }, for_best=False)
    assert ok is True
    assert status["fundamentals"] == "fail"
    assert reasons == []

    ok2, reasons2, status2 = gate_breakout_quality({
        "volume_ratio": 1.5, "rsi": 55, "chase_risk": False,
        "classification": "AVOID_REVIEW", "fundamental_coverage": 0.8,
        "fundamental_score": 15,
    }, for_best=True)
    assert ok2 is False
    assert any("AVOID" in r for r in reasons2)


def test_technical_allows_rsi_between_70_and_82():
    from product.breakout_quality import gate_breakout_quality

    ok, _, status = gate_breakout_quality({
        "volume_ratio": 1.4, "rsi": 76, "chase_risk": False,
    }, for_best=False)
    assert ok is True
    assert status["rsi"] == "elevated"

    ok_best, reasons, _ = gate_breakout_quality({
        "volume_ratio": 1.4, "rsi": 76, "chase_risk": False,
        "classification": "GARP_CANDIDATE", "fundamental_coverage": 0.7,
        "fundamental_score": 70,
    }, for_best=True)
    assert ok_best is False
    assert any("RSI" in r for r in reasons)


def test_gate_passes_solid_setup():
    from product.breakout_quality import gate_breakout_quality

    ok, reasons, status = gate_breakout_quality({
        "volume_ratio": 1.6, "rsi": 58, "chase_risk": False,
        "classification": "GARP_CANDIDATE", "fundamental_coverage": 0.7,
        "fundamental_score": 72, "above_sma50": True,
    })
    assert ok is True
    assert reasons == []
    assert status["volume"] == "pass"
    assert status["fundamentals"] == "pass"


def test_telegram_live_breakouts_skip_thin_volume(tmp_path):
    from research.autonomy.telegram_notifications import TelegramNotifier

    class _Feed:
        def entry_allowed(self, sym):
            return True

        def price(self, sym):
            return 110.0

    class _Engine:
        def is_configured(self):
            return False

        def send(self, msg):
            return False

    n = TelegramNotifier(
        tmp_path,
        engine_factory=_Engine,
        breakout_confirmation_s=8.0,
        breakout_buffer_bps=10.0,
    )
    payload = {
        "records": [
            {
                "symbol": "THIN", "entry": 100, "stop": 95, "target": 120,
                "status": "Ready to trade", "signals": ["BREAKOUT_52W"],
                "volume_ratio": 0.1, "rsi": 50, "score": 90,
            },
            {
                "symbol": "FAT", "entry": 100, "stop": 95, "target": 120,
                "status": "Ready to trade", "signals": ["BREAKOUT_52W"],
                "volume_ratio": 1.8, "rsi": 55, "score": 80,
            },
        ]
    }
    n.observe_live_breakouts(payload, _Feed())
    assert "THIN" not in n.state.get("arms", {})
    assert "FAT" in n.state.get("arms", {})


def test_technical_chase_is_soft_best_among_hard():
    from product.breakout_quality import gate_breakout_quality

    ok, reasons, status = gate_breakout_quality({
        "volume_ratio": 1.2, "rsi": 55, "chase_risk": True,
    }, for_best=False)
    assert ok is True
    assert status["extension"] == "fail"
    assert reasons == []

    ok_best, reasons_best, _ = gate_breakout_quality({
        "volume_ratio": 1.2, "rsi": 55, "chase_risk": True,
        "classification": "GARP_CANDIDATE", "fundamental_coverage": 0.7,
        "fundamental_score": 70,
    }, for_best=True)
    assert ok_best is False
    assert any("chase" in r for r in reasons_best)


def test_technical_allows_volume_above_eased_floor():
    from product.breakout_quality import gate_breakout_quality

    ok, reasons, status = gate_breakout_quality({
        "volume_ratio": 0.75, "rsi": 55, "chase_risk": False,
    }, for_best=False)
    assert ok is True
    assert status["volume"] == "pass"
    assert reasons == []


def test_optional_context_marks_unavailable_without_kite(monkeypatch):
    from product import breakout_quality as bq

    class _Settings:
        kite_access_token = ""
        exchange = "NSE"

    monkeypatch.setattr("data.kite_client.KiteClient", None, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "config", type("M", (), {"settings": _Settings()})())
    import config as cfg
    monkeypatch.setattr(cfg.settings, "kite_access_token", "", raising=False)
    monkeypatch.setattr("data.kite_client._fresh_env", lambda *a, **k: "")
    ctx = bq.enrich_optional_context("RELIANCE")
    assert ctx["order_book"]["status"] == "unavailable"
    assert ctx["concall"]["status"] == "unavailable"
