"""Terminal MARKET_SCAN must reach Telegram (not only autonomy jobs)."""
from __future__ import annotations

from operations import market_ops as MO


def test_notify_market_scan_telegram_uses_shared_notifier(monkeypatch):
    calls = {}

    class FakeNotifier:
        def __init__(self, root):
            calls["root"] = str(root)

        def configured(self):
            return True

        def notify_scan(self, payload, *, phase=""):
            calls["payload_records"] = len(payload.get("records") or [])
            calls["phase"] = phase
            return {"setup": 1, "prebreakout": 1, "briefing": 0, "eod": 0}

    import research.autonomy as autonomy
    import research.autonomy.telegram_notifications as tg

    monkeypatch.setattr(autonomy, "default_root", lambda: "/tmp/qt-autonomy")
    monkeypatch.setattr(tg, "TelegramNotifier", FakeNotifier)
    monkeypatch.setattr(
        "research.autonomy.schedules.session_phase",
        lambda now, holidays=None: "intraday",
    )

    out = MO._notify_market_scan_telegram(
        {
            "records": [{"symbol": "AAA", "verdict": "BUY", "entry": 100}],
            "summary": {"with_any_setup": 1},
        }
    )
    assert out["setup"] == 1
    assert out["prebreakout"] == 1
    assert calls["phase"] == "intraday"
    assert calls["payload_records"] == 1


def test_notify_market_scan_telegram_skips_when_unconfigured(monkeypatch):
    class FakeNotifier:
        def __init__(self, root):
            pass

        def configured(self):
            return False

        def notify_scan(self, *a, **k):
            raise AssertionError("should not notify when unconfigured")

    import research.autonomy as autonomy
    import research.autonomy.telegram_notifications as tg

    monkeypatch.setattr(autonomy, "default_root", lambda: "/tmp/qt-autonomy")
    monkeypatch.setattr(tg, "TelegramNotifier", FakeNotifier)
    out = MO._notify_market_scan_telegram({"records": []})
    assert out.get("skipped") == "not_configured"


def test_strong_buy_is_ready_to_trade():
    from product.scan_store import _record

    row = _record(
        {
            "symbol": "AAA",
            "verdict": "STRONG BUY",
            "signals": ["MOMENTUM"],
            "reasons": ["Volume confirmed"],
            "chase_risk": False,
            "price": 100,
            "entry": 101,
            "stop": 95,
            "target": 110,
            "score": 90,
            "rsi": 60,
            "volume_ratio": 2.0,
            "momentum_5d": 3.0,
        },
        {"AAA": "AAA Ltd"},
        set(),
    )
    assert row["status"] == "Ready to trade"


def test_bootstrap_uses_momentum_scan_path():
    import inspect

    source = inspect.getsource(MO.MarketOperationsWorker._bootstrap)
    assert "latest_momentum_scan.json" in source
    assert "latest_scan.json" not in source
