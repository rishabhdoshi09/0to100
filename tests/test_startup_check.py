from __future__ import annotations

from product.startup_check import build_startup_check, maybe_open_home_browser


def test_startup_check_live_money_locked_and_telegram_optional(monkeypatch):
    monkeypatch.setenv("QT_NONINTERACTIVE", "1")
    payload = build_startup_check(probe_network=False)
    names = [lane["name"] for lane in payload["lanes"]]
    assert "LIVE MONEY" in names
    assert "UI" in names
    assert "FORWARD EVIDENCE" in names
    live = next(lane for lane in payload["lanes"] if lane["name"] == "LIVE MONEY")
    assert live["status"] == "LOCKED"
    assert payload["live_locked"] is True
    assert "Telegram absence is not a product failure." in payload["note"]
    reports = next(lane for lane in payload["lanes"] if lane["name"] == "REPORTS")
    assert reports["required"] is False


def test_browser_open_skipped_when_noninteractive(monkeypatch):
    monkeypatch.setenv("QT_NONINTERACTIVE", "1")
    assert maybe_open_home_browser() is False
    monkeypatch.delenv("QT_NONINTERACTIVE", raising=False)
    monkeypatch.setenv("QT_NO_BROWSER", "1")
    assert maybe_open_home_browser() is False
