"""Telegram is optional. Browser is the desk."""
from __future__ import annotations

from alerts.telegram_status import classify_error, snapshot


def test_telegram_is_never_required():
    payload = snapshot()
    assert payload["required"] is False
    assert payload["desk"] == "browser"
    assert "browser" in payload["note"].lower()
    assert "optional" in payload["note"].lower()


def test_classify_error_does_not_leak_tokens():
    err = classify_error("401 Unauthorized bot123:AASECRET", 401)
    assert "unauthorized" in err
    assert "AASECRET" not in err


def test_html_send_retries_without_parse_mode(monkeypatch, tmp_path):
    from alerts.telegram_alerts import AlertEngine

    monkeypatch.setenv("DEVBLOOM_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    calls = []

    class _Resp:
        def __init__(self, code):
            self.status_code = code
        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"http_{self.status_code}")

    def fake_post(url, json=None, timeout=8):
        calls.append(dict(json or {}))
        if json and json.get("parse_mode"):
            return _Resp(400)
        return _Resp(200)

    monkeypatch.setattr("alerts.telegram_alerts.requests.post", fake_post)
    engine = AlertEngine()
    assert engine.enabled is True
    assert engine.send("<b>hi</b>") is True
    assert len(calls) == 2
    assert calls[0].get("parse_mode") == "HTML"
    assert "parse_mode" not in calls[1]
