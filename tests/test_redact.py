"""Credentials never reach logs (owner's Mac 2026-11-15: the Telegram bot token
was written to simplequant.log on every failed send during a DNS outage)."""
from __future__ import annotations

import requests

from core.redact import REDACTED, redact_secrets

TOKEN = "8000000001:AAFakeFakeFakeFakeFakeFakeFakeFake12"


def test_telegram_url_token_is_masked():
    text = (
        "HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded "
        f"with url: /bot{TOKEN}/sendMessage (Caused by NameResolutionError)"
    )
    out = redact_secrets(text)
    assert TOKEN not in out and "AAFake" not in out
    assert f"/bot{REDACTED}/sendMessage" in out


def test_query_and_header_credentials_are_masked():
    out = redact_secrets("GET /quote?api_key=abc123&access_token=xyz789 {'Authorization': 'token k:s'}")
    assert "abc123" not in out and "xyz789" not in out and "k:s" not in out


def test_plain_text_is_unchanged():
    assert redact_secrets("Kite data is not from the expected NSE session.") == (
        "Kite data is not from the expected NSE session."
    )
    assert redact_secrets(None) == ""


def test_alert_engine_never_logs_the_bot_token(monkeypatch, caplog):
    from alerts.telegram_alerts import AlertEngine

    def offline(url, **kwargs):
        raise requests.exceptions.ConnectionError(
            f"HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded with url: "
            f"/bot{TOKEN}/sendMessage (Caused by NameResolutionError)"
        )

    monkeypatch.setattr(requests, "post", offline)
    engine = AlertEngine.__new__(AlertEngine)
    engine.last_error = ""
    engine.last_message_ids = []
    caplog.set_level("WARNING")
    assert engine._post_chunk(f"https://api.telegram.org/bot{TOKEN}/sendMessage", {"text": "x"}) is False
    assert TOKEN not in engine.last_error
    assert TOKEN not in caplog.text
