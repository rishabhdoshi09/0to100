"""Telegram "sent" must mean Telegram confirmed delivery to the configured chat.

9 Nov incident: logs reported alerts as sent, but delivery could not be
confirmed. AlertEngine counted any HTTP 2xx as delivered and kept no message id,
so an HTML page from a proxy/captive portal, an ``{"ok": false}`` body, or a
reply for a different chat all became "sent". No network: requests.post is faked.
"""
from __future__ import annotations

from datetime import datetime

import pytest

import alerts.telegram_alerts as TA
from research.autonomy.telegram_notifications import TelegramNotifier


class _Resp:
    def __init__(self, status=200, body=None, text=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = text if text is not None else ("" if body is None else str(body))

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


def _engine(monkeypatch, responses, chat_id="123456"):
    monkeypatch.setattr(TA, "_telegram_cred", lambda name: {"TELEGRAM_BOT_TOKEN": "t0k", "TELEGRAM_CHAT_ID": chat_id}[name])
    queue = list(responses)
    sent = []
    monkeypatch.setattr(TA.requests, "post", lambda url, json=None, timeout=None: sent.append(json) or queue.pop(0))
    return TA.AlertEngine(), sent


def _ok(message_id=77, chat=123456, username=None):
    chat_obj = {"id": chat}
    if username:
        chat_obj["username"] = username
    return _Resp(200, {"ok": True, "result": {"message_id": message_id, "chat": chat_obj}})


def test_confirmed_delivery_records_telegram_message_ids(monkeypatch):
    engine, _ = _engine(monkeypatch, [_ok(77)])
    assert engine.send("hello") is True
    assert engine.last_message_ids == [77]
    assert engine.last_error == ""


@pytest.mark.parametrize("response, reason", [
    (_Resp(200, None, text="<html>Sign in to Wi-Fi</html>"), "not JSON"),
    (_Resp(200, {"ok": False, "description": "Bad Request"}), "ok=false"),
    (_Resp(200, {"ok": True, "result": {}}), "no message_id"),
    (_Resp(200, {"ok": True, "result": {"message_id": 5, "chat": {"id": 999}}}), "wrong chat"),
])
def test_http_200_without_telegram_confirmation_is_not_delivery(monkeypatch, response, reason):
    # The plain-text retry gets the same unconfirmed answer.
    engine, sent = _engine(monkeypatch, [response, response])
    assert engine.send("hello") is False, reason
    assert engine.last_message_ids == []
    assert engine.last_error
    assert len(sent) == 2


def test_channel_username_chat_is_matched_by_username(monkeypatch):
    engine, _ = _engine(monkeypatch, [_ok(9, chat=-1001, username="qt_alerts")], chat_id="@qt_alerts")
    assert engine.send("hi") is True and engine.last_message_ids == [9]


def test_multi_chunk_message_is_delivered_only_if_every_chunk_is_confirmed(monkeypatch):
    long = "\n\n".join(["x" * 3000, "y" * 3000])
    engine, _ = _engine(monkeypatch, [_ok(1), _Resp(200, {"ok": False}), _Resp(200, {"ok": False})])
    assert engine.send(long) is False
    assert engine.last_message_ids == [1]          # what did arrive is still on record


class _Engine:
    def __init__(self, ids):
        self.ids = ids
        self.last_error = ""
        self.last_message_ids: list[int] = []

    def is_configured(self):
        return True

    def send(self, message, reply_markup=None):
        self.last_message_ids = list(self.ids)
        return True


def test_notifier_stores_the_delivery_evidence_with_the_sent_key(tmp_path):
    now = lambda: datetime(2026, 11, 9, 10, 0)
    n = TelegramNotifier(tmp_path, engine_factory=lambda: _Engine([501, 502]), now_fn=now)
    assert n._send_status("scan:2026-11-09T10:00", "msg") == "sent"
    assert n.state["message_ids"]["2026-11-09"]["scan:2026-11-09T10:00"] == [501, 502]
    reloaded = TelegramNotifier(tmp_path, engine_factory=lambda: _Engine([]), now_fn=now)
    assert reloaded.state["message_ids"]["2026-11-09"]["scan:2026-11-09T10:00"] == [501, 502]
    assert reloaded._send_status("scan:2026-11-09T10:00", "msg") == "already_sent"


def test_another_process_save_does_not_erase_delivery_receipts(tmp_path):
    """The F&O worker and the autonomy notifier persist the same state file."""
    now = lambda: datetime(2026, 11, 9, 10, 0)
    cash = TelegramNotifier(tmp_path, engine_factory=lambda: _Engine([11]), now_fn=now)
    fno = TelegramNotifier(tmp_path, engine_factory=lambda: _Engine([22]), now_fn=now)
    assert cash._send_status("scan:A", "m") == "sent"
    assert fno._send_status("fno:B", "m") == "sent"          # fno saved from a stale snapshot
    merged = TelegramNotifier(tmp_path, engine_factory=lambda: _Engine([]), now_fn=now)
    assert merged.state["message_ids"]["2026-11-09"] == {"scan:A": [11], "fno:B": [22]}


class _HttpResponse:
    def __init__(self, status, body: bytes):
        self.status = status
        self._body = body

    def read(self, n=-1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("body, delivered", [
    (b'{"ok":true,"result":{"message_id":31,"chat":{"id":42}}}', True),
    (b"<html>captive portal</html>", False),
    (b'{"ok":false,"description":"chat not found"}', False),
    (b'{"ok":true,"result":{"message_id":31,"chat":{"id":7}}}', False),
])
def test_host_operational_alert_needs_a_telegram_receipt(monkeypatch, body, delivered):
    import product.host_alerts as HA
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.delenv("QT_ALERT_WEBHOOK_URL", raising=False)
    contexts = []

    def fake_urlopen(request, timeout=0, context=None):
        contexts.append(context)
        return _HttpResponse(200, body)

    monkeypatch.setattr(HA.urllib.request, "urlopen", fake_urlopen)
    result = HA.send_operational_alert("host check")
    import ssl
    assert contexts and isinstance(contexts[0], ssl.SSLContext)
    assert contexts[0].verify_mode == ssl.CERT_REQUIRED and contexts[0].check_hostname
    assert result.delivered is delivered
    assert result.message_id == (31 if delivered else None)
    assert "123:tok" not in result.detail
