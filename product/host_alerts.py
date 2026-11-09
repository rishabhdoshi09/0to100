"""Low-noise host alerts for unattended QuantTerm PAPER/SHADOW operation.

Alerts are operational notifications only. They never authorize trading and a
notification failure never changes market state. Secret-bearing destination
URLs are never returned in persisted error details.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AlertResult:
    attempted: bool
    delivered: bool
    channel: str = ""
    detail: str = ""
    message_id: int | None = None   # Telegram's own receipt when delivered there

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _configured_destinations() -> tuple[str, str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    webhook = os.environ.get("QT_ALERT_WEBHOOK_URL", "").strip()
    if not (token and chat):
        try:
            from config import settings
            token = token or str(settings.telegram_bot_token or "").strip()
            chat = chat or str(settings.telegram_chat_id or "").strip()
        except Exception:
            pass
    return token, chat, webhook


def _safe_failure(exc: Exception) -> str:
    # urllib exceptions can echo the full request URL. Telegram embeds the bot
    # token in that URL, so persist only the exception class, never str(exc).
    return f"{type(exc).__name__}: alert delivery failed"


def _telegram_receipt(raw: bytes, chat: str) -> tuple[int | None, str]:
    """Telegram's confirmation for the configured chat, or why there is none."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return None, "unconfirmed: response is not Telegram JSON"
    if not isinstance(body, dict) or body.get("ok") is not True:
        return None, "unconfirmed: Telegram ok=false"
    result = body.get("result") if isinstance(body.get("result"), dict) else {}
    message_id = result.get("message_id")
    if isinstance(message_id, bool) or not isinstance(message_id, int):
        return None, "unconfirmed: no message_id"
    got = result.get("chat") if isinstance(result.get("chat"), dict) else {}
    wanted = str(chat or "").strip()
    same = (str(got.get("username") or "").lower() == wanted[1:].lower()
            if wanted.startswith("@") else str(got.get("id")) == wanted)
    if not same:
        return None, "unconfirmed: delivered to a different chat"
    return message_id, "delivered"


def send_operational_alert(message: str, *, timeout: float = 8.0) -> AlertResult:
    token, chat, webhook = _configured_destinations()
    body = str(message or "").strip()[:3500]
    if not body:
        return AlertResult(False, False, detail="empty message")

    if token and chat:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({"chat_id": chat, "text": body}).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                raw = response.read(65536) if 200 <= status < 300 else b""
            if not 200 <= status < 300:
                return AlertResult(True, False, "telegram", "non-2xx response")
            message_id, reason = _telegram_receipt(raw, chat)
            if message_id is None:
                # HTTP 2xx alone is not delivery (proxy/portal page, ok=false, other chat).
                return AlertResult(True, False, "telegram", reason)
            return AlertResult(True, True, "telegram", "delivered", message_id)
        except Exception as exc:
            return AlertResult(True, False, "telegram", _safe_failure(exc))

    if webhook:
        payload = json.dumps({"text": body, "source": "quantterm"}).encode("utf-8")
        request = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                ok = 200 <= int(response.status) < 300
            return AlertResult(True, ok, "webhook", "delivered" if ok else "non-2xx response")
        except Exception as exc:
            return AlertResult(True, False, "webhook", _safe_failure(exc))

    return AlertResult(False, False, detail="no alert destination configured")
