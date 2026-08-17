"""Honest Telegram delivery status — Telegram is optional extra, not the desk.

The phone desk is the browser (Home / Ideas / thesis). Telegram may push
the same evidence to people who already have it. Missing Telegram must
never look like a missing desk.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _status_path() -> Path:
    return Path(os.environ.get("DEVBLOOM_LOG_DIR", "logs")) / "telegram_status.json"

DESK_NOTE = (
    "The desk is this browser — Home, Ideas, and the buy thesis. "
    "Telegram is optional extra. Not everyone has it."
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty() -> dict[str, Any]:
    return {
        "required": False,
        "desk": "browser",
        "configured": False,
        "listener_running": False,
        "bot_reachable": None,
        "last_send_ok": None,
        "last_send_at": "",
        "last_inbound_at": "",
        "last_error": "",
        "note": DESK_NOTE,
    }


def classify_error(exc: BaseException | str, status_code: int | None = None) -> str:
    text = str(exc or "").lower()
    code = int(status_code or 0)
    if code == 401 or code == 403 or "unauthorized" in text:
        return "unauthorized — token or chat id rejected"
    if code == 404 or "not found" in text:
        return "bot_not_found — token rejected"
    if code == 409 or "conflict" in text:
        return "conflict — two listeners polling the same bot"
    if code == 400 or "bad request" in text:
        return "bad_request — Telegram refused the payload"
    if "timeout" in text or "timed out" in text:
        return "timeout — api.telegram.org did not answer"
    if "connection" in text or "name or service" in text or "unreachable" in text:
        return "unreachable — cannot reach api.telegram.org"
    if not text:
        return "send_failed"
    return text[:160]


def _read() -> dict[str, Any]:
    payload = _empty()
    try:
        raw = json.loads(_status_path().read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            payload.update({k: raw[k] for k in payload if k in raw})
    except Exception:
        pass
    return payload


def _write(payload: dict[str, Any]) -> None:
    try:
        path = _status_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


_EXAMPLE_SECRETS = {
    "your_bot_token_here",
    "your_bot_token",
    "your_chat_id_here",
    "your_chat_id",
}


def usable_telegram_secret(value: Any) -> bool:
    """True only for a real token/chat id — example .env placeholders do not count."""
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return False
    low = text.lower()
    if low in _EXAMPLE_SECRETS or low.startswith("your_"):
        return False
    if text.startswith("<") and text.endswith(">"):
        return False
    return True


def telegram_credentials() -> tuple[str, str]:
    """Prefer a real process env pair over .env.example placeholders in settings."""
    env_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    env_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if usable_telegram_secret(env_token) and usable_telegram_secret(env_chat):
        return env_token, env_chat
    file_token, file_chat = "", ""
    try:
        from config import settings
        file_token = str(getattr(settings, "telegram_bot_token", "") or "").strip()
        file_chat = str(getattr(settings, "telegram_chat_id", "") or "").strip()
    except Exception:
        pass
    if usable_telegram_secret(file_token) and usable_telegram_secret(file_chat):
        return file_token, file_chat
    return env_token or file_token, env_chat or file_chat


def configured() -> bool:
    token, chat = telegram_credentials()
    return usable_telegram_secret(token) and usable_telegram_secret(chat)


def listener_flag() -> bool:
    try:
        from alerts import telegram_actions as ta
        return bool(getattr(ta, "_started", False))
    except Exception:
        return False


def snapshot() -> dict[str, Any]:
    with _LOCK:
        payload = _read()
    payload["required"] = False
    payload["desk"] = "browser"
    payload["configured"] = configured()
    payload["listener_running"] = listener_flag()
    payload["note"] = DESK_NOTE
    if not payload["configured"]:
        token, chat = telegram_credentials()
        if (token or chat) and not (usable_telegram_secret(token) and usable_telegram_secret(chat)):
            payload["last_error"] = "example_placeholder"
        else:
            payload["last_error"] = payload.get("last_error") or "not_configured"
    return payload


def record_send(ok: bool, error: str = "") -> None:
    with _LOCK:
        payload = _read()
        payload["last_send_ok"] = bool(ok)
        payload["last_send_at"] = _now()
        payload["last_error"] = "" if ok else (error or "send_failed")
        payload["configured"] = configured()
        _write(payload)


def record_inbound() -> None:
    with _LOCK:
        payload = _read()
        payload["last_inbound_at"] = _now()
        payload["listener_running"] = True
        _write(payload)


def record_listener(running: bool, error: str = "") -> None:
    with _LOCK:
        payload = _read()
        payload["listener_running"] = bool(running)
        if error:
            payload["last_error"] = error
        _write(payload)


def probe_bot(timeout: float = 5.0) -> dict[str, Any]:
    """Reachability only — never sends a chat message."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        try:
            from config import settings
            token = str(getattr(settings, "telegram_bot_token", "") or "").strip()
        except Exception:
            token = ""
    if not token:
        record_listener(False, "not_configured")
        return snapshot()
    try:
        import requests
        resp = requests.get(
            f"https://api.telegram.org/bot{token}/getMe",
            timeout=timeout,
        )
        reachable = resp.ok and bool((resp.json() or {}).get("ok"))
        with _LOCK:
            payload = _read()
            payload["bot_reachable"] = reachable
            payload["configured"] = configured()
            if not reachable:
                payload["last_error"] = classify_error(resp.reason or "not found", resp.status_code)
            elif payload.get("last_error") in ("not_configured", "unauthorized — token or chat id rejected"):
                payload["last_error"] = ""
            _write(payload)
    except Exception as exc:
        with _LOCK:
            payload = _read()
            payload["bot_reachable"] = False
            payload["last_error"] = classify_error(exc)
            _write(payload)
    return snapshot()
