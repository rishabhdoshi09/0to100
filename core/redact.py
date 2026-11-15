"""Strip credentials from text before it reaches a log, a status file or a UI.

On 2026-11-15 the owner's ``simplequant.log`` held the full Telegram bot token
dozens of times: a DNS failure during a dark wake raised a ``requests``
ConnectionError whose message embeds the request URL
(``https://api.telegram.org/bot<id>:<secret>/sendMessage``), and the alert
engine logged ``str(exc)`` verbatim. Any exception text that can carry a URL
or a header must pass through :func:`redact_secrets` first.
"""
from __future__ import annotations

import re

REDACTED = "<redacted>"

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Telegram Bot API path segment: /bot123456789:AA...secret.../method
    (re.compile(r"(/bot)\d{5,}:[A-Za-z0-9_-]{20,}"), r"\1" + REDACTED),
    # Bare bot token (e.g. echoed in a config error)
    (re.compile(r"\b\d{6,}:AA[A-Za-z0-9_-]{30,}\b"), REDACTED),
    # key=value credentials in query strings or messages
    (
        re.compile(
            r"(?i)\b(access_token|request_token|api_key|api_secret|apikey|"
            r"client_secret|password|passwd|secret|token)=([^&\s'\"<>]+)"
        ),
        r"\1=" + REDACTED,
    ),
    # Authorization headers (Kite uses "token key:secret")
    (re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?(?:token|bearer)\s+)[^\s'\",}]+"), r"\1" + REDACTED),
)


def redact_secrets(text: object) -> str:
    """Return ``text`` as a string with known credential shapes masked."""
    out = str(text if text is not None else "")
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    return out
