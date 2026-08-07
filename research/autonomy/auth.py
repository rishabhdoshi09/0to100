"""Bounded, data-only Zerodha authentication health classification."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone

TOKEN_MISSING = "TOKEN_MISSING"
SESSION_VALID = "SESSION_VALID"
SESSION_EXPIRED = "SESSION_EXPIRED"
PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
CONFIG_INVALID = "CONFIG_INVALID"


@dataclass(frozen=True)
class AuthHealth:
    status: str
    checked_at: str
    user_id: str = ""
    broker: str = ""
    error_code: str = ""
    reason: str = ""

    @property
    def valid(self) -> bool:
        return self.status == SESSION_VALID

    def as_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_reason(exc: Exception) -> str:
    # Never serialize request headers/credentials.  Exception class is enough for UI classification;
    # the bounded message is retained only after stripping common credential key names.
    msg = str(exc)[:240]
    lower = msg.lower()
    if any(k in lower for k in ("access_token", "api_secret", "request_token", "authorization")):
        return type(exc).__name__
    return msg or type(exc).__name__


def classify_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if any(x in name for x in ("tokenexception", "permissionexception", "authentication")):
        return SESSION_EXPIRED
    if any(x in msg for x in ("invalid token", "token is invalid", "session expired", "access token")):
        return SESSION_EXPIRED
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return PROVIDER_UNAVAILABLE
    if any(x in name for x in ("timeout", "connection", "network")):
        return PROVIDER_UNAVAILABLE
    if any(x in msg for x in ("timed out", "connection", "dns", "temporarily unavailable", "502", "503", "504")):
        return PROVIDER_UNAVAILABLE
    return CONFIG_INVALID


def probe_auth(*, client_factory=None) -> AuthHealth:
    """Validate the genuine session with ``profile()`` through the data-only boundary."""
    if client_factory is None:
        try:
            from data.kite_client import _fresh_env
            if not _fresh_env("KITE_API_KEY"):
                return AuthHealth(CONFIG_INVALID, _now(), error_code="KITE_API_KEY_MISSING",
                                  reason="Zerodha API key is not configured")
            if not _fresh_env("KITE_ACCESS_TOKEN"):
                return AuthHealth(TOKEN_MISSING, _now(), error_code="KITE_TOKEN_MISSING",
                                  reason="daily Zerodha login is required")
        except Exception as exc:
            return AuthHealth(CONFIG_INVALID, _now(), error_code="CONFIG_LOAD_FAILED",
                              reason=_safe_reason(exc))
        from research.intelligence.data.kite_activation import KiteDataClient
        client_factory = KiteDataClient.from_config

    try:
        client = client_factory()
        profile = client.profile() or {}
        if not profile:
            return AuthHealth(SESSION_EXPIRED, _now(), error_code="EMPTY_PROFILE",
                              reason="Zerodha session did not return a profile")
        return AuthHealth(SESSION_VALID, _now(), user_id=str(profile.get("user_id") or ""),
                          broker=str(profile.get("broker") or ""))
    except Exception as exc:
        status = classify_exception(exc)
        return AuthHealth(status, _now(), error_code=type(exc).__name__.upper(),
                          reason=_safe_reason(exc))
