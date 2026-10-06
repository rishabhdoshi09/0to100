"""Public internet access boundary for QuantTerm's operator API.

Local/private deployments preserve the existing operator workflow by default.
When QT_PUBLIC_READ_ONLY is enabled, every state-changing HTTP method requires
an explicit bearer token. If no operator token is configured, the API is
strictly read-only for mutations.

This module deliberately does not trust client IPs or forwarding headers:
reverse proxies can make remote traffic appear local.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Mapping


_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in _TRUE_VALUES


def public_read_only_enabled() -> bool:
    return _env_flag("QT_PUBLIC_READ_ONLY")


def operator_token() -> str:
    return str(os.environ.get("QT_OPERATOR_TOKEN") or "").strip()


@dataclass(frozen=True)
class MutationAccess:
    allowed: bool
    code: str
    detail: str


def _bearer_token(authorization: str | None) -> str:
    value = str(authorization or "").strip()
    if not value:
        return ""
    scheme, separator, credential = value.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return ""
    return credential.strip()


def authorize_mutation(
    method: str,
    headers: Mapping[str, str] | None = None,
) -> MutationAccess:
    """Authorize one HTTP method without trusting network topology.

    Safe/read-only methods always pass. Outside explicit public mode the
    existing private/local operator behavior is unchanged. In public mode an
    unsafe method requires QT_OPERATOR_TOKEN and a matching Bearer credential.
    """
    verb = str(method or "GET").upper()
    if verb not in _UNSAFE_METHODS:
        return MutationAccess(True, "READ_ONLY_METHOD", "Read-only request.")

    if not public_read_only_enabled():
        return MutationAccess(True, "PRIVATE_OPERATOR_MODE", "Public read-only mode is disabled.")

    expected = operator_token()
    if not expected:
        return MutationAccess(
            False,
            "PUBLIC_READ_ONLY",
            "This public QuantTerm instance is read-only. Operator mutations are disabled.",
        )

    supplied = _bearer_token((headers or {}).get("authorization"))
    if supplied and secrets.compare_digest(supplied, expected):
        return MutationAccess(True, "OPERATOR_AUTHORIZED", "Operator bearer token accepted.")

    return MutationAccess(
        False,
        "OPERATOR_AUTH_REQUIRED",
        "This public QuantTerm instance is read-only unless an operator bearer token is supplied.",
    )


def access_projection() -> dict:
    """Non-secret public description of the active access boundary."""
    enabled = public_read_only_enabled()
    configured = bool(operator_token())
    return {
        "public_read_only": enabled,
        "mutation_policy": (
            "OPERATOR_TOKEN_REQUIRED"
            if enabled and configured
            else "READ_ONLY"
            if enabled
            else "PRIVATE_OPERATOR"
        ),
        "operator_token_configured": configured,
        "unsafe_methods": sorted(_UNSAFE_METHODS),
        "live_money_unlocked": False,
    }
