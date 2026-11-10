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
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_DESK_ORIGINS = frozenset({"http://127.0.0.1:5173", "http://localhost:5173"})


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in _TRUE_VALUES


def public_read_only_enabled() -> bool:
    """Invalid EXPLICIT public-mode configuration must not open mutations.

    Unset preserves the original private/local workflow. Once a deployment
    supplies the variable, only a recognized false value opts out; misspelled
    true values, whitespace and other invalid strings fail CLOSED.
    """
    raw = os.environ.get("QT_PUBLIC_READ_ONLY")
    if raw is None:
        return False
    return str(raw).strip().lower() not in _FALSE_VALUES


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


def _normalize_origin(value: str) -> str:
    return str(value or "").strip().rstrip("/").lower()


def desk_origins() -> frozenset[str]:
    extra = {
        _normalize_origin(item)
        for item in str(os.environ.get("QT_DESK_ORIGINS") or "").split(",")
        if _normalize_origin(item)
    }
    return _DESK_ORIGINS | frozenset(extra)


def _browser_cross_site(headers: Mapping[str, str]) -> bool:
    """True when the browser itself says the request did not come from the desk.

    Only browser-asserted provenance (Origin / Sec-Fetch-Site) is read, and it
    can only refuse a request, never authorize one. Clients that send neither
    header (scripts, internal services) are unaffected.
    """
    origin = headers.get("origin")
    if origin is not None:
        # The desk page on :5173 calls the report API on :8766 directly; the
        # browser labels that same-site, so Origin (exact) decides when present.
        return _normalize_origin(origin) not in desk_origins()
    return str(headers.get("sec-fetch-site") or "").strip().lower() == "cross-site"


def authorize_mutation(
    method: str,
    headers: Mapping[str, str] | None = None,
) -> MutationAccess:
    """Authorize one HTTP method without trusting network topology.

    Safe/read-only methods always pass. A valid operator bearer token always
    authorizes. Otherwise an unsafe method carrying browser provenance from a
    page other than the desk is refused in every mode: the API binds to
    loopback, but CORS only hides responses — any page open in the operator's
    browser can still deliver a simple POST (or DNS-rebind) to it. Outside
    explicit public mode the private operator workflow is otherwise unchanged.
    In public mode an unsafe method requires QT_OPERATOR_TOKEN and a matching
    Bearer credential.
    """
    verb = str(method or "GET").upper()
    if verb not in _UNSAFE_METHODS:
        return MutationAccess(True, "READ_ONLY_METHOD", "Read-only request.")

    headers = headers or {}
    expected = operator_token()
    supplied = _bearer_token(headers.get("authorization"))
    if expected and supplied and secrets.compare_digest(supplied, expected):
        return MutationAccess(True, "OPERATOR_AUTHORIZED", "Operator bearer token accepted.")

    if _browser_cross_site(headers):
        return MutationAccess(
            False,
            "CROSS_SITE_MUTATION",
            "Mutations are accepted only from the QuantTerm desk origin.",
        )

    if not public_read_only_enabled():
        return MutationAccess(True, "PRIVATE_OPERATOR_MODE", "Public read-only mode is disabled.")

    if not expected:
        return MutationAccess(
            False,
            "PUBLIC_READ_ONLY",
            "This public QuantTerm instance is read-only. Operator mutations are disabled.",
        )

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
