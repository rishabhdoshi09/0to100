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

# Reads that expose owner/operator state rather than public market intelligence.
# Prefixes are intentionally explicit: new routes stay public only when they
# are not placed under one of these established private namespaces.
_PRIVATE_MAIN_READ_PREFIXES = (
    "/api/operations",
    "/api/operator-health",
    "/api/system-health-contract",
    "/api/watchlist",
    "/api/autonomous-learning",
    "/api/learning-policies",
    "/api/learning-dashboard",
    "/api/research-director",
    "/api/evolution-",
    "/api/oms",
    "/api/risk-governor",
    "/api/reconciliation",
    "/api/protection",
    "/api/tca",
    "/api/autonomy-desk",
    "/api/desk-pipeline",
    "/api/broker-observer",
    "/api/institutional-readiness",
    "/api/target-portfolio",
    "/api/product-dashboard",
    "/api/product-contract",
    "/api/due-diligence/framework-audit",
)
_PRIVATE_MAIN_READ_EXACT = frozenset({"/docs", "/redoc", "/openapi.json"})
_PRIVATE_REPORT_READ_PREFIXES = ("/evidence/",)


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


def _operator_authorized(headers: Mapping[str, str] | None = None) -> bool:
    expected = operator_token()
    supplied = _bearer_token((headers or {}).get("authorization"))
    return bool(expected and supplied and secrets.compare_digest(supplied, expected))


def _private_read(path: str, service: str) -> bool:
    clean = "/" + str(path or "").lstrip("/")
    if service == "report":
        return any(clean.startswith(prefix) for prefix in _PRIVATE_REPORT_READ_PREFIXES)
    if clean in _PRIVATE_MAIN_READ_EXACT:
        return True
    return any(clean.startswith(prefix) for prefix in _PRIVATE_MAIN_READ_PREFIXES)


def authorize_request(
    method: str,
    path: str,
    headers: Mapping[str, str] | None = None,
    *,
    service: str = "main",
) -> MutationAccess:
    """Authorize a request in public mode.

    Unsafe methods reuse the mutation boundary. Safe methods remain public
    except for explicitly owner/operator-only namespaces. The same bearer token
    can be used by a private admin client for those reads.
    """
    verb = str(method or "GET").upper()
    mutation = authorize_mutation(verb, headers)
    if verb in _UNSAFE_METHODS:
        return mutation
    if not public_read_only_enabled():
        return MutationAccess(True, "PRIVATE_OPERATOR_MODE", "Public read-only mode is disabled.")
    if not _private_read(path, service):
        return MutationAccess(True, "PUBLIC_READ", "Public read is allowed.")
    if _operator_authorized(headers):
        return MutationAccess(True, "OPERATOR_AUTHORIZED", "Operator bearer token accepted.")
    return MutationAccess(
        False,
        "PRIVATE_OPERATOR_SURFACE",
        "This surface is private to the QuantTerm operator.",
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
        "private_read_policy": "OPERATOR_TOKEN_REQUIRED" if enabled else "PRIVATE_OPERATOR",
        "unsafe_methods": sorted(_UNSAFE_METHODS),
        "live_money_unlocked": False,
    }
