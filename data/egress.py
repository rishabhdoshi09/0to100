"""Tell "this provider is broken" apart from "this machine cannot reach anything".

A development container behind a deny-all proxy produced this, over and over:

    nsearchives.nseindia.com  ProxyError: Tunnel connection failed: 403 Forbidden
    www.nseindia.com          ProxyError: Tunnel connection failed: 403 Forbidden
    api.bseindia.com          ProxyError: Tunnel connection failed: 403 Forbidden
    query1.finance.yahoo.com  403 Host not in allowlist

Read one line at a time, that is four independent provider outages. Read
together it is one fact about the machine, and the difference matters twice:
the desk should not tell an operator that NSE, BSE and Yahoo all broke on the
same afternoon, and it should stop retrying providers that cannot be reached
from here until something about the environment changes.

The rule is deliberately conservative. Unrelated providers must ALL fail, and
they must fail at the transport layer — a 404 from one endpoint is a provider
problem no matter how many other things are also down.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

# ── failure classes ────────────────────────────────────────────────────────
PROVIDER_DOWN = "PROVIDER_DOWN"
DNS_FAILURE = "DNS_FAILURE"
TIMEOUT = "TIMEOUT"
AUTH_FAILURE = "AUTH_FAILURE"
PARSER_CHANGED = "PARSER_CHANGED"
RATE_LIMITED = "RATE_LIMITED"
ENVIRONMENT_EGRESS_BLOCKED = "ENVIRONMENT_EGRESS_BLOCKED"
#: General internet works, but the exchanges specifically do not. A selective
#: allowlist looks nothing like an outage and needs its own name: the operator
#: has to change a firewall rule, not wait for NSE to come back.
MARKET_EGRESS_BLOCKED = "MARKET_EGRESS_BLOCKED"
UNCLASSIFIED = "UNCLASSIFIED"

#: Transport-layer failures. These say nothing about whether the provider is
#: healthy — only that we never got to ask it.
_TRANSPORT_CLASSES = frozenset({DNS_FAILURE, TIMEOUT, ENVIRONMENT_EGRESS_BLOCKED})

_PROXY_PATTERNS = (
    "tunnel connection failed",
    "unable to connect to proxy",
    "proxyerror",
    "not in allowlist",
    "connect_rejected",
    "forbidden by proxy",
    "egress",
)
_DNS_PATTERNS = (
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
    "getaddrinfo failed",
    "gaierror",
)
_TIMEOUT_PATTERNS = ("timed out", "timeout", "read timed out")
_AUTH_PATTERNS = (
    "401", "403 client error", "unauthorized", "invalid api key",
    "token is invalid", "access_token", "authentication",
)
_RATE_PATTERNS = ("429", "too many requests", "rate limit")
_PARSER_PATTERNS = (
    "expecting value", "jsondecodeerror", "no columns to parse",
    "unexpected character", "not well-formed", "xmlsyntaxerror",
)


def _has(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def classify_failure(error: str, *, status_code: int | None = None) -> str:
    """Classify ONE failure. Egress blocking is only decided in aggregate.

    A proxy refusal looks the same whether the destination is up or down, so
    it is reported here as a transport failure and only becomes
    ENVIRONMENT_EGRESS_BLOCKED once :func:`classify_environment` sees the same
    thing happening to unrelated providers.
    """
    text = str(error or "").lower()
    if status_code == 429 or _has(text, _RATE_PATTERNS):
        return RATE_LIMITED
    if _has(text, _PROXY_PATTERNS):
        return ENVIRONMENT_EGRESS_BLOCKED
    if _has(text, _DNS_PATTERNS):
        return DNS_FAILURE
    if _has(text, _TIMEOUT_PATTERNS):
        return TIMEOUT
    if status_code in (401, 403) or _has(text, _AUTH_PATTERNS):
        return AUTH_FAILURE
    if _has(text, _PARSER_PATTERNS):
        return PARSER_CHANGED
    if status_code is not None and 500 <= int(status_code) < 600:
        return PROVIDER_DOWN
    if text:
        return PROVIDER_DOWN
    return UNCLASSIFIED


@dataclass(frozen=True)
class ProbeResult:
    provider: str
    host: str
    ok: bool
    failure_class: str = ""
    detail: str = ""


@dataclass(frozen=True)
class EnvironmentVerdict:
    egress_blocked: bool
    reason: str
    blocked_hosts: tuple[str, ...] = ()
    reachable_hosts: tuple[str, ...] = ()
    #: "", ENVIRONMENT_EGRESS_BLOCKED or MARKET_EGRESS_BLOCKED.
    failure_class: str = ""

    @property
    def market_blocked(self) -> bool:
        """True for either shape: the desk cannot get market data either way."""
        return self.failure_class in (ENVIRONMENT_EGRESS_BLOCKED, MARKET_EGRESS_BLOCKED)

    def as_dict(self) -> dict[str, object]:
        return {
            "egress_blocked": self.egress_blocked,
            "failure_class": self.failure_class,
            "market_blocked": self.market_blocked,
            "reason": self.reason,
            "blocked_hosts": list(self.blocked_hosts),
            "reachable_hosts": list(self.reachable_hosts),
        }


#: Fewer than this many independent providers failing is not evidence about
#: the machine. Two unrelated organisations can be down at once; three
#: unrelated organisations failing identically at the transport layer is the
#: machine.
MIN_INDEPENDENT_FOR_ENVIRONMENT = 3


def classify_environment(
    results: Sequence[ProbeResult],
    *,
    control_hosts: Sequence[str] = (),
) -> EnvironmentVerdict:
    """Decide whether the machine, rather than the providers, is the problem.

    ``control_hosts`` are endpoints unrelated to the market. They are what makes
    "we are unplugged" separable from "the firewall allows the internet but not
    the exchanges" — two situations that produce identical provider errors and
    need completely different responses from the operator.
    """
    if not results:
        return EnvironmentVerdict(False, "nothing was probed")

    controls = {str(h) for h in control_hosts}
    market = [r for r in results if r.host not in controls]
    reachable = tuple(r.host for r in results if r.ok)

    if controls and market and all(not r.ok for r in market):
        control_up = [r for r in results if r.host in controls and r.ok]
        transport = [r for r in market if r.failure_class in _TRANSPORT_CLASSES]
        if control_up and len(transport) == len(market):
            return EnvironmentVerdict(
                False,
                "the internet is reachable but every market provider is refused "
                "at the transport layer; this host has a selective egress policy "
                "rather than an outage",
                blocked_hosts=tuple(r.host for r in market),
                reachable_hosts=tuple(r.host for r in control_up),
                failure_class=MARKET_EGRESS_BLOCKED,
            )

    if reachable:
        return EnvironmentVerdict(
            False,
            "at least one provider was reachable, so egress works",
            blocked_hosts=tuple(r.host for r in results if not r.ok),
            reachable_hosts=reachable,
        )

    failed = [r for r in results if not r.ok]
    organisations = {_organisation(r.host) for r in failed}
    if len(organisations) < MIN_INDEPENDENT_FOR_ENVIRONMENT:
        return EnvironmentVerdict(
            False,
            f"only {len(organisations)} independent provider(s) failed; not "
            "enough to blame the machine",
            blocked_hosts=tuple(r.host for r in failed),
        )

    transport = [r for r in failed if r.failure_class in _TRANSPORT_CLASSES]
    if len(transport) != len(failed):
        provider_side = sorted(
            {r.host for r in failed if r.failure_class not in _TRANSPORT_CLASSES}
        )
        return EnvironmentVerdict(
            False,
            "some failures came from the providers themselves, not the "
            f"transport: {', '.join(provider_side)}",
            blocked_hosts=tuple(r.host for r in failed),
        )

    return EnvironmentVerdict(
        True,
        f"{len(organisations)} unrelated providers all failed at the transport "
        "layer; this machine cannot reach the internet",
        blocked_hosts=tuple(r.host for r in failed),
        failure_class=ENVIRONMENT_EGRESS_BLOCKED,
    )


def _organisation(host: str) -> str:
    """Registrable-ish domain, so two NSE hostnames are not two providers."""
    parts = [p for p in str(host or "").lower().split(".") if p]
    if len(parts) <= 2:
        return ".".join(parts)
    # Handles the common two-label public suffixes this desk actually uses.
    if parts[-2] in ("co", "com", "net", "org", "gov") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def retry_is_futile(verdict: EnvironmentVerdict) -> bool:
    """Whether hammering providers again could possibly help.

    Retrying a blocked tunnel is not resilience, it is a busy loop that fills
    the log and hides the one fact worth reading. True for a selective market
    block as well: the exchanges are no more reachable for being adjacent to a
    working internet connection.
    """
    return bool(verdict.market_blocked)
